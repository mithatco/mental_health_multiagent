"""
Utility for evaluating chat logs using Ragas with Ollama models.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

# Try importing the Ragas evaluator, fall back to Ollama evaluator if not available
try:
    from .ragas_evaluation import RagasEvaluator as Evaluator
    USING_RAGAS = True
    print("Using Ragas for evaluation")
except ImportError:
    from .ollama_evaluation import OllamaEvaluator as Evaluator
    USING_RAGAS = False
    print("Ragas not available, falling back to OllamaEvaluator")

class ChatLogEvaluator:
    """Evaluate chat logs using Ragas with Ollama models."""
    
    def __init__(self, logs_dir: str, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:3b"):
        """
        Initialize the chat log evaluator.
        
        Args:
            logs_dir: Directory containing chat logs
            ollama_url: URL for the Ollama API
            model: Name of the model to use for evaluation
        """
        self.logs_dir = logs_dir
        try:
            self.evaluator = Evaluator(ollama_url=ollama_url, model=model)
            self.using_ragas = USING_RAGAS
        except Exception as e:
            print(f"Error initializing evaluator: {e}")
            # If Ragas fails, try to fall back to OllamaEvaluator
            if USING_RAGAS:
                print("Falling back to OllamaEvaluator")
                from .ollama_evaluation import OllamaEvaluator
                self.evaluator = OllamaEvaluator(ollama_url=ollama_url, model=model)
                self.using_ragas = False
    
    def get_log_path(self, log_id: str) -> str:
        """
        Get the full path to a log file.
        
        Args:
            log_id: ID of the log file
        
        Returns:
            Full path to the log file
        """
        # Check if log_id contains .json extension
        if not log_id.endswith('.json'):
            log_id += '.json'
        
        # First check if file exists in the main logs directory
        filepath = os.path.join(self.logs_dir, log_id)
        
        # If not found, check in batch subdirectories
        if not os.path.isfile(filepath):
            for subdir in os.listdir(self.logs_dir):
                subdir_path = os.path.join(self.logs_dir, subdir)
                if os.path.isdir(subdir_path):
                    potential_file = os.path.join(subdir_path, log_id)
                    if os.path.isfile(potential_file):
                        filepath = potential_file
                        break
        
        return filepath
    
    def extract_qa_pairs(self, conversation: List[Dict[str, str]]) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract question-answer pairs from a conversation.
        
        Args:
            conversation: List of conversation messages
        
        Returns:
            Tuple of (questions, responses, contexts)
        """
        questions = []
        responses = []
        contexts = []
        
        # Extract patient profile from metadata or system message
        context = None
        
        # First try to find the patient profile in system messages
        for message in conversation:
            if message.get('role') == 'system':
                system_content = message.get('content', '')
                if system_content:
                    context = system_content
                    break
        
        # If no system message, try to extract from the content
        if not context:
            for message in conversation:
                content = message.get('content', '')
                # Look for typical profile descriptions
                if 'profile' in content.lower() and len(content) < 2000:  # Not too long
                    context = content
                    break
            
        # Extract relevant conversation information as additional context
        diagnosis = ""
        for message in conversation:
            if message.get('role') == 'assistant' and len(message.get('content', '')) > 500:
                # This is likely the diagnosis which provides useful context
                content = message.get('content', '')
                if "diagnosis" in content.lower() or "assessment" in content.lower():
                    diagnosis = content
                    break
        
        if diagnosis and not context:
            context = diagnosis
        elif diagnosis and context:
            context = context + "\n\n" + diagnosis
        
        # Extract question-answer pairs
        for i in range(len(conversation) - 1):
            current = conversation[i]
            next_msg = conversation[i + 1]
            
            # Clean the message content
            current_content = current.get('content', '')
            next_content = next_msg.get('content', '')
            
            # Extract only real question and answer, stripping <think> sections
            if current.get('role') == 'assistant' and next_msg.get('role') == 'patient':
                # Clean assistant question - remove thinking and formatting
                if '<think>' in current_content:
                    clean_question = current_content.split('<think>')[0].strip()
                else:
                    clean_question = current_content
                
                # Clean patient response - remove thinking and formatting
                if '<think>' in next_content:
                    parts = next_content.split('</think>')
                    if len(parts) > 1:
                        clean_response = parts[1].strip()
                    else:
                        clean_response = next_content.replace('<think>', '').replace('</think>', '').strip()
                else:
                    clean_response = next_content
                
                # Skip empty content
                if clean_question and clean_response:
                    questions.append(clean_question)
                    responses.append(clean_response)
                    if context:
                        contexts.append(context)
        
        # If we found questions but no context, create a more meaningful default context
        if questions and not contexts:
            default_context = """
This is a mental health assessment conversation between a clinician and a patient. 
The clinician is asking standardized assessment questions, and the patient is describing 
their symptoms and experiences. The goal is to evaluate the patient's mental health condition 
and provide appropriate support and treatment recommendations.
"""
            contexts = [default_context] * len(questions)
        
        return questions, responses, contexts if contexts else None
    
    def evaluate_log(self, log_id: str) -> Dict[str, Any]:
        """
        Evaluate a chat log.
        
        Args:
            log_id: ID of the log file
        
        Returns:
            Evaluation results
        """
        filepath = self.get_log_path(log_id)
        
        if not os.path.isfile(filepath):
            return {'error': 'Log file not found'}
        
        try:
            # Load log file
            with open(filepath, 'r') as f:
                log_data = json.load(f)
            
            # Extract conversation
            conversation = log_data.get('conversation', [])
            
            # Extract patient profile from metadata if available
            context_from_metadata = None
            if 'metadata' in log_data and 'patient_profile' in log_data['metadata']:
                profile_name = log_data['metadata']['patient_profile']
                # Look for profile file in profiles directory
                profile_dir = os.path.join(os.path.dirname(os.path.dirname(self.logs_dir)), "profiles")
                profile_path = os.path.join(profile_dir, f"{profile_name}.txt")
                if os.path.exists(profile_path):
                    with open(profile_path, 'r') as f:
                        context_from_metadata = f.read()
            
            # Extract question-answer pairs
            questions, responses, contexts = self.extract_qa_pairs(conversation)
            
            # If we have context from metadata but not from conversation, use it
            if context_from_metadata and not contexts:
                contexts = [context_from_metadata] * len(questions)
            
            if not questions or not responses:
                return {'error': 'No question-answer pairs found in conversation'}
            
            # Find the diagnosis message (typically the last message from the assistant)
            diagnosis_index = None
            rag_context = None
            
            for i in range(len(conversation) - 1, -1, -1):
                msg = conversation[i]
                if msg.get('role') == 'assistant' and 'rag_usage' in msg:
                    # Found the diagnosis with RAG context
                    rag_context = msg.get('rag_usage', {}).get('accessed_documents', [])
                    
                    # Find the corresponding index in our responses list
                    # This is a bit tricky since extract_qa_pairs might filter some messages
                    # For simplicity, assume diagnosis is the last response
                    diagnosis_index = len(responses) - 1
                    break
            
            # Run evaluation with progress tracking
            start_time = time.time()
            try:
                # Pass diagnosis index and RAG context to evaluation
                results = self.evaluator.evaluate_responses(
                    questions, responses, contexts, 
                    use_rubrics=True,
                    use_standard_metrics=True,
                    diagnosis_index=diagnosis_index,
                    rag_context=rag_context
                )
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                return {
                    'error': f'Error evaluating log: {str(e)}', 
                    'details': error_details
                }
                    
            evaluation_time = time.time() - start_time
            
            # Add metadata
            eval_results = {
                'log_id': log_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': getattr(self.evaluator, 'model', 'unknown'),
                'using_ragas': self.using_ragas,
                'metrics': results,
                'evaluation_time': evaluation_time,
                'question_count': len(questions)
            }
            
            # Save evaluation results to log file
            log_data['evaluation'] = eval_results
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            return eval_results
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {'error': f'Error evaluating log: {str(e)}', 'details': error_details}
    
    def get_evaluation_status(self, log_id: str) -> Dict[str, Any]:
        """
        Get the evaluation status and results for a log.
        
        Args:
            log_id: ID of the log file
        
        Returns:
            Evaluation status and results if available
        """
        filepath = self.get_log_path(log_id)
        
        if not os.path.isfile(filepath):
            return {'status': 'error', 'message': 'Log file not found'}
        
        try:
            # Load log file
            with open(filepath, 'r') as f:
                log_data = json.load(f)
            
            # Check if evaluation exists
            if 'evaluation' in log_data:
                return {
                    'status': 'completed',
                    'results': log_data['evaluation']
                }
            else:
                return {'status': 'not_evaluated'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
