"""
LLM-based evaluation of mental health conversations.
This module uses an LLM to directly evaluate therapist-client conversations.
"""

import os
import sys
import json
import re
import time
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the Ollama client
try:
    from utils.ollama_client import OllamaClient
except ImportError:
    print("Could not import OllamaClient, trying relative import...")
    try:
        from .ollama_client import OllamaClient
    except ImportError:
        print("ERROR: Could not import OllamaClient")
        OllamaClient = None

# Define the evaluation rubric as a constant
EVALUATION_RUBRIC = """
Evaluation Rubric for Simulated Therapist-Client Conversations
Rate each conversation based on the following criteria (scale 1–5):

1. Completeness of DSM-5 Dimension Coverage

(5) All relevant DSM-5 Level 1 dimensions are thoroughly explored through appropriate questions.
(4) Most DSM-5 dimensions are addressed, with minor omissions.
(3) Some DSM-5 dimensions are missing or only superficially covered.
(2) Few DSM-5 dimensions are explored, leading to significant gaps.
(1) DSM-5 dimensions are barely addressed or completely ignored.

2. Clinical Relevance and Accuracy of Questions

(5) Questions precisely reflect DSM-5 criteria, clearly targeting clinical symptoms.
(4) Questions generally align with DSM-5 criteria with slight inaccuracies or vague phrasing.
(3) Questions somewhat reflect DSM-5 criteria, but several inaccuracies exist.
(2) Questions poorly reflect DSM-5 criteria; most are clinically irrelevant or confusing.
(1) Questions are unrelated or inappropriate for clinical assessment.

3. Consistency and Logical Flow

(5) The dialogue flows logically, each question naturally follows from previous responses.
(4) Minor logical inconsistencies exist but don't significantly disrupt conversation flow.
(3) Noticeable logical inconsistencies occasionally disrupt coherence and understanding.
(2) Frequent inconsistencies severely impact logical flow and conversational coherence.
(1) Dialogue appears random or highly disconnected, lacking logical progression.

4. Diagnostic Justification and Explainability

(5) Diagnoses clearly align with DSM-5 responses, and the reasoning behind each diagnostic decision is explicitly stated and clinically sound.
(4) Diagnoses generally align with responses, with minor ambiguity in justification.
(3) Diagnoses somewhat align but have unclear or partially flawed justifications.
(2) Diagnoses rarely align with responses; justifications are superficial or unclear.
(1) Diagnoses have no clear connection to conversation content or lack justification entirely.

5. Empathy, Naturalness, and Professionalism

(5) Therapist responses show consistent empathy, natural conversational style, and appropriate professional tone.
(4) Mostly empathetic and professional, with minor unnatural or robotic moments.
(3) Occasional empathy; interactions sometimes robotic, impersonal, or inappropriate.
(2) Rarely empathetic or natural; responses generally impersonal, abrupt, or inappropriate.
(1) Completely lacking empathy, professional tone, or natural conversational flow.
"""

class LLMEvaluator:
    """Evaluate mental health conversations using an LLM with a custom rubric."""
    
    def __init__(self, ollama_url="http://localhost:11434", model="qwen2.5:3b"):
        """
        Initialize the LLM evaluator.
        
        Args:
            ollama_url: URL for the Ollama API
            model: Name of the model to use for evaluation
        """
        self.ollama_url = ollama_url
        self.model = model
        self.client = OllamaClient(base_url=ollama_url)
    
    def evaluate_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a conversation log using the LLM.
        
        Args:
            log_data: Dictionary containing the conversation log
        
        Returns:
            Dictionary containing evaluation results
        """
        start_time = time.time()
        
        # Extract conversation and diagnosis
        conversation = log_data.get('conversation', [])
        diagnosis = log_data.get('diagnosis', '')
        
        # Extract questionnaire information and try to load the content
        questionnaire_name = log_data.get('questionnaire', 'Unknown questionnaire')
        questionnaire_content = self._load_questionnaire_content(questionnaire_name)
        
        # Extract patient profile if available
        patient_profile = None
        if 'metadata' in log_data and 'patient_profile' in log_data['metadata']:
            patient_profile = log_data['metadata']['patient_profile']
        
        # Format conversation for evaluation
        formatted_conversation = self._format_conversation(conversation)
        
        # Evaluate based on rubric
        rubric_results = self._evaluate_with_rubric(formatted_conversation, diagnosis, questionnaire_content)
        
        # Check diagnosis accuracy if patient profile is available
        diagnosis_accuracy = None
        if patient_profile:
            diagnosis_accuracy = self._evaluate_diagnosis_accuracy(diagnosis, patient_profile)
        
        # Format results
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': self.model,
            'evaluation_time': time.time() - start_time,
            'rubric_scores': rubric_results.get('scores', {}),
            'explanations': rubric_results.get('explanations', {}),
            'overall_comments': rubric_results.get('overall_comments', ''),
            'average_score': rubric_results.get('average_score', 0),
            'rubric': EVALUATION_RUBRIC
        }
        
        if diagnosis_accuracy:
            results['diagnosis_accuracy'] = diagnosis_accuracy
        
        return results
    
    def _load_questionnaire_content(self, questionnaire_name: str) -> str:
        """
        Load questionnaire content from file.
        
        Args:
            questionnaire_name: Name of the questionnaire file
        
        Returns:
            Content of the questionnaire file or a placeholder if not found
        """
        # Define possible directories where questionnaires might be stored
        possible_dirs = [
            "documents/questionnaires",
            "documents",
            "questionnaires"
        ]
        
        # Try to find the questionnaire file
        for dir_path in possible_dirs:
            # Check relative to current directory
            file_path = os.path.join(dir_path, questionnaire_name)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as f:
                        return f.read()
                except Exception as e:
                    print(f"Error reading questionnaire file {file_path}: {e}")
                    break
            
            # Check relative to project root (assuming we're in a subdirectory)
            root_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), dir_path, questionnaire_name)
            if os.path.isfile(root_file_path):
                try:
                    with open(root_file_path, 'r') as f:
                        return f.read()
                except Exception as e:
                    print(f"Error reading questionnaire file {root_file_path}: {e}")
                    break
        
        # If questionnaire file not found, return a description of what would be expected
        return f"Questionnaire '{questionnaire_name}' (specific content not available)"
    
    def _format_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """
        Format a conversation for LLM evaluation.
        
        Args:
            conversation: List of conversation messages
        
        Returns:
            Formatted conversation as a string
        """
        formatted = []
        
        for msg in conversation:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Clean up content to remove thinking sections
            if '<think>' in content:
                sections = content.split('<think>')
                visible_content = sections[0]
                
                # Check if there's a closing tag
                if '</think>' in content:
                    # Get content after the closing tag as well
                    parts = content.split('</think>')
                    if len(parts) > 1:
                        visible_content += parts[1]
                
                content = visible_content.strip()
            
            # Format based on role
            if role.lower() == 'system':
                # Skip system messages in the evaluation
                continue
            elif role.lower() == 'assistant':
                formatted.append(f"Therapist: {content}")
            elif role.lower() == 'patient':
                formatted.append(f"Patient: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n\n".join(formatted)
    
    def _evaluate_with_rubric(self, conversation: str, diagnosis: str, questionnaire_content: str) -> Dict[str, Any]:
        """
        Evaluate a conversation using the defined rubric.
        
        Args:
            conversation: Formatted conversation
            diagnosis: Diagnosis provided by the therapist
            questionnaire_content: Content of the questionnaire used
        
        Returns:
            Dictionary containing scores and explanations
        """
        prompt = f"""
You are an expert clinical supervisor evaluating a simulated therapeutic assessment conversation between a mental health professional and a patient. 

Please carefully analyze the following conversation and final diagnosis, then rate it according to the rubric below.

QUESTIONNAIRE USED:
----------------
{questionnaire_content}
----------------

The conversation should be evaluated based on how well it covers the questions and areas in this questionnaire.

For each of the 5 criteria, provide:
1. A numeric score (1-5)
2. A brief explanation for your score

CONVERSATION:
----------------
{conversation}
----------------

DIAGNOSIS:
----------------
{diagnosis}
----------------

{EVALUATION_RUBRIC}

INSTRUCTIONS:
- Provide your evaluation in a structured JSON format
- For each criterion, include a "score" (numeric, 1-5) and "explanation" (text)
- Include an "overall_comments" section with your general assessment

Please format your response as valid JSON with the following structure:
{{
  "criteria": {{
    "dsm_coverage": {{ "score": X, "explanation": "..." }},
    "clinical_relevance": {{ "score": X, "explanation": "..." }},
    "consistency": {{ "score": X, "explanation": "..." }},
    "diagnostic_justification": {{ "score": X, "explanation": "..." }},
    "empathy": {{ "score": X, "explanation": "..." }}
  }},
  "overall_comments": "..."
}}

Your evaluation should be objective, fair, and based solely on the provided conversation, questionnaire, and rubric.
"""
        
        try:
            # Fixed: Pass parameters in the correct order for OllamaClient.generate
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            
            # The client returns a dict with 'response' key, so extract that
            response_text = response.get('response', '')
            
            # Extract JSON from the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    results = json.loads(json_str)
                    
                    # Map the scores and explanations for easier access
                    scores = {}
                    explanations = {}
                    
                    if "criteria" in results:
                        for key, value in results["criteria"].items():
                            scores[key] = value.get("score", 0)
                            explanations[key] = value.get("explanation", "")
                    
                    # Calculate average score ourselves instead of relying on the LLM
                    if scores:
                        average_score = sum(scores.values()) / len(scores)
                    else:
                        average_score = 0
                    
                    return {
                        "scores": scores,
                        "explanations": explanations,
                        "overall_comments": results.get("overall_comments", ""),
                        "average_score": round(average_score, 1)  # Round to 1 decimal place
                    }
                except json.JSONDecodeError as e:
                    print(f"Error parsing LLM response JSON: {e}")
                    print(f"Response was: {json_str}")
                    return {
                        "scores": {},
                        "explanations": {},
                        "overall_comments": "Error parsing evaluation results.",
                        "average_score": 0
                    }
            else:
                print("Could not extract JSON from LLM response")
                print(f"Response was: {response_text}")
                return {
                    "scores": {},
                    "explanations": {},
                    "overall_comments": "Error extracting structured evaluation from LLM response.",
                    "average_score": 0
                }
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            return {
                "scores": {},
                "explanations": {},
                "overall_comments": f"Error occurred during evaluation: {str(e)}",
                "average_score": 0
            }
    
    def _evaluate_diagnosis_accuracy(self, diagnosis: str, patient_profile: str) -> Dict[str, Any]:
        """
        Evaluate whether the diagnosis matches the patient profile.
        
        Args:
            diagnosis: Diagnosis text provided by the therapist
            patient_profile: Name of the patient profile (e.g., "depression", "anxiety")
            
        Returns:
            Dictionary containing accuracy assessment
        """
        prompt = f"""
You are an expert psychiatrist EVALUATING a diagnosis's accuracy compared to a known condition.

This is strictly a comparison task between the diagnosis given and the expected profile.

DIAGNOSIS:
----------------
{diagnosis}
----------------

EXPECTED PROFILE: {patient_profile}

Your task: 
1. Determine if the diagnosis EXPLICITLY identifies or aligns with the expected profile ({patient_profile}).
2. Rate your confidence in this assessment (on a scale of 1-5)
3. Provide a brief explanation for why they match or don't match

IMPORTANT:
- This is purely a matching task, NOT a diagnostic assessment
- Focus only on whether the diagnosis correctly identifies {patient_profile} as the primary condition
- Do NOT perform your own diagnosis of symptoms
- If the diagnosis mentions {patient_profile} or equivalent clinical terms for this condition, it's a match

Format your response as valid JSON:
{{
  "matches_profile": true/false,
  "confidence": X,
  "explanation": "your explanation here"
}}
"""
        
        try:
            # Fixed: Pass parameters in the correct order for OllamaClient.generate
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            
            # The client returns a dict with 'response' key, so extract that
            response_text = response.get('response', '')
            
            # Extract JSON from the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    results = json.loads(json_str)
                    return results
                except json.JSONDecodeError as e:
                    print(f"Error parsing diagnosis accuracy JSON: {e}")
                    return {
                        "matches_profile": False,
                        "confidence": 0,
                        "explanation": "Error parsing evaluation results."
                    }
            else:
                print("Could not extract JSON from LLM response")
                return {
                    "matches_profile": False,
                    "confidence": 0,
                    "explanation": "Error extracting structured evaluation from LLM response."
                }
        except Exception as e:
            print(f"Error during diagnosis accuracy evaluation: {e}")
            return {
                "matches_profile": False,
                "confidence": 0,
                "explanation": f"Error occurred during evaluation: {str(e)}"
            }

class ChatLogEvaluator:
    """Evaluate chat logs using the LLM-based evaluator."""
    
    def __init__(self, logs_dir: str, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:3b"):
        """
        Initialize the chat log evaluator.
        
        Args:
            logs_dir: Directory containing chat logs
            ollama_url: URL for the Ollama API
            model: Name of the model to use for evaluation
        """
        self.logs_dir = logs_dir
        self.ollama_url = ollama_url
        self.model = model
        self.evaluator = LLMEvaluator(ollama_url=ollama_url, model=model)
        
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
    
    def evaluate_log(self, log_id: str, model: str = None) -> Dict[str, Any]:
        """
        Evaluate a chat log using the LLM evaluator.
        
        Args:
            log_id: ID of the log file
            model: Optional model override to use for this evaluation
        
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
            
            # Use specified model if provided, otherwise use the default
            eval_model = model if model else self.model
            
            # Create temporary evaluator if needed
            if eval_model != self.model:
                temp_evaluator = LLMEvaluator(ollama_url=self.ollama_url, model=eval_model)
                results = temp_evaluator.evaluate_log(log_data)
            else:
                results = self.evaluator.evaluate_log(log_data)
            
            # Add metadata
            eval_results = {
                'log_id': log_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': eval_model,
                'evaluation': results
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

def example_usage():
    """Example usage of the LLM evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate mental health conversations")
    parser.add_argument("log_id", help="ID of the log file to evaluate")
    parser.add_argument("--logs-dir", type=str, default="./chat_logs", help="Directory containing chat logs")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Model to use for evaluation")
    args = parser.parse_args()
    
    evaluator = ChatLogEvaluator(
        logs_dir=args.logs_dir,
        model=args.model
    )
    
    print(f"Evaluating log {args.log_id}...")
    results = evaluator.evaluate_log(args.log_id)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("Evaluation results:")
        print(f"  Average score: {results['evaluation'].get('average_score', 'N/A')}")
        print("\nScores by criteria:")
        for criterion, score in results['evaluation'].get('rubric_scores', {}).items():
            print(f"  {criterion}: {score}")
        
        print("\nOverall comments:")
        print(results['evaluation'].get('overall_comments', 'No comments provided'))
        
        if 'diagnosis_accuracy' in results['evaluation']:
            accuracy = results['evaluation']['diagnosis_accuracy']
            match_status = "matches" if accuracy.get('matches_profile', False) else "does not match"
            print(f"\nDiagnosis {match_status} expected profile with confidence {accuracy.get('confidence', 0)}/5")
            print(f"Explanation: {accuracy.get('explanation', 'No explanation provided')}")

if __name__ == "__main__":
    example_usage()