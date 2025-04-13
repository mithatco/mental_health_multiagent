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
        Initialize the LLM-based evaluator.
        
        Args:
            ollama_url: URL of the Ollama API
            model: Name of the model to use for evaluation
        """
        from ollama import Client as OllamaClient
        self.client = OllamaClient(host=ollama_url)
        self.model = model
        print(f"Initialized LLM evaluator with model {model}")
    
    def evaluate_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a conversation log using the LLM.
        
        Args:
            log_data: Dictionary containing conversation data
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            start_time = time.time()
            # Extract necessary components from log data
            conversation = log_data.get('conversation', [])
            diagnosis = log_data.get('diagnosis', 'No diagnosis provided')
            questionnaire = log_data.get('questionnaire', 'Unknown questionnaire')
            
            # Extract patient profile from metadata
            patient_profile = 'unknown'
            if 'metadata' in log_data and 'patient_profile' in log_data['metadata']:
                patient_profile = log_data['metadata']['patient_profile']
            
            # Format conversation for evaluation
            formatted_conversation = self._format_conversation(conversation)

            print(f"Formatted conversation")
            
            # Load questionnaire content
            questionnaire_content = self._load_questionnaire_content(questionnaire)

            print(f"Loaded questionnaire content")
            
            # Evaluate based on rubric
            rubric_results = self._evaluate_with_rubric(formatted_conversation, diagnosis, questionnaire_content)
            
            print(f"Evaluated based on rubric")
            
            # Evaluate questions coverage and quality
            question_results = self._evaluate_questions(formatted_conversation, questionnaire_content)

            print(f"Evaluated questions coverage and quality")

            # Evaluate diagnosis accuracy against patient profile
            diagnosis_accuracy = self._evaluate_diagnosis_accuracy(diagnosis, patient_profile)

            print(f"Evaluated diagnosis accuracy")
            
            # Combine results
            results = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': self.model,
                'rubric_scores': rubric_results.get('scores', {}),
                'explanations': rubric_results.get('explanations', {}),
                'overall_comments': rubric_results.get('overall_comments', ''),
                'average_score': rubric_results.get('average_score', 0),
                'diagnosis_accuracy': diagnosis_accuracy,
                'question_evaluation': question_results,
                'evaluation_time': time.time() - start_time
            }

            return results
        except Exception as e:
            print(f"Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _load_questionnaire_content(self, questionnaire_name: str) -> str:
        """
        Load the content of the specified questionnaire.
        
        Args:
            questionnaire_name: Name or path of the questionnaire file
            
        Returns:
            String containing the questionnaire content
        """
        questionnaire_content = "No questionnaire content available"
        
        try:
            # Get project root path
            import os
            script_dir = os.path.dirname(os.path.realpath(__file__))
            project_root = os.path.dirname(script_dir)
            
            # Try different possible locations for the questionnaire
            possible_paths = [
                questionnaire_name,  # Direct path
                os.path.join(project_root, questionnaire_name),  # Relative to project root
                os.path.join(project_root, "documents", "questionnaires", questionnaire_name),  # In questionnaires dir
                os.path.join(project_root, "documents", "questionnaires", os.path.basename(questionnaire_name))  # Just the filename
            ]
            
            # Also try adding .txt extension if not already present
            for path in list(possible_paths):  # Create a copy to avoid modifying during iteration
                if not path.endswith(('.txt', '.pdf')):
                    possible_paths.append(f"{path}.txt")
            
            # Try each path
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found questionnaire at: {path}")
                    
                    if path.endswith('.pdf'):
                        # Use the DocumentProcessor for PDF files
                        try:
                            from utils.document_processor import DocumentProcessor
                            document = DocumentProcessor.load_document(path)
                            if document and document.content:
                                return document.content
                        except Exception as e:
                            print(f"Error loading PDF questionnaire: {e}")
                    else:
                        # Regular text file
                        with open(path, 'r') as f:
                            return f.read()
            
            print(f"Questionnaire not found: {questionnaire_name}")
            print(f"Tried paths: {possible_paths}")
            
        except Exception as e:
            print(f"Error loading questionnaire content: {e}")
        
        return questionnaire_content
    
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

    def _evaluate_questions(self, conversation: str, questionnaire_content: str) -> Dict[str, Any]:
        """
        Evaluate how well the assistant covered the questions from the questionnaire.
        
        Args:
            conversation: Formatted conversation
            questionnaire_content: Content of the questionnaire used
            
        Returns:
            Dictionary containing percentage of questions asked and quality score
        """
        # Use the existing extract_questions_from_text function to get all questions
        try:
            from utils.document_processor import extract_questions_from_text
            extracted_questions = extract_questions_from_text(questionnaire_content)
            
            # Log the extracted questions for debugging
            print(f"Extracted {len(extracted_questions)} questions from questionnaire:")
            for i, q in enumerate(extracted_questions):
                print(f"  Question {i+1}: {q}")
            
            if not extracted_questions or len(extracted_questions) < 5:
                print("Warning: Couldn't extract enough questions from the questionnaire. Falling back to LLM extraction.")
                return self.fallback_evaluation(conversation, questionnaire_content)
            
            # Extract therapist lines from the conversation
            therapist_lines = []
            lines = conversation.split('\n')
            
            # Skip the first line if it's from the therapist (introduction)
            start_index = 1 if lines and lines[0].lower().startswith('therapist:') else 0
            
            # Process all lines except possibly the first and last
            for i, line in enumerate(lines[start_index:], start_index):
                if line.lower().startswith('therapist:'):
                    # Skip the last therapist line if it looks like a closing statement
                    if i == len(lines) - 1 or i == len(lines) - 2:
                        # Check if it's a closing remark (shorter and contains certain keywords)
                        content = line[line.index(':') + 1:].strip().lower()
                        closing_keywords = ['thank', 'appreciation', 'recommend', 'follow up', 'follow-up', 
                                          'appointment', 'session', 'goodbye', 'diagnosis']
                        
                        if any(keyword in content for keyword in closing_keywords) or len(content) < 100:
                            continue  # Skip this line as it's likely a closing remark
                    
                    # Extract the content after "Therapist:"
                    content = line[line.index(':') + 1:].strip()
                    therapist_lines.append(content)
            
            print(f"Extracted {len(therapist_lines)} therapist lines from conversation")
            
            # Now let the LLM evaluate the question coverage
            prompt = f"""
You are an expert clinical supervisor evaluating how well a mental health professional covered a standardized questionnaire in a patient interview.

I need you to:
1. Analyze which questions from the questionnaire were asked during the conversation
2. Calculate what percentage of the questionnaire was covered
3. Rate the quality of how questions were asked on a scale of 1-5

QUESTIONNAIRE QUESTIONS:
---
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(extracted_questions)])}
---

THERAPIST'S QUESTIONS/STATEMENTS:
---
{chr(10).join([f"- {line}" for line in therapist_lines])}
---

For your evaluation, please:

1. For EACH question in the questionnaire, determine if the therapist asked it (or a reasonable equivalent).
   - A question counts as "asked" if the therapist conveyed the same intent, even if phrased differently
   - If you're unsure, mark it as not asked

2. Rate the quality of how questions were asked on a scale of 1-5:
   1: Poor (questions were significantly altered in ways that changed their meaning)
   2: Fair (questions were recognizable but lost important nuance)
   3: Good (questions preserved the essential meaning but had minor changes)
   4: Very Good (questions were asked with minimal alterations)
   5: Excellent (questions were asked verbatim or with appropriate clinical adaptation)

3. Provide a brief explanation of your assessment

FORMAT YOUR RESPONSE AS VALID JSON:
{{
  "questions_asked": [integer number of questions correctly asked],
  "quality_score": [integer 1-5],
  "explanation": [string - your explanation],
  "question_analysis": [
    {{
      "original": "Original question text",
      "asked": true/false,
      "therapist_version": "How the therapist phrased it (if asked)" or "Not asked"
    }},
    ... MAKE SURE TO INCLUDE ALL QUESTIONS ...
  ]
}}
"""
            
            try:
                # Generate response using the LLM
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt
                )
                
                # Extract text from response
                response_text = response.get('response', '')
                
                # Extract JSON from the response
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        results = json.loads(json_str)
                        
                        # Extract the key metrics
                        questions_asked = int(results.get("questions_asked", 0))
                        quality_score = int(results.get("quality_score", 0))
                        explanation = results.get("explanation", "")
                        question_analysis = results.get("question_analysis", [])
                        
                        # Calculate percentage
                        percentage_asked = (questions_asked / len(extracted_questions)) * 100
                        
                        # Log the calculation details
                        # print(f"Question analysis calculation from LLM:")
                        # print(f"Total questions: {len(extracted_questions)}")
                        # print(f"Questions asked: {questions_asked}")
                        # print(f"Percentage asked: {percentage_asked:.2f}%")
                        
                        return {
                            "total_questions": len(extracted_questions),
                            "questions_asked": questions_asked,
                            "percentage_asked": percentage_asked,
                            "quality_score": quality_score,
                            "explanation": explanation,
                            "question_analysis": question_analysis
                        }
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing question evaluation JSON: {e}")
                        print(f"Response was: {json_str}")
                        return self.fallback_evaluation(conversation, questionnaire_content)
                else:
                    print("Could not extract JSON from question evaluation response")
                    print(f"Response was: {response_text}")
                    return self.fallback_evaluation(conversation, questionnaire_content)
                    
            except Exception as e:
                print(f"Error during LLM evaluation: {e}")
                return self.fallback_evaluation(conversation, questionnaire_content)
                
        except Exception as e:
            print(f"Error in question evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "total_questions": len(extracted_questions),
            "questions_asked": 0,
            "percentage_asked": 0,
            "quality_score": 0,
            "explanation": "",
            "question_analysis": []
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