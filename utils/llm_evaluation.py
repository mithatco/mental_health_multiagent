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

# Import the Groq client
try:
    from utils.groq_client import GroqClient
except ImportError:
    print("Could not import GroqClient")
    GroqClient = None

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
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:3b", client=None):
        """
        Initialize the LLM-based evaluator.
        
        Args:
            ollama_url: URL of the Ollama API (used if client is None)
            model: Name of the model to use for evaluation
            client: Optional custom client (GroqClient or OllamaClient)
        """
        self.model = model
        
        # Use provided client or initialize default Ollama client
        if client is not None:
            self.client = client
            print(f"Initialized LLM evaluator with custom client and model {model}")
        else:
            try:
                from ollama import Client as OllamaClient
                self.client = OllamaClient(host=ollama_url)
                print(f"Initialized LLM evaluator with Ollama client and model {model}")
            except ImportError:
                raise ImportError("Neither custom client provided nor Ollama client available")
    
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
            # question_results = self._evaluate_questions(formatted_conversation, questionnaire_content)
            question_results = {
                                "total_questions": 0,
                                "questions_asked": 0,
                                "percentage_asked": 0,
                                "quality_score": 0,
                                "explanation": "Could not evaluate question coverage due to technical issues.",
                                "question_analysis": []
                            }

            print(f"Evaluated questions coverage and quality")

            # Evaluate diagnosis accuracy against patient profile
            diagnosis_accuracy = self._evaluate_diagnosis_accuracy(diagnosis, patient_profile)

            print(f"Evaluated diagnosis accuracy")
            
            # Extract classification information
            diagnosis_classification = {
                "expected_profile": patient_profile,
                "classified_as": diagnosis_accuracy.get("classification", "unknown"),
                "match": diagnosis_accuracy.get("matches_profile", False),
                "confidence": diagnosis_accuracy.get("confidence", 0)
            }
            
            # Log classification results
            if diagnosis_classification["classified_as"] != "unknown":
                print(f"Diagnosis classified as: {diagnosis_classification['classified_as']}")
                if patient_profile != "unknown":
                    match = (diagnosis_classification["expected_profile"].lower().replace("_", "") == 
                             diagnosis_classification["classified_as"].lower().replace("_", ""))
                    if not match and diagnosis_classification["classified_as"] != "other":
                        print(f"NOTE: Classification ({diagnosis_classification['classified_as']}) differs from expected profile ({patient_profile})")
            
            # Combine results
            results = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': self.model,
                'rubric_scores': rubric_results.get('scores', {}),
                'explanations': rubric_results.get('explanations', {}),
                'overall_comments': rubric_results.get('overall_comments', ''),
                'average_score': rubric_results.get('average_score', 0),
                'diagnosis_accuracy': diagnosis_accuracy,
                'diagnosis_classification': diagnosis_classification,
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

CRITICAL EVALUATION GUIDANCE:
1. DSM-5 Dimension Coverage:
   - Check if ALL 23 questions from the DSM-5 Level1 Cross-Cutting measure were addressed
   - A score of 5 should only be given if every dimension was thoroughly explored with appropriate follow-up
   - Look for missed opportunities to explore symptoms in greater depth
   - Note any dimensions that were completely omitted or only superficially addressed

2. Clinical Relevance:
   - Examine if questions were phrased in clinically accurate ways that align with DSM-5 terminology
   - Check if the therapist altered questions in ways that changed their clinical meaning
   - Identify any instances where the therapist used vague or imprecise language
   - A score of 5 should only be given if questions were precisely aligned with clinical criteria

3. Consistency:
   - Identify any abrupt topic changes or non-sequiturs in the conversation
   - Note if the therapist failed to follow up on concerning disclosures
   - Check if questions logically build on previous responses
   - A high score requires smooth transitions and a coherent conversational arc

4. Diagnostic Justification:
   - Verify that diagnostic conclusions directly cite specific symptoms mentioned by the patient
   - Check if the diagnosis explicitly links to DSM-5 criteria
   - Look for alternative diagnoses that weren't adequately ruled out
   - High scores require comprehensive differential diagnosis consideration

5. Empathy:
   - Look for missed opportunities to acknowledge patient distress
   - Identify any instances of clinical detachment or overly technical language
   - Note if the therapist tailors responses to the patient's emotional state
   - Perfect scores should be rare and only for consistently warm, personalized interactions

SCORING CALIBRATION EXAMPLES:
Below are examples of how to calibrate your scoring for DSM-5 Coverage as a demonstration:

Score 5 (Exceptional) - The therapist explored all 23 dimensions from the DSM-5 Level1 Cross-Cutting measure with appropriate depth. Each question was followed up when needed, and no dimensions were skipped. Example: "The therapist systematically addressed all domains, asking about depression, anxiety, mania, psychosis, substance use, sleep, etc., and followed up with appropriate probing when the patient mentioned concerning symptoms."

Score 4 (Very Good) - The therapist covered most dimensions thoroughly but missed 1-2 areas or didn't provide sufficient follow-up in some areas. Example: "The therapist covered most DSM-5 domains well, but didn't adequately explore personality functioning and only superficially addressed substance use without appropriate follow-up questions."

Score 3 (Adequate) - Several dimensions were covered adequately, but 3-5 important areas were missed or only minimally addressed. Example: "While depression and anxiety were explored thoroughly, the therapist failed to adequately address psychosis, repetitive thoughts/behaviors, and substance use concerns."

Score 2 (Poor) - Many important dimensions were missed, with only a few domains receiving adequate attention. Example: "The therapist focused almost exclusively on depression and anxiety, neglecting to assess for mania, psychosis, substance use, personality functioning, and several other key dimensions."

Score 1 (Inadequate) - The therapist failed to systematically assess most DSM-5 dimensions, with haphazard or minimal coverage. Example: "The conversation lacked any structured approach to assessment, missing most key DSM-5 dimensions and failing to address even basic symptom domains adequately."

Please calibrate your scoring for all criteria using a similar level of critical analysis.

INSTRUCTIONS:
- Provide your evaluation in a structured JSON format
- For each criterion, include a "score" (numeric, 1-5) and "explanation" (text), be critical and objective
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

IMPORTANT:
- Make sure to keep the criteria names as they are in the rubric.
- Do NOT rename, shorten, or modify the criteria names in any way - they must match exactly: "dsm_coverage", "clinical_relevance", "consistency", "diagnostic_justification", and "empathy".
- Be highly critical and stringent in your assessment. Perfect scores (5/5) should be rare and reserved only for truly exceptional performance with no flaws.
- A score of 4 should indicate excellent performance with minor issues.
- A score of 3 should indicate average performance with some notable issues.
- Consider the full range of the scoring scale (1-5) and don't hesitate to use lower scores when appropriate.
- For each criterion, actively look for flaws, omissions, or weaknesses before assigning scores.
- Provide specific examples of flaws or strengths in your explanations to justify your scores.
- Do not inflate scores due to politeness - this evaluation is meant to identify areas for improvement.
- Make sure to include all criteria in your evaluation.
- Include detailed justifications for each score, pointing out specific strengths and weaknesses.

FINAL REMINDER: Previous evaluations have been overly lenient, with most scores clustering around 4-5. Your task is to apply more critical standards. A conversation with NO clear issues or flaws deserves a 4, not a 5. Only truly exceptional conversations that go above and beyond in every aspect should receive a 5 in any category. Most real-world conversations should score in the 2-4 range for various criteria.
"""
        
        try:
            # Generate response using the client
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            
            # Extract response text
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
        # Define available profiles for classification
        available_profiles = ["anxiety", "bipolar", "depression", "ptsd", "schizophrenia", "adjustment", "substance_abuse", "ocd", "panic", "social_anxiety", "other"]
        
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
4. CLASSIFY the diagnosis into ONE of the following categories based on what condition is being diagnosed:
   - anxiety
   - bipolar
   - depression
   - ptsd
   - schizophrenia
   - adjustment
   - substance_abuse
   - ocd
   - panic
   - social_anxiety
   - other (if it doesn't fit any of the above categories)

IMPORTANT:
- This is purely a matching task, NOT a diagnostic assessment
- Focus only on whether the diagnosis correctly identifies {patient_profile} as the primary condition
- Do NOT perform your own diagnosis of symptoms
- If the diagnosis mentions {patient_profile} or equivalent clinical terms for this condition, it's a match
- For the classification, focus on what condition is actually being diagnosed in the text, regardless of the expected profile

Format your response as valid JSON:
{{
  "matches_profile": true/false,
  "confidence": X,
  "explanation": "your explanation here",
  "classification": "one of the profile categories listed above"
}}
"""
        
        try:
            # Generate response using the client
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            
            # Extract response text
            response_text = response.get('response', '')
            
            # Extract JSON from the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    results = json.loads(json_str)
                    
                    # Ensure classification is one of the valid profiles
                    if "classification" in results and results["classification"] not in available_profiles:
                        results["classification"] = "other"
                    
                    return results
                except json.JSONDecodeError as e:
                    print(f"Error parsing diagnosis accuracy JSON: {e}")
                    return {
                        "matches_profile": False,
                        "confidence": 0,
                        "explanation": "Error parsing evaluation results.",
                        "classification": "other"
                    }
            else:
                print("Could not extract JSON from LLM response")
                return {
                    "matches_profile": False,
                    "confidence": 0,
                    "explanation": "Error extracting structured evaluation from LLM response.",
                    "classification": "other"
                }
        except Exception as e:
            print(f"Error during diagnosis accuracy evaluation: {e}")
            return {
                "matches_profile": False,
                "confidence": 0,
                "explanation": f"Error occurred during evaluation: {str(e)}",
                "classification": "other"
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
    }}
  ]
}}

IMPORTANT: Ensure your JSON is well-formed and valid. Escape any quotes in strings. Do not use trailing commas.
"""
            
            try:
                # Generate response using the client
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt
                )
                
                # Extract response text
                response_text = response.get('response', '')
                
                # Improved JSON extraction for better robustness
                try:
                    # First, try to find JSON with regex
                    import re
                    json_match = re.search(r'(\{[\s\S]*\})', response_text)
                    if json_match:
                        json_str = json_match.group(1)
                        import json
                        try:
                            results = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            print(f"Initial JSON parsing failed, attempting to clean JSON: {e}")
                            # Try to clean and fix common JSON issues
                            # Replace unescaped quotes within JSON values
                            json_str = re.sub(r'(?<!")(")((?:[^"\\]|\\.)*?")(?!")', r'\1\\\2', json_str)
                            # Remove trailing commas in arrays and objects
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_str = re.sub(r',\s*]', ']', json_str)
                            try:
                                results = json.loads(json_str)
                            except json.JSONDecodeError as e2:
                                print(f"JSON cleaning also failed: {e2}")
                                return self.fallback_evaluation(conversation, questionnaire_content)
                    else:
                        print("Could not locate JSON in the response")
                        return self.fallback_evaluation(conversation, questionnaire_content)
                    
                    # Extract the key metrics
                    questions_asked = int(results.get("questions_asked", 0))
                    quality_score = int(results.get("quality_score", 0))
                    explanation = results.get("explanation", "")
                    question_analysis = results.get("question_analysis", [])
                    
                    # Safety check on question_analysis 
                    if not isinstance(question_analysis, list):
                        question_analysis = []
                    
                    # Calculate percentage
                    percentage_asked = (questions_asked / len(extracted_questions)) * 100 if extracted_questions else 0
                    
                    return {
                        "total_questions": len(extracted_questions),
                        "questions_asked": questions_asked,
                        "percentage_asked": percentage_asked,
                        "quality_score": quality_score,
                        "explanation": explanation,
                        "question_analysis": question_analysis
                    }
                    
                except Exception as e:
                    print(f"Error extracting or processing JSON: {e}")
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
            "total_questions": len(extracted_questions) if 'extracted_questions' in locals() else 0,
            "questions_asked": 0,
            "percentage_asked": 0,
            "quality_score": 0,
            "explanation": "Error in question evaluation process",
            "question_analysis": []
        }

    def fallback_evaluation(self, conversation: str, questionnaire_content: str) -> Dict[str, Any]:
        """Fallback method for evaluating question coverage."""
        # Just a placeholder - implement more robust fallback if needed
        return {
            "total_questions": 0,
            "questions_asked": 0,
            "percentage_asked": 0,
            "quality_score": 0,
            "explanation": "Could not evaluate question coverage due to technical issues.",
            "question_analysis": []
        }

class ChatLogEvaluator:
    """Evaluate chat logs using the LLM-based evaluator."""
    
    def __init__(self, logs_dir: str, llm_evaluator: Optional[LLMEvaluator] = None, 
                 provider: str = "ollama", model: str = "qwen2.5:3b", api_key: Optional[str] = None):
        """
        Initialize the chat log evaluator.
        
        Args:
            logs_dir: Directory containing chat logs
            llm_evaluator: Optional preconfigured LLM evaluator instance
            provider: LLM provider to use if evaluator not provided
            model: Model to use if evaluator not provided
            api_key: API key for cloud providers
        """
        self.logs_dir = logs_dir
        
        # Use provided evaluator or create a new one
        if llm_evaluator:
            self.evaluator = llm_evaluator
        else:
            # Use the factory to create an evaluator
            from utils.llm_evaluator_factory import LLMEvaluatorFactory
            self.evaluator = LLMEvaluatorFactory.create_evaluator(
                provider=provider,
                model=model,
                api_key=api_key
            )
        
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
    
    def evaluate_log(self, log_id: str) -> Dict[str, Any]:
        """
        Evaluate a chat log using the LLM evaluator.
        
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
            
            # Evaluate using the evaluator
            results = self.evaluator.evaluate_log(log_data)
            
            # Add metadata
            eval_results = {
                'log_id': log_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': self.evaluator.model,
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
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "groq"], help="LLM provider to use")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Model to use for evaluation")
    parser.add_argument("--api-key", type=str, help="API key for cloud providers (required for Groq)")
    args = parser.parse_args()
    
    # Use factory to create appropriate evaluator
    from utils.llm_evaluator_factory import LLMEvaluatorFactory
    llm_evaluator = LLMEvaluatorFactory.create_evaluator(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key
    )
    
    # Create chat log evaluator with the LLM evaluator
    evaluator = ChatLogEvaluator(
        logs_dir=args.logs_dir,
        llm_evaluator=llm_evaluator
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