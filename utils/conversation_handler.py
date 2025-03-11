import time
from typing import List, Dict, Any

class ConversationHandler:
    def __init__(self, assistant, patient):
        """
        Initialize the conversation handler.
        
        Args:
            assistant (MentalHealthAssistant): The mental health assistant agent
            patient (Patient): The patient agent
        """
        self.assistant = assistant
        self.patient = patient
        self.conversation_log = []
    
    def run(self):
        """
        Run the conversation between the mental health assistant and patient.
        
        Returns:
            str: Final diagnosis from the mental health assistant
        """
        print("=== Starting Mental Health Assessment ===\n")
        
        # Start with the assistant's first question
        print("[DEBUG] Getting first message from assistant")
        question = self.assistant.get_next_message()
        
        # Check if the response is a dictionary with RAG usage info
        if isinstance(question, dict) and "content" in question:
            question_content = question["content"]
            rag_usage = question.get("rag_usage")
            
            # Add to conversation log with RAG info
            self.conversation_log.append({
                "role": "assistant",
                "content": question_content,
                "rag_usage": rag_usage
            })
            
            print(f"Assistant: {question_content}")
            
            # Display RAG usage info if available
            if rag_usage and rag_usage.get("count", 0) > 0:
                print(f"\n[RAG Engine accessed {rag_usage['count']} documents]")
                for i, doc in enumerate(rag_usage['accessed_documents']):
                    print(f"  - Doc {i+1}: {doc.get('title', 'Unknown')}")
                
            question = question_content  # Use just the content for patient interaction
        else:
            # Legacy format support
            self.conversation_log.append({
                "role": "assistant",
                "content": question
            })
            print(f"Assistant: {question}")
        
        # Track if we've gone through all questions
        all_questions_asked = False
        
        # Check if the question is incomplete (missing a question mark or only fragments)
        if not self._is_complete_question(question):
            print("[DEBUG] Detected incomplete question. Fixing...")
            question = self._fix_incomplete_question(question, self.assistant.current_question_idx - 1)
        
        while True:
            # If we've already gone through all questions and received the patient's last response,
            # it's time to generate the diagnosis
            if all_questions_asked:
                print("[DEBUG] All questions have been asked. Requesting final diagnosis...")
                # Force the assistant to generate a diagnosis
                diagnosis = self.assistant.generate_diagnosis()
                
                # Check if the diagnosis is a dictionary with RAG usage info
                if isinstance(diagnosis, dict) and "content" in diagnosis:
                    diagnosis_content = diagnosis["content"]
                    rag_usage = diagnosis.get("rag_usage")
                    
                    # Add to conversation log with RAG info
                    self.conversation_log.append({
                        "role": "assistant",
                        "content": diagnosis_content,
                        "rag_usage": rag_usage
                    })
                    
                    print(f"\n=== Final Diagnosis ===\n")
                    print(f"Assistant: {diagnosis_content}")
                    
                    return diagnosis_content
                else:
                    # Legacy format support
                    self.conversation_log.append({
                        "role": "assistant", 
                        "content": diagnosis
                    })
                    print(f"\n=== Final Diagnosis ===\n")
                    print(f"Assistant: {diagnosis}")
                    return diagnosis
            
            # Get patient's response to the current question
            patient_response = self.patient.respond_to_question(question)
            self.conversation_log.append({
                "role": "patient",
                "content": patient_response
            })
            
            print(f"Patient: {patient_response}")
            
            # Small delay for more natural conversation flow
            time.sleep(1)
            
            # Check if we've asked the last question
            if self.assistant.current_question_idx >= len(self.assistant.questions):
                print("[DEBUG] Just asked the final question. Next step will be diagnosis.")
                all_questions_asked = True
            
            print("[DEBUG] Getting next message from assistant")
            # Get next question from assistant
            question = self.assistant.get_next_message(patient_response)
            
            # Check if the question is incomplete
            if not all_questions_asked and not self._is_complete_question(question):
                print("[DEBUG] Detected incomplete question. Fixing...")
                question = self._fix_incomplete_question(question, self.assistant.current_question_idx - 1)
            
            # Check if the response is a dictionary with RAG usage info
            if isinstance(question, dict) and "content" in question:
                question_content = question["content"]
                rag_usage = question.get("rag_usage")
                
                print("[DEBUG] Assistant response includes RAG info")
                
                # Add to conversation log with RAG info
                self.conversation_log.append({
                    "role": "assistant",
                    "content": question_content,
                    "rag_usage": rag_usage
                })
                
                # Display response to user
                print(f"Assistant: {question_content}")
                
                # Display RAG usage info if available
                if rag_usage and rag_usage.get("count", 0) > 0:
                    print(f"\n[RAG Engine accessed {rag_usage['count']} documents]")
                    for i, doc in enumerate(rag_usage['accessed_documents']):
                        print(f"  - Doc {i+1}: {doc.get('title', 'Unknown')}")
                
                # If this is the diagnosis, return it
                if all_questions_asked:
                    return question_content
                
                question = question_content  # Use just the content for next iteration
            else:
                # Legacy format support
                print("[DEBUG] Assistant response does NOT include RAG info")
                self.conversation_log.append({
                    "role": "assistant",
                    "content": question
                })
                print(f"Assistant: {question}")
                
                # If this is the diagnosis, return it
                if all_questions_asked:
                    return question
    
    def _is_complete_question(self, text):
        """Check if a question appears complete."""
        if not text:
            return False
            
        # Questions with complete sentences typically end with these punctuation marks
        if text.strip().endswith('?'):
            return True
            
        # Check if it's at least a complete sentence
        if text.strip().endswith('.') or text.strip().endswith('!'):
            # Make sure it has a reasonable length
            words = text.split()
            return len(words) >= 3
            
        # Check for fragments that are clearly incomplete
        if text.strip().startswith(', ') or text.strip().startswith('or '):
            return False
            
        # If it has reasonable length and structure, consider it complete
        words = text.split()
        return len(words) >= 5
    
    def _fix_incomplete_question(self, text, question_idx):
        """Fix an incomplete question by referencing the original questionnaire."""
        # If question index is valid, get the original question
        if 0 <= question_idx < len(self.assistant.questions):
            original = self.assistant.questions[question_idx]
            print(f"[DEBUG] Using original question from questionnaire: {original}")
            return original
            
        # If we can't get the original, try to make the fragment more question-like
        if text.strip().startswith(', ') or text.strip().startswith('or '):
            fixed = f"Do you experience {text.strip()}?"
            print(f"[DEBUG] Fixed fragment to: {fixed}")
            return fixed
            
        # Add a question mark if it's missing
        if not text.strip().endswith('?'):
            fixed = f"{text.strip()}?"
            print(f"[DEBUG] Added question mark: {fixed}")
            return fixed
            
        return text
    
    def get_conversation_log(self) -> List[Dict[str, str]]:
        """
        Get the conversation log.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_log
