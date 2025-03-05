import os
from utils.ollama_client import OllamaClient
from utils.rag_engine import RAGEngine

class MentalHealthAssistant:
    def __init__(self, ollama_url, model, questions, rag_engine=None):
        """
        Initialize the Mental Health Assistant agent.
        
        Args:
            ollama_url (str): URL for the Ollama API
            model (str): Ollama model to use
            questions (list): List of questions from the questionnaire
            rag_engine (RAGEngine, optional): RAG engine for document retrieval
        """
        self.client = OllamaClient(base_url=ollama_url)
        self.model = model
        self.questions = questions
        self.responses = []
        self.current_question_idx = 0
        self.context = None
        self.rag_engine = rag_engine
        
        # Load system prompt from file
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "prompts", "mental_health_assistant_prompt.txt")
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()
        
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    def get_next_message(self, patient_response=None):
        """
        Get the next message from the assistant.
        
        Args:
            patient_response (str): Response from the patient
            
        Returns:
            str: Next message from the assistant
        """
        if patient_response:
            # Add patient's response to conversation history
            self.conversation_history.append({"role": "user", "content": patient_response})
            self.responses.append((self.questions[self.current_question_idx-1], patient_response))
        
        if self.current_question_idx < len(self.questions):
            # Ask the next question
            next_question = self.questions[self.current_question_idx]
            
            # Enhance with RAG if available
            enhanced_question = next_question
            if self.rag_engine:
                context = self.rag_engine.get_context_for_question(next_question)
                if context:
                    enhanced_question = f"{next_question}\n\nI have this additional context to consider: {context}"
            
            self.current_question_idx += 1
            
            # Add question to conversation history
            self.conversation_history.append({"role": "assistant", "content": next_question})
            
            return next_question
        else:
            # All questions have been asked, generate diagnosis
            return self.generate_diagnosis()
    
    def generate_diagnosis(self):
        """
        Generate a diagnosis based on the patient's responses.
        
        Returns:
            str: Diagnosis from the assistant
        """
        # Create a prompt for diagnosis
        diagnosis_prompt = f"""
        Based on the questionnaire responses, please provide a comprehensive mental health assessment.
        
        Questionnaire responses:
        {self._format_responses()}
        
        Please analyze these responses and provide:
        1. A potential diagnosis or assessment
        2. Explanation of the reasoning behind this assessment
        3. Recommended next steps or treatment options
        """
        
        # Enhance with RAG if available
        if self.rag_engine:
            # Get context for general mental health diagnosis
            context = self.rag_engine.retrieve("mental health diagnosis criteria DSM-5", top_k=3)
            if context:
                context_str = "\n\n".join(context)
                diagnosis_prompt += f"\n\nAdditional reference information:\n{context_str}"
        
        self.conversation_history.append({"role": "user", "content": diagnosis_prompt})
        
        result = self.client.chat(self.model, self.conversation_history)
        diagnosis = result['response']
        self.context = result['context']
        
        # Add diagnosis to conversation history
        self.conversation_history.append({"role": "assistant", "content": diagnosis})
        
        return diagnosis
    
    def _format_responses(self):
        """Format the patient's responses for diagnosis."""
        formatted = ""
        for i, (question, response) in enumerate(self.responses, 1):
            formatted += f"Q{i}: {question}\nA{i}: {response}\n\n"
        return formatted
