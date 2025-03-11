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
        self.has_introduced = False  # Flag to track if introduction has been given
        
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
            if self.current_question_idx > 0:  # Only add to responses if we've asked a question
                self.responses.append((self.questions[self.current_question_idx-1], patient_response))
        
        # If this is the first interaction, provide an introduction
        if not self.has_introduced:
            self.has_introduced = True
            
            # Generate introduction based on the number and nature of questions
            questionnaire_type = self._determine_questionnaire_type()
            intro_message = self._generate_introduction(questionnaire_type)
            
            # Add introduction to conversation history
            self.conversation_history.append({"role": "assistant", "content": intro_message})
            
            return intro_message
        
        # Continue with regular question flow
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
    
    def _determine_questionnaire_type(self):
        """Determine the type of questionnaire based on the questions."""
        # Look for keywords in the questions
        keywords = {
            "depression": ["depression", "mood", "sad", "interest", "pleasure", "hopeless"],
            "anxiety": ["anxiety", "worry", "nervous", "panic", "fear", "edge"],
            "general": ["mental health", "wellness", "well-being", "symptoms", "feelings"],
            "psychiatric": ["hallucination", "psychosis", "delusion", "paranoid", "voices"]
        }
        
        # Count occurrences of keywords
        counts = {category: 0 for category in keywords}
        for question in self.questions:
            question_lower = question.lower()
            for category, terms in keywords.items():
                for term in terms:
                    if term in question_lower:
                        counts[category] += 1
        
        # Return the category with the most matches, defaulting to general
        max_count = 0
        questionnaire_type = "general"
        for category, count in counts.items():
            if count > max_count:
                max_count = count
                questionnaire_type = category
        
        return questionnaire_type
    
    def _generate_introduction(self, questionnaire_type):
        """Generate an appropriate introduction based on the questionnaire type."""
        
        # Base introduction
        intro = "Hello, I'm a mental health professional, and I'll be conducting an assessment today. "
        
        # Add questionnaire-specific information
        if questionnaire_type == "depression":
            intro += "We'll be going through a depression screening questionnaire to better understand your mood and experiences. "
        elif questionnaire_type == "anxiety":
            intro += "Today we'll complete an anxiety assessment to help understand your experiences with worry and stress. "
        elif questionnaire_type == "psychiatric":
            intro += "I'll be asking you questions from a psychiatric evaluation to help us understand your experiences. "
        else:  # general
            intro += "Today we'll complete a general mental health assessment to understand how you've been feeling. "
        
        # Add explanation of process
        intro += "I'll ask you several questions, and your honest responses will help me provide a preliminary assessment. " 
        intro += "Everything you share is confidential, and this is a safe space to discuss your concerns. "
        
        # Add first question prompt
        intro += "Let's begin with the first question:\n\n"
        
        return intro
    
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
        
        Please analyze these responses and provide a professional assessment that includes:
        1. A compassionate summary of what you've heard from the patient
        2. A potential diagnosis or clinical impression based on the symptoms
        3. Explanation of the reasoning behind this assessment
        4. Recommended next steps or treatment options
        5. Close with an empathetic statement that validates the patient's experiences
        
        Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
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
