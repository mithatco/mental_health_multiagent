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
            str or dict: Next message from the assistant, possibly with RAG info
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
            # Ask the next question - use the exact question from the questionnaire without RAG
            next_question = self.questions[self.current_question_idx]
            
            print(f"[DEBUG] Asking questionnaire question #{self.current_question_idx + 1}: {next_question[:50]}...")
            
            self.current_question_idx += 1
            
            # Add question to conversation history
            self.conversation_history.append({"role": "assistant", "content": next_question})
            
            # Return the exact question from the questionnaire without RAG enhancement
            return next_question
        else:
            # All questions have been asked, generate diagnosis
            print("[DEBUG] All questions asked. Generating final diagnosis with RAG assistance.")
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
        
        # Enhance with RAG if available - ONLY use RAG during diagnosis phase
        if self.rag_engine:
            print("[DEBUG] Now using RAG for diagnosis...")
            
            # Build a query based on the patient's symptoms from their responses
            symptoms_query = self._extract_symptoms_for_query()
            print(f"[DEBUG] Querying RAG using extracted symptoms: {symptoms_query[:100]}...")
            
            # Get context for general mental health diagnosis
            context = self.rag_engine.retrieve(symptoms_query, top_k=5)
            if context:
                print(f"[DEBUG] Found {len(context)} relevant documents for diagnosis")
                
                # Don't include raw context in the prompt, instead use system message
                self.conversation_history.append({
                    "role": "system", 
                    "content": f"Use this additional reference information to help inform your diagnosis, but don't include raw reference text in your response: {' '.join(context)}"
                })
        
        self.conversation_history.append({"role": "user", "content": diagnosis_prompt})
        
        result = self.client.chat(self.model, self.conversation_history)
        diagnosis = result['response']
        self.context = result['context']
        
        # Add diagnosis to conversation history
        self.conversation_history.append({"role": "assistant", "content": diagnosis})
        
        return diagnosis
    
    def _extract_symptoms_for_query(self):
        """Extract key symptoms from patient responses to create a better RAG query."""
        symptoms = []
        for question, response in self.responses:
            # Add both question and response to get context
            symptoms.append(f"{question} {response}")
        
        # Join all symptoms into one query string
        combined = " ".join(symptoms)
        
        # Include common mental health terminology to improve RAG retrieval
        query = f"mental health assessment for patient with symptoms: {combined}"
        return query

    def _format_responses(self):
        """Format the patient's responses for diagnosis."""
        formatted = ""
        for i, (question, response) in enumerate(self.responses, 1):
            formatted += f"Q{i}: {question}\nA{i}: {response}\n\n"
        return formatted

    def respond(self, message, conversation_history=None):
        """
        Generate a response to the patient's message.
        
        Args:
            message: Message from the patient
            conversation_history: Optional conversation history to use
            
        Returns:
            dict: Response with content and RAG usage information
        """
        # Use provided conversation history or default to self.conversation_history
        history = conversation_history or self.conversation_history
        
        # Create a copy to avoid modifying the original
        history_copy = history.copy()
        
        # Add the user message
        history_copy.append({"role": "user", "content": message})
        
        print(f"[DEBUG] Generating response to: {message[:50]}..." if len(message) > 50 else message)
        
        # Query RAG engine if available
        if self.rag_engine:
            print("[DEBUG] Querying RAG engine for relevant context")
            context = self.rag_engine.retrieve(message, top_k=3)
            
            if context:
                print(f"[DEBUG] Found {len(context)} relevant documents")
                context_str = "\n\n".join(context)
                # Add context to the message
                system_message = {
                    "role": "system", 
                    "content": f"Consider this additional information when responding:\n{context_str}"
                }
                history_copy.append(system_message)
        
        # Generate response
        result = self.client.chat(self.model, history_copy)
        response = result['response']
        
        print(f"[DEBUG] Generated response: {response[:50]}..." if len(response) > 50 else response)
        
        # If RAG was used, get the accessed documents
        rag_usage = None
        if self.rag_engine and hasattr(self.rag_engine, 'get_accessed_documents'):
            accessed_docs = self.rag_engine.get_accessed_documents()
            if accessed_docs:
                print(f"[DEBUG] RAG used: found {len(accessed_docs)} relevant documents")
                rag_usage = {
                    "accessed_documents": accessed_docs,
                    "count": len(accessed_docs)
                }
                # Clear the accessed documents for next query
                self.rag_engine.clear_accessed_documents()
        
        # Return both the response and RAG usage information
        return {
            "content": response,
            "rag_usage": rag_usage
        }
