import os
from utils.llm_client_base import LLMClient
from utils.rag_engine import RAGEngine

class MentalHealthAssistant:
    def __init__(self, provider="ollama", provider_options=None, model="qwen3:4b", 
                 questions=None, rag_engine=None, questionnaire_name=None):
        """
        Initialize the Mental Health Assistant agent.
        
        Args:
            provider (str): LLM provider to use (ollama or groq)
            provider_options (dict, optional): Options to pass to the provider client
            model (str): Model to use with the provider
            questions (list): List of questions from the questionnaire
            rag_engine (RAGEngine, optional): RAG engine for document retrieval
            questionnaire_name (str, optional): Name of the questionnaire being used
        """
        # Initialize default provider options
        if provider_options is None:
            provider_options = {}
            
            # Set default options based on provider
            if provider == "ollama":
                provider_options["base_url"] = "http://localhost:11434"
                
        # Create the client using the factory method
        self.client = LLMClient.create(provider, **provider_options)
        self.model = model
        self.questions = questions
        self.responses = []
        self.current_question_idx = 0
        self.context = None
        self.rag_engine = rag_engine
        self.has_introduced = False  # Flag to track if introduction has been given
        self.questionnaire_name = questionnaire_name
        
        # Load system prompt from file
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "prompts", "mental_health_assistant_prompt.txt")
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()
        
        # Track documents that have already been seen to avoid duplication
        self.seen_documents = set()
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
            
            # Generate introduction based on the questionnaire content directly
            intro_message = self._generate_introduction()
            
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
    
    def _generate_introduction(self):
        """Generate an introduction for the mental health assessment."""
        print("[DEBUG] Retrieving full questionnaire document for introduction...")
        
        # Get the full questionnaire document if available
        full_questionnaire_content = ""
        if self.questionnaire_name and self.rag_engine:
            specific_docs = self.rag_engine.get_context_for_question(f"full text of {self.questionnaire_name}")
            
            # Handle both old format (list) and new format (dictionary)
            if isinstance(specific_docs, dict) and "content" in specific_docs:
                # New format from enhanced RAG engine
                full_questionnaire_content = specific_docs["content"]
            elif isinstance(specific_docs, list) and len(specific_docs) > 0:
                # Old format (list of strings)
                full_questionnaire_content = specific_docs[0]
            else:
                # Default empty string if no content found
                full_questionnaire_content = ""
        else:
            full_questionnaire_content = ""
        
        # Create a prompt for the introduction generation
        intro_prompt = f"""
        You are a professional mental health clinician about to conduct an assessment using a mental health questionnaire.
        
        Here is the full questionnaire document you will be administering:
        
        ```
        {full_questionnaire_content}
        ```
        
        IMPORTANT INSTRUCTIONS:
        - Use ONLY the actual name of the questionnaire as shown in the document above
        - DO NOT introduce this as a "Somatic Symptom Disorder questionnaire" unless that is explicitly the name in the document
        - DO NOT assume what specific condition is being assessed
        - The questionnaire may be a general mental health assessment or focused on various conditions
        - Your role is to administer the questionnaire without making diagnostic assumptions up front
        
        Based on this questionnaire document, please generate a warm, professional introduction to the patient that:
        1. Introduces yourself as a mental health professional
        2. Identifies the specific questionnaire you're using by name (from the document)
        3. Explains the purpose of this specific assessment 
        4. Reassures the patient about confidentiality and creating a safe space
        5. Briefly explains how the assessment will proceed ({len(self.questions)} questions)
        6. Indicates you're ready to begin with the first question
        
        Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
        Make sure to correctly identify and name the specific questionnaire you're administering.
        For the first interaction, provide a complete introduction followed by your first question.
        
        This is real-time conversation with a human patient, so make your introduction engaging, natural, and conversational.
        """
        
        # Create a temporary conversation for generating the introduction
        temp_conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": intro_prompt}
        ]
        
        # Generate the introduction using the LLM
        result = self.client.chat(self.model, temp_conversation)
        introduction = result['response']
        
        return introduction
    
    def generate_diagnosis(self):
        """
        Generate a diagnosis based on the patient's responses.
        
        Returns:
            dict: Diagnosis from the assistant with RAG usage information
        """
        # First, use AI to summarize observations from the patient responses
        observations = self._summarize_observations()
        print(f"[DEBUG] Generated clinical observations: {observations[:100]}...")
        
        # Create a prompt for diagnosis that includes the observations
        diagnosis_prompt = f"""
        Based on the questionnaire responses, please provide a comprehensive mental health assessment.
        
        Questionnaire responses:
        {self._format_responses()}
        
        Clinical observations and potential concerns:
        {observations}
        
        IMPORTANT DIAGNOSTIC CONSIDERATIONS:
        - Consider multiple possible diagnoses that could explain the symptoms
        - Do not default to Somatic Symptom Disorder unless clearly warranted by the symptoms
        - Be open to various diagnostic possibilities including anxiety disorders, mood disorders, trauma-related disorders, etc.
        - Make your diagnosis based solely on the symptoms presented, not on assumptions
        - If symptoms are insufficient for a definitive diagnosis, indicate this is a provisional impression
        
        Please analyze these responses and observations and provide a professional assessment that MUST follow this EXACT structure:

        1. First paragraph: Write a compassionate summary of what you've heard from the patient, showing empathy for their situation.
        
        2. After that, include a section with the heading "**Diagnosis:**" (exactly as shown, with the asterisks)
           - On the same line, immediately after the heading, provide the specific diagnosis or clinical impression
           - Do not add extra newlines between the heading and the diagnosis
        
        3. Next, include a section with the heading "**Reasoning:**" (exactly as shown, with the asterisks)
           - Immediately after this heading, explain your rationale for the diagnosis/impression
           - Do not add extra newlines between the heading and your explanation
        
        4. Finally, include a section with the heading "**Recommended Next Steps/Treatment Options:**" (exactly as shown, with the asterisks)
           - List specific numbered recommendations (1., 2., 3., etc.)
           - Make each recommendation clear and actionable
        
        When writing your assessment, use these special tags:
        - Wrap medical terms and conditions in <med>medical term</med> tags
        - Wrap symptoms in <sym>symptom</sym> tags
        - Wrap patient quotes or paraphrases in <quote>patient quote</quote> tags
        
        EXTREMELY IMPORTANT:
        1. Do NOT include any introductory statements answering the prompt" 
        2. Do NOT begin with phrases like "Okay, here's a clinical assessment..."
        3. Start DIRECTLY with the compassionate summary paragraph without any preamble
        4. Never include meta-commentary about what you're about to write
        5. Include all four components in the exact order specified
        6. Format section headings consistently with double asterisks
        7. Maintain proper spacing between sections (one blank line)
        8. Do not add extra newlines within sections
        9. Always wrap medical terms, symptoms, and quotes in the specified tags

        Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
        """
        
        # Initialize RAG usage information
        rag_usage = None
        
        # Enhance with RAG if available - ONLY use RAG during diagnosis phase
        if self.rag_engine:
            print("[DEBUG] Now using RAG for diagnosis...")
            
            # Query RAG using the observations instead of raw responses
            print(f"[DEBUG] Querying RAG using clinical observations...")
            
            # Create a more focused query using the observations
            rag_query = f"mental health diagnosis for patient with symptoms: {observations}"
            
            # Get context for general mental health diagnosis
            rag_result = self.rag_engine.retrieve(rag_query, top_k=5)
            
            if isinstance(rag_result, dict) and "content_list" in rag_result:
                # New RAG format
                documents = rag_result.get("documents", [])
                filtered_content = []
                newly_accessed_docs = []
                
                # Only use documents we haven't seen before
                for i, doc in enumerate(documents):
                    doc_id = doc.get("title", "") + "|" + doc.get("highlight", "")[:50]
                    if doc_id not in self.seen_documents:
                        self.seen_documents.add(doc_id)
                        # Add content only if it's new
                        if i < len(rag_result["content_list"]):
                            filtered_content.append(rag_result["content_list"][i])
                        newly_accessed_docs.append(doc)
                
                if filtered_content:
                    print(f"[DEBUG] Found {len(filtered_content)} new relevant documents for diagnosis")
                    # Don't include raw context in the prompt, instead use system message
                    self.conversation_history.append({
                        "role": "system", 
                        "content": f"Use this additional reference information to help inform your diagnosis, but don't include raw reference text in your response: {' '.join(filtered_content)}"
                    })
                    
                    # Track RAG usage information for new documents only
                    rag_usage = {
                        "documents": newly_accessed_docs,
                        "stats": rag_result.get("stats", {}),
                        "count": len(newly_accessed_docs)
                    }
            else:
                # Legacy format
                context = rag_result
                if context:
                    print(f"[DEBUG] Found {len(context)} relevant documents for diagnosis")
                    
                    # Don't include raw context in the prompt, instead use system message
                    self.conversation_history.append({
                        "role": "system", 
                        "content": f"Use this additional reference information to help inform your diagnosis, but don't include raw reference text in your response: {' '.join(context)}"
                    })
                    
                    # Track RAG usage information
                    if hasattr(self.rag_engine, 'get_accessed_documents'):
                        accessed_docs = self.rag_engine.get_accessed_documents()
                        if accessed_docs:
                            print(f"[DEBUG] RAG used: captured {len(accessed_docs)} relevant documents for diagnosis")
                            rag_usage = {
                                "accessed_documents": accessed_docs,
                                "count": len(accessed_docs)
                            }
                            # Clear the accessed documents for next query
                            self.rag_engine.clear_accessed_documents()
        
        self.conversation_history.append({"role": "user", "content": diagnosis_prompt})
        
        result = self.client.chat(self.model, self.conversation_history)
        diagnosis = result['response']
        self.context = result['context']
        
        # Add diagnosis to conversation history
        self.conversation_history.append({"role": "assistant", "content": diagnosis})
        
        # Return both the diagnosis and RAG usage information
        return {
            "content": diagnosis,
            "rag_usage": rag_usage
        }
    
    def _summarize_observations(self) -> str:
        """
        Summarize clinical observations from patient responses.
        
        Returns:
            str: Clinical observations and potential concerns
        """
        # Create formatted responses for the summarization
        formatted_responses = self._format_responses()
        
        # Create a prompt for the observation summarization
        summarization_prompt = f"""
        You are a mental health professional reviewing patient responses to a questionnaire.
        
        Here are the patient's responses:
        {formatted_responses}
        
        Based on these responses, please:
        1. Identify the main symptoms and concerns
        2. Note patterns in the patient's responses
        3. List potential areas of clinical significance
        4. Highlight any risk factors or warning signs
        5. Summarize your observations in clinical language
        
        Format your response as a concise clinical observation summary using professional terminology.
        Focus on extracting the most relevant clinical information while avoiding speculation.
        """
        
        # Create a temporary conversation for generating the observations
        temp_conversation = [
            {"role": "system", "content": "You are a clinical mental health professional conducting an assessment."},
            {"role": "user", "content": summarization_prompt}
        ]
        
        # Generate the clinical observations using the LLM
        result = self.client.chat(self.model, temp_conversation)
        observations = result['response']
        
        return observations

    def _extract_symptoms_for_query(self):
        """Extract key symptoms from patient responses to create a better RAG query."""
        # This method is kept for backwards compatibility
        # The preferred approach is now to use _summarize_observations() for RAG queries
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
            rag_result = self.rag_engine.retrieve(message, top_k=3)
            
            if isinstance(rag_result, dict) and "content_list" in rag_result:
                # New RAG format
                documents = rag_result.get("documents", [])
                filtered_content = []
                newly_accessed_docs = []
                
                # Only use documents we haven't seen before
                for i, doc in enumerate(documents):
                    doc_id = doc.get("title", "") + "|" + doc.get("highlight", "")[:50]
                    if doc_id not in self.seen_documents:
                        self.seen_documents.add(doc_id)
                        # Add content only if it's new
                        if i < len(rag_result["content_list"]):
                            filtered_content.append(rag_result["content_list"][i])
                        newly_accessed_docs.append(doc)
                
                if filtered_content:
                    context_str = "\n\n".join(filtered_content)
                    # Add context to the message
                    system_message = {
                        "role": "system", 
                        "content": f"Consider this additional information when responding:\n{context_str}"
                    }
                    history_copy.append(system_message)
                    
                    rag_usage = {
                        "documents": newly_accessed_docs,
                        "stats": rag_result.get("stats", {}),
                        "count": len(newly_accessed_docs)
                    }
            else:
                # Legacy format handling
                context = rag_result
                if context:
                    print(f"[DEBUG] Found {len(context)} relevant documents")
                    context_str = "\n\n".join(context)
                    # Add context to the message
                    system_message = {
                        "role": "system", 
                        "content": f"Consider this additional information when responding:\n{context_str}"
                    }
                    history_copy.append(system_message)
                    
                    rag_usage = {
                        "accessed_documents": [{"title": "Unknown"}] if context else [],
                        "count": 1 if context else 0
                    }
        
        # Generate response
        result = self.client.chat(self.model, history_copy)
        response = result['response']
        
        print(f"[DEBUG] Generated response: {response[:50]}..." if len(response) > 50 else response)
        
        # If RAG was used, get the accessed documents
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

def get_context_for_question(self, question: str) -> dict:
    """Get relevant context for a question from the RAG engine."""
    # Get context from RAG engine
    context = self.rag_engine.get_context_for_question(question)
    
    # Handle both old format (string/list) and new format (dictionary)
    if isinstance(context, dict) and "content" in context:
        # New format - already has all we need
        rag_usage = {
            "documents": context.get("documents", []),
            "stats": context.get("stats", {}),
        }
        return {
            "content": context["content"],
            "rag_usage": rag_usage
        }
    else:
        # Old format - convert to new format
        rag_usage = {
            "count": 1 if context else 0,
            "accessed_documents": [{"title": "Unknown"}] if context else []
        }
        # If it's a list, join items with newlines
        if isinstance(context, list):
            context = "\n\n".join(context)
            
        return {
            "content": context,
            "rag_usage": rag_usage
        }
