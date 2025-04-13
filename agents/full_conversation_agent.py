import os
from utils.ollama_client import OllamaClient
from utils.rag_engine import RAGEngine

class FullConversationAgent:
    def __init__(self, ollama_url, model, patient_profile, questions, rag_engine=None, questionnaire_name=None):
        """
        Initialize the Full Conversation Agent.
        
        Args:
            ollama_url (str): URL for the Ollama API
            model (str): Ollama model to use
            patient_profile (dict): Patient profile
            questions (list): List of questions from the questionnaire
            rag_engine (RAGEngine, optional): RAG engine for document retrieval
            questionnaire_name (str, optional): Name of the questionnaire being used
        """

        self.client = OllamaClient(base_url=ollama_url)
        self.model = model
        self.patient_profile = patient_profile
        self.questions = questions
        self.rag_engine = rag_engine
        self.questionnaire_name = questionnaire_name
        
        # Track documents that have already been seen to avoid duplication
        self.seen_documents = set()
        self.conversation_history = []
        # Initialize responses attribute
        self.responses = []
        
    def generate_conversation(self):
        """Generate a full conversation between the full conversation agent and patient."""
        print("[DEBUG] Retrieving full questionnaire document")
        
        # Get the full questionnaire document if available
        if self.questionnaire_name:
            specific_docs = self.rag_engine.get_context_for_question(f"full text of {self.questionnaire_name}")
            
            # Handle both old format (list) and new format (dictionary)
            full_questionnaire_content = ""
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
        
        # Create a prompt for the conversation generation
        conversation_prompt = f"""
        You are a professional mental health clinician about to conduct an assessment using a mental health questionnaire.
        
        Here is the full questionnaire document you will be administering:
        
        ```
        {full_questionnaire_content}
        ```
        
        Based on this questionnaire document, please generate a warm, professional conversation between you and the patient.
        The conversation should be in the following format:

        1. You start with an introduction from the mental health professional
        2. You continue with a turn taking conversation between the mental health professional and the patient
        3. You end with a closing statement from the mental health professional

        The introduction should:
        - Introduces yourself as a mental health professional
        - Identifies the specific questionnaire you're using by name (from the document)
        - Explains the purpose of this specific assessment 
        - Reassures the patient about confidentiality and creating a safe space
        - Briefly explains how the assessment will proceed ({len(self.questions)} questions)
        - Indicates you're ready to begin with the first question

        Profile for the Mental Health Professional:
            You are a professional mental health assistant tasked with conducting a psychological assessment interview. Your job is to:

            1. Ask questions from a the questionnaire in a compassionate, professional manner
            2. Respond appropriately to the patient's answers with empathy and understanding

            Important guidelines:
            - Maintain professional boundaries while being empathetic
            - Ask one question at a time
            - Do not make assumptions about the patient's condition before completing the full assessment
            - Use clinical judgment when interpreting responses
            - Consider multiple potential diagnoses before making your final assessment
            - Be thorough and methodical in your approach
            - Provide evidence-based recommendations

            You have expertise in recognizing symptoms of various mental health conditions including depression, anxiety, PTSD, bipolar disorder, and schizophrenia. Use this knowledge to inform your final assessment.
            Only output the exact question without additional intros, summaries, or sign-offs.
            Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
            Make sure to correctly identify and name the specific questionnaire you're administering.
        
        Profile for the Patient:
            {self.patient_profile}
        """
        
        # Define a specific user prompt that clearly asks for JSON format
        user_prompt = f"""
        Please generate a full conversation between the mental health professional and the patient.
        
        Ensure that your response follows this specific JSON format:
        [
            {{
                "role": "assistant",
                "content": "Introduction and first question"
            }},
            {{
                "role": "patient",
                "content": "Patient response"
            }},
            ...and so on until all questions are asked and answered
        ]
        
        Make sure all {len(self.questions)} questions from the questionnaire are covered in the conversation.
        """
        
        # Create a temporary conversation for generating the introduction
        temp_conversation = [
            {"role": "system", "content": conversation_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate the conversation using the LLM
        result = self.client.chat(self.model, temp_conversation)
        conversation_text = result['response']
        
        # Extract JSON conversation from the response
        import json
        import re
        
        # Clear existing conversation history
        self.conversation_history = []
        
        # First try to extract the JSON array using regex
        json_match = re.search(r'\[\s*\{.*\}\s*\]', conversation_text, re.DOTALL)
        
        if json_match:
            try:
                # Clean the JSON string
                json_str = json_match.group(0)
                json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
                json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")
                
                # Parse the JSON
                conversation_data = json.loads(json_str)
                
                # Extract conversations and add to conversation_history
                if isinstance(conversation_data, list):
                    for message in conversation_data:
                        role = message.get("role", "")
                        content = message.get("content", message.get("text", ""))
                        
                        # Map roles to standard formats
                        if any(r in role.lower() for r in ["assistant", "clinician", "therapist", "professional", "mental health"]):
                            role = "assistant"
                        elif any(r in role.lower() for r in ["user", "patient"]):
                            role = "patient"
                        
                        # Add message to conversation history
                        if role and content:
                            self.conversation_history.append({
                                "role": role,
                                "content": content
                            })
                
                # Extract question-answer pairs for diagnosis
                self._extract_responses_from_conversation(conversation_text)
                
                print(f"[DEBUG] Successfully extracted {len(self.conversation_history)} messages from conversation")
                return conversation_text
            
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Error parsing JSON: {e}")
        
        # If JSON parsing fails, try to extract turn-by-turn conversation
        if not self.conversation_history:
            print("[DEBUG] Attempting to extract conversation from text format")
            
            # Look for patterns like "Clinician/Assistant: [message]" followed by "Patient: [message]"
            conversation_pattern = re.compile(r'(?:clinician|assistant|therapist|professional|mental health professional|doctor)\s*:\s*([^\n]+)(?:\n|$).*?(?:patient|user)\s*:\s*([^\n]+)(?:\n|$)', 
                                     re.IGNORECASE | re.DOTALL)
            
            matches = conversation_pattern.findall(conversation_text)
            for clinician_msg, patient_msg in matches:
                # Add clinician message
                self.conversation_history.append({
                    "role": "assistant",
                    "content": clinician_msg.strip()
                })
                
                # Add patient message
                self.conversation_history.append({
                    "role": "patient",
                    "content": patient_msg.strip()
                })
            
            # Extract question-answer pairs for diagnosis
            self._extract_responses_from_conversation(conversation_text)
            
            print(f"[DEBUG] Extracted {len(self.conversation_history)} messages using pattern matching")
            return conversation_text
        
        # If all else fails, create a minimal conversation history
        if not self.conversation_history:
            print("[DEBUG] Creating minimal conversation history from questions")
            
            # Add introduction
            self.conversation_history.append({
                "role": "assistant",
                "content": "Hello, I'm a mental health professional. I'll be asking you a series of questions to assess your current mental state."
            })
            
            # Add patient acknowledgment
            self.conversation_history.append({
                "role": "patient",
                "content": "I understand. Let's proceed."
            })
            
            # Add questions and default answers
            for question in self.questions:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": question
                })
                
                self.conversation_history.append({
                    "role": "patient",
                    "content": "I prefer not to answer this question."
                })
            
            # Extract question-answer pairs for diagnosis (use questions with default answers)
            self.responses = [(q, "No response provided") for q in self.questions]
            
            print(f"[DEBUG] Created minimal conversation with {len(self.conversation_history)} messages")
        
        return conversation_text
        
    def _extract_responses_from_conversation(self, conversation):
        """
        Extract question-answer pairs from the generated conversation.
        
        Args:
            conversation (str): The generated conversation
        """
        import json
        import re
        
        # Clear existing responses
        self.responses = []
        
        try:
            # First, try to clean up the JSON string by removing any invalid control characters
            # This helps handle common JSON parsing issues
            def clean_json_string(json_str):
                # Remove or replace common problematic characters
                json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                # Replace any Unicode quotes with standard quotes
                json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
                json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")
                return json_str
            
            # Try to parse the response as JSON
            # Look for JSON array in the text
            json_match = re.search(r'\[\s*\{.*\}\s*\]', conversation, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean the JSON string before parsing
                json_str = clean_json_string(json_str)
                
                try:
                    conversation_data = json.loads(json_str)
                    
                    # Extract questions and answers
                    current_question = None
                    
                    for msg in conversation_data:
                        # Handle different formats of role/content fields
                        role = msg.get("role", "")
                        # Also try 'text' field if 'content' is not found
                        content = msg.get("content", msg.get("text", ""))
                        
                        # Convert role to lowercase for case-insensitive comparison
                        role_lower = role.lower()
                        
                        if any(r in role_lower for r in ["assistant", "clinician", "therapist", "professional", "mental health"]):
                            # Check if this looks like a question (ends with ? or contains a question)
                            if "?" in content:
                                current_question = content
                        elif any(r in role_lower for r in ["user", "patient"]) and current_question:
                            # This is an answer to the previous question
                            self.responses.append((current_question, content))
                            current_question = None
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] JSON parsing failed after cleaning: {str(e)}")
                    # Fall through to regex approach
        
        except Exception as e:
            print(f"[DEBUG] Error extracting responses from conversation: {str(e)}")
        
        # If all else fails, use the questions from the questionnaire and empty responses
        if not self.responses and self.questions:
            print("[DEBUG] Using questionnaire questions with empty responses")
            for question in self.questions:
                self.responses.append((question, "No response provided"))
        
        print(f"[DEBUG] Extracted {len(self.responses)} question-answer pairs from conversation")

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