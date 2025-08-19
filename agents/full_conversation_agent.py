import os
import time
from utils.llm_client_base import LLMClient
from utils.rag_engine import RAGEngine
import json
import re

class FullConversationAgent:
    def __init__(self, provider="ollama", provider_options=None, model="qwen3:4b", 
                 patient_profile=None, questions=None, rag_engine=None, questionnaire_name=None,
                 disable_rag_evaluation=False):
        """
        Initialize the Full Conversation Agent.
        
        Args:
            provider (str): LLM provider to use (ollama or groq)
            provider_options (dict, optional): Options to pass to the provider client
            model (str): Model to use with the provider
            patient_profile (str): Name of the patient profile to use
            questions (list): List of questions from the questionnaire
            rag_engine (RAGEngine, optional): RAG engine for document retrieval
            questionnaire_name (str, optional): Name of the questionnaire being used
            disable_rag_evaluation (bool): Whether to disable RAG evaluation
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
        # Store the original profile name
        self.profile_name = patient_profile
        # Load the actual profile content from file
        self.patient_profile = self._load_profile(patient_profile)
        self.questions = questions
        self.rag_engine = rag_engine
        self.questionnaire_name = questionnaire_name
        self.disable_rag_evaluation = disable_rag_evaluation
        
        # Store the provider name for special handling
        self.provider = provider
        
        # Track documents that have already been seen to avoid duplication
        self.seen_documents = set()
        self.conversation_history = []
        # Initialize responses attribute
        self.responses = []
        
    def _load_profile(self, profile_name):
        """Load a patient profile from file."""
        if not profile_name:
            return "No specific patient profile provided."
        
        profiles_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
        
        # Try to load from the profiles directory
        profile_path = os.path.join(profiles_dir, f"{profile_name}.txt")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile_content = f.read()
            return profile_content
        else:
            print(f"Warning: Profile '{profile_name}' not found. Using default profile.")
            return f"Profile named {profile_name} (profile details not found)"
        
    def generate_conversation(self):
        """Generate a full conversation between the full conversation agent and patient."""
        print("[DEBUG] Retrieving full questionnaire document")
        
        # Get the full questionnaire document if available
        full_questionnaire_content = ""
        if self.questionnaire_name:
            # Try direct file reading approach first if this is a filename with .txt extension
            if self.questionnaire_name.endswith('.txt'):
                try:
                    # Construct path to questionnaire file
                    questionnaire_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents", "questionnaires")
                    questionnaire_path = os.path.join(questionnaire_dir, self.questionnaire_name)
                    
                    # Read file directly
                    if os.path.exists(questionnaire_path):
                        print(f"[DEBUG] Reading questionnaire directly from file: {questionnaire_path}")
                        with open(questionnaire_path, 'r') as f:
                            full_questionnaire_content = f.read()
                        print(f"[DEBUG] Successfully loaded questionnaire file ({len(full_questionnaire_content)} chars)")
                except Exception as e:
                    print(f"[DEBUG] Error reading questionnaire file directly: {str(e)}")
            
            # Fall back to RAG if direct file read failed or if not a .txt file
            if not full_questionnaire_content and self.rag_engine:
                print("[DEBUG] Falling back to RAG engine for questionnaire retrieval")
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
        
        # Create a prompt for the conversation generation
        base_conversation_prompt = f"""
        You are an agent which generates a full conversation between a mental health professional and a patient according to a questionnaire.
        
        IMPORTANT INSTRUCTIONS:
        - Use ONLY the actual name of the questionnaire as provided in the user message
        - The questionnaire may be a general mental health assessment or focused on various conditions
        - Your role is to administer the questionnaire without making diagnostic assumptions up front
        
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
        """
        
        # Special handling for Groq provider - more structured prompt
        if hasattr(self, 'provider') and self.provider == "groq":
            conversation_prompt = base_conversation_prompt + """
            
            YOUR RESPONSE MUST BE STRUCTURED AS A VALID JSON ARRAY OF OBJECTS.
            Each object must have 'role' and 'content' fields.
            The 'role' must be either 'assistant' for the mental health professional or 'patient' for the patient.
            The 'content' field must contain the actual message.
            
            You MUST adhere to proper JSON syntax with double quotes around property names and string values.
            Do not include any explanatory text, commentary, or code blocks around the JSON.
            """
        else:
            conversation_prompt = base_conversation_prompt

        # Define a specific user prompt that clearly asks for JSON format
        base_user_prompt = f"""
        Please generate a full conversation between the mental health professional and the patient.
        
        Here is the full questionnaire document you will be administering:
        
        ```
        {full_questionnaire_content}
        ```
        
        Profile for the Patient:
        {self.patient_profile}
        
        EXTREMELY IMPORTANT:
        - The mental health professional MUST ask ALL the questions from the questionnaire in order
        - Use the EXACT wording of the questions as they appear in the questionnaire
        - Do NOT skip any questions or add additional diagnostic questions
        - Do not stop after the first question
        - Make sure all {len(self.questions)} questions from the questionnaire are covered in the conversation
        - The patient should respond in NATURAL CONVERSATIONAL LANGUAGE, not with numerical ratings
        - Patient responses should be descriptive and elaborate on their experiences, not just "3" or "4"
        - Patient should describe their symptoms in their own words while addressing the severity implied by the questionnaire
        
        YOUR RESPONSE MUST BE A VALID JSON ARRAY of objects, each with 'role' and 'content' fields.
        
        The format MUST be exactly as follows, with proper JSON syntax:
        ```json
        [
            {{
                "role": "assistant",
                "content": "Introduction and first question"
            }},
            {{
                "role": "patient",
                "content": "Patient response in natural language, NOT numerical ratings"
            }},
            ...and so on until all questions are asked and answered
        ]
        ```
        
        DO NOT include any text before or after the JSON array.
        DO NOT include backticks or "json" markers.
        ONLY return the JSON array itself.
        
        The introduction should:
        - Introduces yourself as a mental health professional
        - Identifies the specific questionnaire you're using by name (from the document)
        - Explains the purpose of this specific assessment 
        - Reassures the patient about confidentiality and creating a safe space
        - Briefly explains how the assessment will proceed ({len(self.questions)} questions)
        - Indicates you're ready to begin with the first question
        """
        
        # Special handling for Groq provider
        if hasattr(self, 'provider') and self.provider == "groq":
            user_prompt = f"""
            Generate a full conversation between the mental health professional and patient as a JSON array.
            
            Here is the full questionnaire document you will be administering:
            
            ```
            {full_questionnaire_content}
            ```
            
            Profile for the Patient:
            {self.patient_profile}
            
            EXTREMELY IMPORTANT:
            - The mental health professional MUST ask ALL the questions from the questionnaire in order
            - Use the EXACT wording of the questions as they appear in the questionnaire
            - Do NOT skip any questions or add additional diagnostic questions
            - Make sure all {len(self.questions)} questions from the questionnaire are covered in the conversation
            - The patient should respond in NATURAL CONVERSATIONAL LANGUAGE, not with numerical ratings
            - Patient responses should be descriptive and elaborate on their experiences, not just "3" or "4"
            - Patient should describe their symptoms in their own words while addressing the severity implied by the questionnaire
            
            The introduction should:
            - Introduces yourself as a mental health professional
            - Identifies the specific questionnaire you're using by name (from the document above)
            - Explains the purpose of this specific assessment 
            - Reassures the patient about confidentiality and creating a safe space
            - Briefly explains how the assessment will proceed ({len(self.questions)} questions)
            - Indicates you're ready to begin with the first question
            
            CRITICAL: Return ONLY a raw JSON array with no text before or after. 
            Each object must have 'role' and 'content' fields. Include all {len(self.questions)} questions.
            
            Format: [{{\"role\":\"assistant\",\"content\":\"...\"}},{{\"role\":\"patient\",\"content\":\"Patient response in natural language, NOT numerical ratings\"}},...]
            """
        else:
            user_prompt = base_user_prompt
        
        # Create a temporary conversation for generating the introduction
        temp_conversation = [
            {"role": "system", "content": conversation_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate the conversation using the LLM
        result = self.client.chat(self.model, temp_conversation)
        conversation_text = result['response']
        
        # Clear existing conversation history
        self.conversation_history = []
        
        # Enhanced JSON extraction and cleaning
        def extract_and_clean_json(text):
            print(f"DEBUG: Raw conversation text length: {len(text)}")
            # Log the first and last 100 characters to see start/end format
            if len(text) > 200:
                print(f"DEBUG: Text starts with: {text[:100]}")
                print(f"DEBUG: Text ends with: {text[-100:]}")
            else:
                print(f"DEBUG: Full text: {text}")
            
            # Clean the response text to get valid JSON
            cleaned_text = text.strip()
            
            # Remove backticks, "json" markers, and other common prefixes/suffixes
            cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
            cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            
            # First approach: Try to extract array with regex
            # Remove any text before the first '[' and after the last ']'
            array_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
            if array_match:
                cleaned_text = array_match.group(0)
                print(f"DEBUG: Found JSON array with regex")
            
            # Clean control characters and Unicode quotes
            cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_text)
            cleaned_text = cleaned_text.replace('\u201c', '"').replace('\u201d', '"')
            cleaned_text = cleaned_text.replace('\u2018', "'").replace('\u2019', "'")
            
            # Fix common JSON syntax errors
            # Replace single quotes with double quotes (only for keys and string values)
            cleaned_text = re.sub(r'([{,])\s*\'([^\']+)\'\s*:', r'\1"\2":', cleaned_text)
            cleaned_text = re.sub(r':\s*\'([^\']+)\'\s*([,}])', r':"\1"\2', cleaned_text)
            
            # If there's still no proper JSON array structure, try harder to extract one
            if not (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
                print("DEBUG: No proper JSON array found, trying harder extraction")
                # Try to find anything that looks like JSON objects and build an array
                objects = re.findall(r'{.*?}', cleaned_text, re.DOTALL)
                if objects:
                    cleaned_text = "[" + ",".join(objects) + "]"
                    print(f"DEBUG: Built JSON array from {len(objects)} extracted objects")
            
            return cleaned_text
            
        # Clean and try to parse JSON
        cleaned_text = extract_and_clean_json(conversation_text)
        
        try:
            # Try to parse the cleaned JSON
            print(f"DEBUG: Attempting to parse JSON of length {len(cleaned_text)}")
            conversation_data = json.loads(cleaned_text)
            print(f"DEBUG: Successfully parsed JSON with {len(conversation_data)} items")
            
            # Validate that it's a list of properly structured messages
            if isinstance(conversation_data, list):
                for i, message in enumerate(conversation_data):
                    if not isinstance(message, dict):
                        print(f"DEBUG: Item {i} is not a dictionary: {message}")
                        continue
                    
                    if "role" not in message or "content" not in message:
                        print(f"DEBUG: Item {i} missing role or content: {message}")
                        continue
                    
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    # Validate and standardize roles
                    if any(r in role.lower() for r in ["assistant", "clinician", "therapist", "professional", "mental health"]):
                        role = "assistant"
                    elif any(r in role.lower() for r in ["user", "patient"]):
                        role = "patient"
                    else:
                        print(f"DEBUG: Invalid role '{role}' at item {i}, defaulting to 'assistant'")
                        role = "assistant"
                    
                    # Ensure content is a string
                    if not isinstance(content, str):
                        content = str(content)
                    
                    # Add the message to the conversation history
                    self.conversation_history.append({
                        "role": role,
                        "content": content
                    })
                
                print(f"DEBUG: Added {len(self.conversation_history)} messages to conversation history")
                
                # Extract responses from conversation history after successful parsing
                self._extract_responses_from_conversation_history()
            else:
                print(f"DEBUG: Parsed JSON is not a list: {type(conversation_data)}")
                # Try to handle single message case
                if isinstance(conversation_data, dict) and "role" in conversation_data and "content" in conversation_data:
                    print(f"DEBUG: Found single message dict, adding to conversation")
                    self.conversation_history.append({
                        "role": conversation_data["role"],
                        "content": conversation_data["content"]
                    })
                elif isinstance(conversation_data, dict):
                    # Try to extract role-content pairs from flat dict structure
                    for role_key, content in conversation_data.items():
                        role = role_key.lower()
                        if "assistant" in role or "clinician" in role or "therapist" in role:
                            std_role = "assistant"
                        elif "patient" in role or "user" in role or "client" in role:
                            std_role = "patient"
                        else:
                            continue
                        
                        if isinstance(content, str):
                            print(f"DEBUG: Adding message with role {std_role} from dict")
                            self.conversation_history.append({
                                "role": std_role,
                                "content": content
                            })
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse conversation as JSON: {e}")
            # If JSON parsing fails completely, try text-based extraction
            self._extract_conversation_from_text(conversation_text)
            # Extract responses from the conversation history after text extraction
            if self.conversation_history:
                self._extract_responses_from_conversation_history()
        
        # If we didn't get any messages, try to extract turn-by-turn conversation as a last resort
        if not self.conversation_history:
            print("Warning: Could not parse conversation in standard format, attempting fallback extraction")
            self._extract_responses_from_conversation(conversation_text)
        
        # Check if we have sufficient question-answer pairs
        # If not, regenerate the conversation
        MAX_REGENERATION_ATTEMPTS = 2
        regeneration_attempts = 0
        
        while len(self.responses) < 5 and regeneration_attempts < MAX_REGENERATION_ATTEMPTS:
            print(f"WARNING: Generated conversation only has {len(self.responses)} Q&A pairs, fewer than the required 5. Regenerating...")
            regeneration_attempts += 1
            
            # Clear existing data before regenerating
            self.conversation_history = []
            self.responses = []
            
            # Regenerate with a more explicit prompt
            enhanced_prompt = user_prompt + f"""
            
            CRITICAL REQUIREMENT: Your response MUST include a FULL conversation covering ALL {len(self.questions)} questions from the questionnaire. 
            The current generation only produced {len(self.responses)} question-answer pairs, which is insufficient.
            Make sure the mental health professional asks ALL questions and the patient responds to each one.
            """
            
            # Create a temporary conversation for regeneration
            regeneration_conversation = [
                {"role": "system", "content": conversation_prompt},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            # Generate a new conversation
            result = self.client.chat(self.model, regeneration_conversation)
            conversation_text = result['response']
            
            try:
                # Clean and try to parse JSON
                cleaned_text = extract_and_clean_json(conversation_text)
                
                # Try to parse the cleaned JSON
                conversation_data = json.loads(cleaned_text)
                
                # Validate and process the conversation data
                if isinstance(conversation_data, list):
                    # Process the conversation data as before
                    for i, message in enumerate(conversation_data):
                        if not isinstance(message, dict):
                            continue
                        
                        if "role" not in message or "content" not in message:
                            continue
                        
                        role = message.get("role", "")
                        content = message.get("content", "")
                        
                        # Validate and standardize roles
                        if any(r in role.lower() for r in ["assistant", "clinician", "therapist", "professional", "mental health"]):
                            role = "assistant"
                        elif any(r in role.lower() for r in ["user", "patient"]):
                            role = "patient"
                        else:
                            role = "assistant"
                        
                        # Ensure content is a string
                        if not isinstance(content, str):
                            content = str(content)
                        
                        # Add the message to the conversation history
                        self.conversation_history.append({
                            "role": role,
                            "content": content
                        })
                    
                    # Extract responses from conversation history after parsing
                    self._extract_responses_from_conversation_history()
                else:
                    # Try text-based extraction as a fallback
                    self._extract_conversation_from_text(conversation_text)
                    if self.conversation_history:
                        self._extract_responses_from_conversation_history()
            except json.JSONDecodeError:
                # If JSON parsing fails, try text-based extraction
                self._extract_conversation_from_text(conversation_text)
                if self.conversation_history:
                    self._extract_responses_from_conversation_history()
            
            # If we still don't have enough responses, try another method
            if not self.conversation_history or len(self.responses) < 5:
                self._extract_responses_from_conversation(conversation_text)
            
            print(f"Regeneration attempt {regeneration_attempts}: now have {len(self.responses)} Q&A pairs")
        
        if len(self.responses) < 5:
            print(f"WARNING: After {MAX_REGENERATION_ATTEMPTS} attempts, still only generated {len(self.responses)} Q&A pairs. Skipping this conversation attempt as instructed.")
        
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
            # First, try to clean up the JSON string to handle common parsing issues
            def clean_json_string(json_str):
                # Start with basic whitespace cleanup
                json_str = json_str.strip()
                
                # Remove markdown code markers
                json_str = re.sub(r'^```json\s*', '', json_str)
                json_str = re.sub(r'^```\s*', '', json_str)
                json_str = re.sub(r'\s*```$', '', json_str)
                
                # Extract just the JSON array if there's other text
                match = re.search(r'\[.*\]', json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                
                # Remove or replace common problematic characters
                json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                
                # Replace any Unicode quotes with standard quotes
                json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
                json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")
                
                return json_str
            
            # Clean and try to parse as JSON
            cleaned_text = clean_json_string(conversation)
            
            try:
                conversation_data = json.loads(cleaned_text)
                
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
                        # Consider all assistant messages as potential questions
                        # Rather than requiring a question mark which excludes prompts/statements
                        current_question = content
                    elif any(r in role_lower for r in ["user", "patient"]) and current_question:
                        # This is an answer to the previous question
                        self.responses.append((current_question, content))
                        current_question = None
                        
                print(f"[DEBUG] Successfully parsed JSON and extracted {len(self.responses)} Q&A pairs")
                
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON parsing failed after cleaning: {str(e)}")
                print(f"[DEBUG] Attempted to parse: {cleaned_text[:100]}...")
                # Fall through to regex approach below
        
        except Exception as e:
            print(f"[DEBUG] Error extracting responses from conversation: {str(e)}")
        
        # If we couldn't extract responses from JSON, try pattern matching
        if not self.responses:
            try:
                print("[DEBUG] Trying pattern matching to extract question-answer pairs")
                
                # Try to find pairs of assistant/patient messages
                # Pattern for "Assistant/Clinician: [content]" followed by "Patient: [answer]"
                # No longer requiring content to end with a question mark
                qa_pattern = re.compile(r'(?:assistant|clinician|therapist|professional|doctor)\s*:\s*([^\n]+)[^\n]*\n+(?:patient|user)\s*:\s*([^\n]+)', re.IGNORECASE)
                
                matches = qa_pattern.findall(conversation)
                for question, answer in matches:
                    self.responses.append((question.strip(), answer.strip()))
                    
                print(f"[DEBUG] Extracted {len(self.responses)} Q&A pairs using pattern matching")
            except Exception as e:
                print(f"[DEBUG] Error in pattern matching: {str(e)}")
        
        # If all else fails, use the questions from the questionnaire and empty responses
        if not self.responses and self.questions:
            print("[DEBUG] Using questionnaire questions with empty responses")
            for question in self.questions:
                self.responses.append((question, "No response provided"))
        
        print(f"[DEBUG] Final extraction result: {len(self.responses)} question-answer pairs")

    def generate_diagnosis(self):
        """
        Generate a diagnosis based on the patient's responses.
        
        Returns:
            dict: Diagnosis from the assistant with RAG usage information
        """

        # print("--------------------------------")
        # print("Responses:")
        # print(self.responses)
        # print(f"Number of Q&A pairs extracted: {len(self.responses)}")
        # print("--------------------------------")

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
        
        # Enhance with RAG if available and not disabled - ONLY use RAG during diagnosis phase
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
        else:
            # Skip RAG if it's disabled
            if self.disable_rag_evaluation:
                print("[DEBUG] RAG evaluation disabled for diagnosis generation")
        
        self.conversation_history.append({"role": "user", "content": diagnosis_prompt})
        
        result = self.client.chat(self.model, self.conversation_history)
        diagnosis = result['response']
        self.context = result.get('context')
        
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

    def _extract_conversation_from_text(self, text):
        """Extract conversation turns from plain text when JSON parsing fails."""
        import re
        print("DEBUG: Attempting text-based conversation extraction")
        
        # Define patterns to identify speaker turns
        patterns = [
            # Look for patterns like "Assistant: message" or "Patient: message"
            r'(?:^|\n)(assistant|patient|clinician|therapist|doctor|user|client):\s*(.*?)(?=\n(?:assistant|patient|clinician|therapist|doctor|user|client):|$)',
            # Alternative pattern with quotes or brackets
            r'(?:^|\n)[\'"]?(assistant|patient|clinician|therapist|doctor|user|client)[\'"]?\s*[:\-]\s*[\'"]?(.*?)[\'"]?(?=\n|$)',
            # JSON-like format without proper syntax
            r'role[\'"]?\s*:\s*[\'"]?(assistant|patient|clinician|therapist|doctor|user|client)[\'"]?[,\s]+[\'"]?content[\'"]?\s*:\s*[\'"]?(.*?)[\'"]?(?=[,\}]|$)'
        ]
        
        # Try each pattern until we get some results
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                print(f"DEBUG: Found {len(matches)} conversation turns with pattern")
                
                for role_text, content in matches:
                    # Standardize role
                    role = role_text.lower()
                    if role in ["assistant", "clinician", "therapist", "doctor"]:
                        std_role = "assistant"
                    else:
                        std_role = "patient"
                    
                    # Clean up content
                    content = content.strip()
                    if content:
                        self.conversation_history.append({
                            "role": std_role,
                            "content": content
                        })
                
                print(f"DEBUG: Extracted {len(self.conversation_history)} messages from text")
                return
        
        # Last resort: just treat the whole thing as a single assistant message
        if not self.conversation_history:
            print("DEBUG: No patterns matched, treating entire text as assistant message")
            self.conversation_history.append({
                "role": "assistant",
                "content": text
            })
            
    def _extract_responses_from_conversation_history(self):
        """
        Extract question-answer pairs directly from the parsed conversation history.
        This method is called after successful JSON parsing to populate the responses.
        """
        # Clear existing responses
        self.responses = []
        
        print("DEBUG: Extracting responses from conversation history...")
        
        current_question = None
        
        # Process conversation history to extract Q&A pairs
        for i in range(len(self.conversation_history)):
            msg = self.conversation_history[i]
            
            if msg["role"] == "assistant":
                # This is a potential question/prompt from the assistant
                current_question = msg["content"]
            elif msg["role"] == "patient" and current_question:
                # This is an answer to the previous question/prompt
                self.responses.append((current_question, msg["content"]))
                current_question = None
        
        print(f"DEBUG: Extracted {len(self.responses)} Q&A pairs from conversation history")