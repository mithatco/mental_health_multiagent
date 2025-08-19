import os
import random
from utils.llm_client_base import LLMClient

class Patient:
    """Agent representing a patient with mental health concerns."""
    
    def __init__(self, provider="ollama", provider_options=None, model="qwen3:4b", profile_name=None):
        """
        Initialize the patient agent.
        
        Args:
            provider (str): LLM provider to use (ollama or groq)
            provider_options (dict, optional): Options to pass to the provider client
            model (str): Model to use with the provider
            profile_name (str, optional): Name of the patient profile to use
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
        
        # Set up patient profile
        self.profile_name = profile_name
        self.profile = self._load_profile(profile_name)
        
        # Set up conversation history with system prompt
        strong_system_prompt = self._create_strong_system_prompt()
        self.conversation_history = [{"role": "system", "content": strong_system_prompt}]
    
    def _create_strong_system_prompt(self):
        """Create a strong system prompt that clearly establishes the patient role."""
        base_prompt = """You are roleplaying as a patient with specific symptoms in a mental health assessment.
        
                    IMPORTANT: You must ALWAYS respond AS THE PATIENT, never break character or respond as an AI assistant.

                    You are participating in a mental health assessment interview. A mental health professional will ask you questions, and you must answer them based on your patient profile.

                    Your responses should:
                    1. ALWAYS be in first person
                    2. NEVER ask questions back to the interviewer unless clarifying a question
                    3. NEVER explain that you're an AI or that you're roleplaying
                    4. NEVER refuse to answer based on AI limitations 
                    5. NEVER mention what your disorder is 
                    5. ALWAYS express the symptoms listed in your profile
                    6. ALWAYS stay consistent with your character's experiences and background

                    You should express genuine emotional responses matching your condition and should appear to be seeking help.
                    """

        if self.profile:
            # Add profile-specific information
            profile_prompt = f"""
                Your specific patient profile is: {self.profile_name}

                You have these symptoms and characteristics:
                {self.profile}

                Remember that you ARE this patient right now. Answer all questions as this person would, based on their symptoms and experiences.
            """
            return base_prompt + profile_prompt
        else:
            # Generic patient with mild symptoms if no profile specified
            return base_prompt + """
                        You are experiencing mild symptoms of anxiety and depression, including occasional worry, some trouble sleeping, and decreased interest in activities you used to enjoy.
                    """
    
    def _load_profile(self, profile_name):
        """Load a patient profile from file."""
        if not profile_name:
            return None
        
        profiles_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
        
        # Try to load from the profiles directory
        profile_path = os.path.join(profiles_dir, f"{profile_name}.txt")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile_content = f.read()
            return profile_content
        else:
            print(f"Warning: Profile '{profile_name}' not found. Using default profile.")
            return None
    
    @staticmethod
    def list_available_profiles():
        """List all available patient profiles."""
        profiles_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
        
        if not os.path.exists(profiles_dir):
            os.makedirs(profiles_dir)
            return []
        
        profile_files = [f[:-4] for f in os.listdir(profiles_dir) 
                         if f.endswith('.txt')]
        return profile_files
    
    def respond_to_question(self, question):
        """
        Generate a response to a question from the mental health professional.
        
        Args:
            question (str): Question from the professional
            
        Returns:
            str: Patient's response
        """
        # Add the question to conversation history
        self.conversation_history.append({"role": "user", "content": f"Mental Health Professional: {question}"})
        
        # Generate short context reminder to stay in character
        reminder = f"Remember to respond AS THE PATIENT with {self.profile_name or 'mental health issues'}. Express the symptoms in your profile. Answer directly as this patient, expressing your symptoms and experiences."
        temp_history = self.conversation_history.copy()
        temp_history.append({"role": "system", "content": reminder})
        
        # Generate response - removed the system_prompt parameter that was causing the error
        result = self.client.chat(
            self.model, 
            temp_history
        )
        
        response = result['response']
        
        # Clean up response to ensure it's patient-like
        response = self._ensure_patient_response(response)
        
        # Add the response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _ensure_patient_response(self, response):
        """Ensure the response is appropriate for a patient, removing AI-like language."""
        # Remove common AI/assistant phrases
        ai_phrases = [
            "As an AI, I",
            "As an assistant",
            "I'm an AI",
            "I am an AI",
            "I'm a language model",
            "I am a language model",
            "I'm not a real patient",
            "I don't actually have",
            "I cannot provide",
            "I don't have personal experiences"
        ]
        
        for phrase in ai_phrases:
            if phrase.lower() in response.lower():
                # Replace with a more patient-appropriate phrase
                response = response.replace(phrase, "I")
        
        # Check if the response is asking a question back
        if response.strip().endswith("?") and not "?" in response.strip()[:-1]:
            response = response.strip()[:-1] + "."
            
        return response
