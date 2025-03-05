import os
from utils.ollama_client import OllamaClient

class Patient:
    def __init__(self, ollama_url, model, profile_name=None):
        """
        Initialize the Patient agent.
        
        Args:
            ollama_url (str): URL for the Ollama API
            model (str): Ollama model to use
            profile_name (str, optional): Name of the patient profile to use
        """
        self.client = OllamaClient(base_url=ollama_url)
        self.model = model
        self.context = None
        
        # Load system prompt from profile or default
        self.system_prompt = self._load_profile(profile_name)
        
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    def _load_profile(self, profile_name):
        """
        Load a patient profile from a file.
        
        Args:
            profile_name (str, optional): Name of the profile to load
            
        Returns:
            str: The profile content or default profile if not found
        """
        # Set default profile directory
        profile_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
        
        # If directory doesn't exist, create it
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
        
        # If no profile specified, use the default
        if not profile_name:
            profile_name = "depression"
        
        # If .txt extension not provided, add it
        if not profile_name.endswith('.txt'):
            profile_name += '.txt'
        
        # Try to load the specified profile
        profile_path = os.path.join(profile_dir, profile_name)
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                return f.read()
        
        # If profile doesn't exist, fall back to the default prompt
        default_prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                        "prompts", "patient_prompt.txt")
        
        with open(default_prompt_path, 'r') as f:
            return f.read()
    
    def respond_to_question(self, question):
        """
        Generate a response to the mental health assistant's question.
        
        Args:
            question (str): Question from the mental health assistant
            
        Returns:
            str: Patient's response to the question
        """
        # Add question to conversation history
        self.conversation_history.append({"role": "user", "content": question})
        
        result = self.client.chat(self.model, self.conversation_history)
        patient_response = result['response']
        self.context = result['context']
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": patient_response})
        
        return patient_response
    
    @staticmethod
    def list_available_profiles():
        """
        List all available patient profiles.
        
        Returns:
            list: List of available profile names (without extension)
        """
        profile_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
        
        if not os.path.exists(profile_dir):
            return []
        
        return [f.replace('.txt', '') for f in os.listdir(profile_dir) 
                if f.endswith('.txt') and os.path.isfile(os.path.join(profile_dir, f))]
