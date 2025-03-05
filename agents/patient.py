import os
from utils.ollama_client import OllamaClient

class Patient:
    def __init__(self, ollama_url, model):
        """
        Initialize the Patient agent.
        
        Args:
            ollama_url (str): URL for the Ollama API
            model (str): Ollama model to use
        """
        self.client = OllamaClient(base_url=ollama_url)
        self.model = model
        self.context = None
        
        # Load system prompt from file
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "prompts", "patient_prompt.txt")
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()
        
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
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
