from abc import ABC, abstractmethod

class LLMClient(ABC):
    """
    Base class for LLM clients.
    This abstract class defines the common interface that
    all LLM clients should implement.
    """
    
    @abstractmethod
    def generate(self, model, prompt, system=None, context=None):
        """
        Generate a response from the model.
        
        Args:
            model (str): Name of the model to use
            prompt (str): The prompt to send to the model
            system (str, optional): System message for the model
            context (list, optional): Previous conversation context
            
        Returns:
            dict: Response from the model including generated text
        """
        pass
        
    @abstractmethod
    def chat(self, model, messages, stream=False):
        """
        Generate a chat response using the model.
        
        Args:
            model (str): Name of the model to use
            messages (list): List of message dictionaries with 'role' and 'content'
            stream (bool): Whether to stream the response
            
        Returns:
            dict: Response from the model
        """
        pass
    
    @staticmethod
    def create(provider_name, **kwargs):
        """
        Factory method to create an LLM client instance.
        
        Args:
            provider_name (str): Provider name ("ollama", "groq", or "openai")
            **kwargs: Additional arguments to pass to the client constructor
            
        Returns:
            LLMClient: An instance of the appropriate LLM client
        """
        if provider_name.lower() == "ollama":
            from utils.ollama_client import OllamaClient
            return OllamaClient(**kwargs)
        elif provider_name.lower() == "groq":
            from utils.groq_client import GroqClient
            return GroqClient(**kwargs)
        elif provider_name.lower() == "openai":
            from utils.openai_client import OpenAIClient
            return OpenAIClient(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider_name}. Supported providers: ollama, groq, openai") 