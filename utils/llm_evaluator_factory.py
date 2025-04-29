"""
Factory to create LLM evaluators with different backends.
"""

import os
from typing import Dict, Any, Optional

class LLMEvaluatorFactory:
    """Factory for creating LLM evaluators with different backends."""
    
    @staticmethod
    def create_evaluator(provider: str = "ollama", 
                        model: str = "qwen2.5:3b", 
                        api_key: Optional[str] = None, 
                        api_url: Optional[str] = None) -> Any:
        """
        Create an LLM evaluator with the specified backend.
        
        Args:
            provider: LLM provider ('ollama' or 'groq')
            model: Model name to use
            api_key: API key for cloud providers like Groq
            api_url: URL for the API (optional)
            
        Returns:
            An instance of the appropriate LLM evaluator
        """
        # Import here to avoid circular imports
        from utils.llm_evaluation import LLMEvaluator
        
        if provider.lower() == "ollama":
            # Default to http://localhost:11434 if not provided
            ollama_url = api_url or "http://localhost:11434"
            return LLMEvaluator(ollama_url=ollama_url, model=model)
        
        elif provider.lower() == "groq":
            # Create a Groq-based evaluator
            from utils.groq_client import GroqClient
            
            # Get API key from args or environment
            groq_api_key = api_key or os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("Groq API key not provided and GROQ_API_KEY environment variable not set")
            
            # Create client
            groq_client = GroqClient(api_key=groq_api_key)
            
            # We need to map model names between Ollama and Groq
            # For example, map qwen2.5:3b to an appropriate Groq model
            groq_model_mapping = {
                "qwen2.5:3b": "llama3-8b-8192",  # Example mapping
                "qwen2.5:7b": "llama3-8b-8192",
                "qwen2.5:72b": "llama3-70b-8192",
                "llama3:8b": "llama3-8b-8192",
                "llama3:70b": "llama3-70b-8192",
                "gemma:7b": "gemma-7b-it"
            }
            
            # Use mapped model or just pass through the model name
            groq_model = groq_model_mapping.get(model, model)
            
            # Create the evaluator with the Groq client and model
            return LLMEvaluator(client=groq_client, model=groq_model)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    @staticmethod
    def get_available_providers() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available providers.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "ollama": {
                "name": "Ollama",
                "description": "Local LLM runtime with various models",
                "requires_api_key": False,
                "default_url": "http://localhost:11434",
                "recommended_models": ["qwen2.5:3b", "llama3:8b"]
            },
            "groq": {
                "name": "Groq",
                "description": "Cloud LLM API with fast inference",
                "requires_api_key": True,
                "default_url": "https://api.groq.com/openai/v1",
                "recommended_models": ["llama3-8b-8192", "llama3-70b-8192", "gemma-7b-it"]
            }
        } 