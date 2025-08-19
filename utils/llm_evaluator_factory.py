"""
Factory to create LLM evaluators with different backends.
"""

import os
from typing import Dict, Any, Optional

class LLMEvaluatorFactory:
    """Factory for creating LLM evaluators with different backends."""
    
    @staticmethod
    def create_evaluator(provider: str = "ollama", 
                        model: str = "qwen3:4b", 
                        api_key: Optional[str] = None, 
                        api_url: Optional[str] = None) -> Any:
        """
        Create an LLM evaluator with the specified backend.
        
        Args:
            provider: LLM provider ('ollama', 'groq', or 'openai')
            model: Model name to use
            api_key: API key for cloud providers like Groq or OpenAI
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
                "qwen3:4b": "llama3-8b-8192",  # Example mapping
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
            
        elif provider.lower() == "openai":
            # Create an OpenAI-based evaluator
            from utils.openai_client import OpenAIClient
            
            # Get API key from args or environment
            openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
            
            # Create client
            openai_client = OpenAIClient(api_key=openai_api_key, api_base=api_url)
            
            # We need to map model names if needed
            openai_model_mapping = {
                "qwen3:4b": "gpt-4o-mini",  # Map to an appropriate OpenAI model
                "qwen2.5:7b": "gpt-4.1-mini", 
                "qwen2.5:72b": "gpt-4.1",
                "llama3:8b": "gpt-4.1-mini",
                "llama3:70b": "gpt-4.1"
            }
            
            # Use mapped model or just pass through the model name
            openai_model = openai_model_mapping.get(model, model)
            
            # Create the evaluator with the OpenAI client and model
            return LLMEvaluator(client=openai_client, model=openai_model)
        
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
                "recommended_models": ["qwen3:4b", "llama3:8b"]
            },
            "groq": {
                "name": "Groq",
                "description": "Cloud LLM API with fast inference",
                "requires_api_key": True,
                "default_url": "https://api.groq.com/openai/v1",
                "recommended_models": ["llama3-8b-8192", "llama3-70b-8192", "gemma-7b-it"]
            },
            "openai": {
                "name": "OpenAI",
                "description": "OpenAI's API for language models",
                "requires_api_key": True,
                "default_url": "https://api.openai.com/v1",
                "recommended_models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano"]
            }
        } 