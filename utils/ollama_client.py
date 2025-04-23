import requests
import json
from utils.llm_client_base import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, base_url="http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url (str): Base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_chat = f"{base_url}/api/chat"
    
    def generate(self, model, prompt, system=None, context=None):
        """
        Generate a response using the Ollama API.
        
        Args:
            model (str): Name of the model to use
            prompt (str): The prompt to send to the model
            system (str, optional): System message for the model
            context (list, optional): Previous conversation context
            
        Returns:
            dict: Response from the model including generated text and context
        """
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        if system:
            payload["system"] = system
        
        if context:
            payload["context"] = context
        
        try:
            response = requests.post(self.api_generate, json=payload)
            response.raise_for_status()
            
            # Ollama streams responses, so we need to parse the JSON lines
            full_response = ""
            for line in response.text.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    full_response += data.get('response', '')
            
            # Return the context if provided in the response
            final_data = json.loads(response.text.strip().split('\n')[-1])
            context = final_data.get('context', None)
            
            return {
                "response": full_response,
                "context": context
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {str(e)}")
            return {"response": "Error communicating with Ollama", "context": None}
        
    def chat(self, model, messages, stream=False):
        """
        Generate a chat response using the Ollama API.
        
        Args:
            model (str): Name of the model to use
            messages (list): List of message dictionaries with 'role' and 'content'
            stream (bool): Whether to stream the response
            
        Returns:
            dict: Response from the model
        """
        # Extract system message if it exists
        system_message = None
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)
        
        # Format the conversation history into a prompt for Ollama
        prompt = ""
        for msg in chat_messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        
        # Add the latest prompt if the last message is from user
        if chat_messages and chat_messages[-1]["role"] == "user":
            prompt = chat_messages[-1]["content"]
        
        return self.generate(model, prompt, system=system_message)
