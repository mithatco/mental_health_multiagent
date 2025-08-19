import os
import time
from collections import deque
from datetime import datetime
from utils.llm_client_base import LLMClient

# Import OpenAI library
from openai import OpenAI
from openai.types.chat import ChatCompletion
import openai

class OpenAIClient(LLMClient):
    # Class-level variable for rate limiting across all instances
    _api_call_timestamps = deque()
    _rate_limit = 3500  # Default RPM for most OpenAI models
    _rate_window = 60  # 1 minute in seconds
    
    # Rate limiting statistics
    _total_calls = 0
    _total_wait_time = 0
    _rate_limit_hits = 0
    
    def __init__(self, api_key=None, api_base=None, model_mapping=None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to load from 
                                    environment variable OPENAI_API_KEY
            api_base (str, optional): Base URL for the OpenAI API. If not provided, 
                                     will use the default OpenAI API endpoint
            model_mapping (dict, optional): Mapping of shorthand model names to full model names
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=api_base  # Will use default if None
        )
        
        # Default model mapping
        self.model_mapping = model_mapping or {
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini",
            "gpt-4.1-nano": "gpt-4.1-nano"
        }
    
    def _wait_for_rate_limit(self):
        """
        Check if we've hit the rate limit and wait if necessary.
        """
        current_time = time.time()
        OpenAIClient._total_calls += 1
        
        # Remove timestamps older than the rate window
        while OpenAIClient._api_call_timestamps and OpenAIClient._api_call_timestamps[0] < current_time - OpenAIClient._rate_window:
            OpenAIClient._api_call_timestamps.popleft()
        
        # Check if we've reached the rate limit
        if len(OpenAIClient._api_call_timestamps) >= OpenAIClient._rate_limit:
            # Calculate how long to wait
            oldest_call = OpenAIClient._api_call_timestamps[0]
            wait_time = oldest_call + OpenAIClient._rate_window - current_time
            
            if wait_time > 0:
                # Update statistics
                OpenAIClient._rate_limit_hits += 1
                OpenAIClient._total_wait_time += wait_time
                
                print(f"OpenAI rate limit reached ({OpenAIClient._rate_limit} calls per {OpenAIClient._rate_window}s).")
                print(f"Waiting {wait_time:.2f} seconds...")
                print(f"Total API calls: {OpenAIClient._total_calls}, Total wait time: {OpenAIClient._total_wait_time:.2f}s")
                
                time.sleep(wait_time)
                # Recursive call to check again after waiting
                return self._wait_for_rate_limit()
        
        # Add the current timestamp to our record
        OpenAIClient._api_call_timestamps.append(current_time)
        return
    
    def _resolve_model_name(self, model):
        """
        Resolve the model name to the full OpenAI model name.
        
        Args:
            model (str): Model name or shorthand
            
        Returns:
            str: Full model name for the OpenAI API
        """
        return self.model_mapping.get(model, model)
    
    def generate(self, model, prompt, system=None, context=None):
        """
        Generate a response using the OpenAI API.
        
        Args:
            model (str): Name of the model to use
            prompt (str): The prompt to send to the model
            system (str, optional): System message for the model
            context (list, optional): Previous conversation context (ignored for OpenAI)
            
        Returns:
            dict: Response from the model including generated text
        """
        # Apply rate limiting
        self._wait_for_rate_limit()
        
        resolved_model = self._resolve_model_name(model)
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Use the OpenAI client to create a chat completion
            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=0.7
            )
            
            # Extract the response text
            text_response = response.choices[0].message.content
            
            # Extract usage information if available
            usage = {}
            if hasattr(response, 'usage'):
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            return {
                "response": text_response,
                "context": None,  # OpenAI doesn't return context like Ollama
                "usage": usage
            }
            
        except openai.RateLimitError as e:
            print(f"OpenAI API rate limit exceeded: {str(e)}")
            # Extract retry-after information if available
            wait_time = 20.0  # Default wait time
            retry_after = getattr(e, 'retry_after', None)
            if retry_after:
                wait_time = float(retry_after)
                
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            # Retry the request
            return self.generate(model, prompt, system, context)
            
        except openai.APIError as e:
            print(f"OpenAI API error: {str(e)}")
            # Check if this is a server error (HTTP 5xx)
            if getattr(e, 'status_code', 0) >= 500:
                wait_time = 5.0
                print(f"OpenAI API server error. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                # Retry the request
                return self.generate(model, prompt, system, context)
            
            return {"response": f"Error communicating with OpenAI: {str(e)}", "context": None}
            
        except Exception as e:
            print(f"Unexpected error with OpenAI API: {str(e)}")
            return {"response": f"Unexpected error with OpenAI: {str(e)}", "context": None}
        
    def chat(self, model, messages, stream=False):
        """
        Generate a chat response using the OpenAI API.
        
        Args:
            model (str): Name of the model to use
            messages (list): List of message dictionaries with 'role' and 'content'
            stream (bool): Whether to stream the response (not implemented yet)
            
        Returns:
            dict: Response from the model
        """
        # Apply rate limiting
        self._wait_for_rate_limit()
        
        resolved_model = self._resolve_model_name(model)
        
        # Format messages for the OpenAI API
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        try:
            # Use the OpenAI client to create a chat completion
            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=formatted_messages,
                temperature=0.7,
                stream=stream
            )
            
            # Handle streaming responses if requested
            if stream:
                # Not fully implemented yet - this is a placeholder
                # The caller would need to handle the stream
                return {"response": response, "streaming": True}
            
            # Extract the response text
            text_response = response.choices[0].message.content
            
            # Extract usage information if available
            usage = {}
            if hasattr(response, 'usage'):
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            return {
                "response": text_response,
                "context": None,
                "usage": usage
            }
            
        except openai.RateLimitError as e:
            print(f"OpenAI API rate limit exceeded: {str(e)}")
            # Extract retry-after information if available
            wait_time = 20.0  # Default wait time
            retry_after = getattr(e, 'retry_after', None)
            if retry_after:
                wait_time = float(retry_after)
                
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            # Retry the request
            return self.chat(model, messages, stream)
            
        except openai.APIError as e:
            print(f"OpenAI API error: {str(e)}")
            # Check if this is a server error (HTTP 5xx)
            if getattr(e, 'status_code', 0) >= 500:
                wait_time = 5.0
                print(f"OpenAI API server error. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                # Retry the request
                return self.chat(model, messages, stream)
            
            return {"response": f"Error communicating with OpenAI: {str(e)}", "context": None}
            
        except Exception as e:
            print(f"Unexpected error with OpenAI API: {str(e)}")
            return {"response": f"Unexpected error with OpenAI: {str(e)}", "context": None} 