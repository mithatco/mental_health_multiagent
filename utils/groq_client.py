import os
import requests
import json
import time
from collections import deque
from datetime import datetime
from utils.llm_client_base import LLMClient

class GroqClient(LLMClient):
    # Class-level variable for rate limiting across all instances
    _api_call_timestamps = deque()
    _rate_limit = 1000  # 1000 calls per minute
    _rate_window = 60  # 1 minute in seconds
    
    # Rate limiting statistics
    _total_calls = 0
    _total_wait_time = 0
    _rate_limit_hits = 0
    
    # Backoff settings
    _backoff_base = 1.0  # Base seconds to wait
    _backoff_factor = 1.5  # Multiplier for each consecutive rate limit
    _max_backoff = 30.0  # Maximum seconds to wait
    _consecutive_rate_limits = 0  # Counter for consecutive rate limits

    def __init__(self, api_key=None):
        """
        Initialize the Groq client.
        
        Args:
            api_key (str, optional): Groq API key. If not provided, will try to load from 
                                    environment variable GROQ_API_KEY
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set as GROQ_API_KEY environment variable")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _wait_for_rate_limit(self):
        """
        Check if we've hit the rate limit and wait if necessary.
        Uses exponential backoff for repeated rate limit hits.
        """
        current_time = time.time()
        GroqClient._total_calls += 1
        
        # Remove timestamps older than the rate window
        while GroqClient._api_call_timestamps and GroqClient._api_call_timestamps[0] < current_time - GroqClient._rate_window:
            GroqClient._api_call_timestamps.popleft()
        
        # Check if we've reached the rate limit
        if len(GroqClient._api_call_timestamps) >= GroqClient._rate_limit:
            # Calculate how long to wait using exponential backoff
            oldest_call = GroqClient._api_call_timestamps[0]
            base_wait_time = oldest_call + GroqClient._rate_window - current_time
            
            # Apply backoff if we're hitting limits consecutively
            GroqClient._consecutive_rate_limits += 1
            backoff_multiplier = min(
                GroqClient._backoff_factor ** (GroqClient._consecutive_rate_limits - 1),
                GroqClient._max_backoff / max(base_wait_time, GroqClient._backoff_base)
            )
            
            wait_time = max(base_wait_time, GroqClient._backoff_base) * backoff_multiplier
            wait_time = min(wait_time, GroqClient._max_backoff)  # Cap at max_backoff
            
            if wait_time > 0:
                # Update statistics
                GroqClient._rate_limit_hits += 1
                GroqClient._total_wait_time += wait_time
                
                print(f"Rate limit reached ({GroqClient._rate_limit} calls per {GroqClient._rate_window}s).")
                print(f"Waiting {wait_time:.2f} seconds (hit #{GroqClient._rate_limit_hits}, consecutive #{GroqClient._consecutive_rate_limits})...")
                print(f"Total API calls: {GroqClient._total_calls}, Total wait time: {GroqClient._total_wait_time:.2f}s")
                
                time.sleep(wait_time)
                # Recursive call to check again after waiting
                return self._wait_for_rate_limit()
        else:
            # Reset consecutive counter if we're not at the limit
            GroqClient._consecutive_rate_limits = 0
        
        # Add the current timestamp to our record
        GroqClient._api_call_timestamps.append(current_time)
        return
        
    @staticmethod
    def get_rate_limit_stats():
        """
        Get the current rate limiting statistics.
        
        Returns:
            dict: Statistics about rate limiting
        """
        return {
            "total_calls": GroqClient._total_calls,
            "rate_limit_hits": GroqClient._rate_limit_hits,
            "total_wait_time": GroqClient._total_wait_time,
            "current_window_calls": len(GroqClient._api_call_timestamps),
            "consecutive_rate_limits": GroqClient._consecutive_rate_limits,
            "rate_limit": GroqClient._rate_limit,
            "rate_window_seconds": GroqClient._rate_window
        }
    
    def generate(self, model, prompt, system=None, context=None):
        """
        Generate a response using the Groq API.
        
        Args:
            model (str): Name of the model to use (e.g., "llama3-70b-8192")
            prompt (str): The prompt to send to the model
            system (str, optional): System message for the model
            context (list, optional): Previous conversation context (ignored for Groq)
            
        Returns:
            dict: Response from the model including generated text
        """
        # Apply rate limiting
        self._wait_for_rate_limit()
        
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Create the payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,  # Default temperature
            "max_tokens": 1024   # Default max tokens
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            response_data = response.json()
            text_response = response_data["choices"][0]["message"]["content"]
            
            return {
                "response": text_response,
                "context": None  # Groq doesn't return context like Ollama
            }
            
        except requests.exceptions.RequestException as e:
            # Check if this is a rate limit error (HTTP 429)
            if hasattr(e, 'response') and e.response.status_code == 429:
                # Check for retry-after header
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                        print(f"Groq API rate limit exceeded. Server requested wait time: {wait_time} seconds")
                    except (ValueError, TypeError):
                        # If header exists but can't be converted to float, use default
                        wait_time = 5.0
                        print(f"Groq API rate limit exceeded. Using default wait time: {wait_time} seconds")
                else:
                    # Default if no retry-after header
                    wait_time = 5.0
                    print(f"Groq API rate limit exceeded. No retry-after header. Using default wait time: {wait_time} seconds")
                
                # Update statistics
                GroqClient._rate_limit_hits += 1
                GroqClient._total_wait_time += wait_time
                
                # Wait before retrying
                time.sleep(wait_time)
                
                # Retry the request (recursive call)
                return self.generate(model, prompt, system, context)
            else:
                print(f"Error calling Groq API: {str(e)}")
                return {"response": f"Error communicating with Groq: {str(e)}", "context": None}
        
    def chat(self, model, messages, stream=False):
        """
        Generate a chat response using the Groq API.
        
        Args:
            model (str): Name of the model to use (e.g., "llama3-70b-8192")
            messages (list): List of message dictionaries with 'role' and 'content'
            stream (bool): Whether to stream the response (not implemented for Groq yet)
            
        Returns:
            dict: Response from the model
        """
        # Apply rate limiting
        self._wait_for_rate_limit()
        
        # Format messages for the Groq API
        formatted_messages = []
        
        # Check if we need to enhance the system message for conversation formatting
        # This is important for full conversation generation to ensure proper JSON structure
        needs_json_format = False
        for msg in messages:
            if msg.get("role") == "system" and "questionnaire" in msg.get("content", "").lower() and "conversation" in msg.get("content", "").lower():
                needs_json_format = True
                break
        
        # Add improved system message for JSON formatting if needed
        if needs_json_format:
            print("Enhancing system message for Groq JSON formatting")
            # Look for existing system message
            has_system = any(msg.get("role") == "system" for msg in messages)
            
            if has_system:
                # Modify existing system messages
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        # Enhance the system message with JSON formatting instructions
                        formatted_messages.append({
                            "role": "system",
                            "content": msg.get("content", "") + "\n\nIMPORTANT: Your response MUST be a valid JSON array of objects. Each object MUST have 'role' and 'content' fields. The 'role' must be either 'assistant' for the mental health professional or 'patient' for the patient. Format: [{\"role\":\"assistant\",\"content\":\"message\"},{\"role\":\"patient\",\"content\":\"message\"}]. Do NOT include any text before or after the JSON array.\n\nCRITICAL: DO NOT introduce the mental health assessment as a \"Somatic Symptom Disorder questionnaire\" or any specific disorder questionnaire unless explicitly named as such in the instructions. Simply use the actual questionnaire name as provided."
                        })
                    else:
                        # Copy other messages as-is
                        formatted_messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
            else:
                # Add new system message at the beginning
                formatted_messages.append({
                    "role": "system",
                    "content": "IMPORTANT: Your response MUST be a valid JSON array of objects. Each object MUST have 'role' and 'content' fields. The 'role' must be either 'assistant' for the mental health professional or 'patient' for the patient. Format: [{\"role\":\"assistant\",\"content\":\"message\"},{\"role\":\"patient\",\"content\":\"message\"}]. Do NOT include any text before or after the JSON array.\n\nCRITICAL: DO NOT introduce the mental health assessment as a \"Somatic Symptom Disorder questionnaire\" or any specific disorder questionnaire unless explicitly named as such in the instructions. Simply use the actual questionnaire name as provided."
                })
                # Add all other messages
                for msg in messages:
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
        else:
            # Just copy the messages if we don't need special formatting
            for msg in messages:
                # Ensure role is one of system, user, or assistant
                if msg["role"] in ["system", "user", "assistant"]:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Create the payload
        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 2048  # Increased for full conversations
        }
        
        # Handle streaming if requested (not implemented yet)
        if stream:
            print("Warning: Streaming not yet implemented for Groq client")
            
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            response_data = response.json()
            text_response = response_data["choices"][0]["message"]["content"]
            
            # Log the first 100 characters of the response for debugging
            if len(text_response) > 100:
                print(f"Groq response preview: {text_response[:100]}...")
            else:
                print(f"Groq response: {text_response}")
            
            return {
                "response": text_response,
                "context": None  # Groq doesn't return context like Ollama
            }
            
        except requests.exceptions.RequestException as e:
            # Check if this is a rate limit error (HTTP 429)
            if hasattr(e, 'response') and e.response.status_code == 429:
                # Check for retry-after header
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                        print(f"Groq API rate limit exceeded. Server requested wait time: {wait_time} seconds")
                    except (ValueError, TypeError):
                        # If header exists but can't be converted to float, use default
                        wait_time = 5.0
                        print(f"Groq API rate limit exceeded. Using default wait time: {wait_time} seconds")
                else:
                    # Default if no retry-after header
                    wait_time = 5.0
                    print(f"Groq API rate limit exceeded. No retry-after header. Using default wait time: {wait_time} seconds")
                
                # Update statistics
                GroqClient._rate_limit_hits += 1
                GroqClient._total_wait_time += wait_time
                
                # Wait before retrying
                time.sleep(wait_time)
                
                # Retry the request (recursive call)
                return self.chat(model, messages, stream)
            else:
                print(f"Error calling Groq API: {str(e)}")
                return {"response": f"Error communicating with Groq: {str(e)}", "context": None} 