"""
Utility for evaluating chat logs using LLM-based evaluation.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

# Import the LLM evaluator directly - no fallbacks or adapters
from .llm_evaluation import LLMEvaluator, ChatLogEvaluator as LLMChatLogEvaluator

class ChatLogEvaluator:
    """Evaluate chat logs using LLM-based evaluation."""
    
    def __init__(self, logs_dir: str, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:3b"):
        """
        Initialize the chat log evaluator.
        
        Args:
            logs_dir: Directory containing chat logs
            ollama_url: URL for the Ollama API
            model: Name of the model to use for evaluation
        """
        self.logs_dir = logs_dir
        try:
            # Use LLM evaluator directly
            self.llm_evaluator = LLMChatLogEvaluator(logs_dir=logs_dir, ollama_url=ollama_url, model=model)
            print(f"Using LLM evaluator with model {model}")
        except Exception as e:
            print(f"Error initializing LLM evaluator: {e}")
            self.llm_evaluator = None
    
    def get_log_path(self, log_id: str) -> str:
        """
        Get the full path to a log file.
        
        Args:
            log_id: ID of the log file
        
        Returns:
            Full path to the log file
        """
        # Use the LLM evaluator's get_log_path method
        if self.llm_evaluator:
            return self.llm_evaluator.get_log_path(log_id)
        
        # Fallback implementation if LLM evaluator is not available
        if not log_id.endswith('.json'):
            log_id += '.json'
        
        filepath = os.path.join(self.logs_dir, log_id)
        
        if not os.path.isfile(filepath):
            for subdir in os.listdir(self.logs_dir):
                subdir_path = os.path.join(self.logs_dir, subdir)
                if os.path.isdir(subdir_path):
                    potential_file = os.path.join(subdir_path, log_id)
                    if os.path.isfile(potential_file):
                        filepath = potential_file
                        break
        
        return filepath
    
    def evaluate_log(self, log_id: str) -> Dict[str, Any]:
        """
        Evaluate a chat log.
        
        Args:
            log_id: ID of the log file
        
        Returns:
            Evaluation results
        """
        # Simply use the LLM evaluator's evaluate_log method
        if self.llm_evaluator:
            return self.llm_evaluator.evaluate_log(log_id)
        
        return {'error': 'LLM evaluator is not initialized'}
    
    def get_evaluation_status(self, log_id: str) -> Dict[str, Any]:
        """
        Get the evaluation status and results for a log.
        
        Args:
            log_id: ID of the log file
        
        Returns:
            Evaluation status and results if available
        """
        # Use the LLM evaluator's get_evaluation_status method
        if self.llm_evaluator:
            status = self.llm_evaluator.get_evaluation_status(log_id)
            
            # Add debugging
            print(f"LLM evaluator returned status: {status}")
            
            # Check if an evaluation exists and ensure the results are properly formatted
            if 'status' in status and status['status'] == 'completed' and 'results' in status:
                # Make sure the results are returned in a consistent format
                results = status['results']
                
                # Check if the evaluation data is available under a nested key
                if 'evaluation' in results:
                    # Print the structure for debugging
                    print(f"Evaluation structure: {results['evaluation']}")
                    
                    # Return properly structured data
                    return {
                        'status': 'completed',
                        'results': results
                    }
                else:
                    # If there's no nested evaluation, return the results directly
                    print(f"Results structure: {results}")
                    return {
                        'status': 'completed',
                        'results': results
                    }
            
            return status
        
        return {'status': 'error', 'message': 'LLM evaluator is not initialized'}
