"""
Evaluation utilities using local Ollama models instead of OpenAI API.
This module provides Ragas-like metrics but uses local models via Ollama.
"""

import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class OllamaEvaluator:
    """Evaluate mental health agent responses using local Ollama models."""
    
    def __init__(self, ollama_url="http://localhost:11434", model="qwen2.5:3b"):
        """
        Initialize the evaluator with an Ollama model.
        
        Args:
            ollama_url: URL for the Ollama API (default: http://localhost:11434)
            model: Name of the model to use (default: qwen2.5:3b)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Ollama client."""
        try:
            # Try to import from the utils folder first
            try:
                from utils.ollama_client import OllamaClient
                return OllamaClient(base_url=self.ollama_url)
            except ImportError:
                # Try importing from parent directory
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from utils.ollama_client import OllamaClient
                return OllamaClient(base_url=self.ollama_url)
        except ImportError as e:
            print(f"Error importing OllamaClient: {e}")
            print("Using requests library as fallback")
            # Fallback to using requests directly
            import requests
            
            class SimpleOllamaClient:
                def __init__(self, base_url):
                    self.base_url = base_url
                
                def chat(self, model, messages):
                    response = requests.post(
                        f"{self.base_url}/api/chat",
                        json={"model": model, "messages": messages}
                    )
                    return response.json()
            
            return SimpleOllamaClient(base_url=self.ollama_url)
    
    def _generate_with_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate text with Ollama.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt for the model
        
        Returns:
            The generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat(self.model, messages)
            return response.get('response', '')
        except Exception as e:
            print(f"Error generating with Ollama: {e}")
            return ""
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant an answer is to the question using Ollama.
        
        Args:
            question: The question
            answer: The answer to evaluate
        
        Returns:
            Score between 0 and 1 representing relevancy
        """
        system_prompt = """You are an objective evaluator assessing how relevant an answer is to a question.
        Score the relevancy on a scale from 0 to 1, where:
        - 0 means completely irrelevant
        - 0.5 means somewhat relevant
        - 1 means perfectly relevant and directly addresses the question
        Provide only a single decimal number as your response, nothing else."""
        
        prompt = f"""Question: {question}
        
        Answer: {answer}
        
        How relevant is this answer to the question on a scale from 0 to 1?
        Your response must be a single number between 0 and 1."""
        
        response = self._generate_with_ollama(prompt, system_prompt)
        
        # Extract the score from the response
        try:
            # Extract first number between 0 and 1
            matches = re.findall(r'0\.\d+|[01]', response)
            if matches:
                return float(matches[0])
            else:
                # If no decimal found, check for integer 0 or 1
                if "0" in response:
                    return 0.0
                elif "1" in response:
                    return 1.0
                else:
                    print(f"Could not extract score from response: {response}")
                    return 0.5  # Default middle value
        except Exception as e:
            print(f"Error extracting score: {e}")
            return 0.5  # Default middle value
    
    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """
        Evaluate how faithful an answer is to the provided context.
        
        Args:
            context: The context information
            answer: The answer to evaluate
        
        Returns:
            Score between 0 and 1 representing faithfulness
        """
        system_prompt = """You are an objective evaluator assessing how faithful an answer is to the provided context.
        Faithfulness measures whether the answer contains information that is supported by the context.
        
        Score the faithfulness on a scale from 0 to 1, where:
        - 0 means the answer contains primarily information NOT found in the context
        - 0.5 means about half of the answer is supported by the context
        - 1 means the answer is entirely supported by the context
        
        Provide only a single decimal number as your response, nothing else."""
        
        prompt = f"""Context: {context}
        
        Answer: {answer}
        
        How faithful is this answer to the provided context on a scale from 0 to 1?
        Your response must be a single number between 0 and 1."""
        
        response = self._generate_with_ollama(prompt, system_prompt)
        
        # Extract the score from the response
        try:
            matches = re.findall(r'0\.\d+|[01]', response)
            if matches:
                return float(matches[0])
            else:
                if "0" in response:
                    return 0.0
                elif "1" in response:
                    return 1.0
                else:
                    print(f"Could not extract score from response: {response}")
                    return 0.5
        except Exception as e:
            print(f"Error extracting score: {e}")
            return 0.5
    
    def evaluate_context_precision(self, question: str, context: str) -> float:
        """
        Evaluate how relevant the context is to the question.
        
        Args:
            question: The question
            context: The context to evaluate
        
        Returns:
            Score between 0 and 1 representing context precision
        """
        system_prompt = """You are an objective evaluator assessing how relevant a context is to a question.
        Context precision measures whether the context contains information relevant to answering the question.
        
        Score the context precision on a scale from 0 to 1, where:
        - 0 means the context is completely irrelevant to the question
        - 0.5 means the context has some relevant information
        - 1 means the context is perfectly relevant to answering the question
        
        Provide only a single decimal number as your response, nothing else."""
        
        prompt = f"""Question: {question}
        
        Context: {context}
        
        How relevant is this context to the question on a scale from 0 to 1?
        Your response must be a single number between 0 and 1."""
        
        response = self._generate_with_ollama(prompt, system_prompt)
        
        # Extract the score from the response
        try:
            matches = re.findall(r'0\.\d+|[01]', response)
            if matches:
                return float(matches[0])
            else:
                if "0" in response:
                    return 0.0
                elif "1" in response:
                    return 1.0
                else:
                    print(f"Could not extract score from response: {response}")
                    return 0.5
        except Exception as e:
            print(f"Error extracting score: {e}")
            return 0.5
    
    def evaluate_context_recall(self, context: str, answer: str) -> float:
        """
        Evaluate how much of the relevant information from the context is used in the answer.
        
        Args:
            context: The context information
            answer: The answer to evaluate
        
        Returns:
            Score between 0 and 1 representing context recall
        """
        system_prompt = """You are an objective evaluator assessing how much of the relevant information from the context is used in the answer.
        Context recall measures whether the answer includes the important information from the context.
        
        Score the context recall on a scale from 0 to 1, where:
        - 0 means none of the important information from the context is used in the answer
        - 0.5 means about half of the important information is used
        - 1 means all important information from the context is used in the answer
        
        Provide only a single decimal number as your response, nothing else."""
        
        prompt = f"""Context: {context}
        
        Answer: {answer}
        
        How much of the important information from the context is used in the answer on a scale from 0 to 1?
        Your response must be a single number between 0 and 1."""
        
        response = self._generate_with_ollama(prompt, system_prompt)
        
        # Extract the score from the response
        try:
            matches = re.findall(r'0\.\d+|[01]', response)
            if matches:
                return float(matches[0])
            else:
                if "0" in response:
                    return 0.0
                elif "1" in response:
                    return 1.0
                else:
                    print(f"Could not extract score from response: {response}")
                    return 0.5
        except Exception as e:
            print(f"Error extracting score: {e}")
            return 0.5

    def evaluate_responses(
        self,
        questions: List[str],
        responses: List[str],
        context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate responses from a mental health agent using Ollama models.
        
        Args:
            questions: List of questions posed to the agent
            responses: List of responses from the agent
            context: Optional list of context information used for responses
                    (e.g. patient profiles or therapeutic guidelines)
        
        Returns:
            Dictionary containing evaluation results with metrics
        """
        results = {
            'answer_relevancy': []
        }
        
        # Always evaluate answer relevancy for each question-response pair
        print(f"Evaluating answer relevancy for {len(questions)} questions...")
        for i, (q, r) in enumerate(zip(questions, responses)):
            print(f"  Question {i+1}/{len(questions)}...")
            relevancy = self.evaluate_answer_relevancy(q, r)
            results['answer_relevancy'].append(relevancy)
        
        # Add context-based metrics if context is provided
        if context and len(context) > 0:
            results['faithfulness'] = []
            results['context_precision'] = []
            results['context_recall'] = []
            
            print(f"Evaluating context-based metrics for {len(questions)} questions...")
            for i, (q, r) in enumerate(zip(questions, responses)):
                print(f"  Question {i+1}/{len(questions)}...")
                # Use corresponding context or last one if fewer contexts than responses
                ctx = context[min(i, len(context) - 1)] if context else ""
                
                if ctx:  # Only evaluate if we have valid context
                    # Evaluate faithfulness
                    print(f"    Evaluating faithfulness...")
                    faithfulness = self.evaluate_faithfulness(ctx, r)
                    results['faithfulness'].append(faithfulness)
                    
                    # Evaluate context precision
                    print(f"    Evaluating context precision...")
                    ctx_precision = self.evaluate_context_precision(q, ctx)
                    results['context_precision'].append(ctx_precision)
                    
                    # Evaluate context recall
                    print(f"    Evaluating context recall...")
                    ctx_recall = self.evaluate_context_recall(ctx, r)
                    results['context_recall'].append(ctx_recall)
                else:
                    # If context is empty for this item, use default values
                    results['faithfulness'].append(0.5)  # Neutral score
                    results['context_precision'].append(0.5)  # Neutral score
                    results['context_recall'].append(0.5)  # Neutral score
        
        # Calculate aggregated metrics - Fix: Create a copy of keys first to avoid modifying during iteration
        metric_keys = list(results.keys())
        for metric in metric_keys:
            if results[metric]:  # Only calculate if we have values
                results[f'avg_{metric}'] = sum(results[metric]) / len(results[metric])
        
        return results
    
    @staticmethod
    def get_metric_descriptions() -> Dict[str, str]:
        """Return descriptions of metrics."""
        return {
            'answer_relevancy': "Measures how relevant the response is to the question",
            'faithfulness': "Measures if the response contains information not supported by context",
            'context_precision': "Measures how relevant the context is to the question",
            'context_recall': "Measures how much relevant info from context is used in the response"
        }


def example_usage():
    """Example of how to use the Ollama evaluator with the Patient agent."""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agents.patient import Patient
    except ImportError as e:
        print(f"Error importing Patient agent: {e}")
        # Use mock data for example
        questions = [
            "How have you been sleeping lately?",
            "Do you ever feel anxious in social situations?",
            "Have you had any changes in appetite?"
        ]
        responses = [
            "I've been having trouble sleeping. I keep waking up in the middle of the night with racing thoughts.",
            "Yes, I get very anxious in crowds. Sometimes I avoid going to social events because of it.",
            "I've lost my appetite recently. Food just doesn't seem appealing anymore."
        ]
        context = ["Patient has moderate anxiety with sleep disturbances and social anxiety."]
        
        # Evaluate
        print("Using mock data for example...")
        evaluator = OllamaEvaluator(model="qwen2.5:3b")
        results = evaluator.evaluate_responses(questions, responses, context)
        
        print("\nEvaluation results:")
        for metric, values in results.items():
            if isinstance(values, list):
                print(f"  {metric}: {', '.join([f'{v:.3f}' for v in values])}")
            else:
                print(f"  {metric}: {values:.3f}")
        
        return
    
    # Setup
    patient = Patient(ollama_url="http://localhost:11434", 
                      model="qwen2.5:3b", 
                      profile_name="anxiety_moderate")
    
    # Generate some responses
    questions = [
        "How have you been sleeping lately?",
        "Do you ever feel anxious in social situations?",
        "Have you had any changes in appetite?"
    ]
    
    responses = []
    for q in questions:
        responses.append(patient.respond_to_question(q))
    
    # Get the patient profile as context
    context = [patient.profile] if patient.profile else None
    
    # Evaluate using same Ollama model
    evaluator = OllamaEvaluator(
        ollama_url="http://localhost:11434",
        model="qwen2.5:3b"
    )
    results = evaluator.evaluate_responses(questions, responses, context)
    
    print("\nEvaluation results using Ollama:")
    for metric, values in results.items():
        if isinstance(values, list):
            print(f"  {metric}: {', '.join([f'{v:.3f}' for v in values])}")
        else:
            print(f"  {metric}: {values:.3f}")


if __name__ == "__main__":
    print("Running evaluation example using local Ollama models...")
    example_usage()
