"""
Evaluation utilities using Ollama models when Ragas is not available.
This is a fallback evaluator for basic metrics.
"""

import time
from typing import List, Dict, Any, Optional
import json
import os

class OllamaEvaluator:
    """Evaluate agent responses using Ollama models without Ragas."""
    
    def __init__(self, ollama_url="http://localhost:11434", model="qwen2.5:3b"):
        """
        Initialize the evaluator with an Ollama model.
        
        Args:
            ollama_url: URL for the Ollama API
            model: Name of the model to use
        """
        self.ollama_url = ollama_url
        self.model = model
        
        # Import here to avoid dependency issues
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("The requests library is required for OllamaEvaluator")
    
    def chat(self, messages):
        """Send a request to Ollama chat API."""
        try:
            # Set stream=false to ensure we get a complete response, not streaming
            response = self.requests.post(
                f"{self.ollama_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
                headers={"Content-Type": "application/json"}
            )
            
            # Check if response is valid
            if not response.ok:
                print(f"Error from Ollama API: {response.status_code} - {response.text}")
                return {"error": f"HTTP error: {response.status_code}", "content": ""}
            
            # Try to parse the response as JSON
            try:
                return response.json()
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to extract just the first valid JSON object
                print(f"JSON decode error: {e}")
                text = response.text.strip()
                
                # Debug the response
                print(f"Response from Ollama (first 100 chars): {text[:100]}...")
                
                # Try to find and parse just the first complete JSON object
                try:
                    # Look for first { and matching }
                    start = text.find('{')
                    if start >= 0:
                        # Simple approach: count braces to find matching end
                        brace_count = 0
                        for i, char in enumerate(text[start:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found complete JSON object
                                    json_text = text[start:start+i+1]
                                    return json.loads(json_text)
                
                    # If no valid JSON found, look for the content directly
                    if '"content":' in text:
                        content_start = text.find('"content":') + 10
                        quote_type = text[content_start]  # " or '
                        content_value_start = content_start + 1
                        content_end = text.find(quote_type, content_value_start)
                        if content_end > content_value_start:
                            content = text[content_value_start:content_end]
                            return {"message": {"content": content}}
                except Exception as nested_e:
                    print(f"Failed to extract JSON: {nested_e}")
                
                # Return a fallback response
                return {"message": {"content": "0.5", "role": "assistant"}}
                
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            # Return a fallback response with 0.5 (neutral score)
            return {"message": {"content": "0.5", "role": "assistant"}}
    
    def evaluate_responses(
        self, 
        questions: List[str], 
        responses: List[str], 
        context: Optional[List[str]] = None,
        **kwargs  # Added to maintain compatibility with Ragas evaluator
    ) -> Dict[str, Any]:
        """
        Evaluate responses from a mental health agent.
        
        Args:
            questions: List of questions posed to the agent
            responses: List of responses from the agent
            context: Optional list of context information
            **kwargs: Extra parameters (for compatibility with Ragas evaluator)
        
        Returns:
            Dictionary containing evaluation results with metrics
        """
        if not questions or not responses:
            return {
                "error": "No questions or responses provided for evaluation"
            }
        
        if len(questions) != len(responses):
            return {
                "error": f"Number of questions ({len(questions)}) does not match number of responses ({len(responses)})"
            }
        
        # Ensure context is available for all QA pairs
        if context and len(context) < len(questions):
            context.extend([context[-1]] * (len(questions) - len(context)))
        elif not context:
            context = [""] * len(questions)
        
        # Basic metrics we'll calculate
        answer_relevancy_scores = []
        faithfulness_scores = []
        context_precision_scores = []
        context_recall_scores = []
        
        # Process each question-answer pair
        print(f"Evaluating {len(questions)} question-answer pairs...")
        for i, (question, response, ctx) in enumerate(zip(questions, responses, context)):
            print(f"Evaluating pair {i+1}/{len(questions)}...")
            
            # For answer relevancy (how relevant the response is to the question)
            relevancy_prompt = [
                {"role": "system", "content": "You are an evaluator assessing how relevant an answer is to a question. Score from 0 (completely irrelevant) to 1 (perfectly relevant)."},
                {"role": "user", "content": f"Question: {question}\n\nResponse: {response}\n\nHow relevant is this response to the question on a scale from 0 to 1? Provide just the number."}
            ]
            relevancy_result = self.chat(relevancy_prompt)
            try:
                relevancy_score = self._extract_score(relevancy_result.get("message", {}).get("content", "0"))
                answer_relevancy_scores.append(relevancy_score)
            except:
                answer_relevancy_scores.append(0)
            
            # For faithfulness (whether answer is supported by context)
            if ctx:
                faithfulness_prompt = [
                    {"role": "system", "content": "You are an evaluator assessing if a response contains only information from the provided context. Score from 0 (not supported) to 1 (fully supported)."},
                    {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {question}\n\nResponse: {response}\n\nIs this response faithful to the context on a scale from 0 to 1? Provide just the number."}
                ]
                faithfulness_result = self.chat(faithfulness_prompt)
                try:
                    faithfulness_score = self._extract_score(faithfulness_result.get("message", {}).get("content", "0"))
                    faithfulness_scores.append(faithfulness_score)
                except:
                    faithfulness_scores.append(0)
                    
                # For context precision (how relevant the context is to the question)
                precision_prompt = [
                    {"role": "system", "content": "You are an evaluator assessing how relevant the context is to answering a question. Score from 0 (irrelevant) to 1 (highly relevant)."},
                    {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {question}\n\nHow relevant is this context to the question on a scale from 0 to 1? Provide just the number."}
                ]
                precision_result = self.chat(precision_prompt)
                try:
                    precision_score = self._extract_score(precision_result.get("message", {}).get("content", "0"))
                    context_precision_scores.append(precision_score)
                except:
                    context_precision_scores.append(0)
                
                # For context recall (how much of the relevant context is in the answer)
                recall_prompt = [
                    {"role": "system", "content": "You are an evaluator assessing how much relevant information from the context appears in the answer. Score from 0 (none) to 1 (all relevant info)."},
                    {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {question}\n\nResponse: {response}\n\nHow much relevant information from context is used in the response? Score from 0 to 1."}
                ]
                recall_result = self.chat(recall_prompt)
                try:
                    recall_score = self._extract_score(recall_result.get("message", {}).get("content", "0"))
                    context_recall_scores.append(recall_score)
                except:
                    context_recall_scores.append(0)
            else:
                # No context provided
                faithfulness_scores.append(0)
                context_precision_scores.append(0)
                context_recall_scores.append(0)
        
        # Return results
        results = {
            "answer_relevancy": answer_relevancy_scores,
            "avg_answer_relevancy": sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0,
        }
        
        # Only include other metrics if we have context
        if any(context):
            results["faithfulness"] = faithfulness_scores
            results["avg_faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
            results["context_precision"] = context_precision_scores
            results["avg_context_precision"] = sum(context_precision_scores) / len(context_precision_scores) if context_precision_scores else 0
            results["context_recall"] = context_recall_scores
            results["avg_context_recall"] = sum(context_recall_scores) / len(context_recall_scores) if context_recall_scores else 0
        
        return results
    
    def _extract_score(self, text):
        """Extract a numeric score from text response."""
        # Try to find a number between 0 and 1 in the text
        import re
        matches = re.findall(r'0\.\d+|[01]\.0|[01]', text)
        if matches:
            try:
                return float(matches[0])
            except:
                pass
        
        # If no match, try to analyze the text
        text_lower = text.lower()
        if any(word in text_lower for word in ["excellent", "perfect", "completely", "fully"]):
            return 1.0
        elif any(word in text_lower for word in ["good", "mostly", "largely"]):
            return 0.75
        elif any(word in text_lower for word in ["moderate", "partial", "somewhat"]):
            return 0.5
        elif any(word in text_lower for word in ["poor", "minimal", "barely"]):
            return 0.25
        else:
            return 0.0
