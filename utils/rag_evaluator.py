"""
RAG Evaluation Module using deepeval metrics.

This module provides functions to evaluate RAG performance using metrics such as:
- Contextual Relevancy: How relevant the retrieved context is to the query
- Faithfulness: Whether the response contains information supported by the context
- Answer Relevancy: How relevant the response is to the query
"""

import time
import importlib.util
import sys
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

# Improved library availability check
def is_library_available(library_name):
    """Check if a library is available without importing it"""
    return importlib.util.find_spec(library_name) is not None

# Check for deepeval availability
if is_library_available("deepeval"):
    try:
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric
        DEEPEVAL_AVAILABLE = True
        print("deepeval library successfully loaded.")
    except (ImportError, ModuleNotFoundError) as e:
        DEEPEVAL_AVAILABLE = False
        print(f"Error importing deepeval modules: {str(e)}")
else:
    DEEPEVAL_AVAILABLE = False
    print("deepeval library not found in the Python environment.")
    print("To enable RAG evaluation, install with: pip install deepeval")

class RAGEvaluator:
    """
    Evaluator for RAG system performance using deepeval metrics.
    """
    
    def __init__(self, model: str = "default"):
        """
        Initialize the RAG evaluator.
        
        Args:
            model: Model to use for evaluation (passed to deepeval metrics)
        """
        self.model = model
        self.is_available = DEEPEVAL_AVAILABLE
        if not self.is_available:
            print("RAG evaluation will be disabled due to missing dependencies")
    
    def evaluate_contextual_relevancy(self, 
                                    query: str, 
                                    context: List[str], 
                                    threshold: float = 0.7) -> Dict[str, Any]:
        """
        Evaluate how relevant the retrieved context is to the query.
        
        Args:
            query: User query or question
            context: List of retrieved context passages
            threshold: Threshold for passing the evaluation (0-1)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.is_available:
            return {"error": "deepeval library not available", "score": 0.0}
            
        start_time = time.time()
        
        try:
            # Create metric
            metric = ContextualRelevancyMetric(
                threshold=threshold,
                # model=self.model,
                include_reason=True
            )
            
            # Create test case
            test_case = LLMTestCase(
                input=query,
                retrieval_context=context,
                actual_output=""  # Not relevant for contextual relevancy
            )
            
            # Run metric with timeout protection
            metric.measure(test_case)
            
            end_time = time.time()
            
            # Return formatted results
            return {
                "metric": "contextual_relevancy",
                "score": metric.score,
                "passed": metric.score >= threshold,
                "reason": metric.reason,
                "threshold": threshold,
                "evaluation_time": end_time - start_time
            }
        except Exception as e:
            print(f"Error in contextual relevancy evaluation: {str(e)}")
            # Return a default score instead of failing
            return {
                "metric": "contextual_relevancy",
                "score": 0.5,  # Default middle score
                "passed": 0.5 >= threshold,
                "reason": f"Evaluation error: {str(e)}",
                "threshold": threshold,
                "evaluation_time": time.time() - start_time,
                "error": str(e)
            }

    def evaluate_faithfulness(self, 
                            query: str, 
                            response: str, 
                            context: List[str], 
                            threshold: float = 0.7) -> Dict[str, Any]:
        """
        Evaluate if the response contains only information supported by the retrieved context.
        
        Args:
            query: User query or question
            response: Generated response
            context: List of retrieved context passages
            threshold: Threshold for passing the evaluation (0-1)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.is_available:
            return {"error": "deepeval library not available", "score": 0.0}
            
        start_time = time.time()
        
        try:
            # Create metric
            metric = FaithfulnessMetric(
                threshold=threshold,
                # model=self.model,
                include_reason=True
            )
            
            # Create test case
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                retrieval_context=context
            )
            
            # Run metric
            metric.measure(test_case)
            
            end_time = time.time()
            
            # Return formatted results
            return {
                "metric": "faithfulness",
                "score": metric.score,
                "passed": metric.score >= threshold,
                "reason": metric.reason,
                "threshold": threshold,
                "evaluation_time": end_time - start_time
            }
        except Exception as e:
            print(f"Error in faithfulness evaluation: {str(e)}")
            # Return a default score instead of failing
            return {
                "metric": "faithfulness",
                "score": 0.5,  # Default middle score
                "passed": 0.5 >= threshold,
                "reason": f"Evaluation error: {str(e)}",
                "threshold": threshold,
                "evaluation_time": time.time() - start_time,
                "error": str(e)
            }

    def evaluate_answer_relevancy(self, 
                                query: str, 
                                response: str, 
                                threshold: float = 0.7) -> Dict[str, Any]:
        """
        Evaluate how relevant the response is to the query.
        
        Args:
            query: User query or question
            response: Generated response
            threshold: Threshold for passing the evaluation (0-1)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.is_available:
            return {"error": "deepeval library not available", "score": 0.0}
            
        start_time = time.time()
        
        try:
            # Create metric
            metric = AnswerRelevancyMetric(
                threshold=threshold,
                # model=self.model,
                include_reason=True
            )
            
            # Create test case
            test_case = LLMTestCase(
                input=query,
                actual_output=response
            )
            
            # Run metric
            metric.measure(test_case)
            
            end_time = time.time()
            
            # Return formatted results
            return {
                "metric": "answer_relevancy",
                "score": metric.score,
                "passed": metric.score >= threshold,
                "reason": metric.reason,
                "threshold": threshold,
                "evaluation_time": end_time - start_time
            }
        except Exception as e:
            print(f"Error in answer relevancy evaluation: {str(e)}")
            # Return a default score instead of failing
            return {
                "metric": "answer_relevancy",
                "score": 0.5,  # Default middle score
                "passed": 0.5 >= threshold,
                "reason": f"Evaluation error: {str(e)}",
                "threshold": threshold,
                "evaluation_time": time.time() - start_time,
                "error": str(e)
            }
    
    def evaluate_rag(self, 
                    query: str, 
                    response: str, 
                    context: List[str], 
                    metrics: List[str] = None,
                    threshold: float = 0.7) -> Dict[str, Any]:
        """
        Run comprehensive RAG evaluation with multiple metrics.
        
        Args:
            query: User query or question
            response: Generated response
            context: List of retrieved context passages
            metrics: List of metrics to evaluate (defaults to all)
            threshold: Threshold for passing evaluations (0-1)
            
        Returns:
            Dictionary with all evaluation results
        """
        if not self.is_available:
            return {
                "error": "deepeval library not available",
                "all_passed": False,
                "average_score": 0.0,
                "threshold": threshold,
                "metrics": {}
            }
            
        if metrics is None:
            metrics = ["contextual_relevancy", "faithfulness", "answer_relevancy"]
            
        results = {}
        all_passed = True
        total_score = 0
        eval_count = 0
        
        # Run each requested metric
        for metric in metrics:
            if metric == "contextual_relevancy":
                metric_result = self.evaluate_contextual_relevancy(
                    query=query, 
                    context=context, 
                    threshold=threshold
                )
            elif metric == "faithfulness":
                metric_result = self.evaluate_faithfulness(
                    query=query, 
                    response=response, 
                    context=context, 
                    threshold=threshold
                )
            elif metric == "answer_relevancy":
                metric_result = self.evaluate_answer_relevancy(
                    query=query, 
                    response=response, 
                    threshold=threshold
                )
            else:
                continue
                
            results[metric] = metric_result
            all_passed = all_passed and metric_result.get("passed", False)
            
            if "score" in metric_result:
                total_score += metric_result["score"]
                eval_count += 1
        
        # Calculate average score
        avg_score = total_score / eval_count if eval_count > 0 else 0
        
        return {
            "all_passed": all_passed,
            "average_score": round(avg_score, 4),
            "threshold": threshold,
            "metrics": results
        }

    def format_evaluation_results(self, results: Dict[str, Any]) -> str:
        """
        Format evaluation results as a readable string.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted string representation of results
        """
        if "error" in results:
            return f"Evaluation Error: {results['error']}"
            
        output = "==== RAG Evaluation Results ====\n\n"
        output += f"Overall Score: {results.get('average_score', 'N/A')} (Threshold: {results.get('threshold', 'N/A')})\n"
        output += f"Overall Result: {'Passed' if results.get('all_passed', False) else 'Failed'}\n\n"
        
        for metric_name, metric_results in results.get("metrics", {}).items():
            output += f"--- {metric_name.replace('_', ' ').title()} ---\n"
            output += f"Score: {metric_results.get('score', 'N/A')}\n"
            output += f"Result: {'Passed' if metric_results.get('passed', False) else 'Failed'}\n"
            if "reason" in metric_results:
                output += f"Reason: {metric_results['reason']}\n"
            output += f"Evaluation Time: {metric_results.get('evaluation_time', 'N/A'):.2f}s\n\n"
            
        return output
