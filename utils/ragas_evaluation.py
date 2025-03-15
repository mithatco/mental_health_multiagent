"""
Evaluation utilities using Ragas library with local Ollama models.
This module provides Ragas metrics but uses local models via Ollama.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Union, Tuple

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import rubrics from separate file
from rubrics.rubrics import MENTAL_HEALTH_RUBRICS, METRIC_DESCRIPTIONS

# Try to import directly first
try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    # Import Ragas components
    from ragas import SingleTurnSample  # Add this import
    from ragas.metrics import (
        Faithfulness, 
        ContextPrecision, 
        ContextRecall,
        ResponseRelevancy,
    )

    from langchain_ollama import OllamaLLM
    from langchain_ollama import OllamaEmbeddings  # Add this import for embeddings
    
    # Import RubricsScore for custom evaluation
    try:
        from ragas.metrics import RubricsScore
    except ImportError:
        try:
            from ragas.metrics.critique import RubricsScore
        except ImportError:
            RubricsScore = None
            print("Could not import RubricsScore")
        
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Direct import failed - {e}")
    RAGAS_AVAILABLE = False

# Add this class to make OllamaLLM compatible with Ragas
class RagasCompatibleLLM:
    """A wrapper around OllamaLLM to make it compatible with Ragas expectations."""
    
    def __init__(self, base_llm):
        """Initialize with a base LLM."""
        self.base_llm = base_llm
    
    async def generate(self, prompts, stop=None, callbacks=None, **kwargs):
        """
        Handle the generation request in a way that works with Ragas.
        
        Args:
            prompts: The prompt(s) to generate from. Could be StringPromptValue or other types.
            stop: Optional stop sequences
            callbacks: Optional callbacks
            **kwargs: Additional arguments
            
        Returns:
            A generation response that Ragas can process
        """
        from langchain_core.outputs import LLMResult, Generation
        
        # Convert StringPromptValue to string if needed
        if hasattr(prompts, "to_string"):
            prompt_str = prompts.to_string()
        elif isinstance(prompts, list) and len(prompts) > 0:
            # If it's a list, take the first item (Ragas usually sends a list with one item)
            if hasattr(prompts[0], "to_string"):
                prompt_str = prompts[0].to_string()
            else:
                prompt_str = str(prompts[0])
        else:
            prompt_str = str(prompts)
        
        # Call the base LLM
        try:
            # Use invoke for simple string response
            response = self.base_llm.invoke(prompt_str)
            
            # Create a properly formatted result that matches what Ragas expects
            generations = [[Generation(text=response)]]
            result = LLMResult(generations=generations)
            return result
        except Exception as e:
            print(f"Error in RagasCompatibleLLM: {e}")
            # Return an empty result on error
            return LLMResult(generations=[[Generation(text="")]])
    
    # Add minimal required methods to mimic an LLM
    def bind(self, **kwargs):
        """Support binding like a regular LLM."""
        return self
        
    def __getattr__(self, name):
        """Pass through any other attributes to the base LLM."""
        return getattr(self.base_llm, name)

class RagasEvaluator:
    """Evaluate mental health agent responses using Ragas metrics with local Ollama models."""
    
    def __init__(self, ollama_url="http://localhost:11434", model="qwen2.5:3b", embedding_model="nomic-embed-text"):
        """
        Initialize the evaluator with an Ollama model.
        
        Args:
            ollama_url: URL for the Ollama API (default: http://localhost:11434)
            model: Name of the model to use (default: qwen2.5:3b)
            embedding_model: Name of the embedding model to use (default: nomic-embed-text)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.embedding_model = embedding_model
        
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "Ragas and langchain libraries are required for this evaluator. "
                "Install with: pip install ragas langchain-ollama"
            )
        
        # Initialize Langchain Ollama models
        self._init_ollama_models()
        
        # Initialize Ragas metrics
        self._init_metrics()
        
        # Initialize rubric scorers
        self._init_rubric_scorers()
    
    def _init_ollama_models(self):
        """Initialize Ollama models for use with Ragas."""
        # Set base URL for Ollama API
        os.environ["OLLAMA_API_BASE"] = self.ollama_url
        
        # Create the base LLM
        base_llm = OllamaLLM(model=self.model, temperature=0)
        
        # Create the embeddings model
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_url
        )
        
        # Wrap the LLM with our compatible wrapper for Ragas
        self.llm = RagasCompatibleLLM(base_llm)
        
        # Keep a reference to the original LLM for other uses
        self.base_llm = base_llm
    
    def _init_metrics(self):
        """Initialize Ragas metrics with our compatible LLM wrapper."""
        # Configure metrics with local models
        self.metrics = {}
        
        # Get the Ragas version to adapt the code accordingly
        import ragas
        ragas_version = getattr(ragas, "__version__", "unknown")
        print(f"Detected Ragas version: {ragas_version}")
        
        # Add the metrics based on Ragas version
        try:
            print(f"Initializing metrics with compatible LLM wrapper: {self.model}")
            
            # Use debug prints to track initialization
            print("Initializing Response Relevancy...")
            # Note: Now we pass both the LLM and embeddings
            self.metrics["answer_relevancy"] = ResponseRelevancy(llm=self.llm, embeddings=self.embeddings)
            print("  Success!")
            
            print("Initializing Faithfulness...")
            self.metrics["faithfulness"] = Faithfulness(llm=self.llm)
            print("  Success!")
            
            print("Initializing Context Precision...")
            self.metrics["context_precision"] = ContextPrecision(llm=self.llm)
            print("  Success!")
            
            print("Initializing Context Recall...")
            self.metrics["context_recall"] = ContextRecall(llm=self.llm)
            print("  Success!")
            
        except Exception as e:
            print(f"Error initializing standard metrics: {e}")
            import traceback
            traceback.print_exc()
            print("No metrics initialized. Evaluations will only use custom rubrics if enabled.")
    
    def _init_rubric_scorers(self):
        """Initialize rubric-based scorers."""
        self.rubric_scorers = {}
        
        if not RubricsScore:
            print("RubricsScore is not available, skipping rubric scorer initialization")
            return
        
        try:
            for rubric_key, rubric_data in MENTAL_HEALTH_RUBRICS.items():
                # Extract the scoring descriptions
                rubrics = {
                    f"score{i}_description": rubric_data[f"score{i}_description"] 
                    for i in range(1, 6)
                }
                
                # Create a scorer for each rubric using specific formatting for compatibility
                try:
                    self.rubric_scorers[rubric_key] = RubricsScore(
                        rubrics=rubrics,
                        llm=self.llm,  # Use LLM instead of chat_model for compatibility
                        name=rubric_data["name"]
                    )
                    print(f"Initialized {rubric_key} rubric scorer")
                except Exception as e:
                    print(f"Error initializing {rubric_key} rubric: {e}")
                    continue
                
            print(f"Initialized {len(self.rubric_scorers)} rubric scorers")
        except Exception as e:
            print(f"Warning: Could not initialize rubric scorers: {e}")
            self.rubric_scorers = {}
    
    def _evaluate_metric_safely(self, metric, metric_name, question, response, context=None):
        """
        Safely evaluate a metric using the SingleTurnSample API.
        
        Args:
            metric: The Ragas metric object
            metric_name: Name of the metric (for logging)
            question: The question/prompt
            response: The response to evaluate
            context: Optional context information
            
        Returns:
            Float score or None if evaluation fails
        """
        try:
            # Import needed for async operations
            import asyncio
            
            # Function to run async evaluation with SingleTurnSample
            async def run_async_eval():
                try:
                    # Create the appropriate SingleTurnSample with the required reference field
                    contexts = [context] if context and context.strip() else []
                    sample = SingleTurnSample(
                        user_input=question,
                        response=response,
                        retrieved_contexts=contexts,
                        reference=""  # Required field even though we don't have a reference answer
                    )
                    
                    # Call the single_turn_ascore with the sample
                    result = await metric.single_turn_ascore(sample)
                    return result
                except Exception as e:
                    print(f"  Error in async evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            # Run the async function and get the result
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(run_async_eval())
            
            # Handle different return types
            if isinstance(result, pd.Series):
                return float(result.iloc[0])
            elif isinstance(result, list) and result:
                return float(result[0]) 
            elif result is not None:
                return float(result)
            
            return None
        except Exception as e:
            print(f"  Error evaluating {metric_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_responses(
        self,
        questions: List[str],
        responses: List[str],
        context: Optional[List[str]] = None,
        use_rubrics: bool = True,
        use_standard_metrics: bool = True,
        diagnosis_index: Optional[int] = None,  # New parameter to identify the diagnosis response
        rag_context: Optional[List[Dict[str, Any]]] = None  # New parameter for RAG context
    ) -> Dict[str, Any]:
        """
        Evaluate responses from a mental health agent using Ragas metrics.
        
        Args:
            questions: List of questions posed to the agent
            responses: List of responses from the agent
            context: Optional list of context information used for responses
                    (e.g. patient profiles or therapeutic guidelines)
            use_rubrics: Whether to include rubric-based evaluation
            use_standard_metrics: Whether to include standard Ragas metrics
            diagnosis_index: Optional index of the diagnosis response in the responses list
                            (if None, standard metrics applied to all responses)
            rag_context: Optional RAG context used for the diagnosis
                        (list of dicts with 'title', 'excerpt', etc.)
        
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
            # Replicate the last context if fewer contexts than QA pairs
            context.extend([context[-1]] * (len(questions) - len(context)))
        elif not context:
            # Create empty contexts if none provided
            context = [""] * len(questions)
        
        # Check if any context has actual content
        has_meaningful_context = any(c.strip() for c in context)
        
        # Initialize results dict
        results = {}
        
        # Run standard metrics and collect results - only for diagnosis or for all responses
        if use_standard_metrics:
            print("Running standard Ragas evaluations...")
            
            # Define metrics to evaluate
            metrics_to_evaluate = [
                {"name": "answer_relevancy", "requires_context": True},
                {"name": "faithfulness", "requires_context": True},
                {"name": "context_precision", "requires_context": True},
                {"name": "context_recall", "requires_context": True}
            ]
            
            # Prepare indices to evaluate based on diagnosis_index
            if diagnosis_index is not None and 0 <= diagnosis_index < len(responses):
                # Only evaluate the diagnosis
                indices_to_evaluate = [diagnosis_index]
                print(f"Evaluating only the diagnosis (response #{diagnosis_index+1})")
            else:
                # Evaluate all responses
                indices_to_evaluate = range(len(questions))
                print(f"Evaluating all {len(questions)} responses")
            
            # Process each selected response
            for i in indices_to_evaluate:
                q = questions[i]
                r = responses[i]
                
                # For the diagnosis, use the RAG context if provided
                if i == diagnosis_index and rag_context:
                    # Format the RAG context as a single string
                    ctx = self._format_rag_context_for_evaluation(rag_context)
                    print(f"Using RAG context for diagnosis evaluation ({len(ctx)} characters)")
                else:
                    ctx = context[i] if i < len(context) else ""
                
                # Only evaluate with metrics if there's proper context
                if not ctx.strip() and diagnosis_index is not None:
                    print(f"Skipping metrics for response {i+1} (no context)")
                    continue
                
                print(f"Evaluating response {i+1}/{len(responses)}...")
                
                # Evaluate each metric
                for metric_info in metrics_to_evaluate:
                    metric_name = metric_info["name"]
                    requires_context = metric_info["requires_context"]
                    
                    # Skip context-based metrics if no context
                    if requires_context and not ctx.strip():
                        print(f"  Skipping {metric_name} (no context)")
                        continue
                    
                    # Skip metrics that aren't available
                    if metric_name not in self.metrics:
                        print(f"  Skipping {metric_name} (not available)")
                        continue
                    
                    try:
                        print(f"  Evaluating {metric_name}...")
                        
                        # Evaluate metric using SingleTurnSample
                        result = self._evaluate_metric_safely(
                            metric=self.metrics[metric_name],
                            metric_name=metric_name,
                            question=q,
                            response=r,
                            context=ctx
                        )
                        
                        # Store result
                        if result is not None:
                            if metric_name not in results:
                                results[metric_name] = []
                            results[metric_name].append(result)
                            print(f"    Score: {result:.3f}")
                        else:
                            print(f"    Failed to evaluate {metric_name}")
                    except Exception as e:
                        print(f"    Error evaluating {metric_name}: {str(e)}")
                        import traceback
                        traceback.print_exc()
            
            # Calculate average scores
            for metric_name in list(results.keys()):
                if isinstance(results[metric_name], list) and results[metric_name]:
                    scores = [score for score in results[metric_name] if score is not None]
                    if scores:
                        results[f"avg_{metric_name}"] = sum(scores) / len(scores)
                        print(f"Average {metric_name}: {results[f'avg_{metric_name}']:.3f}")
        
        # Custom rubric evaluation implementation - always applied to all responses
        if use_rubrics and self.rubric_scorers:
            print("Running rubric-based evaluations...")
            results["rubric_scores"] = {}
            
            # For each question-answer pair
            for i in range(len(questions)):
                question = questions[i]
                answer = responses[i]
                ctx = context[i] if i < len(context) else ""
                
                # For each rubric scorer
                for rubric_key, scorer in self.rubric_scorers.items():
                    try:
                        print(f"  Evaluating {rubric_key} for Q{i+1}...")
                        
                        if rubric_key not in results["rubric_scores"]:
                            results["rubric_scores"][rubric_key] = []
                        
                        # Instead of using the Ragas API directly, let's build our own evaluation prompt
                        # This helps avoid compatibility issues
                        score = self._evaluate_with_rubric_manually(
                            rubric_key=rubric_key,
                            rubric_data=MENTAL_HEALTH_RUBRICS[rubric_key],
                            question=question,
                            answer=answer,
                            context=ctx
                        )
                        
                        results["rubric_scores"][rubric_key].append(float(score) if score is not None else None)
                    except Exception as e:
                        print(f"    Error evaluating rubric {rubric_key} for Q{i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        results["rubric_scores"][rubric_key].append(None)
            
            # Calculate averages for each rubric
            for rubric_key in MENTAL_HEALTH_RUBRICS.keys():
                if rubric_key in results["rubric_scores"]:
                    values = [v for v in results["rubric_scores"][rubric_key] if v is not None]
                    if values:
                        results["rubric_scores"][f"avg_{rubric_key}"] = sum(values) / len(values)
            
            # Include rubric descriptions for reference
            results["rubric_descriptions"] = {
                key: {
                    "name": data["name"],
                    "description": data["description"],
                }
                for key, data in MENTAL_HEALTH_RUBRICS.items()
            }
        
        # For diagnosis mode, include standard metric descriptions if any were calculated
        if any(k.startswith("avg_") and k != "avg_harmfulness" and not k.startswith("avg_rubric") for k in results):
            results["metric_descriptions"] = self.get_metric_descriptions()
        
        return results

    def _evaluate_with_rubric_manually(self, rubric_key, rubric_data, question, answer, context):
        """
        Manually evaluate a response using a rubric by crafting our own LLM prompt.
        This avoids using the Ragas API directly which might have compatibility issues.
        
        Args:
            rubric_key: Key identifying the rubric
            rubric_data: Dictionary containing rubric descriptions
            question: Question posed to the agent
            answer: Response from the agent
            context: Context information used for the response
            
        Returns:
            Float score between 1.0 and 5.0, or None if evaluation fails
        """
        rubric_prompt = f"""
You are an expert evaluator tasked with scoring a mental health professional's response based on a specific criterion.

CRITERION: {rubric_data['name']} - {rubric_data['description']}

Here is the detailed scoring rubric:
1: {rubric_data['score1_description']}
2: {rubric_data['score2_description']}
3: {rubric_data['score3_description']}
4: {rubric_data['score4_description']}
5: {rubric_data['score5_description']}

CONTEXT: {context if context else "No specific context provided."}

QUESTION: {question}

RESPONSE TO EVALUATE: {answer}

Based on the criterion and rubric above, assign a score from 1 to 5.
Provide your score as a single number between 1 and 5, with no other text.
"""
        try:
            # Use the raw LLM for direct completion
            result = self.base_llm.invoke(rubric_prompt)
            
            # Extract the score
            score_match = re.search(r'([1-5](\.\d+)?)', str(result))
            if score_match:
                return float(score_match.group(1))
            else:
                # If no exact match, look for any number
                number_match = re.search(r'\d+(\.\d+)?', str(result))
                if number_match:
                    score = float(number_match.group(0))
                    # Ensure score is between 1 and 5
                    return max(1.0, min(5.0, score))
                else:
                    print(f"Could not extract score from: {result}")
                    return None
        except Exception as e:
            print(f"Error during manual rubric evaluation: {e}")
            return None
    
    @staticmethod
    def get_metric_descriptions() -> Dict[str, str]:
        """Return descriptions of Ragas metrics."""
        return METRIC_DESCRIPTIONS
    
    @staticmethod
    def get_rubric_descriptions() -> Dict[str, Dict[str, str]]:
        """Return descriptions of available rubrics."""
        return {k: {"name": v["name"], "description": v["description"]} 
                for k, v in MENTAL_HEALTH_RUBRICS.items()}

    def _format_rag_context_for_evaluation(self, rag_context: List[Dict[str, Any]]) -> str:
        """
        Format RAG context into a string for evaluation purposes.
        
        Args:
            rag_context: List of context documents used by RAG
            
        Returns:
            Formatted string representation of the RAG context
        """
        if not rag_context:
            return ""
        
        # Format each document with title and excerpt
        formatted_context = []
        for doc in rag_context:
            doc_title = doc.get("title", "Untitled Document")
            doc_excerpt = doc.get("excerpt", "").strip()
            if doc_excerpt:
                formatted_context.append(f"Document: {doc_title}\n{doc_excerpt}\n")
        
        # Join all formatted documents
        return "\n".join(formatted_context)

def example_usage():
    """Example of how to use the Ragas evaluator."""
    try:
        # Local patient module import
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agents.patient import Patient
    except ImportError as e:
        print(f"Could not import Patient agent: {e}")
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
        try:
            evaluator = RagasEvaluator(model="qwen2.5:3b")
            # Add use_rubrics parameter
            results = evaluator.evaluate_responses(questions, responses, context, use_rubrics=True)
            
            print("\nEvaluation results:")
            # Print standard metrics
            for metric, values in results.items():
                if metric not in ["rubric_scores", "rubric_descriptions"]:
                    if isinstance(values, list):
                        print(f"  {metric}: {', '.join([f'{v:.3f}' for v in values])}")
                    else:
                        print(f"  {metric}: {values:.3f}")
            
            # Print rubric scores if available
            if "rubric_scores" in results:
                print("\nRubric scores:")
                for rubric, scores in results["rubric_scores"].items():
                    if not rubric.startswith("avg_"):
                        # Skip averages for now
                        print(f"  {rubric}: {', '.join([f'{v:.1f}' if v is not None else 'N/A' for v in scores])}")
                
                print("\nAverage rubric scores:")
                for rubric, score in results["rubric_scores"].items():
                    if rubric.startswith("avg_"):
                        print(f"  {rubric[4:]}: {score:.2f}")
        except ImportError:
            print("Could not run example due to missing dependencies.")
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
    
    # Evaluate using Ragas with Ollama
    try:
        evaluator = RagasEvaluator(
            ollama_url="http://localhost:11434",
            model="qwen2.5:3b"
        )
        # Add use_rubrics parameter
        results = evaluator.evaluate_responses(questions, responses, context, use_rubrics=True)
        
        print("\nRagas evaluation results:")
        # Print standard metrics
        for metric, values in results.items():
            if metric not in ["rubric_scores", "rubric_descriptions"]:
                if isinstance(values, list):
                    print(f"  {metric}: {', '.join([f'{v:.3f}' for v in values])}")
                else:
                    print(f"  {metric}: {values:.3f}")
        
        # Print rubric scores if available
        if "rubric_scores" in results:
            print("\nRubric scores:")
            for rubric, scores in results["rubric_scores"].items():
                if not rubric.startswith("avg_"):
                    # Skip averages for now
                    print(f"  {rubric}: {', '.join([f'{v:.1f}' if v is not None else 'N/A' for v in scores])}")
            
            print("\nAverage rubric scores:")
            for rubric, score in results["rubric_scores"].items():
                if rubric.startswith("avg_"):
                    print(f"  {rubric[4:]}: {score:.2f}")
    except ImportError:
        print("Could not run example due to missing dependencies.")


if __name__ == "__main__":
    print("Running Ragas evaluation example using local Ollama models...")
    example_usage()