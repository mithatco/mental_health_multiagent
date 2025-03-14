"""
Evaluation utilities using Ragas library with local Ollama models.
This module provides Ragas metrics but uses local models via Ollama.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Union, Tuple

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Try to import directly first
try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    # Import Ragas components
    from ragas.metrics import (
        faithfulness, 
        context_precision, 
        context_recall
    )
    
    # Handle the renamed metric
    try:
        from ragas.metrics import answer_relevancy
        print("Using answer_relevancy from ragas.metrics")
    except (ImportError, AttributeError):
        try:
            from ragas.metrics import response_relevancy as answer_relevancy
            print("Using response_relevancy as answer_relevancy")
        except ImportError:
            answer_relevancy = None
            print("Could not import answer_relevancy or response_relevancy")
    
    # Import RubricsScore for custom evaluation
    try:
        from ragas.metrics import RubricsScore
    except ImportError:
        try:
            from ragas.metrics.critique import RubricsScore
        except ImportError:
            RubricsScore = None
            print("Could not import RubricsScore")
        
    from ragas.dataset_schema import SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Direct import failed - {e}")
    RAGAS_AVAILABLE = False

# If direct import fails, try using absolute imports
if not RAGAS_AVAILABLE:
    try:
        print("Trying absolute imports...")
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        
        # Updated import for newer Ragas versions
        from ragas.metrics import (
            faithfulness, 
            context_precision, 
            context_recall
        )
        
        # Handle the renamed metric
        try:
            from ragas.metrics import answer_relevancy
            AnswerRelevancyClass = answer_relevancy  # Old version
        except (ImportError, AttributeError):
            try:
                from ragas.metrics import response_relevancy
                AnswerRelevancyClass = response_relevancy  # New version
            except ImportError:
                AnswerRelevancyClass = None
        
        from ragas.metrics import RubricsScore
        from ragas.dataset_schema import SingleTurnSample
        RAGAS_AVAILABLE = True
        print("Absolute imports successful")
    except ImportError as e:
        print(f"Warning: Ragas library not available: {e}")
        print("Install with: pip install ragas langchain")
        RAGAS_AVAILABLE = False

# Try to import LangChain components separately, as they might have different import paths
if RAGAS_AVAILABLE:
    try:
        # Use updated imports from langchain_ollama
        from langchain_ollama import ChatOllama
        from langchain_ollama import OllamaLLM  # Renamed from Ollama
        print("Successfully imported from langchain_ollama")
        
        # Also try importing the specific LangChain components needed for Ragas
        from langchain.schema import SystemMessage, HumanMessage, AIMessage
        from langchain.schema.output_parser import StrOutputParser
    except ImportError:
        try:
            # Fall back to community imports if needed
            from langchain_community.chat_models import ChatOllama
            from langchain_community.llms import Ollama
            print("Warning: Using langchain_community imports. Consider installing langchain_ollama.")
        except ImportError as e:
            try:
                # Fall back to old imports if needed, but with a warning
                print("Warning: Using deprecated langchain imports. Please install langchain_ollama.")
                from langchain.chat_models import ChatOllama
                from langchain.llms import Ollama
            except ImportError as e:
                print(f"Warning: LangChain imports failed: {e}")
                print("Install with: pip install langchain-ollama")
                RAGAS_AVAILABLE = False

# Define mental health specific rubrics
MENTAL_HEALTH_RUBRICS = {
    # Empathy and rapport building
    "empathy": {
        "name": "Empathy & Rapport",
        "description": "Evaluates how well the response demonstrates empathy and builds rapport with the patient",
        "score1_description": "Shows no empathy or rapport building; dismissive or insensitive",
        "score2_description": "Minimal empathy with limited acknowledgment of patient concerns",
        "score3_description": "Basic empathy shown, but lacks personalization or deep understanding",
        "score4_description": "Good empathy with clear acknowledgment of feelings and experiences",
        "score5_description": "Excellent empathy with deep understanding, validation, and rapport building"
    },
    
    # Clinical accuracy
    "clinical_accuracy": {
        "name": "Clinical Accuracy",
        "description": "Evaluates the clinical accuracy and appropriateness of the response",
        "score1_description": "Clinically inaccurate, potentially harmful advice or assessment",
        "score2_description": "Contains significant clinical inaccuracies or inappropriate guidance",
        "score3_description": "Generally sound but with some inaccuracies or oversimplifications",
        "score4_description": "Clinically accurate with minor imprecisions or omissions",
        "score5_description": "Excellent clinical accuracy, aligned with best practices and evidence-based approach"
    },
    
    # Therapeutic techniques
    "therapeutic_approach": {
        "name": "Therapeutic Approach",
        "description": "Evaluates the appropriate use of therapeutic techniques in response",
        "score1_description": "No apparent therapeutic approach; counterproductive or inappropriate",
        "score2_description": "Limited therapeutic value with minimal structure or technique",
        "score3_description": "Basic therapeutic elements but generic or lacking customization",
        "score4_description": "Good use of appropriate therapeutic techniques for patient needs",
        "score5_description": "Excellent application of tailored therapeutic techniques matching patient's specific needs"
    },
    
    # Safety and risk assessment
    "risk_assessment": {
        "name": "Safety & Risk Assessment",
        "description": "Evaluates how well the response addresses safety concerns or risk factors",
        "score1_description": "Completely ignores critical safety concerns or increases risk",
        "score2_description": "Inadequate attention to potential risks or safety issues",
        "score3_description": "Basic recognition of risks but incomplete or formulaic response",
        "score4_description": "Good assessment of risks with appropriate response and guidance",
        "score5_description": "Excellent risk assessment with comprehensive safety planning when needed"
    },
    
    # Communication clarity
    "clarity": {
        "name": "Communication Clarity",
        "description": "Evaluates the clarity, accessibility, and appropriateness of language used",
        "score1_description": "Highly confusing, jargon-filled, or inappropriate language",
        "score2_description": "Unclear communication with excessive terminology or poor structure",
        "score3_description": "Generally understandable but could be clearer or more accessible",
        "score4_description": "Clear communication with appropriate language for the patient",
        "score5_description": "Exceptionally clear, accessible, and well-structured communication"
    }
}

class RagasEvaluator:
    """Evaluate mental health agent responses using Ragas metrics with local Ollama models."""
    
    def __init__(self, ollama_url="http://localhost:11434", model="qwen2.5:3b"):
        """
        Initialize the evaluator with an Ollama model.
        
        Args:
            ollama_url: URL for the Ollama API (default: http://localhost:11434)
            model: Name of the model to use (default: qwen2.5:3b)
        """
        self.ollama_url = ollama_url
        self.model = model
        
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
        
        # Create LLM instances for different metrics
        # Configure temperature=0 for more reliable evaluation results
        self.chat_model = ChatOllama(model=self.model, temperature=0)
        
        # Use OllamaLLM instead of Ollama if available
        try:
            # New import
            self.llm = OllamaLLM(model=self.model, temperature=0)
        except NameError:
            # Fallback to old import
            self.llm = Ollama(model=self.model, temperature=0)
    
    def _init_metrics(self):
        """Initialize Ragas metrics with our models."""
        # Configure metrics with local models
        self.metrics = {}
        
        # Get the Ragas version to adapt the code accordingly
        import ragas
        ragas_version = getattr(ragas, "__version__", "unknown")
        print(f"Detected Ragas version: {ragas_version}")
        
        # Add the metrics based on Ragas version
        try:
            # Try to initialize metrics with newer Ragas API
            if answer_relevancy:
                self.metrics["answer_relevancy"] = answer_relevancy.AnswerRelevancy(llm=self.llm)
                print(f"Initialized answer_relevancy metric")
            
            self.metrics["faithfulness"] = faithfulness.Faithfulness(llm=self.llm)
            print(f"Initialized faithfulness metric")
            
            self.metrics["context_precision"] = context_precision.ContextPrecision(llm=self.llm)
            print(f"Initialized context_precision metric")
            
            self.metrics["context_recall"] = context_recall.ContextRecall(llm=self.llm)
            print(f"Initialized context_recall metric")
            
            # Try to initialize harmfulness metric if available
            try:
                from ragas.metrics.critique import harmfulness
                self.metrics["harmfulness"] = harmfulness.Harmfulness(llm=self.llm)
                print(f"Initialized harmfulness metric")
            except ImportError:
                print(f"Harmfulness metric not available in this version of Ragas")
        except Exception as e:
            print(f"Error initializing standard metrics: {e}")
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
    
    def evaluate_responses(
        self,
        questions: List[str],
        responses: List[str],
        context: Optional[List[str]] = None,
        use_rubrics: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate responses from a mental health agent using Ragas metrics.
        
        Args:
            questions: List of questions posed to the agent
            responses: List of responses from the agent
            context: Optional list of context information used for responses
                    (e.g. patient profiles or therapeutic guidelines)
            use_rubrics: Whether to include rubric-based evaluation
        
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
        
        # Create dataframe for evaluation
        data = {
            "question": questions,
            "answer": responses,
            "contexts": [[c] for c in context]  # Ragas expects a list of passages
        }
        
        eval_df = pd.DataFrame(data)
        
        # Initialize results dict
        results = {}
        
        # Run standard metrics and collect results
        print("Running standard Ragas evaluations...")
        for metric_name, metric in self.metrics.items():
            print(f"  Evaluating {metric_name}...")
            try:
                # Skip context-based metrics if no meaningful context is provided
                if metric_name in ["context_precision", "context_recall", "faithfulness"] and not any(context):
                    print(f"    Skipping {metric_name} due to empty context")
                    continue
                
                # Run the metric
                scores = metric.score(eval_df)
                if isinstance(scores, pd.Series):
                    # Convert series to list
                    results[metric_name] = scores.tolist()
                    # Calculate average
                    results[f"avg_{metric_name}"] = np.mean(scores)
                else:
                    # Handle case where metric returns a single value
                    results[metric_name] = [scores] * len(questions)
                    results[f"avg_{metric_name}"] = scores
            except Exception as e:
                print(f"    Error evaluating {metric_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Custom rubric evaluation implementation
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
            result = self.llm.invoke(rubric_prompt)
            
            # Extract the score
            import re
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
        return {
            'answer_relevancy': "Measures how relevant the response is to the question",
            'faithfulness': "Measures if the response contains information not supported by context",
            'context_precision': "Measures how relevant the context is to the question",
            'context_recall': "Measures how much relevant info from context is used in the response",
            'harmfulness': "Detects potential harmful content in responses"
        }
    
    @staticmethod
    def get_rubric_descriptions() -> Dict[str, Dict[str, str]]:
        """Return descriptions of available rubrics."""
        return {k: {"name": v["name"], "description": v["description"]} 
                for k, v in MENTAL_HEALTH_RUBRICS.items()}


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
