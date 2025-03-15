"""
Test script to verify that both standard metrics and rubric evaluation are working correctly.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ragas_evaluation import RagasEvaluator

def test_metrics():
    """Test that both standard metrics and rubrics are working."""
    print("Testing Ragas metrics and rubrics...")
    
    # Create a simple test
    questions = [
        "How have you been feeling lately?",
        "Have you had any thoughts of harming yourself?"
    ]
    
    responses = [
        "I've been feeling quite depressed recently. It's hard to get out of bed in the morning.",
        "Sometimes I think about it, but I don't have any specific plans."
    ]
    
    # Context is important for some metrics
    context = ["Patient has a history of major depressive disorder with suicidal ideation."]
    
    # Create evaluator
    evaluator = RagasEvaluator(
        ollama_url="http://localhost:11434",
        model="qwen2.5:3b"  # Use a smaller model for testing
    )
    
    # Run evaluation with both standard metrics and rubrics
    print("\nRunning evaluation with both standard metrics and rubrics...")
    results = evaluator.evaluate_responses(
        questions=questions,
        responses=responses,
        context=context,
        use_rubrics=True,
        use_standard_metrics=True
    )
    
    # Check which metrics were calculated
    print("\nEvaluation results:")
    
    # Print standard metrics
    standard_metrics = [k for k in results if k.startswith("avg_") and not "rubric" in k]
    if standard_metrics:
        print("\nStandard metrics:")
        for metric in standard_metrics:
            print(f"  {metric}: {results[metric]:.3f}")
    else:
        print("\nNo standard metrics were calculated!")
        if "metrics_debug" in results:
            print("Debug info:", json.dumps(results["metrics_debug"], indent=2))
    
    # Print rubric scores
    if "rubric_scores" in results:
        print("\nRubric scores:")
        rubric_avgs = [k for k in results["rubric_scores"] if k.startswith("avg_")]
        for rubric in rubric_avgs:
            print(f"  {rubric}: {results['rubric_scores'][rubric]:.3f}")
    else:
        print("\nNo rubric scores were calculated!")
    
    # Print all calculated metrics to a JSON file for inspection
    results_path = Path("test_metrics_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved detailed results to {results_path}")
    
    return results

if __name__ == "__main__":
    test_metrics()
