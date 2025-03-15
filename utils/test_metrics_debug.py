"""
Debug script for Ragas metrics. This script tests each metric individually 
to identify any issues with the evaluation pipeline.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import Ragas components directly to test them
    from ragas.metrics import (
        Faithfulness, 
        ContextPrecision, 
        ContextRecall,
        ResponseRelevancy,
    )
    from langchain_ollama import OllamaLLM
except ImportError as e:
    print(f"Could not import required libraries: {e}")
    sys.exit(1)

def test_individual_metrics():
    """Test each Ragas metric individually to identify issues."""
    print("Testing individual Ragas metrics...")
    
    # Set up Ollama API base URL
    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
    
    # Create a test LLM
    print("Creating LLM...")
    llm = OllamaLLM(model="qwen2.5:3b", temperature=0)
    
    # Create a simple test dataset - use the column names that Ragas expects
    data = {
        "user_input": [
            "How have you been feeling lately?",
            "Have you had any thoughts of harming yourself?"
        ],
        "response": [
            "I've been feeling quite depressed recently. It's hard to get out of bed in the morning.",
            "Sometimes I think about it, but I don't have any specific plans."
        ],
        "contexts": [
            ["Patient has a history of major depressive disorder with suicidal ideation."],
            ["Patient has a history of major depressive disorder with suicidal ideation."]
        ],
        # Additional columns that some metrics might need
        "retrieved_contexts": [
            ["Patient has a history of major depressive disorder with suicidal ideation."],
            ["Patient has a history of major depressive disorder with suicidal ideation."]
        ],
        "reference": [
            "",
            ""
        ]
    }
    
    df = pd.DataFrame(data)
    print(f"Test dataframe columns: {df.columns.tolist()}")
    
    # Test each metric individually
    metrics_to_test = [
        ("ResponseRelevancy", ResponseRelevancy(llm=llm)),
        ("Faithfulness", Faithfulness(llm=llm)),
        ("ContextPrecision", ContextPrecision(llm=llm)),
        ("ContextRecall", ContextRecall(llm=llm))
    ]
    
    results = {}
    
    for name, metric in metrics_to_test:
        print(f"\nTesting {name}...")
        try:
            # Try to evaluate with this metric
            print(f"  Calling metric.score()...")
            scores = metric.single_turn_ascore(df)
            print(f"  Success! Result: {scores}")
            
            # Store results
            results[name] = {
                "success": True,
                "scores": scores.tolist() if hasattr(scores, 'tolist') else scores
            }
        except Exception as e:
            print(f"  Error testing {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Store error
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary of results
    print("\nTest Summary:")
    for name, result in results.items():
        if result["success"]:
            print(f"  ✅ {name}: Success")
        else:
            print(f"  ❌ {name}: Failed - {result['error']}")
    
    # Save detailed results to a JSON file
    results_path = Path("metric_debug_results.json")
    with open(results_path, "w") as f:
        # Need to clean up non-serializable objects
        clean_results = {}
        for name, result in results.items():
            if result["success"]:
                if hasattr(result["scores"], 'tolist'):
                    clean_results[name] = {
                        "success": True,
                        "scores": result["scores"].tolist()
                    }
                else:
                    clean_results[name] = {
                        "success": True,
                        "scores": str(result["scores"])
                    }
            else:
                clean_results[name] = {
                    "success": False,
                    "error": result["error"]
                }
                
        json.dump(clean_results, f, indent=2)
    
    print(f"\nSaved detailed results to {results_path}")
    
    return results

if __name__ == "__main__":
    test_individual_metrics()
