"""
Utility to inspect available metrics in the installed Ragas version.
This helps identify the correct metric names for your specific Ragas version.
"""

import importlib
import inspect
import sys

def inspect_ragas_package():
    """Inspect the installed Ragas package and print available metrics."""
    try:
        import ragas
        print(f"Ragas version: {ragas.__version__}")
        
        # Try to import metrics module
        try:
            from ragas import metrics
            print("\nAvailable metrics in ragas.metrics:")
            
            # Get all public objects from the metrics module
            public_objects = [name for name in dir(metrics) if not name.startswith('_')]
            
            # Filter for classes and functions
            for name in public_objects:
                obj = getattr(metrics, name)
                if inspect.isclass(obj):
                    print(f"  - {name} (Class)")
                elif inspect.isfunction(obj):
                    print(f"  - {name} (Function)")
                else:
                    print(f"  - {name}")
                    
            # Attempt to get detailed info about each metric
            print("\nDetailed metric information:")
            for name in public_objects:
                try:
                    obj = getattr(metrics, name)
                    if hasattr(obj, '__doc__') and obj.__doc__:
                        doc = obj.__doc__.strip().split('\n')[0]
                        print(f"  - {name}: {doc}")
                except Exception as e:
                    print(f"  - {name}: Error getting info - {e}")
        
        except ImportError as e:
            print(f"Could not import ragas.metrics: {e}")
            
        # Check if there are other metric-related modules
        print("\nSearching for other metric-related modules in ragas:")
        ragas_modules = {name: module for name, module in sys.modules.items() 
                         if name.startswith('ragas.') and 'metric' in name.lower()}
        for name, module in ragas_modules.items():
            print(f"  - Found module: {name}")
            
    except ImportError:
        print("Ragas is not installed.")
        
def check_specific_metrics():
    """Check if specific metrics we're trying to use are available."""
    target_metrics = [
        'faithfulness', 
        'answer_relevancy', 
        'context_relevancy', 
        'context_recall'
    ]
    
    print("\nChecking for specific metrics we need:")
    for metric_name in target_metrics:
        try:
            # Try different import paths
            found = False
            
            # Try from ragas.metrics
            try:
                module = importlib.import_module('ragas.metrics')
                if hasattr(module, metric_name):
                    print(f"  ✅ {metric_name} found in ragas.metrics")
                    found = True
            except ImportError:
                pass
            
            # If not found, try other potential locations
            if not found:
                potential_paths = [
                    f'ragas.metrics.{metric_name}',
                    f'ragas.{metric_name}'
                ]
                
                for path in potential_paths:
                    try:
                        importlib.import_module(path)
                        print(f"  ✅ {metric_name} found as module {path}")
                        found = True
                        break
                    except ImportError:
                        continue
            
            if not found:
                print(f"  ❌ {metric_name} not found")
                
        except Exception as e:
            print(f"  ❌ Error checking for {metric_name}: {e}")
    
if __name__ == "__main__":
    print("Inspecting Ragas package to find available metrics...\n")
    inspect_ragas_package()
    check_specific_metrics()
    
    print("\nRecommendations:")
    print("1. Update your ragas_evaluation.py script to use the metrics available in your Ragas version")
    print("2. Consider upgrading Ragas with: pip install --upgrade 'ragas[all]'")
    print("3. Check Ragas documentation for your specific version: https://docs.ragas.io/")
