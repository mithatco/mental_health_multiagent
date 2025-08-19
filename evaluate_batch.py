#!/usr/bin/env python3

"""
Command-line script to evaluate a batch of mental health conversations.
"""

import os
import sys
import argparse
from typing import List
import glob
import re
from datetime import datetime

from utils.llm_evaluator_factory import LLMEvaluatorFactory
from utils.batch_evaluator import BatchEvaluator

def get_batch_paths(base_dir: str, batch_pattern: str = None) -> List[str]:
    """
    Get paths to batch directories matching a pattern.
    
    Args:
        base_dir: Base directory to search in
        batch_pattern: Optional regex pattern to match batch directories
        
    Returns:
        List of batch directory paths
    """
    # Ensure base dir exists
    if not os.path.isdir(base_dir):
        raise ValueError(f"Base directory not found: {base_dir}")
    
    # Find all batch directories
    batch_dirs = []
    
    # If base_dir is itself a batch directory, return it
    if os.path.basename(base_dir).startswith("batch_"):
        return [base_dir]
    
    # Find all subdirectories that look like batch directories
    potential_batches = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("batch_")]
    
    # Filter by pattern if provided
    if batch_pattern:
        pattern = re.compile(batch_pattern)
        potential_batches = [d for d in potential_batches if pattern.search(d)]
    
    # Sort by date (newest first)
    potential_batches.sort(reverse=True)
    
    # Convert to full paths
    batch_dirs = [os.path.join(base_dir, d) for d in potential_batches]
    
    return batch_dirs

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate mental health conversation batches")
    
    # Main arguments
    parser.add_argument("dir", help="Directory containing batch directories or a specific batch directory")
    parser.add_argument("--batch", "-b", help="Optional batch name pattern to match (regex)")
    parser.add_argument("--latest", "-l", action="store_true", help="Only evaluate the latest batch")
    parser.add_argument("--max-workers", "-w", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force re-evaluation of all logs, even if they already have evaluations")
    parser.add_argument("--fix-only", action="store_true",
                       help="Only fix existing evaluations without re-evaluating")
    
    # LLM provider options
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "groq", "openai"],
                       help="LLM provider to use for evaluation (ollama, groq, or openai)")
    parser.add_argument("--model", "-m", type=str, default="qwen2.5:3b", 
                       help="Model to use for LLM evaluation")
    parser.add_argument("--api-key", "-k", type=str, 
                       help="API key for cloud providers (required for Groq or OpenAI)")
    parser.add_argument("--api-url", "-u", type=str,
                       help="API URL override (optional)")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Check for required API key with cloud providers
        if args.provider.lower() == "groq" and not args.api_key and not os.environ.get("GROQ_API_KEY"):
            print("Error: Groq API key must be provided (use --api-key or set GROQ_API_KEY environment variable)")
            return 1
        
        if args.provider.lower() == "openai" and not args.api_key and not os.environ.get("OPENAI_API_KEY"):
            print("Error: OpenAI API key must be provided (use --api-key or set OPENAI_API_KEY environment variable)")
            return 1
        
        # Get batch directories
        batch_dirs = get_batch_paths(args.dir, args.batch)
        
        if not batch_dirs:
            print(f"No batch directories found in {args.dir}")
            return 1
        
        # If --latest, only evaluate the most recent batch
        if args.latest:
            batch_dirs = batch_dirs[:1]
        
        print(f"Found {len(batch_dirs)} batch directories to evaluate")
        
        # Display provider and model
        print(f"Using {args.provider.upper()} provider with model: {args.model}")
        
        # Display force flag if enabled
        if args.force:
            print("Force re-evaluation enabled - will re-evaluate all logs even if they have evaluations")
            
        # Display fix-only flag if enabled
        if args.fix_only:
            print("Fix-only mode enabled - will only fix existing evaluations without re-evaluating")
            if args.force:
                print("Warning: --force has no effect in fix-only mode")
        
        # Evaluate each batch
        for i, batch_dir in enumerate(batch_dirs):
            print(f"\nProcessing batch {i+1}/{len(batch_dirs)}: {os.path.basename(batch_dir)}")
            
            # Create an evaluator with the specified provider
            evaluator = BatchEvaluator(
                batch_dir=batch_dir,
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                api_url=args.api_url,
                max_workers=args.max_workers,
                force_reevaluation=args.force
            )
            
            # Run evaluation or fix only, based on args
            if args.fix_only:
                print(f"Fixing existing evaluations in {os.path.basename(batch_dir)}")
                results = evaluator.fix_batch_evaluations()
                print(f"Fixed {results['fixed_files']} evaluations")
            else:
                print(f"Evaluating conversations in {os.path.basename(batch_dir)}")
                results = evaluator.evaluate_batch()
            
            # Print summary
            summary = results.get('summary', {})
            
            print("\nBatch Summary:")
            print(f"Total conversations evaluated: {summary.get('total_evaluated', 0)}")
            
            # Diagnosis accuracy
            diag_data = summary.get('diagnosis', {})
            print(f"\nDiagnosis accuracy: {diag_data.get('accuracy', 0):.2f} ({diag_data.get('correct_matches', 0)}/{diag_data.get('total_diagnoses', 0)})")
            
            # Output paths for saved results
            eval_dir = os.path.join(batch_dir, f"batch_eval_{args.provider}")
            if os.path.exists(eval_dir):
                results_file = f"{args.model.replace(':', '-')}_results.json"
                json_path = os.path.join(eval_dir, results_file)
                csv_path = os.path.join(eval_dir, f"{args.model.replace(':', '-')}_summary.csv")
            else:
                base_name = f"batch_eval_{args.provider}_{args.model.replace(':', '-')}"
                json_path = os.path.join(batch_dir, f"{base_name}_results.json")
                csv_path = os.path.join(batch_dir, f"{base_name}_summary.csv")
            
            print(f"\nResults saved to:")
            if os.path.exists(json_path):
                print(f"  - {json_path}")
            if os.path.exists(csv_path):
                print(f"  - {csv_path}")
            
        return 0
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 