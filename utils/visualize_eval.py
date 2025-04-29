"""
Visualization tools for batch evaluation results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import argparse

def load_batch_results(batch_dir: str) -> Dict[str, Any]:
    """
    Load batch evaluation results from a directory.
    
    Args:
        batch_dir: Path to the batch directory
        
    Returns:
        Batch evaluation results dictionary
    """
    # Try to load the batch_eval_results.json file
    results_path = os.path.join(batch_dir, "batch_eval_results.json")
    
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Batch evaluation results not found at {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results

def plot_diagnosis_confusion_matrix(results: Dict[str, Any], output_dir: str = None) -> None:
    """
    Plot the diagnosis confusion matrix.
    
    Args:
        results: Batch evaluation results
        output_dir: Directory to save the plot (if None, display instead)
    """
    # Extract confusion matrix and labels
    summary = results.get("summary", {})
    diagnosis_metrics = summary.get("diagnosis", {}).get("metrics", {})
    
    if not diagnosis_metrics or "confusion_matrix" not in diagnosis_metrics:
        print("No confusion matrix found in results")
        return
    
    conf_matrix = np.array(diagnosis_metrics.get("confusion_matrix", []))
    labels = diagnosis_metrics.get("label_names", [])
    
    if len(conf_matrix) == 0 or len(labels) == 0:
        print("Empty confusion matrix or labels")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Diagnosis Confusion Matrix')
    plt.tight_layout()
    
    # Save or display
    if output_dir:
        output_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(output_path)
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()

def plot_rubric_scores(results: Dict[str, Any], output_dir: str = None) -> None:
    """
    Plot the LLM rubric scores.
    
    Args:
        results: Batch evaluation results
        output_dir: Directory to save the plot (if None, display instead)
    """
    # Extract rubric scores
    summary = results.get("summary", {})
    rubric_scores = summary.get("llm_rubric", {})
    
    if not rubric_scores:
        print("No rubric scores found in results")
        return
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Sort items by score for better visualization
    sorted_items = sorted(rubric_scores.items(), key=lambda x: x[1])
    criteria = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Plot bar chart
    ax = sns.barplot(x=scores, y=criteria, palette='viridis')
    
    # Add value labels
    for i, score in enumerate(scores):
        ax.text(score + 0.1, i, f"{score:.2f}", va='center')
    
    plt.xlabel('Score (1-5)')
    plt.title('LLM Rubric Evaluation Scores')
    plt.xlim(0, 5.5)  # Set limit to 5.5 to make room for labels
    plt.tight_layout()
    
    # Save or display
    if output_dir:
        output_path = os.path.join(output_dir, "rubric_scores.png")
        plt.savefig(output_path)
        print(f"Saved rubric scores to {output_path}")
    else:
        plt.show()

def plot_roc_curves(results: Dict[str, Any], output_dir: str = None) -> None:
    """
    Plot ROC curves for diagnosis evaluation.
    
    Args:
        results: Batch evaluation results
        output_dir: Directory to save the plot (if None, display instead)
    """
    # Extract ROC data
    summary = results.get("summary", {})
    diagnosis_metrics = summary.get("diagnosis", {}).get("metrics", {})
    
    if not diagnosis_metrics or "roc_auc" not in diagnosis_metrics:
        print("No ROC data found in results")
        return
    
    roc_auc = diagnosis_metrics.get("roc_auc", {})
    
    if not roc_auc:
        print("Empty ROC AUC data")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Sort labels by AUC for better visibility in legend
    sorted_labels = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
    
    # Since we don't have the actual ROC curve data, we'll create a simple bar chart of AUC values
    labels = [item[0] for item in sorted_labels]
    auc_values = [item[1] for item in sorted_labels]
    
    # Plot bar chart
    sns.barplot(x=labels, y=auc_values, palette='viridis')
    
    plt.xlabel('Diagnosis')
    plt.ylabel('AUC')
    plt.title('ROC AUC by Diagnosis')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save or display
    if output_dir:
        output_path = os.path.join(output_dir, "roc_auc.png")
        plt.savefig(output_path)
        print(f"Saved ROC AUC to {output_path}")
    else:
        plt.show()

def plot_nlp_metrics(results: Dict[str, Any], output_dir: str = None) -> None:
    """
    Plot NLP metrics from evaluation.
    
    Args:
        results: Batch evaluation results
        output_dir: Directory to save the plot (if None, display instead)
    """
    # Extract NLP metrics
    summary = results.get("summary", {})
    nlp_metrics = summary.get("nlp_metrics", {})
    
    if not nlp_metrics:
        print("No NLP metrics found in results")
        return
    
    # Create a DataFrame with flattened metrics
    metrics_data = {}
    
    # Extract coherence metrics
    coherence = nlp_metrics.get("coherence", {})
    for key, value in coherence.items():
        metrics_data[f"Coherence: {key}"] = value
    
    # Extract readability metrics
    readability = nlp_metrics.get("readability", {})
    for key, value in readability.items():
        metrics_data[f"Readability: {key}"] = value
    
    # Extract diversity metrics
    diversity = nlp_metrics.get("diversity", {})
    for key, value in diversity.items():
        metrics_data[f"Diversity: {key}"] = value
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. Plot coherence metrics
    coherence_keys = [k for k in metrics_data.keys() if k.startswith("Coherence")]
    coherence_values = [metrics_data[k] for k in coherence_keys]
    axs[0].bar(coherence_keys, coherence_values, color='skyblue')
    axs[0].set_title('Semantic Coherence Metrics')
    axs[0].set_ylim(0, 1)  # Coherence is typically between 0 and 1
    axs[0].tick_params(axis='x', rotation=45)
    
    # 2. Plot readability metrics
    readability_keys = [k for k in metrics_data.keys() if k.startswith("Readability")]
    readability_values = [metrics_data[k] for k in readability_keys]
    axs[1].bar(readability_keys, readability_values, color='lightgreen')
    axs[1].set_title('Readability Metrics')
    axs[1].tick_params(axis='x', rotation=45)
    
    # 3. Plot diversity metrics
    diversity_keys = [k for k in metrics_data.keys() if k.startswith("Diversity")]
    diversity_values = [metrics_data[k] for k in diversity_keys]
    axs[2].bar(diversity_keys, diversity_values, color='salmon')
    axs[2].set_title('Diversity Metrics')
    axs[2].set_ylim(0, 1)  # Distinct-N is typically between 0 and 1
    axs[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save or display
    if output_dir:
        output_path = os.path.join(output_dir, "nlp_metrics.png")
        plt.savefig(output_path)
        print(f"Saved NLP metrics to {output_path}")
    else:
        plt.show()

def generate_html_report(batch_dir: str, results: Dict[str, Any], output_path: Optional[str] = None) -> None:
    """
    Generate an HTML report of the evaluation results.
    
    Args:
        batch_dir: Path to the batch directory
        results: Batch evaluation results
        output_path: Optional specific path for the HTML report
    """
    # First, generate all plots
    output_dir = os.path.join(batch_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_diagnosis_confusion_matrix(results, output_dir)
    plot_rubric_scores(results, output_dir)
    plot_roc_curves(results, output_dir)
    plot_nlp_metrics(results, output_dir)
    
    # Extract key summary data
    summary = results.get("summary", {})
    batch_name = results.get("batch_name", "Unknown Batch")
    timestamp = results.get("timestamp", "Unknown Time")
    total_evaluated = summary.get("total_evaluated", 0)
    provider = summary.get("provider", "Unknown Provider")
    model = summary.get("model", "Unknown Model")
    
    diagnosis_data = summary.get("diagnosis", {})
    diagnosis_accuracy = diagnosis_data.get("accuracy", 0)
    correct_matches = diagnosis_data.get("correct_matches", 0)
    total_diagnoses = diagnosis_data.get("total_diagnoses", 0)
    
    rubric_scores = summary.get("llm_rubric", {})
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Batch Evaluation Report: {batch_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #f2f2f2;
            }}
            .plot-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .summary-box {{
                padding: 15px;
                background-color: #e8f4f8;
                border-left: 5px solid #7cb9e8;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Batch Evaluation Report: {batch_name}</h1>
            <p>Generated on: {timestamp}</p>
            <p>Provider: <strong>{provider}</strong>, Model: <strong>{model}</strong></p>
            
            <div class="summary-box">
                <h2>Executive Summary</h2>
                <p>Total conversations evaluated: <strong>{total_evaluated}</strong></p>
                <p>Diagnosis accuracy: <strong>{diagnosis_accuracy:.2f}</strong> ({correct_matches}/{total_diagnoses})</p>
            </div>
            
            <div class="section">
                <h2>Diagnosis Evaluation</h2>
                <div class="plot-container">
                    <h3>Confusion Matrix</h3>
                    <img src="plots/confusion_matrix.png" alt="Confusion Matrix">
                </div>
                
                <div class="plot-container">
                    <h3>ROC AUC by Diagnosis</h3>
                    <img src="plots/roc_auc.png" alt="ROC AUC">
                </div>
            </div>
            
            <div class="section">
                <h2>LLM Rubric Evaluation</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Criterion</th>
                        <th>Score (1-5)</th>
                    </tr>
    """
    
    # Add rubric scores to the table
    for criterion, score in rubric_scores.items():
        html_content += f"""
                    <tr>
                        <td>{criterion}</td>
                        <td>{score:.2f}</td>
                    </tr>
        """
    
    # Continue HTML content
    html_content += f"""
                </table>
                
                <div class="plot-container">
                    <h3>Rubric Scores Visualization</h3>
                    <img src="plots/rubric_scores.png" alt="Rubric Scores">
                </div>
            </div>
            
            <div class="section">
                <h2>NLP Metrics</h2>
                <div class="plot-container">
                    <h3>NLP Metrics Overview</h3>
                    <img src="plots/nlp_metrics.png" alt="NLP Metrics">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report to file
    if output_path is None:
        report_path = os.path.join(batch_dir, "evaluation_report.html")
    else:
        report_path = output_path
        
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated HTML report at {report_path}")

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize batch evaluation results")
    
    # Main arguments
    parser.add_argument("batch_dir", help="Path to the batch directory")
    parser.add_argument("--html", "-H", action="store_true", help="Generate HTML report")
    parser.add_argument("--all-plots", "-a", action="store_true", help="Generate all plots")
    parser.add_argument("--confusion", "-c", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--rubric", "-r", action="store_true", help="Plot rubric scores")
    parser.add_argument("--roc", "-R", action="store_true", help="Plot ROC curves")
    parser.add_argument("--nlp", "-n", action="store_true", help="Plot NLP metrics")
    parser.add_argument("--output-dir", "-o", help="Directory to save plots")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load batch results
        results = load_batch_results(args.batch_dir)
        
        # Set output directory
        output_dir = args.output_dir
        if not output_dir and args.html:
            output_dir = os.path.join(args.batch_dir, "plots")
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate HTML report (which generates all plots)
        if args.html:
            generate_html_report(args.batch_dir, results)
            return 0
        
        # Generate individual plots
        if args.all_plots or args.confusion:
            plot_diagnosis_confusion_matrix(results, output_dir)
        
        if args.all_plots or args.rubric:
            plot_rubric_scores(results, output_dir)
        
        if args.all_plots or args.roc:
            plot_roc_curves(results, output_dir)
        
        if args.all_plots or args.nlp:
            plot_nlp_metrics(results, output_dir)
        
        # If no specific plots were requested, show help
        if not (args.all_plots or args.confusion or args.rubric or args.roc or args.nlp or args.html):
            parser.print_help()
        
        return 0
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 