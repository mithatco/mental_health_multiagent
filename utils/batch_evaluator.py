"""
Batch evaluation of mental health conversations.
This module provides functionality to evaluate batches of conversations.
"""

import os
import sys
import json
import time
import glob
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import copy

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utilities
from utils.llm_evaluation import LLMEvaluator
from utils.llm_evaluator_factory import LLMEvaluatorFactory
from utils.nlp_metrics import (
    semantic_coherence, 
    readability_metrics, 
    diversity_metrics, 
    diagnosis_metrics,
    clean_text
)

class BatchEvaluator:
    """Evaluate a batch of mental health conversations."""
    
    def __init__(self, batch_dir: str, provider: str = "ollama", model: str = "qwen2.5:3b", 
                api_key: Optional[str] = None, api_url: Optional[str] = None, max_workers: int = 4,
                force_reevaluation: bool = False):
        """
        Initialize the batch evaluator.
        
        Args:
            batch_dir: Path to the batch directory
            provider: LLM provider ('ollama', 'groq', or 'openai')
            model: Model to use for LLM evaluation
            api_key: API key for cloud providers (Groq or OpenAI)
            api_url: URL for the API (optional)
            max_workers: Maximum number of worker threads
            force_reevaluation: If True, re-evaluate all logs even if they have evaluations
        """
        self.batch_dir = batch_dir
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_url = api_url
        self.max_workers = max_workers
        self.force_reevaluation = force_reevaluation
        
        # Initialize LLM evaluator using the factory
        self.llm_evaluator = LLMEvaluatorFactory.create_evaluator(
            provider=provider,
            model=model,
            api_key=api_key,
            api_url=api_url
        )
        
        # Ensure batch dir exists
        if not os.path.isdir(batch_dir):
            raise ValueError(f"Batch directory not found: {batch_dir}")
        
        # Get batch name from directory
        self.batch_name = os.path.basename(batch_dir)
        
    def _fix_diagnosis_metrics(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix diagnosis metrics in a log file by removing outdated structure.
        
        Args:
            log_data: The log data to fix
            
        Returns:
            Updated log data
        """
        # Check if we have evaluation data
        if 'evaluation' not in log_data:
            return log_data
        
        # Check if we have diagnosis evaluation
        if 'diagnosis_evaluation' not in log_data['evaluation']:
            return log_data
        
        # Remove metrics if they exist
        if 'metrics' in log_data['evaluation']['diagnosis_evaluation']:
            del log_data['evaluation']['diagnosis_evaluation']['metrics']
            print("Removed outdated metrics from diagnosis evaluation")
        
        return log_data

    def evaluate_conversation(self, log_path: str) -> Dict[str, Any]:
        """
        Evaluate a single conversation.
        
        Args:
            log_path: Path to the conversation log file
            
        Returns:
            Evaluation results dictionary
        """
        try:
            # Load log data
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            # Check if evaluation already exists and is complete
            needs_evaluation = True
            
            # Skip the check if force_reevaluation is True
            if not self.force_reevaluation and 'evaluation' in log_data:
                evaluation = log_data['evaluation']
                if (evaluation.get('provider', '') == self.provider and 
                    evaluation.get('model', '') == self.model):
                    
                    # Check if the evaluation contains all required metrics
                    has_llm_eval = 'llm_evaluation' in evaluation and evaluation['llm_evaluation']
                    has_ext_metrics = 'extended_metrics' in evaluation and evaluation['extended_metrics']
                    has_diag_eval = 'diagnosis_evaluation' in evaluation and evaluation['diagnosis_evaluation']
                    
                    # Check diagnosis evaluation specifically
                    diag_eval_complete = False
                    if has_diag_eval:
                        diag_eval = evaluation['diagnosis_evaluation']
                        diag_eval_complete = ('expected' in diag_eval and 
                                             'predicted' in diag_eval and 
                                             'matches' in diag_eval)
                    
                    # Only skip evaluation if everything is complete
                    if has_llm_eval and has_ext_metrics and has_diag_eval and diag_eval_complete:
                        print(f"Complete evaluation already exists for {os.path.basename(log_path)}")
                        return evaluation
                    else:
                        print(f"Incomplete evaluation found for {os.path.basename(log_path)}, re-evaluating...")
            elif self.force_reevaluation and 'evaluation' in log_data:
                print(f"Force re-evaluating {os.path.basename(log_path)}")
            
            # Extract relevant data
            conversation = log_data.get('conversation', [])
            diagnosis = log_data.get('diagnosis', '')
            
            # If diagnosis is not directly available, try to extract it from the final assistant message
            if not diagnosis and conversation:
                last_message = conversation[-1].get('content', '')
                diagnosis_match = re.search(r'(?i)diagnosis:?\s*(.+?)(?:\n|$)', last_message)
                if diagnosis_match:
                    diagnosis = diagnosis_match.group(1).strip()
            
            # Get expected diagnosis from metadata
            expected_diagnosis = ''
            if 'metadata' in log_data and 'patient_profile' in log_data['metadata']:
                expected_diagnosis = log_data['metadata']['patient_profile']
            
            # Get conversation as text for metrics
            conversation_text = self._format_conversation_for_metrics(conversation)
            therapist_text = self._extract_therapist_text(conversation)
            
            # 1. LLM-based evaluation
            llm_eval_results = self.llm_evaluator.evaluate_log(log_data)
            
            # 2. NLP Metrics
            nlp_metrics = self._calculate_nlp_metrics(conversation, therapist_text, conversation_text)
            
            # 3. Diagnosis evaluation - simple matching only, metrics calculated at batch level
            diag_eval = {
                "expected": expected_diagnosis,
                "predicted": diagnosis,
                "matches": self._diagnoses_match(diagnosis, expected_diagnosis)
            }
            
            # Add classification from LLM evaluation if available
            if 'diagnosis_classification' in llm_eval_results:
                classification = llm_eval_results.get('diagnosis_classification', {})
                diag_eval['classification'] = classification.get('classified_as', 'unknown')
                # Ensure we have the confidence level as well
                if 'confidence' in classification:
                    diag_eval['classification_confidence'] = classification.get('confidence', 0)
            
            # Combine all results
            evaluation_results = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "provider": self.provider,
                "model": self.model,
                "llm_evaluation": llm_eval_results,
                "extended_metrics": nlp_metrics,
                "diagnosis_evaluation": diag_eval
            }
            
            # Save evaluation back to the log file
            log_data['evaluation'] = evaluation_results
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            return evaluation_results
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error evaluating conversation {os.path.basename(log_path)}: {str(e)}")
            print(error_details)
            
            return {
                "error": str(e),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _diagnoses_match(self, predicted: str, expected: str) -> bool:
        """
        Check if the predicted diagnosis matches the expected one.
        
        Args:
            predicted: Predicted diagnosis
            expected: Expected diagnosis
            
        Returns:
            True if the diagnoses match, False otherwise
        """
        if not predicted or not expected:
            return False
            
        # Clean and normalize
        pred_clean = predicted.lower().strip()
        expected_clean = expected.lower().strip()
        
        # Direct match
        if expected_clean in pred_clean:
            return True
            
        # Handle common variants (e.g., "anxiety" matches "anxiety disorder")
        if expected_clean == "anxiety" and ("anxiety" in pred_clean):
            return True
        if expected_clean == "depression" and any(term in pred_clean for term in ["depression", "depressive", "mdd"]):
            return True
        if expected_clean == "ptsd" and any(term in pred_clean for term in ["ptsd", "post-traumatic", "post traumatic"]):
            return True
        if expected_clean == "bipolar" and "bipolar" in pred_clean:
            return True
        if expected_clean == "schizophrenia" and "schizophrenia" in pred_clean:
            return True
            
        return False
    
    def _calculate_nlp_metrics(self, conversation: List[Dict[str, str]], 
                              therapist_text: str, conversation_text: str) -> Dict[str, Any]:
        """
        Calculate NLP metrics for a conversation.
        
        Args:
            conversation: List of conversation messages
            therapist_text: Text of therapist messages
            conversation_text: Full conversation text
            
        Returns:
            Dictionary with NLP metrics
        """
        # Semantic coherence
        coherence = semantic_coherence(conversation)
        
        # Readability metrics
        readability = readability_metrics(therapist_text)
        
        # Diversity metrics
        diversity = diversity_metrics(therapist_text)
        
        return {
            "coherence": coherence,
            "readability": readability,
            "diversity": diversity
        }
    
    def _format_conversation_for_metrics(self, conversation: List[Dict[str, str]]) -> str:
        """
        Format conversation for metrics calculation.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Formatted conversation as string
        """
        formatted = []
        
        for msg in conversation:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Clean content
            content = clean_text(content)
            
            # Format based on role
            if role.lower() == 'system':
                continue
            elif role.lower() == 'assistant':
                formatted.append(f"Therapist: {content}")
            elif role.lower() == 'patient':
                formatted.append(f"Patient: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n\n".join(formatted)
    
    def _extract_therapist_text(self, conversation: List[Dict[str, str]]) -> str:
        """
        Extract just the therapist messages from a conversation.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Concatenated therapist messages
        """
        therapist_msgs = []
        
        for msg in conversation:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            if role == 'assistant':
                # Clean content
                content = clean_text(content)
                therapist_msgs.append(content)
        
        return " ".join(therapist_msgs)
    
    def evaluate_batch(self) -> Dict[str, Any]:
        """
        Evaluate all conversations in the batch.
            
        Returns:
            Batch evaluation results
        """
        # Get all conversation files in the batch
        conversation_files = glob.glob(os.path.join(self.batch_dir, "chat_*.json"))
        
        if not conversation_files:
            raise ValueError(f"No conversation files found in {self.batch_dir}")
        
        print(f"Found {len(conversation_files)} conversation files in {self.batch_dir}")
        
        # Prepare results container
        results = {
            "batch_name": self.batch_name,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "provider": self.provider,
            "model": self.model,
            "total_conversations": len(conversation_files),
            "evaluations": [],
            "summary": {}
        }
        
        # Use threads to parallelize evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.evaluate_conversation, file_path): file_path 
                             for file_path in conversation_files}
            
            for future in tqdm(as_completed(future_to_file), total=len(conversation_files), 
                              desc="Evaluating conversations"):
                file_path = future_to_file[future]
                try:
                    eval_result = future.result()
                    # Add file name to results
                    eval_result["file_name"] = os.path.basename(file_path)
                    
                    # Remove any metrics from individual diagnosis evaluations
                    # (they should only be calculated at batch level)
                    if 'diagnosis_evaluation' in eval_result and isinstance(eval_result['diagnosis_evaluation'], dict):
                        if 'metrics' in eval_result['diagnosis_evaluation']:
                            del eval_result['diagnosis_evaluation']['metrics']
                            
                            # Also update the file to ensure consistency
                            try:
                                with open(file_path, 'r') as f:
                                    log_data = json.load(f)
                                
                                if ('evaluation' in log_data and 
                                    'diagnosis_evaluation' in log_data['evaluation'] and 
                                    'metrics' in log_data['evaluation']['diagnosis_evaluation']):
                                    
                                    del log_data['evaluation']['diagnosis_evaluation']['metrics']
                                    
                                    with open(file_path, 'w') as f:
                                        json.dump(log_data, f, indent=2)
                                    
                                    print(f"Removed individual metrics from {os.path.basename(file_path)}")
                            except Exception as e:
                                print(f"Warning: Could not update file after removing metrics: {e}")
                    
                    results["evaluations"].append(eval_result)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        # Generate batch summary
        results["summary"] = self._generate_batch_summary(results["evaluations"])
        
        # Save batch results
        self._save_batch_results(results)
        
        return results
    
    def _generate_batch_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for batch evaluations.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Summary statistics
        """
        if not evaluations:
            return {
                "error": "No evaluations found",
                "total_evaluated": 0
            }
        
        # Filter out evaluations with errors
        valid_evals = [e for e in evaluations if 'error' not in e]
        
        if not valid_evals:
            return {
                "error": "No valid evaluations found",
                "total_evaluated": 0
            }
        
        # Extract diagnosis evaluation data
        true_diagnoses = []
        pred_diagnoses = []
        classified_diagnoses = []  # New list for LLM classifications
        matches = 0
        
        # Get unique profile/diagnosis types
        all_profiles = set()
        
        print("\nExtracting diagnosis data for batch metrics calculation...")
        
        for eval_data in valid_evals:
            diag_eval = eval_data.get('diagnosis_evaluation', {})
            if diag_eval:
                expected = diag_eval.get('expected', '')
                predicted = diag_eval.get('predicted', '')
                
                # Get classification if available (from LLM evaluation)
                classification = diag_eval.get('classification', 'unknown')
                
                if expected and predicted:
                    # Add to lists for metrics calculation
                    true_diagnoses.append(expected.lower().strip())
                    pred_diagnoses.append(predicted.lower().strip())
                    
                    # Add classification for confusion matrix
                    if classification and classification != 'unknown':
                        classified_diagnoses.append(classification.lower().strip())
                    else:
                        # Use predicted if no classification available
                        classified_diagnoses.append(predicted.lower().strip())
                    
                    # Debug
                    file_name = eval_data.get('file_name', 'unknown')
                    print(f"File: {file_name}, Expected: {expected}, Predicted: {predicted}, Classification: {classification}, Match: {diag_eval.get('matches', False)}")
                    
                    if diag_eval.get('matches', False):
                        matches += 1
                
                if expected:
                    all_profiles.add(expected.lower().strip())
        
        # Make sure we have the common profiles that might not appear in this batch
        standard_profiles = {"depression", "anxiety", "ptsd", "bipolar", "schizophrenia"}
        all_profiles.update([p for p in standard_profiles])
        
        # Sort profiles for consistent ordering
        all_profiles = sorted(all_profiles)
        
        print(f"\nBatch diagnosis data: {len(true_diagnoses)} diagnoses, {len(all_profiles)} unique profiles")
        print(f"Profiles: {all_profiles}")
        
        # Calculate diagnosis metrics if we have valid diagnoses
        diagnosis_metrics_data = {}
        if true_diagnoses and pred_diagnoses:
            try:
                print(f"Calculating diagnosis metrics with {len(true_diagnoses)} diagnoses...")
                
                # Debug data
                for i, (true, pred, classified) in enumerate(zip(true_diagnoses, pred_diagnoses, classified_diagnoses)):
                    print(f"  {i+1}. True: {true}, Predicted: {pred}, Classification: {classified}")
                
                # Import here to make sure we have the latest version
                from utils.nlp_metrics import diagnosis_metrics
                
                # Use classifications for metrics when available
                if classified_diagnoses and len(classified_diagnoses) == len(true_diagnoses):
                    print("Using LLM classifications for diagnosis metrics")
                    diagnosis_metrics_data = diagnosis_metrics(true_diagnoses, classified_diagnoses, list(all_profiles))
                else:
                    diagnosis_metrics_data = diagnosis_metrics(true_diagnoses, pred_diagnoses, list(all_profiles))
                
                print("Successfully calculated diagnosis metrics")
                print(f"Metrics: {diagnosis_metrics_data.keys()}")
                
            except Exception as e:
                import traceback
                print(f"Warning: Error calculating diagnosis metrics in summary: {e}")
                traceback.print_exc()
                
                # Create a default empty structure
                diagnosis_metrics_data = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "confusion_matrix": [[0 for _ in range(len(all_profiles))] for _ in range(len(all_profiles))],
                    "label_names": list(all_profiles),
                    "roc_auc": {profile: 0.0 for profile in all_profiles},
                    "avg_auc": 0.0,
                    "error": str(e)
                }
        
        # Calculate average LLM rubric scores
        avg_rubric_scores = {}
        count_rubric = 0
        
        for eval_data in valid_evals:
            llm_eval = eval_data.get('llm_evaluation', {})
            if llm_eval and 'rubric_scores' in llm_eval and isinstance(llm_eval['rubric_scores'], dict):
                count_rubric += 1
                # Normalize the criterion names before adding to average
                normalized_scores = self._normalize_rubric_criterion_names(llm_eval['rubric_scores'])
                for criterion, score in normalized_scores.items():
                    if isinstance(score, (int, float)):  # Ensure score is numeric
                        if criterion not in avg_rubric_scores:
                            avg_rubric_scores[criterion] = 0
                        avg_rubric_scores[criterion] += score
        
        if count_rubric > 0:
            for criterion in avg_rubric_scores:
                avg_rubric_scores[criterion] /= count_rubric
        
        # Calculate average NLP metrics
        avg_coherence = {"global_coherence": 0, "local_coherence": 0, "window_coherence": 0}
        avg_readability = {"flesch_kincaid_grade": 0, "gunning_fog_index": 0, "flesch_reading_ease": 0}
        avg_diversity = {"distinct_1": 0, "distinct_2": 0, "distinct_3": 0}
        
        count_metrics = 0
        
        for eval_data in valid_evals:
            ext_metrics = eval_data.get('extended_metrics', {})
            if ext_metrics:
                count_metrics += 1
                
                # Coherence
                if 'coherence' in ext_metrics and isinstance(ext_metrics['coherence'], dict):
                    for key in avg_coherence:
                        value = ext_metrics['coherence'].get(key, 0)
                        if isinstance(value, (int, float)):  # Ensure value is numeric
                            avg_coherence[key] += value
                
                # Readability
                if 'readability' in ext_metrics and isinstance(ext_metrics['readability'], dict):
                    for key in avg_readability:
                        value = ext_metrics['readability'].get(key, 0)
                        if isinstance(value, (int, float)):  # Ensure value is numeric
                            avg_readability[key] += value
                
                # Diversity
                if 'diversity' in ext_metrics and isinstance(ext_metrics['diversity'], dict):
                    for key in avg_diversity:
                        value = ext_metrics['diversity'].get(key, 0)
                        if isinstance(value, (int, float)):  # Ensure value is numeric
                            avg_diversity[key] += value
        
        if count_metrics > 0:
            for key in avg_coherence:
                avg_coherence[key] /= count_metrics
            for key in avg_readability:
                avg_readability[key] /= count_metrics
            for key in avg_diversity:
                avg_diversity[key] /= count_metrics
        
        # Compile final summary
        summary = {
            "total_evaluated": len(valid_evals),
            "provider": self.provider,
            "model": self.model,
            "diagnosis": {
                "total_diagnoses": len(true_diagnoses),
                "correct_matches": matches,
                "accuracy": matches / len(true_diagnoses) if true_diagnoses else 0,
                "metrics": diagnosis_metrics_data
            },
            "llm_rubric": avg_rubric_scores,
            "nlp_metrics": {
                "coherence": avg_coherence,
                "readability": avg_readability,
                "diversity": avg_diversity
            }
        }
        
        return summary
    
    def _save_batch_results(self, results: Dict[str, Any]) -> None:
        """
        Save batch evaluation results to file.
        
        Args:
            results: Batch evaluation results
        """
        # Create a filename that includes the provider and model
        base_name = f"batch_eval_{self.provider}_{self.model.replace(':', '-')}"
        
        # Create subdirectory for evaluation results if it doesn't exist
        eval_dir = os.path.join(self.batch_dir, f"batch_eval_{self.provider}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save detailed JSON results
        json_path = os.path.join(eval_dir, f"{self.model.replace(':', '-')}_results.json")
        
        # Ensure the directory exists (just to be extra safe)
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Saved detailed evaluation results to {json_path}")
        except Exception as e:
            print(f"Warning: Could not save JSON results: {e}")
            # Fallback to saving in the main batch directory
            fallback_path = os.path.join(self.batch_dir, f"{base_name}_results.json")
            try:
                with open(fallback_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved detailed evaluation results to fallback location: {fallback_path}")
            except Exception as e2:
                print(f"Error: Failed to save results to fallback location: {e2}")
        
        # Save summary CSV with per-label recall metrics
        try:
            csv_path = os.path.join(eval_dir, f"{self.model.replace(':', '-')}_summary.csv")
            
            # Extract key metrics for each file
            records = []
            
            for eval_data in results["evaluations"]:
                if 'error' in eval_data:
                    continue
                    
                record = {
                    "file_name": eval_data.get("file_name", "unknown"),
                    "timestamp": eval_data.get("timestamp", ""),
                    "provider": self.provider,
                    "model": self.model
                }
                
                # LLM rubric scores
                llm_eval = eval_data.get('llm_evaluation', {})
                if llm_eval and 'rubric_scores' in llm_eval:
                    for criterion, score in llm_eval['rubric_scores'].items():
                        record[f"rubric_{criterion}"] = score
                        
                    record["rubric_avg"] = llm_eval.get('average_score', 0)
                
                # Diagnosis evaluation
                diag_eval = eval_data.get('diagnosis_evaluation', {})
                if diag_eval:
                    record["expected_diagnosis"] = diag_eval.get('expected', '')
                    record["predicted_diagnosis"] = diag_eval.get('predicted', '')
                    record["diagnosis_match"] = diag_eval.get('matches', False)
                    # Add classification if available
                    record["diagnosis_classification"] = diag_eval.get('classification', '')
                    record["classification_confidence"] = diag_eval.get('classification_confidence', 0)
                
                # NLP metrics - select important ones
                ext_metrics = eval_data.get('extended_metrics', {})
                if ext_metrics:
                    # Coherence
                    if 'coherence' in ext_metrics:
                        record["global_coherence"] = ext_metrics['coherence'].get('global_coherence', 0)
                        record["local_coherence"] = ext_metrics['coherence'].get('local_coherence', 0)
                    
                    # Readability
                    if 'readability' in ext_metrics:
                        record["flesch_kincaid_grade"] = ext_metrics['readability'].get('flesch_kincaid_grade', 0)
                        record["flesch_reading_ease"] = ext_metrics['readability'].get('flesch_reading_ease', 0)
                
                records.append(record)
            
            # Initialize CSV
            if not records:
                print("No records to save to CSV")
                return
            
            # Save to CSV
            import csv
            with open(csv_path, 'w', newline='') as f:
                # Get all field names
                fieldnames = set()
                for record in records:
                    fieldnames.update(record.keys())
                
                # Create writer and write records
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(records)
            
            print(f"Saved summary CSV to {csv_path}")
            
            # Also create a diagnosis-specific summary CSV
            diag_summary_path = os.path.join(eval_dir, f"{self.model.replace(':', '-')}_diagnosis_summary.csv")
            
            # Extract diagnosis info for summary
            summary = results.get("summary", {})
            diag_summary = summary.get("diagnosis", {})
            metrics = diag_summary.get("metrics", {})
            recalls_by_label = metrics.get("recalls_by_label", {})
            precisions_by_label = metrics.get("precisions_by_label", {})
            f1_by_label = metrics.get("f1_by_label", {})
            
            if recalls_by_label or precisions_by_label or f1_by_label:
                with open(diag_summary_path, 'w', newline='') as f:
                    # Create fields for diagnosis summary
                    fields = ["model", "total_diagnoses", "correct_matches", "accuracy", 
                             "precision", "recall", "f1"]
                    
                    # Add fields for per-label metrics
                    for label in sorted(set(list(recalls_by_label.keys()) + 
                                         list(precisions_by_label.keys()) + 
                                         list(f1_by_label.keys()))):
                        if recalls_by_label:
                            fields.append(f"recall_{label}")
                        if precisions_by_label:
                            fields.append(f"precision_{label}")
                        if f1_by_label:
                            fields.append(f"f1_{label}")
                    
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    
                    # Create row with summary data
                    row = {
                        "model": self.model,
                        "total_diagnoses": diag_summary.get("total_diagnoses", 0),
                        "correct_matches": diag_summary.get("correct_matches", 0),
                        "accuracy": diag_summary.get("accuracy", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "f1": metrics.get("f1", 0),
                    }
                    
                    # Add per-label metrics
                    for label in sorted(set(list(recalls_by_label.keys()) + 
                                         list(precisions_by_label.keys()) + 
                                         list(f1_by_label.keys()))):
                        if label in recalls_by_label:
                            row[f"recall_{label}"] = recalls_by_label[label]
                        if label in precisions_by_label:
                            row[f"precision_{label}"] = precisions_by_label[label]
                        if label in f1_by_label:
                            row[f"f1_{label}"] = f1_by_label[label]
                    
                    writer.writerow(row)
                
                print(f"Saved diagnosis summary CSV to {diag_summary_path}")
            
        except Exception as e:
            import traceback
            print(f"Warning: Could not save CSV summary: {e}")
            traceback.print_exc()
        
        # Also save an HTML report if visualize_eval is available
        try:
            from utils.visualize_eval import generate_html_report
            report_path = os.path.join(eval_dir, f"{self.model.replace(':', '-')}_report.html")
            generate_html_report(self.batch_dir, results, report_path)
            print(f"Generated HTML report at {report_path}")
        except Exception as e:
            print(f"Warning: Could not generate HTML report: {e}")
            if isinstance(e, ImportError):
                print("Visualization module not available, skipping HTML report generation")
            else:
                print(f"Error during report generation: {e}")

    def fix_batch_evaluations(self) -> Dict[str, Any]:
        """
        Fix existing evaluations in a batch without re-evaluating.
        This is useful for correcting missing or incomplete metrics in
        evaluations that have already been run.
        
        Returns:
            Dictionary with fixed batch evaluation results
        """
        # Get all conversation files in the batch
        conversation_files = glob.glob(os.path.join(self.batch_dir, "chat_*.json"))
        
        if not conversation_files:
            raise ValueError(f"No conversation files found in {self.batch_dir}")
        
        print(f"Found {len(conversation_files)} conversation files in {self.batch_dir}")
        print(f"Checking for evaluations to fix...")
        
        fixed_files = 0
        total_files_with_evals = 0
        
        # Prepare results container
        results = {
            "batch_name": self.batch_name,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "provider": self.provider,
            "model": self.model,
            "total_conversations": len(conversation_files),
            "evaluations": [],
            "summary": {},
            "fixed_files": 0
        }
        
        # Process each file
        for file_path in tqdm(conversation_files, desc="Fixing evaluations"):
            try:
                # Load log data
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                
                # Check if file has evaluation
                if 'evaluation' in log_data:
                    total_files_with_evals += 1
                    evaluation = log_data['evaluation']
                    
                    # Only fix evaluations for the current provider and model
                    if (evaluation.get('provider', '') == self.provider and 
                        evaluation.get('model', '') == self.model):
                        
                        # Save original state
                        original_log_data = copy.deepcopy(log_data)
                        needs_fix = False
                        
                        # Check if diagnosis evaluation exists but is incomplete
                        if 'diagnosis_evaluation' in evaluation:
                            diag_eval = evaluation['diagnosis_evaluation']
                            
                            # Check if basic diagnosis fields are missing
                            required_fields = ['expected', 'predicted', 'matches']
                            missing_fields = [field for field in required_fields if field not in diag_eval]
                            
                            if missing_fields:
                                print(f"Fixing missing diagnosis fields for {os.path.basename(file_path)}: {missing_fields}")
                                
                                # Extract diagnosis data from log if needed
                                if 'predicted' in missing_fields or not diag_eval.get('predicted'):
                                    # Extract predicted diagnosis from the log
                                    conversation = log_data.get('conversation', [])
                                    diagnosis = log_data.get('diagnosis', '')
                                    
                                    if not diagnosis and conversation:
                                        last_message = conversation[-1].get('content', '')
                                        diagnosis_match = re.search(r'(?i)diagnosis:?\s*(.+?)(?:\n|$)', last_message)
                                        if diagnosis_match:
                                            diagnosis = diagnosis_match.group(1).strip()
                                    
                                    diag_eval['predicted'] = diagnosis
                                
                                if 'expected' in missing_fields or not diag_eval.get('expected'):
                                    # Extract expected diagnosis from metadata
                                    expected_diagnosis = ''
                                    if 'metadata' in log_data and 'patient_profile' in log_data['metadata']:
                                        expected_diagnosis = log_data['metadata']['patient_profile']
                                    
                                    diag_eval['expected'] = expected_diagnosis
                                
                                if 'matches' in missing_fields or 'matches' not in diag_eval:
                                    # Calculate match
                                    expected = diag_eval.get('expected', '')
                                    predicted = diag_eval.get('predicted', '')
                                    diag_eval['matches'] = self._diagnoses_match(predicted, expected)
                                
                                # Remove any metrics if they exist (as they should only be calculated at batch level)
                                if 'metrics' in diag_eval:
                                    del diag_eval['metrics']
                                
                                # Update the evaluation in the log data
                                evaluation['diagnosis_evaluation'] = diag_eval
                                log_data['evaluation'] = evaluation
                                needs_fix = True
                            
                            # Check for missing classification if LLM evaluation has it
                            if ('classification' not in diag_eval and 
                                'llm_evaluation' in evaluation and 
                                'diagnosis_classification' in evaluation['llm_evaluation']):
                                
                                print(f"Adding missing classification for {os.path.basename(file_path)}")
                                classification = evaluation['llm_evaluation']['diagnosis_classification']
                                diag_eval['classification'] = classification.get('classified_as', 'unknown')
                                
                                if 'confidence' in classification:
                                    diag_eval['classification_confidence'] = classification.get('confidence', 0)
                                
                                # Update the evaluation in the log data
                                evaluation['diagnosis_evaluation'] = diag_eval
                                log_data['evaluation'] = evaluation
                                needs_fix = True
                                
                        # Fix inconsistent rubric criterion names in llm_evaluation if they exist
                        if 'llm_evaluation' in evaluation and 'rubric_scores' in evaluation['llm_evaluation']:
                            rubric_scores = evaluation['llm_evaluation']['rubric_scores']
                            if rubric_scores:
                                # Check if any criteria names need normalization
                                normalized_scores = self._normalize_rubric_criterion_names(rubric_scores)
                                if normalized_scores != rubric_scores:
                                    print(f"Normalizing rubric criterion names for {os.path.basename(file_path)}")
                                    evaluation['llm_evaluation']['rubric_scores'] = normalized_scores
                                    log_data['evaluation'] = evaluation
                                    needs_fix = True
                        
                        # Save the fixed data if changes were made
                        if needs_fix and log_data != original_log_data:
                            with open(file_path, 'w') as f:
                                json.dump(log_data, f, indent=2)
                            fixed_files += 1
                            print(f"Saved fixed evaluation to {os.path.basename(file_path)}")
                        
                        # Add to results
                        eval_result = log_data['evaluation']
                        eval_result["file_name"] = os.path.basename(file_path)
                        results["evaluations"].append(eval_result)
            
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")
        
        # Generate batch summary
        results["summary"] = self._generate_batch_summary(results["evaluations"])
        results["fixed_files"] = fixed_files
        
        # Save batch results
        self._save_batch_results(results)
        
        print(f"\nFixed {fixed_files} out of {total_files_with_evals} evaluations")
        
        return results

    def _normalize_rubric_criterion_names(self, rubric_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize rubric criterion names to ensure consistency.
        
        Args:
            rubric_scores: Dictionary containing rubric scores with potentially inconsistent keys
            
        Returns:
            Dictionary with normalized criterion names
        """
        # Define the standard criterion names and their potential variations
        standard_names = {
            "dsm_coverage": ["dsm_coverage", "dsm coverage", "completeness_of_dsm-5_dimension_coverage", 
                           "completeness_of_dsm_5_dimension_coverage", "completeness_of_dsm_dimension_coverage"],
            "clinical_relevance": ["clinical_relevance", "clinical relevance", "clinical_relevance_and_accuracy", 
                                 "clinical_relevance_and_accuracy_of_questions"],
            "consistency": ["consistency", "consistency_and_logical_flow", "logical_flow"],
            "diagnostic_justification": ["diagnostic_justification", "diagnostic justification", 
                                       "diagnostic_justification_and_explainability"],
            "empathy": ["empathy", "empathy_naturalness_and_professionalism", "empathy_and_professionalism"]
        }
        
        # Create mapping from variations to standard names
        name_mapping = {}
        for standard, variations in standard_names.items():
            for variation in variations:
                name_mapping[variation.lower()] = standard
        
        # Normalize the keys in the scores dictionary
        normalized_scores = {}
        for key, value in rubric_scores.items():
            # Convert key to lowercase for case-insensitive matching
            key_lower = key.lower()
            if key_lower in name_mapping:
                standard_key = name_mapping[key_lower]
                normalized_scores[standard_key] = value
            else:
                # Keep the original key if no mapping is found
                normalized_scores[key] = value
                print(f"Warning: Unknown criterion name '{key}' - keeping as is")
        
        return normalized_scores

def example_usage():
    """Example usage of the batch evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate batches of mental health conversations")
    parser.add_argument("batch_path", help="Path to the batch directory")
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "groq", "openai"],
                       help="LLM provider to use for evaluation (ollama, groq, or openai)")
    parser.add_argument("--model", "-m", type=str, default="qwen2.5:3b", 
                       help="Model to use for LLM evaluation")
    parser.add_argument("--api-key", "-k", type=str, 
                       help="API key for cloud providers (required for Groq or OpenAI)")
    parser.add_argument("--max-workers", "-w", type=int, default=4, 
                       help="Maximum number of worker threads")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force re-evaluation of all logs, even if they already have evaluations")
    args = parser.parse_args()
    
    print(f"Evaluating batch {args.batch_path} with {args.provider} provider and model {args.model}...")
    if args.force:
        print("Force re-evaluation enabled, will re-evaluate all logs")
    
    # Create and run the evaluator
    evaluator = BatchEvaluator(
        batch_dir=args.batch_path,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        max_workers=args.max_workers,
        force_reevaluation=args.force
    )
    
    results = evaluator.evaluate_batch()
    
    # Print summary information
    summary = results.get('summary', {})
    
    print("\nBatch Evaluation Summary:")
    print(f"Total conversations evaluated: {summary.get('total_evaluated', 0)}")
    
    # Diagnosis accuracy
    diag_data = summary.get('diagnosis', {})
    diag_metrics = diag_data.get('metrics', {})
    print(f"\nDiagnosis Performance:")
    print(f"  Accuracy: {diag_data.get('accuracy', 0):.2f} ({diag_data.get('correct_matches', 0)}/{diag_data.get('total_diagnoses', 0)})")
    print(f"  Precision: {diag_metrics.get('precision', 0):.2f}")
    print(f"  Recall: {diag_metrics.get('recall', 0):.2f}")
    print(f"  F1 Score: {diag_metrics.get('f1', 0):.2f}")
    
    # LLM rubric scores
    llm_rubric = summary.get('llm_rubric', {})
    if llm_rubric:
        print("\nLLM Rubric Scores:")
        for criterion, score in llm_rubric.items():
            print(f"  {criterion}: {score:.2f}")
    
    # NLP metrics highlights
    nlp_metrics = summary.get('nlp_metrics', {})
    if nlp_metrics:
        print("\nNLP Metrics Highlights:")
        
        coherence = nlp_metrics.get('coherence', {})
        if coherence:
            print(f"  Global Coherence: {coherence.get('global_coherence', 0):.3f}")
            print(f"  Local Coherence: {coherence.get('local_coherence', 0):.3f}")
        
        readability = nlp_metrics.get('readability', {})
        if readability:
            print(f"  Flesch-Kincaid Grade: {readability.get('flesch_kincaid_grade', 0):.2f}")
            print(f"  Gunning Fog Index: {readability.get('gunning_fog_index', 0):.2f}")
        
        diversity = nlp_metrics.get('diversity', {})
        if diversity:
            print(f"  Distinct-1: {diversity.get('distinct_1', 0):.3f}")
            print(f"  Distinct-2: {diversity.get('distinct_2', 0):.3f}")

def fix_rubric_names_in_results_file(batch_dir: str, provider: str, model: str):
    """
    Fix the criterion names in an existing batch results file without requiring LLM API access.
    
    Args:
        batch_dir: Directory containing the batch
        provider: Provider name
        model: Model name
    """
    import os
    import json
    import copy
    from typing import Dict, Any
    
    # Create a normalized model name for file paths
    model_file_name = model.replace(':', '-')
    
    # Path to the results file
    eval_dir = os.path.join(batch_dir, f"batch_eval_{provider}")
    results_file = os.path.join(eval_dir, f"{model_file_name}_results.json")
    
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}")
        return
    
    # Define the standard criterion names and their potential variations
    standard_names = {
        "dsm_coverage": ["dsm_coverage", "dsm coverage", "completeness_of_dsm-5_dimension_coverage", 
                      "completeness_of_dsm_5_dimension_coverage", "completeness_of_dsm_dimension_coverage"],
        "clinical_relevance": ["clinical_relevance", "clinical relevance", "clinical_relevance_and_accuracy", 
                            "clinical_relevance_and_accuracy_of_questions"],
        "consistency": ["consistency", "consistency_and_logical_flow", "logical_flow"],
        "diagnostic_justification": ["diagnostic_justification", "diagnostic justification", 
                                  "diagnostic_justification_and_explainability"],
        "empathy": ["empathy", "empathy_naturalness_and_professionalism", "empathy_and_professionalism"]
    }
    
    # Create mapping from variations to standard names
    name_mapping = {}
    for standard, variations in standard_names.items():
        for variation in variations:
            name_mapping[variation.lower()] = standard
    
    # Function to normalize the keys in the scores dictionary
    def normalize_rubric_criterion_names(rubric_scores: Dict[str, Any]) -> Dict[str, Any]:
        normalized_scores = {}
        for key, value in rubric_scores.items():
            # Convert key to lowercase for case-insensitive matching
            key_lower = key.lower()
            if key_lower in name_mapping:
                standard_key = name_mapping[key_lower]
                normalized_scores[standard_key] = value
            else:
                # Keep the original key if no mapping is found
                normalized_scores[key] = value
                print(f"Warning: Unknown criterion name '{key}' - keeping as is")
        
        return normalized_scores
    
    try:
        # Load the results file
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        original_results = copy.deepcopy(results)
        changes_made = False
        
        # Process individual evaluations
        if 'evaluations' in results:
            for i, eval_data in enumerate(results['evaluations']):
                if ('llm_evaluation' in eval_data and 
                    'rubric_scores' in eval_data['llm_evaluation'] and 
                    isinstance(eval_data['llm_evaluation']['rubric_scores'], dict)):
                    
                    rubric_scores = eval_data['llm_evaluation']['rubric_scores']
                    normalized_scores = normalize_rubric_criterion_names(rubric_scores)
                    
                    if normalized_scores != rubric_scores:
                        print(f"Normalizing rubric scores for evaluation {i+1}")
                        eval_data['llm_evaluation']['rubric_scores'] = normalized_scores
                        changes_made = True
                        
                    # Also update explanations to match
                    if 'explanations' in eval_data['llm_evaluation']:
                        explanations = eval_data['llm_evaluation']['explanations']
                        normalized_explanations = {}
                        for key, value in explanations.items():
                            key_lower = key.lower()
                            if key_lower in name_mapping:
                                standard_key = name_mapping[key_lower]
                                normalized_explanations[standard_key] = value
                            else:
                                normalized_explanations[key] = value
                        
                        if normalized_explanations != explanations:
                            eval_data['llm_evaluation']['explanations'] = normalized_explanations
                            changes_made = True
        
        # Process summary
        if 'summary' in results and 'llm_rubric' in results['summary']:
            rubric_scores = results['summary']['llm_rubric']
            normalized_scores = normalize_rubric_criterion_names(rubric_scores)
            
            if normalized_scores != rubric_scores:
                print("Normalizing summary rubric scores")
                results['summary']['llm_rubric'] = normalized_scores
                changes_made = True
        
        # Save if changes were made
        if changes_made:
            print(f"Saving normalized results to {results_file}")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print("Normalization complete!")
        else:
            print("No changes were needed.")
    
    except Exception as e:
        import traceback
        print(f"Error normalizing results: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    example_usage() 