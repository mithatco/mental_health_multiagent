#!/usr/bin/env python3
"""
Batch Analysis Tool

This tool analyzes the results of batch conversation generation.
"""

import os
import sys
import json
import argparse
import csv
from collections import defaultdict, Counter
import re

def load_batch_results(batch_dir):
    """Load batch results from a directory."""
    # Try to load the summary file
    summary_path = os.path.join(batch_dir, "batch_summary.csv")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            reader = csv.DictReader(f)
            summary = list(reader)
    else:
        summary = None
    
    # Try to load the detailed results
    results_path = os.path.join(batch_dir, "batch_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = None
    
    # Load individual conversation logs
    logs = []
    for filename in os.listdir(batch_dir):
        if filename.endswith('.json') and not filename == "batch_results.json":
            with open(os.path.join(batch_dir, filename), 'r') as f:
                logs.append(json.load(f))
    
    return {
        "summary": summary,
        "results": results,
        "logs": logs
    }

def analyze_profiles(batch_data):
    """Analyze distribution of patient profiles."""
    profile_counts = Counter()
    
    for log in batch_data["logs"]:
        profile = log.get("metadata", {}).get("patient_profile", "unknown")
        profile_counts[profile] += 1
    
    return profile_counts

def extract_diagnoses(batch_data):
    """Extract and categorize diagnoses."""
    diagnoses = []
    
    for log in batch_data["logs"]:
        diagnosis = log.get("diagnosis", "")
        profile = log.get("metadata", {}).get("patient_profile", "unknown")
        
        # Try to extract condition from diagnosis
        conditions = []
        condition_patterns = [
            r"diagnosis of ([^\.]+)",
            r"diagnosed with ([^\.]+)",
            r"symptoms of ([^\.]+)",
            r"consistent with ([^\.]+)",
            r"suggestive of ([^\.]+)"
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, diagnosis, re.IGNORECASE)
            conditions.extend(matches)
        
        diagnoses.append({
            "profile": profile,
            "extracted_conditions": conditions,
            "diagnosis": diagnosis
        })
    
    return diagnoses

def calculate_statistics(batch_data):
    """Calculate statistics about the conversations."""
    statistics = {}
    
    if not batch_data["logs"]:
        return statistics
    
    # Calculate average conversation length
    message_counts = []
    for log in batch_data["logs"]:
        conversation = log.get("conversation", [])
        message_counts.append(len(conversation))
    
    statistics["message_counts"] = message_counts
    statistics["avg_message_count"] = sum(message_counts) / len(message_counts)
    statistics["min_message_count"] = min(message_counts)
    statistics["max_message_count"] = max(message_counts)
    
    # Calculate average duration
    durations = []
    for log in batch_data["logs"]:
        duration = log.get("metadata", {}).get("duration")
        if duration:
            durations.append(float(duration))
    
    if durations:
        statistics["durations"] = durations
        statistics["avg_duration"] = sum(durations) / len(durations)
        statistics["min_duration"] = min(durations)
        statistics["max_duration"] = max(durations)
    
    return statistics

def main():
    parser = argparse.ArgumentParser(description="Batch Analysis Tool")
    parser.add_argument('batch_dir', help="Directory containing batch results")
    args = parser.parse_args()
    
    if not os.path.isdir(args.batch_dir):
        print(f"Error: {args.batch_dir} is not a directory")
        sys.exit(1)
    
    print(f"Analyzing batch results in: {args.batch_dir}")
    batch_data = load_batch_results(args.batch_dir)
    
    if not batch_data["logs"]:
        print("No conversation logs found in the batch directory")
        sys.exit(1)
    
    print(f"\nFound {len(batch_data['logs'])} conversations")
    
    # Profile analysis
    profile_counts = analyze_profiles(batch_data)
    print("\n=== Profile Distribution ===")
    for profile, count in profile_counts.most_common():
        print(f"{profile}: {count} conversations ({count/len(batch_data['logs'])*100:.1f}%)")
    
    # Diagnosis analysis
    diagnoses = extract_diagnoses(batch_data)
    print("\n=== Diagnosis Analysis ===")
    condition_counter = Counter()
    for diagnosis in diagnoses:
        for condition in diagnosis["extracted_conditions"]:
            condition_counter[condition.lower()] += 1
    
    print("\nExtracted conditions:")
    for condition, count in condition_counter.most_common(10):
        print(f"- {condition}: {count}")
    
    # Profile-diagnosis correlation
    print("\n=== Profile-Diagnosis Correlation ===")
    profile_conditions = defaultdict(Counter)
    
    for diagnosis in diagnoses:
        profile = diagnosis["profile"]
        for condition in diagnosis["extracted_conditions"]:
            profile_conditions[profile][condition.lower()] += 1
    
    for profile, conditions in profile_conditions.items():
        print(f"\n{profile}:")
        for condition, count in conditions.most_common(3):
            print(f"- {condition}: {count}")
    
    # Statistics
    stats = calculate_statistics(batch_data)
    print("\n=== Conversation Statistics ===")
    print(f"Average message count: {stats.get('avg_message_count', 'N/A'):.1f}")
    print(f"Message count range: {stats.get('min_message_count', 'N/A')} to {stats.get('max_message_count', 'N/A')}")
    
    if 'avg_duration' in stats:
        print(f"Average duration: {stats.get('avg_duration', 'N/A'):.1f} seconds")
        print(f"Duration range: {stats.get('min_duration', 'N/A'):.1f} to {stats.get('max_duration', 'N/A'):.1f} seconds")

if __name__ == "__main__":
    main()
