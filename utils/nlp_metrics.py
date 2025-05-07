"""
Natural Language Processing metrics for conversation evaluation.
This module provides various NLP metrics to evaluate the quality of mental health conversations.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from functools import lru_cache

# Function to clean text for analysis
def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, extra spaces, etc.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Remove special tags like <think></think>
    text = re.sub(r'<think>.*?</think>', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
    
    # Remove special characters but keep sentence structure
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\']', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Semantic Coherence Metrics

# Global variable to store the SentenceTransformer model instance
_sentence_transformer_model = None
# Simple cache for embeddings
_embedding_cache = {}

def _get_fallback_embedding(text: str) -> np.ndarray:
    """
    Create a simple fallback embedding using TF-IDF-like approach when SentenceTransformer fails.
    This is a very basic embedding that won't be semantically meaningful but will allow 
    the system to continue functioning.
    
    Args:
        text: Text to embed
        
    Returns:
        Simple embedding vector
    """
    # Ensure text is a string
    if not isinstance(text, str):
        return np.zeros(768)
    
    # Clean and normalize text
    text = text.lower()
    
    # Create a simple hash-based embedding
    # This is not semantically meaningful but provides a consistent vector
    embedding = np.zeros(768)
    
    # Use character and word n-grams as features
    words = text.split()
    
    # Simple word embedding
    for i, word in enumerate(words):
        # Use hash of word to determine indices to modify
        word_hash = hash(word) % 768
        # Set values based on word position and length
        embedding[word_hash] += 1.0 / (i + 1)
    
    # Character trigrams for more signal
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        if len(trigram.strip()) > 0:
            tri_hash = hash(trigram) % 768
            embedding[tri_hash] += 0.5
    
    # Normalize the vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def get_embedding(text: str, model=None) -> np.ndarray:
    """
    Get embeddings for text using a model. Uses nomic-embed-text through Ollama by default.
    
    Args:
        text: Text to embed
        model: Embedding model to use (default: nomic-embed-text through Ollama)
        
    Returns:
        Embedding vector
    """
    # Handle empty text
    if not text or len(text.strip()) == 0:
        # Return zero vector of appropriate dimension
        return np.zeros(768)  # Default dimension for nomic-embed-text
    
    # Use hash of text for caching
    text_hash = hash(text)
    
    # Check cache first
    if text_hash in _embedding_cache:
        return _embedding_cache[text_hash]
    
    # Directly use fallback embedding if the text is very short - avoid transformer issues
    if len(text.strip().split()) < 3:
        print(f"Text too short, using fallback embedding: '{text}'")
        embedding = _get_fallback_embedding(text)
        _embedding_cache[text_hash] = embedding
        return embedding
    
    # Try using Ollama with nomic-embed-text with fallback options
    for attempt in range(3):  # Try up to 3 different approaches
        try:
            if attempt == 0:
                # First attempt: Use Ollama with nomic-embed-text
                print("Using nomic-embed-text via Ollama for embedding...")
                import requests
                import json
                
                # Call Ollama API
                try:
                    response = requests.post(
                        'http://localhost:11434/api/embeddings',
                        json={
                            'model': 'nomic-embed-text',
                            'prompt': text
                        },
                        timeout=10  # 10 seconds timeout to avoid hanging
                    )
                    
                    if response.status_code == 200:
                        embedding_data = response.json()
                        if 'embedding' in embedding_data:
                            # Get the embedding from the response
                            embedding = np.array(embedding_data['embedding'])
                            print(f"Successfully generated embedding with shape: {embedding.shape}")
                        else:
                            raise ValueError(f"No embedding found in response: {embedding_data}")
                    else:
                        raise ValueError(f"Error from Ollama API: {response.text} (Status: {response.status_code})")
                    
                except requests.RequestException as e:
                    raise ConnectionError(f"Error connecting to Ollama: {e}")
                
            elif attempt == 1:
                # Second attempt: Direct approach using transformers AutoModel
                print("Trying direct transformers approach...")
                import torch
                from transformers import AutoModel, AutoTokenizer
                
                # Load base model components separately
                tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                base_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                
                # Move to CPU explicitly
                base_model = base_model.to('cpu')
                
                # Process text directly
                encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                # Move inputs to CPU
                encoded_input = {k: v.to('cpu') for k, v in encoded_input.items()}
                
                # Get model outputs
                with torch.no_grad():
                    model_output = base_model(**encoded_input)
                    
                # Use mean pooling to get the sentence embedding
                token_embeddings = model_output[0]  # First element contains token embeddings
                attention_mask = encoded_input['attention_mask']
                
                # Multiply token embeddings by attention mask to zero out padding tokens
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings_sum = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Mean pooling
                sentence_embedding = embeddings_sum / sum_mask
                
                # Convert to numpy array
                embedding = sentence_embedding.numpy()[0]  # Take first (and only) sentence
                
            else:
                # Third attempt: Use fallback embedding method
                print("Using fallback embedding method")
                embedding = _get_fallback_embedding(text)
            
            # Cache the result (limit cache size to prevent memory issues)
            if len(_embedding_cache) > 10000:  # Limit cache to 10,000 entries
                # Clear half the cache when it gets too big
                for k in list(_embedding_cache.keys())[:5000]:
                    del _embedding_cache[k]
            _embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            # Log the error
            error_type = type(e).__name__
            print(f"Embedding attempt {attempt+1} failed with {error_type}: {str(e)}")
            
            # If it's the Meta tensor error, immediately go to next method
            if "Meta tensor" in str(e) or "abstract impl" in str(e) or "NotImplementedError" in str(e):
                continue
                
            # If it's the last attempt, use the fallback method
            if attempt == 2:
                print("All embedding attempts failed, using simple fallback")
                embedding = _get_fallback_embedding(text)
                _embedding_cache[text_hash] = embedding
                return embedding
    
    # If we get here, use fallback method
    print("Fallback to simple embedding method")
    embedding = _get_fallback_embedding(text)
    _embedding_cache[text_hash] = embedding
    return embedding

def semantic_coherence(conversation: List[Dict[str, str]], window_size: int = 3) -> Dict[str, float]:
    """
    Calculate the semantic coherence of a conversation using embeddings.
    
    Args:
        conversation: List of conversation messages
        window_size: Size of sliding window for local coherence
        
    Returns:
        Dictionary with semantic coherence metrics
    """
    try:
        # Extract just the content from each message
        texts = [msg.get('content', '') for msg in conversation]
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Handle empty conversations
        if not cleaned_texts or all(not text for text in cleaned_texts):
            print("Warning: Empty conversation or all messages are empty. Returning default values.")
            return {
                "global_coherence": 0.0,
                "local_coherence": 0.0,
                "window_coherence": 0.0
            }
        
        # Get embeddings with better error handling
        embeddings = []
        embedding_errors = 0
        for text in cleaned_texts:
            try:
                embedding = get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                embedding_errors += 1
                # Use fallback embedding
                embeddings.append(_get_fallback_embedding(text))
        
        if embedding_errors > 0:
            print(f"Warning: {embedding_errors} out of {len(cleaned_texts)} embeddings had errors and used fallbacks.")
        
        # Ensure we have enough embeddings to proceed
        if len(embeddings) < 2:
            print("Warning: Not enough valid messages to calculate coherence. Returning default values.")
            return {
                "global_coherence": 0.0,
                "local_coherence": 0.0,
                "window_coherence": 0.0
            }
        
        # Calculate global coherence (average cosine similarity between all pairs)
        global_coherence = 0.0
        pair_count = 0
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                try:
                    # Cosine similarity with safety checks
                    norm_i = np.linalg.norm(embeddings[i])
                    norm_j = np.linalg.norm(embeddings[j])
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
                        # Clamp similarity values to valid range
                        similarity = max(-1.0, min(1.0, similarity))
                        global_coherence += similarity
                        pair_count += 1
                except Exception as e:
                    print(f"Error calculating cosine similarity: {e}")
        
        if pair_count > 0:
            global_coherence /= pair_count
        
        # Calculate local coherence (average similarity between adjacent messages)
        local_coherence = 0.0
        local_pair_count = 0
        
        for i in range(len(embeddings) - 1):
            try:
                norm_i = np.linalg.norm(embeddings[i])
                norm_i1 = np.linalg.norm(embeddings[i+1])
                
                if norm_i > 0 and norm_i1 > 0:
                    similarity = np.dot(embeddings[i], embeddings[i+1]) / (norm_i * norm_i1)
                    # Clamp similarity values to valid range
                    similarity = max(-1.0, min(1.0, similarity))
                    local_coherence += similarity
                    local_pair_count += 1
            except Exception as e:
                print(f"Error calculating local coherence: {e}")
        
        if local_pair_count > 0:
            local_coherence /= local_pair_count
        
        # Calculate window coherence
        window_coherence = 0.0
        window_count = 0
        
        # Ensure we have enough messages for window coherence
        if len(embeddings) >= window_size:
            for i in range(len(embeddings) - window_size + 1):
                window_embeddings = embeddings[i:i+window_size]
                window_sim = 0.0
                window_pairs = 0
                
                for j in range(len(window_embeddings)):
                    for k in range(j+1, len(window_embeddings)):
                        try:
                            norm_j = np.linalg.norm(window_embeddings[j])
                            norm_k = np.linalg.norm(window_embeddings[k])
                            
                            if norm_j > 0 and norm_k > 0:
                                similarity = np.dot(window_embeddings[j], window_embeddings[k]) / (norm_j * norm_k)
                                # Clamp similarity values to valid range
                                similarity = max(-1.0, min(1.0, similarity))
                                window_sim += similarity
                                window_pairs += 1
                        except Exception as e:
                            print(f"Error calculating window coherence: {e}")
                
                if window_pairs > 0:
                    window_sim /= window_pairs
                    window_coherence += window_sim
                    window_count += 1
        
        if window_count > 0:
            window_coherence /= window_count
        
        return {
            "global_coherence": float(global_coherence),
            "local_coherence": float(local_coherence),
            "window_coherence": float(window_coherence)
        }
    except Exception as e:
        print(f"Critical error in semantic_coherence: {e}")
        # Return default values as a fallback
        return {
            "global_coherence": 0.0,
            "local_coherence": 0.0,
            "window_coherence": 0.0
        }

# Readability Metrics

def count_syllables(word: str) -> int:
    """
    Count the number of syllables in a word.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Number of syllables
    """
    # Remove non-alphanumeric characters
    word = re.sub(r'[^a-zA-Z]', '', word.lower())
    
    # Count vowel groups
    if not word:
        return 0
        
    # Special cases
    if word[-1] == 'e':
        word = word[:-1]
        
    # Count vowel groups
    syllables = len(re.findall(r'[aeiouy]+', word))
    
    return max(1, syllables)  # Every word has at least one syllable

def flesch_kincaid_grade(text: str) -> float:
    """
    Calculate the Flesch-Kincaid Grade Level for a text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Flesch-Kincaid Grade Level score
    """
    text = clean_text(text)
    
    # Split text into sentences and words
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = text.split()
    words = [w for w in words if w.strip()]
    
    if not sentences or not words:
        return 0.0
    
    # Count total syllables
    total_syllables = sum(count_syllables(word) for word in words)
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Flesch-Kincaid Grade Level formula
    grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    
    return round(grade_level, 2)

def gunning_fog_index(text: str) -> float:
    """
    Calculate the Gunning Fog Index for a text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Gunning Fog Index score
    """
    text = clean_text(text)
    
    # Split text into sentences and words
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = text.split()
    words = [w for w in words if w.strip()]
    
    if not sentences or not words:
        return 0.0
    
    # Count complex words (words with 3+ syllables)
    complex_words = [w for w in words if count_syllables(w) >= 3]
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences)
    percent_complex_words = len(complex_words) / len(words) * 100
    
    # Gunning Fog formula
    fog_index = 0.4 * (avg_sentence_length + percent_complex_words / 100)
    
    return round(fog_index, 2)

def flesch_reading_ease(text: str) -> float:
    """
    Calculate the Flesch Reading Ease score for a text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Flesch Reading Ease score
    """
    text = clean_text(text)
    
    # Split text into sentences and words
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = text.split()
    words = [w for w in words if w.strip()]
    
    if not sentences or not words:
        return 0.0
    
    # Count total syllables
    total_syllables = sum(count_syllables(word) for word in words)
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Flesch Reading Ease formula
    reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    
    return round(reading_ease, 2)

def readability_metrics(text: str) -> Dict[str, float]:
    """
    Calculate multiple readability metrics for a text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with readability metrics
    """
    return {
        "flesch_kincaid_grade": flesch_kincaid_grade(text),
        "gunning_fog_index": gunning_fog_index(text),
        "flesch_reading_ease": flesch_reading_ease(text)
    }

# Distinct-N Metrics

def distinct_n(text: str, n: int = 1) -> float:
    """
    Calculate the Distinct-N metric for a text.
    
    Args:
        text: Text to analyze
        n: N-gram size
        
    Returns:
        Distinct-N score
    """
    text = clean_text(text)
    words = text.split()
    
    if len(words) < n:
        return 0.0
    
    # Generate n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    # Calculate distinct ratio
    distinct_ngrams = set(ngrams)
    
    if not ngrams:
        return 0.0
        
    return len(distinct_ngrams) / len(ngrams)

def diversity_metrics(text: str) -> Dict[str, float]:
    """
    Calculate diversity metrics for a text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with diversity metrics
    """
    return {
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3)
    }

# Diagnosis Evaluation Metrics

def diagnosis_metrics(true_labels: List[str], pred_labels: List[str], label_names: List[str] = None) -> Dict[str, Any]:
    """
    Calculate diagnosis evaluation metrics.
    
    Args:
        true_labels: List of true diagnoses
        pred_labels: List of predicted diagnoses
        label_names: List of label names for confusion matrix
        
    Returns:
        Dictionary with diagnosis evaluation metrics
    """
    # Handle empty input case
    if not true_labels or not pred_labels:
        print("Warning: Empty label lists provided to diagnosis_metrics")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [],
            "label_names": label_names or [],
            "roc_auc": {},
            "avg_auc": 0.0
        }
    
    # Ensure labels are in the same format
    true_labels = [label.lower().strip() for label in true_labels]
    pred_labels = [label.lower().strip() for label in pred_labels]
    
    # Debug
    print(f"Input to diagnosis_metrics: {len(true_labels)} true labels, {len(pred_labels)} predicted labels")
    
    # Get unique labels if not provided
    if label_names is None:
        label_names = sorted(set(true_labels + pred_labels))
    else:
        # Make sure label_names are also in the same format
        label_names = [label.lower().strip() for label in label_names]
    
    # Ensure we have at least one label
    if not label_names:
        print("Warning: No label names found in diagnosis_metrics")
        label_names = ["unknown"]
    
    # Debug
    print(f"Using {len(label_names)} label names: {label_names}")
    
    # Function to check if diagnoses match (similar to BatchEvaluator._diagnoses_match)
    def diagnoses_match(predicted, expected):
        if not predicted or not expected:
            return False
            
        # Clean and normalize
        pred_clean = predicted.lower().strip()
        expected_clean = expected.lower().strip()
        
        # Direct match
        if expected_clean in pred_clean:
            return True
            
        # Handle common variants (e.g., "anxiety" matches "anxiety disorder")
        if expected_clean == "social_anxiety" and "social_anxiety" in pred_clean:
            return True
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
        if expected_clean == "adjustment" and "adjustment" in pred_clean:
            return True
        if expected_clean == "substance_abuse" and "substance_abuse" in pred_clean:
            return True
        if expected_clean == "ocd" and "ocd" in pred_clean:
            return True
        if expected_clean == "panic" and "panic" in pred_clean:
            return True
            
        return False
    
    # Perform simple standardization to handle common variations
    # This matches how the BatchEvaluator._diagnoses_match method works
    standardized_true = []
    standardized_pred = []
    
    for true, pred in zip(true_labels, pred_labels):
        # Standardize common terms
        if "social_anxiety" in true:
            std_true = "social_anxiety"
        elif "anxiety" in true:
            std_true = "anxiety"
        elif "depress" in true or "mdd" in true:
            std_true = "depression"
        elif "ptsd" in true or "post-traumatic" in true or "post traumatic" in true:
            std_true = "ptsd"
        elif "bipolar" in true:
            std_true = "bipolar"
        elif "schizophrenia" in true:
            std_true = "schizophrenia"
        elif "adjustment" in true:
            std_true = "adjustment"
        elif "substance_abuse" in true:
            std_true = "substance_abuse"
        elif "ocd" in true:
            std_true = "ocd"
        elif "panic" in true:
            std_true = "panic"
        else:
            std_true = true

        if "social_anxiety" in pred:
            std_pred = "social_anxiety"    
        elif "anxiety" in pred:
            std_pred = "anxiety"
        elif "depress" in pred or "mdd" in pred:
            std_pred = "depression"
        elif "ptsd" in pred or "post-traumatic" in pred or "post traumatic" in pred:
            std_pred = "ptsd"
        elif "bipolar" in pred:
            std_pred = "bipolar"
        elif "schizophrenia" in pred:
            std_pred = "schizophrenia"
        elif "somatic" in pred:
            std_pred = "somatic symptom disorder"
        elif "adjustment" in pred:
            std_pred = "adjustment"
        elif "substance_abuse" in pred:
            std_pred = "substance_abuse"
        elif "ocd" in pred:
            std_pred = "ocd"
        elif "panic" in pred:
            std_pred = "panic"
        else:
            std_pred = pred
        
        standardized_true.append(std_true)
        standardized_pred.append(std_pred)
        
        # Debug what we're doing with the standardization
        if std_true != true or std_pred != pred:
            print(f"Standardized: '{true}' -> '{std_true}', '{pred}' -> '{std_pred}'")
    
    # Make sure all standardized labels are in label_names
    for label in set(standardized_true + standardized_pred):
        if label not in label_names:
            label_names.append(label)
            print(f"Added missing label to label_names: '{label}'")
    
    # Map text labels to numeric indices
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    
    # Debug the mapping
    print(f"Label to index mapping: {label_to_idx}")
    
    # Map to indices, using -1 for anything that doesn't match
    true_indices = []
    pred_indices = []
    
    for true, pred in zip(standardized_true, standardized_pred):
        true_idx = label_to_idx.get(true, -1)
        pred_idx = label_to_idx.get(pred, -1)
        
        # Debug any missing mappings
        if true_idx == -1:
            print(f"Warning: Could not map true label '{true}' to any index")
        if pred_idx == -1:
            print(f"Warning: Could not map predicted label '{pred}' to any index")
            
        true_indices.append(true_idx)
        pred_indices.append(pred_idx)
    
    # Filter out any invalid indices
    valid_pairs = [(i, j, true, pred) for i, j, true, pred in 
                  zip(true_indices, pred_indices, standardized_true, standardized_pred) 
                  if i != -1 and j != -1]
    
    if not valid_pairs:
        print("Warning: No valid label pairs found in diagnosis_metrics")
        # Return a structured empty result
        empty_cm = [[0 for _ in range(len(label_names))] for _ in range(len(label_names))]
        empty_roc = {label: 0.0 for label in label_names}
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": empty_cm,
            "label_names": label_names,
            "roc_auc": empty_roc,
            "avg_auc": 0.0
        }
    
    # Unpack valid pairs
    valid_true_indices = [i for i, _, _, _ in valid_pairs]
    valid_pred_indices = [j for _, j, _, _ in valid_pairs]
    valid_true_labels = [t for _, _, t, _ in valid_pairs]
    valid_pred_labels = [p for _, _, _, p in valid_pairs]
    
    # Debug valid pairs
    print(f"Valid label pairs: {len(valid_pairs)}")
    for i, (true_idx, pred_idx, true, pred) in enumerate(valid_pairs):
        print(f"  {i+1}. '{true}' (idx {true_idx}) - '{pred}' (idx {pred_idx})")
        
    # Track all original pairs for diagnosing issues
    print(f"All pairs (original format):")
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        print(f"  {i+1}. '{true}' - '{pred}' - Match: {diagnoses_match(pred, true)}")
    
    try:
        # Calculate confusion matrix for disorders
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        # Multi-class confusion matrix for detailed disorder breakdown
        cm = confusion_matrix(valid_true_indices, valid_pred_indices, labels=range(len(label_names)))
        
        # Debug confusion matrix
        print("Confusion matrix shape:", cm.shape)
        
        # Calculate precision, recall and F1 for each label separately using the matching logic
        precisions_by_label = {}
        recalls_by_label = {}
        f1_by_label = {}
        
        for label in label_names:
            # Find all instances where this is the true label
            true_indices_for_label = [i for i, true_label in enumerate(true_labels) 
                                     if diagnoses_match(true_label, label)]
            
            # Find all instances where this is the predicted label
            pred_indices_for_label = [i for i, pred_label in enumerate(pred_labels)
                                     if diagnoses_match(pred_label, label)]
            
            # Count true positives
            true_positives = sum(1 for i in pred_indices_for_label 
                                if i in true_indices_for_label)
            
            # Calculate precision
            if pred_indices_for_label:
                precision = true_positives / len(pred_indices_for_label)
            else:
                precision = 0.0
            
            # Calculate recall
            if true_indices_for_label:
                recall = true_positives / len(true_indices_for_label)
            else:
                recall = 0.0
            
            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Store metrics
            precisions_by_label[label] = precision
            recalls_by_label[label] = recall
            f1_by_label[label] = f1
            
            print(f"Metrics for '{label}':")
            print(f"  Precision: {precision:.4f} ({true_positives}/{len(pred_indices_for_label) if pred_indices_for_label else 0})")
            print(f"  Recall: {recall:.4f} ({true_positives}/{len(true_indices_for_label) if true_indices_for_label else 0})")
            print(f"  F1: {f1:.4f}")
        
        # Calculate mean metrics across all labels
        precision = sum(precisions_by_label.values()) / len(precisions_by_label) if precisions_by_label else 0.0
        recall = sum(recalls_by_label.values()) / len(recalls_by_label) if recalls_by_label else 0.0
        
        # Calculate mean F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        print(f"Mean precision: {precision:.4f}")
        print(f"Mean recall: {recall:.4f}")
        print(f"Mean F1: {f1:.4f}")
        
        # ROC curves and AUC (one-vs-rest approach) - Keep this for detailed analysis
        roc_auc = {}
        for i, label in enumerate(label_names):
            # Create binary labels for this class
            true_binary = [1 if idx == i else 0 for idx in valid_true_indices]
            pred_binary = [1 if idx == i else 0 for idx in valid_pred_indices]
            
            # Skip if we don't have any positive examples for this class
            if sum(true_binary) == 0:
                print(f"Skipping ROC for '{label}' - no positive examples")
                roc_auc[label] = 0.0
                continue
                
            # Skip if all predictions are the same (ROC undefined)
            if all(p == pred_binary[0] for p in pred_binary):
                print(f"Skipping ROC for '{label}' - all predictions are {pred_binary[0]}")
                roc_auc[label] = 0.0
                continue
            
            try:
                fpr, tpr, _ = roc_curve(true_binary, pred_binary)
                roc_auc[label] = float(auc(fpr, tpr))  # Ensure it's a regular float
                print(f"ROC AUC for '{label}': {roc_auc[label]:.4f}")
            except Exception as e:
                print(f"Warning: ROC calculation failed for label '{label}': {e}")
                roc_auc[label] = 0.0
        
        # Average AUC across all classes
        avg_auc = sum(roc_auc.values()) / len(roc_auc) if roc_auc else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "precisions_by_label": precisions_by_label,
            "recalls_by_label": recalls_by_label,
            "f1_by_label": f1_by_label,
            "confusion_matrix": cm.tolist(),
            "label_names": label_names,
            "roc_auc": roc_auc,
            "avg_auc": float(avg_auc)
        }
    
    except Exception as e:
        import traceback
        print(f"Error in diagnosis_metrics calculation: {e}")
        traceback.print_exc()
        
        # Return a structured empty result with the error
        empty_cm = [[0 for _ in range(len(label_names))] for _ in range(len(label_names))]
        empty_roc = {label: 0.0 for label in label_names}
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": empty_cm,
            "label_names": label_names,
            "roc_auc": empty_roc,
            "avg_auc": 0.0,
            "error": str(e)
        } 