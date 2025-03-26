"""
Chat Log Viewer Flask Application

A simple web interface for viewing mental health assessment chat logs.
"""
import os
import argparse
import time
import subprocess
import uuid
import sys
from flask import Flask, render_template, jsonify, request, send_file, Response, send_from_directory, url_for
from .chat_log_manager import ChatLogManager
import io
import json
import datetime
import csv
import re
import threading
import requests
import shutil
import tempfile

# Update the import to use our ChatLogEvaluator directly
from utils.chat_evaluator import ChatLogEvaluator

# Add path to the parent directory to import main.py functionality
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.document_processor import DocumentProcessor
from utils.rag_engine import RAGEngine
from agents.patient import Patient

app = Flask(__name__)

# Create singletons
chat_log_manager = None
chat_evaluator = None

# Active conversations dictionary
active_conversations = {}

# Track ongoing evaluations with timestamps for cleanup
ongoing_evaluations = {}

# Active batch generations dictionary
active_batches = {}

# Fix the cleanup_stale_evaluations function
def cleanup_stale_evaluations():
    """Clean up evaluations that have been running for too long."""
    global ongoing_evaluations
    
    # Current time
    current_time = time.time()
    
    # Find evaluations to remove (older than 5 minutes or completed/error)
    to_remove = []
    for log_id, eval_data in ongoing_evaluations.items():
        # Remove if older than 5 minutes
        if 'start_time' in eval_data and current_time - eval_data['start_time'] > 300:
            to_remove.append(log_id)
            print(f"Removing stale evaluation for {log_id} (timeout)")
        # Remove if status is completed or error and older than 2 minutes
        elif (eval_data.get('status') in ['completed', 'error'] and 
              'finish_time' in eval_data and current_time - eval_data['finish_time'] > 120):
            to_remove.append(log_id)
            print(f"Removing completed evaluation for {log_id}")
    
    # Remove old evaluations
    for log_id in to_remove:
        del ongoing_evaluations[log_id]

# Clean up stale conversations
def cleanup_stale_conversations():
    """Clean up conversations that have been inactive for too long."""
    global active_conversations
    
    # Current time
    current_time = time.time()
    
    # Find conversations to remove (older than 30 minutes)
    to_remove = []
    for conv_id, conv_data in active_conversations.items():
        # Remove if older than 30 minutes and not completed
        if (conv_data.get('status') not in ['completed', 'error'] and
            'start_time' in conv_data and current_time - conv_data['start_time'] > 1800):
            to_remove.append(conv_id)
            print(f"Removing stale conversation {conv_id} (timeout)")
            
            # Try to terminate the process if it's still running
            if 'process' in conv_data and conv_data['process'].poll() is None:
                try:
                    conv_data['process'].terminate()
                except:
                    pass
    
    # Remove old conversations
    for conv_id in to_remove:
        del active_conversations[conv_id]

# Clean up stale batches
def cleanup_stale_batches():
    """Clean up batch generations that have been inactive for too long."""
    global active_batches
    
    # Current time
    current_time = time.time()
    
    # Find batches to remove (older than 60 minutes)
    to_remove = []
    for batch_id, batch_data in active_batches.items():
        # Remove if older than 60 minutes and not completed
        if (batch_data.get('status') not in ['completed', 'error'] and
            'start_time' in batch_data and current_time - batch_data['start_time'] > 3600):
            to_remove.append(batch_id)
            print(f"Removing stale batch {batch_id} (timeout)")
            
            # Try to terminate the process if it's still running
            if 'process' in batch_data and batch_data['process'].poll() is None:
                try:
                    batch_data['process'].terminate()
                except:
                    pass
    
    # Remove old batches
    for batch_id in to_remove:
        del active_batches[batch_id]

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/conversation')
def conversation():
    """Render the conversation page."""
    return render_template('conversation.html')

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = []
    profiles = set()
    
    # Look for JSON files directly in the logs directory
    for filename in os.listdir(chat_log_manager.get_logs_directory()):
        if filename.endswith('.json') and not filename.startswith('.'):
            filepath = os.path.join(chat_log_manager.get_logs_directory(), filename)
            try:
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    
                    # Extract timestamp
                    timestamp = data.get('timestamp', '')
                    
                    # Format date for display
                    formatted_date = "Unknown"
                    try:
                        date_obj = datetime.datetime.fromisoformat(timestamp)
                        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                    
                    # Get patient profile
                    profile = "Unknown"
                    if 'metadata' in data and 'patient_profile' in data['metadata']:
                        profile = data['metadata']['patient_profile']
                        profiles.add(profile)
                    
                    # Get questionnaire
                    questionnaire = data.get('questionnaire', 'Unknown')
                    
                    logs.append({
                        'id': filename.replace('.json', ''),
                        'filename': filename,
                        'timestamp': timestamp,
                        'formatted_date': formatted_date,
                        'profile': profile,
                        'questionnaire': questionnaire
                    })
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    # Sort logs by timestamp (newest first)
    logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify({
        'logs': logs,
        'profiles': list(profiles)
    })

@app.route('/api/logs/<log_id>', methods=['GET'])
def get_log_details(log_id):
    # Check if log_id contains .json extension
    if not log_id.endswith('.json'):
        log_id += '.json'
    
    # First check if file exists in the main logs directory
    filepath = os.path.join(chat_log_manager.get_logs_directory(), log_id)
    
    # If not found, check in batch subdirectories
    if not os.path.isfile(filepath):
        for subdir in os.listdir(chat_log_manager.get_logs_directory()):
            subdir_path = os.path.join(chat_log_manager.get_logs_directory(), subdir)
            if os.path.isdir(subdir_path):
                potential_file = os.path.join(subdir_path, log_id)
                if os.path.isfile(potential_file):
                    filepath = potential_file
                    break
    
    if not os.path.isfile(filepath):
        return jsonify({'error': 'Log file not found'}), 404
    
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Error reading log file: {str(e)}'}), 500

@app.route('/api/logs/<log_id>/export', methods=['GET'])
def export_log(log_id):
    # Check if log_id contains .json extension
    if not log_id.endswith('.json'):
        log_id += '.json'
    
    filepath = os.path.join(chat_log_manager.get_logs_directory(), log_id)
    
    # If not found, check in batch subdirectories
    if not os.path.isfile(filepath):
        for subdir in os.listdir(chat_log_manager.get_logs_directory()):
            subdir_path = os.path.join(subdir, log_id)
            if os.path.isdir(subdir_path):
                potential_file = os.path.join(subdir_path, log_id)
                if os.path.isfile(potential_file):
                    filepath = potential_file
                    break
    
    if not os.path.isfile(filepath):
        return jsonify({'error': 'Log file not found'}), 404
    
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            
        # Generate text format
        text_content = f"Conversation Log - {data.get('timestamp', 'Unknown')}\n"
        text_content += f"Questionnaire: {data.get('questionnaire', 'Unknown')}\n"
        text_content += "-" * 80 + "\n\n"
        
        for message in data.get('conversation', []):
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            text_content += f"{role.upper()}:\n{content}\n\n"
        
        text_content += "-" * 80 + "\n\n"
        text_content += "DIAGNOSIS:\n" + data.get('diagnosis', 'No diagnosis available')
        
        # Create a text file
        export_filename = f"export_{log_id.replace('.json', '.txt')}"
        export_filepath = os.path.join(app.static_folder, 'exports', export_filename)
        
        # Ensure exports directory exists
        os.makedirs(os.path.dirname(export_filepath), exist_ok=True)
        
        with open(export_filepath, 'w') as f:
            f.write(text_content)
        
        return send_from_directory(os.path.join(app.static_folder, 'exports'), export_filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': f'Error exporting log file: {str(e)}'}), 500

@app.route('/api/batches', methods=['GET'])
def get_batches():
    batches = []
    
    # Look for directories in the logs directory (potential batches)
    for dirname in os.listdir(chat_log_manager.get_logs_directory()):
        dirpath = os.path.join(chat_log_manager.get_logs_directory(), dirname)
        if os.path.isdir(dirpath) and not dirname.startswith('.') and 'batch_' in dirname:
            # Check if this directory contains batch files
            batch_summary_path = os.path.join(dirpath, 'batch_summary.csv')
            batch_results_path = os.path.join(dirpath, 'batch_results.json')
            
            # Only consider it a batch if it has the necessary files
            if os.path.isfile(batch_summary_path) or os.path.isfile(batch_results_path):
                # Parse batch timestamp from directory name
                timestamp_match = re.search(r'batch_(\d{8}_\d{6})', dirname)
                timestamp = None
                
                if timestamp_match:
                    # Convert directory name timestamp format to ISO format
                    try:
                        timestamp = datetime.datetime.strptime(timestamp_match.group(1), '%Y%m%d_%H%M%S').isoformat()
                    except Exception as e:
                        print(f"Error parsing timestamp for {dirname}: {str(e)}")
                        timestamp = None
                
                # Count conversation files in the batch
                conversation_count = 0
                profile = None
                
                try:
                    # Try to get conversation count from batch_results.json
                    if os.path.isfile(batch_results_path):
                        with open(batch_results_path, 'r') as f:
                            results_data = json.load(f)
                            if 'results' in results_data and isinstance(results_data['results'], list):
                                conversation_count = len(results_data['results'])
                                # Get profile if all conversations have same profile
                                profiles = {result.get('profile') for result in results_data['results'] if 'profile' in result}
                                if len(profiles) == 1:
                                    profile = next(iter(profiles))
                except Exception as e:
                    print(f"Error reading batch results for {dirname}: {str(e)}")
                
                # Fall back to counting JSON files if needed
                if conversation_count == 0:
                    conversation_count = sum(1 for f in os.listdir(dirpath) if f.endswith('.json') and f != 'batch_results.json')
                
                batches.append({
                    'id': dirname,
                    'timestamp': timestamp,
                    'conversation_count': conversation_count,
                    'profile': profile
                })
    
    # Sort batches by timestamp (newest first)
    batches.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify({'batches': batches})

@app.route('/api/batches/<batch_id>', methods=['GET'])
def get_batch_details(batch_id):
    batch_dir = os.path.join(chat_log_manager.get_logs_directory(), batch_id)
    
    if not os.path.isdir(batch_dir):
        return jsonify({'error': 'Batch not found'}), 404
    
    batch_data = {
        'id': batch_id,
        'conversation_count': 0,
        'timestamp': None,
        'avg_duration': None,
        'profile': None,
        'summary': {},
        'results': []
    }
    
    # Parse batch timestamp from directory name
    timestamp_match = re.search(r'batch_(\d{8}_\d{6})', batch_id)
    if timestamp_match:
        try:
            batch_data['timestamp'] = datetime.datetime.strptime(timestamp_match.group(1), '%Y%m%d_%H%M%S').isoformat()
        except:
            pass
    
    # Load batch results
    batch_results_path = os.path.join(batch_dir, 'batch_results.json')
    if os.path.isfile(batch_results_path):
        try:
            with open(batch_results_path, 'r') as f:
                results_data = json.load(f)
                if 'results' in results_data and isinstance(results_data['results'], list):
                    batch_data['results'] = results_data['results']
                    batch_data['conversation_count'] = len(results_data['results'])
                    
                    # Calculate average duration
                    durations = [result.get('duration', 0) for result in results_data['results'] if 'duration' in result]
                    if durations:
                        batch_data['avg_duration'] = sum(durations) / len(durations)
                    
                    # Get profile if all conversations have same profile
                    profiles = {result.get('profile') for result in results_data['results'] if 'profile' in result}
                    if len(profiles) == 1:
                        batch_data['profile'] = next(iter(profiles))
        except Exception as e:
            print(f"Error reading batch results for {batch_id}: {str(e)}")
    
    # Load batch summary
    batch_summary_path = os.path.join(batch_dir, 'batch_summary.csv')
    if os.path.isfile(batch_summary_path):
        try:
            with open(batch_summary_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    batch_data['summary'] = {
                        'headers': rows[0],
                        'rows': rows[1:]
                    }
        except Exception as e:
            print(f"Error reading batch summary for {batch_id}: {str(e)}")
    
    return jsonify(batch_data)

@app.route('/api/batches/<batch_id>/export', methods=['GET'])
def export_batch_summary(batch_id):
    batch_dir = os.path.join(chat_log_manager.get_logs_directory(), batch_id)
    
    if not os.path.isdir(batch_dir):
        return jsonify({'error': 'Batch not found'}), 404
    
    # Look for batch summary file
    batch_summary_path = os.path.join(batch_dir, 'batch_summary.csv')
    if os.path.isfile(batch_summary_path):
        return send_from_directory(os.path.dirname(batch_summary_path), 
                                 os.path.basename(batch_summary_path), 
                                 as_attachment=True)
    
    # If no summary file, create one from batch results
    batch_results_path = os.path.join(batch_dir, 'batch_results.json')
    if os.path.isfile(batch_results_path):
        try:
            with open(batch_results_path, 'r') as f:
                results_data = json.load(f)
                
            # Create a CSV from the results
            if 'results' in results_data and isinstance(results_data['results'], list):
                # Create temporary CSV file
                export_filename = f"batch_{batch_id}_summary.csv"
                export_filepath = os.path.join(app.static_folder, 'exports', export_filename)
                
                # Ensure exports directory exists
                os.makedirs(os.path.dirname(export_filepath), exist_ok=True)
                
                with open(export_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(['Conversation ID', 'Profile', 'Questions', 'Duration (s)'])
                    
                    # Write rows
                    for result in results_data['results']:
                        writer.writerow([
                            result.get('conversation_id', ''),
                            result.get('profile', ''),
                            result.get('question_count', ''),
                            f"{result.get('duration', 0):.2f}"
                        ])
                
                return send_from_directory(os.path.join(app.static_folder, 'exports'), 
                                        export_filename, as_attachment=True)
        except Exception as e:
            return jsonify({'error': f'Error exporting batch results: {str(e)}'}), 500
    
    return jsonify({'error': 'No batch summary or results found'}), 404

# Improved function to run evaluation
@app.route('/api/logs/<log_id>/evaluate', methods=['POST'])
def evaluate_log(log_id):
    """Evaluate a chat log using Ollama."""
    global ongoing_evaluations
    
    # Clean up stale evaluations
    cleanup_stale_evaluations()
    
    # Check if log_id contains .json extension
    if not log_id.endswith('.json'):
        log_id += '.json'
    
    # Check if evaluation is already in progress
    if log_id in ongoing_evaluations and ongoing_evaluations[log_id].get('status') == 'in_progress':
        return jsonify({
            'status': 'in_progress',
            'message': 'Evaluation already in progress'
        })
    
    # Get model from request data, default to the evaluator's default model if not provided
    try:
        request_data = request.get_json() or {}
    except:
        request_data = {}
    
    model = request_data.get('model')
    
    # Start evaluation in a background thread
    def run_evaluation():
        try:
            # Run evaluation with specified model if provided
            if model:
                results = chat_evaluator.evaluate_log(log_id.replace('.json', ''), model=model)
            else:
                results = chat_evaluator.evaluate_log(log_id.replace('.json', ''))
            
            # Store results
            if 'error' in results:
                ongoing_evaluations[log_id] = {
                    'status': 'error',
                    'message': results['error'],
                    'details': results.get('details', ''),
                    'finish_time': time.time()
                }
            else:
                ongoing_evaluations[log_id] = {
                    'status': 'completed',
                    'results': results,
                    'finish_time': time.time()
                }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            ongoing_evaluations[log_id] = {
                'status': 'error',
                'message': str(e),
                'details': error_details,
                'finish_time': time.time()
            }
    
    # Mark evaluation as in progress with start time and model
    ongoing_evaluations[log_id] = {
        'status': 'in_progress', 
        'start_time': time.time(),
        'progress': 0,
        'model': model
    }
    
    thread = threading.Thread(target=run_evaluation)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Evaluation started' + (f' with model {model}' if model else '')
    })

# Improve the status endpoint to handle caching
@app.route('/api/logs/<log_id>/evaluation', methods=['GET'])
def get_evaluation(log_id):
    """Get evaluation status and results for a chat log."""
    # Clean up stale evaluations
    cleanup_stale_evaluations()
    
    # Set cache control headers to prevent caching
    response_headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    
    # Check if log_id contains .json extension
    if not log_id.endswith('.json'):
        log_id += '.json'
    
    # Add debug logging
    print(f"Getting evaluation status for {log_id}")
    
    # Check if evaluation is in progress
    if log_id in ongoing_evaluations:
        evaluation_data = ongoing_evaluations[log_id]
        # print(f"Found ongoing evaluation: {evaluation_data}")
        return jsonify(evaluation_data), 200, response_headers
    
    # Check for existing evaluation in the log file
    status = chat_evaluator.get_evaluation_status(log_id.replace('.json', ''))
    print(f"Evaluation status from file: {status}")
    
    # If the evaluation is completed, include the results directly
    if status.get('status') == 'completed' and 'results' in status:
        # Add detailed logging of the results structure
        print(f"Returning completed evaluation")
        print(f"Results structure: {type(status['results'])}")
        
        # Ensure results are properly returned
        if isinstance(status['results'], dict):
            for key in status['results']:
                print(f"Top-level key in results: {key}")
            
            # Fix: If the result has a nested evaluation field, flatten it
            if 'evaluation' in status['results'] and isinstance(status['results']['evaluation'], dict):
                print("Found nested evaluation structure")
                evaluation_data = status['results']['evaluation']
                
                # Add evaluation keys to the top level
                for key, value in evaluation_data.items():
                    if key not in status['results']:
                        status['results'][key] = value
    
    return jsonify(status), 200, response_headers

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model['name'] for model in response.json().get('models', [])]
            return jsonify({'models': models})
        else:
            return jsonify({'error': f'Failed to fetch models from Ollama API: {response.status_code}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error fetching models: {str(e)}'}), 500

@app.route('/api/questionnaires', methods=['GET'])
def get_questionnaires():
    """Get available questionnaires."""
    try:
        # Initialize the RAG engine to access questionnaires
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docs_dir = os.path.join(project_root, "documents")
        questionnaires_dir = os.path.join(docs_dir, "questionnaires")
        
        # Print debugging information
        print(f"Looking for questionnaires in: {questionnaires_dir}")
        
        # Check if the directory exists and list files
        if os.path.exists(questionnaires_dir):
            files = [f for f in os.listdir(questionnaires_dir) 
                    if os.path.isfile(os.path.join(questionnaires_dir, f)) 
                    and not f.startswith('.')]
            print(f"Files found in questionnaires directory: {files}")
        else:
            print(f"Questionnaires directory does not exist: {questionnaires_dir}")
            # Create the directory
            os.makedirs(questionnaires_dir, exist_ok=True)
            files = []
        
        # If no files are found through normal means, try direct file loading as fallback
        if not files:
            # As a fallback, directly check for any PDF files and use them
            questionnaires = []
            for file in os.listdir(questionnaires_dir):
                if file.endswith('.pdf') and not file.startswith('.'):
                    filepath = os.path.join(questionnaires_dir, file)
                    try:
                        # Try to manually load and extract questions
                        from utils.document_processor import DocumentProcessor, extract_questions_from_text
                        document = DocumentProcessor.load_document(filepath)
                        if document:
                            questions = extract_questions_from_text(document.content)
                            if questions:
                                print(f"Manually extracted {len(questions)} questions from {file}")
                                questionnaires.append({
                                    'id': filepath,  # Use full path for manual loading
                                    'name': os.path.splitext(file)[0],
                                    'question_count': len(questions)
                                })
                            else:
                                print(f"No questions found in {file}")
                    except Exception as e:
                        print(f"Error manually processing {file}: {str(e)}")
            
            if questionnaires:
                print(f"Fallback method found {len(questionnaires)} questionnaires")
                return jsonify({"questionnaires": questionnaires})
        
        # If files exist, proceed with RAG engine
        if files:
            # Initialize RAG engine
            rag_engine = RAGEngine(docs_dir, questionnaire_dir=questionnaires_dir)
            
            # Get questionnaires with debug
            print("Calling RAG engine get_questionnaires()")
            questionnaires_dict = rag_engine.get_questionnaires()
            print(f"Questionnaires found: {len(questionnaires_dict)}")
            
            # If RAG engine returned no questionnaires, try manual loading
            if not questionnaires_dict:
                # Try direct file loading as a fallback (same as above)
                print("RAG engine found no questionnaires, trying direct loading...")
                questionnaires = []
                for file in files:
                    if file.endswith('.pdf'):
                        filepath = os.path.join(questionnaires_dir, file)
                        try:
                            # Try to manually load and extract questions
                            from utils.document_processor import DocumentProcessor, extract_questions_from_text
                            document = DocumentProcessor.load_document(filepath)
                            if document:
                                questions = extract_questions_from_text(document.content)
                                if questions:
                                    print(f"Manually extracted {len(questions)} questions from {file}")
                                    questionnaires.append({
                                        'id': filepath,  # Use full path for manual loading
                                        'name': os.path.splitext(file)[0],
                                        'question_count': len(questions)
                                    })
                        except Exception as e:
                            print(f"Error manually processing {file}: {str(e)}")
                
                if questionnaires:
                    print(f"Manual extraction found {len(questionnaires)} questionnaires")
                    return jsonify({"questionnaires": questionnaires})
                else:
                    print("No questionnaires could be extracted")
                    return jsonify({"error": "No questions could be extracted from the PDF files"})
            
            # Convert to a list format suitable for the frontend
            questionnaires = []
            for name, questions in questionnaires_dict.items():
                # Handle relative vs absolute paths
                if not os.path.isabs(name):
                    file_path = os.path.join(questionnaires_dir, name)
                else:
                    file_path = name
                    
                questionnaires.append({
                    'id': file_path,
                    'name': os.path.splitext(os.path.basename(name))[0],
                    'question_count': len(questions)
                })
            
            print(f"Final questionnaires list: {[q['name'] for q in questionnaires]}")
            return jsonify({"questionnaires": questionnaires})
        else:
            print("No questionnaire files found")
            return jsonify({"questionnaires": [], "warning": "No questionnaire files found"})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in get_questionnaires(): {str(e)}")
        print(error_trace)
        return jsonify({"error": str(e), "traceback": error_trace}), 500

@app.route('/api/profiles', methods=['GET'])
def get_profiles():
    """Get available patient profiles."""
    try:
        # Use the Patient class to get available profiles
        available_profiles = Patient.list_available_profiles()
        return jsonify({"profiles": sorted(available_profiles)})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/conversations/start', methods=['POST'])
def start_conversation():
    """Start a new conversation."""
    global active_conversations
    
    # Clean up any stale conversations
    cleanup_stale_conversations()
    
    try:
        # Get request data with better error handling
        content_type = request.headers.get('Content-Type', '')
        print(f"Request Content-Type: {content_type}")
        
        if 'application/json' not in content_type:
            print("Warning: Content-Type is not application/json")
        
        # Try to get the raw data
        raw_data = request.get_data()
        print(f"Raw request data ({len(raw_data)} bytes): {raw_data[:200]}...")
        
        # Now try to parse as JSON
        try:
            data = request.get_json(force=True)
            print(f"Parsed JSON data: {data}")
        except Exception as e:
            return jsonify({
                "error": f"Failed to parse JSON data: {str(e)}. Check your Content-Type header and request body."
            }), 400
        
        if not data:
            print("Empty data received in request")
            return jsonify({"error": "No data provided. Request body is empty or not valid JSON."}), 400
        
        # Extract parameters
        questionnaire = data.get('questionnaire')
        profile = data.get('profile')
        assistant_model = data.get('assistant_model', 'qwen2.5:3b')
        patient_model = data.get('patient_model', 'qwen2.5:3b')
        save_logs = data.get('save_logs', True)
        refresh_cache = data.get('refresh_cache', False)
        
        print(f"Extracted parameters: questionnaire={questionnaire}, profile={profile}, "
              f"assistant_model={assistant_model}, patient_model={patient_model}, "
              f"save_logs={save_logs}, refresh_cache={refresh_cache}")
        
        if not questionnaire:
            return jsonify({"error": "No questionnaire selected"}), 400
        
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Create a temporary directory for logs if not saving
        logs_dir = chat_log_manager.get_logs_directory()
        if not save_logs:
            logs_dir = tempfile.mkdtemp()
        
        # Prepare command to run main.py in a subprocess
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_script = os.path.join(project_root, "main.py")
        
        cmd = [
            sys.executable,
            main_script,
            '--pdf_path', questionnaire,
            '--assistant_model', assistant_model,
            '--patient_model', patient_model,
        ]
        
        if profile:
            cmd.extend(['--patient_profile', profile])
        
        if refresh_cache:
            cmd.append('--refresh_cache')
        
        if not save_logs:
            cmd.append('--no-save')
        else:
            cmd.extend(['--logs-dir', logs_dir])
        
        # Create a file to capture output
        output_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        
        # Create a file to store conversation state
        state_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json')
        state_path = state_file.name
        state_file.close()
        
        # Initialize conversation state
        conversation_state = {
            "conversation": [],
            "status": "starting",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(state_path, 'w') as f:
            json.dump(conversation_state, f)
            
        # Add special arguments to tell main.py to update the state file
        cmd.extend(['--state-file', state_path])
        
        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Store process information
        active_conversations[conversation_id] = {
            'process': process,
            'output_file': output_file,
            'state_file': state_path,
            'questionnaire': questionnaire,
            'profile': profile,
            'assistant_model': assistant_model,
            'patient_model': patient_model,
            'save_logs': save_logs,
            'logs_dir': logs_dir,
            'status': 'in_progress',
            'start_time': time.time()
        }
        
        return jsonify({
            "conversation_id": conversation_id,
            "message": "Conversation started"
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/conversations/<conversation_id>/status', methods=['GET'])
def get_conversation_status(conversation_id):
    """Get the status of a conversation."""
    global active_conversations
    
    if conversation_id not in active_conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conv_data = active_conversations[conversation_id]
    
    try:
        # Check if the process is still running
        process = conv_data['process']
        if process.poll() is not None:
            # Process has ended
            return_code = process.returncode
            
            if return_code != 0:
                # Process ended with error
                with open(conv_data['output_file'].name, 'r') as f:
                    error_output = f.read()
                
                # Update status
                conv_data['status'] = 'error'
                
                return jsonify({
                    "status": "error",
                    "error": f"Conversation process ended with return code {return_code}",
                    "output": error_output
                })
            
            # Process completed successfully
            conv_data['status'] = 'completed'
            
            # Find the log file if saved
            log_id = None
            log_saved = conv_data['save_logs']
            
            if log_saved:
                # Try to find the most recently created log file
                logs_dir = conv_data['logs_dir']
                log_files = [f for f in os.listdir(logs_dir) 
                           if os.path.isfile(os.path.join(logs_dir, f)) and f.endswith('.json')]
                
                if log_files:
                    # Sort by creation time (newest first)
                    log_files.sort(key=lambda f: os.path.getctime(os.path.join(logs_dir, f)), reverse=True)
                    log_id = log_files[0].replace('.json', '')
            
            # Read final state from state file
            with open(conv_data['state_file'], 'r') as f:
                state = json.load(f)
            
            conversation = state.get('conversation', [])
            diagnosis = state.get('diagnosis', 'No diagnosis generated')
            
            return jsonify({
                "status": "completed",
                "conversation": conversation,
                "diagnosis": diagnosis,
                "log_saved": log_saved,
                "log_id": log_id
            })
        
        # Process is still running - get current state
        try:
            with open(conv_data['state_file'], 'r') as f:
                state = json.load(f)
            
            return jsonify({
                "status": "in_progress",
                "conversation": state.get('conversation', [])
            })
        except (json.JSONDecodeError, FileNotFoundError):
            # State file might be empty or not yet created
            return jsonify({
                "status": "in_progress",
                "conversation": []
            })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/api/conversations/<conversation_id>/stop', methods=['POST'])
def stop_conversation(conversation_id):
    """Stop a conversation."""
    global active_conversations
    
    if conversation_id not in active_conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conv_data = active_conversations[conversation_id]
    
    try:
        # Check if process is still running
        process = conv_data['process']
        if process.poll() is None:
            # Process is still running - terminate it
            process.terminate()
            
            # Wait a short time for it to terminate
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it didn't terminate
                process.kill()
        
        # Update status
        conv_data['status'] = 'stopped'
        
        # Close output file
        if 'output_file' in conv_data and not conv_data['output_file'].closed:
            conv_data['output_file'].close()
        
        # If using temporary logs directory and no longer needed, clean it up
        if not conv_data['save_logs'] and 'logs_dir' in conv_data:
            try:
                shutil.rmtree(conv_data['logs_dir'])
            except:
                pass
        
        return jsonify({"success": True, "message": "Conversation stopped"})
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/api/batches/start', methods=['POST'])
def start_batch():
    """Start a new batch generation."""
    global active_batches
    
    # Clean up any stale batches
    cleanup_stale_batches()
    
    try:
        # Get request data with better error handling
        content_type = request.headers.get('Content-Type', '')
        print(f"Batch request - Content-Type: {content_type}")
        
        # Try to get the raw data
        raw_data = request.get_data()
        print(f"Batch raw request data ({len(raw_data)} bytes): {raw_data.decode('utf-8', 'ignore')[:200]}...")
        
        # Now try to parse as JSON
        try:
            data = request.get_json(force=True)
            print(f"Parsed JSON batch data: {data}")
        except Exception as e:
            return jsonify({
                "error": f"Failed to parse JSON data: {str(e)}. Check your Content-Type header and request body."
            }), 400
        
        if not data:
            print("Empty data received in request")
            return jsonify({"error": "No data provided. Request body is empty or not valid JSON."}), 400
        
        # Extract parameters with explicit type conversion and validation
        questionnaire = data.get('questionnaire')
        if not questionnaire:
            return jsonify({"error": "No questionnaire selected"}), 400
        
        profile = data.get('profile')
        assistant_model = data.get('assistant_model', 'qwen2.5:3b')
        patient_model = data.get('patient_model', 'qwen2.5:3b')
        
        # Convert boolean values with explicit checks
        save_logs = bool(data.get('save_logs', True))
        refresh_cache = bool(data.get('refresh_cache', False))
        randomize_profiles = bool(data.get('randomize_profiles', False))
        
        # Handle batch count with validation
        try:
            batch_count = int(data.get('batch_count', 5))
            if batch_count < 1:
                batch_count = 1
            print(f"Using batch count: {batch_count}")
        except (ValueError, TypeError):
            print("Invalid batch count, defaulting to 5")
            batch_count = 5
        
        print(f"Extracted batch parameters: questionnaire={questionnaire}, profile={profile}, "
              f"batch_count={batch_count}, randomize_profiles={randomize_profiles}, "
              f"assistant_model={assistant_model}, patient_model={patient_model}, "
              f"save_logs={save_logs}, refresh_cache={refresh_cache}")
        
        # Generate a unique batch ID
        batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a directory for batch logs
        logs_dir = os.path.join(chat_log_manager.get_logs_directory(), batch_id)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare command to run main.py in a subprocess with batch mode
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_script = os.path.join(project_root, "main.py")
        
        # Ensure main script exists
        if not os.path.exists(main_script):
            return jsonify({"error": f"Main script not found at {main_script}"}), 500
        
        # Construct the command with careful argument formatting
        cmd = [
            sys.executable,
            main_script,
            '--pdf_path', str(questionnaire),
            '--assistant_model', str(assistant_model),
            '--patient_model', str(patient_model),
            '--batch', str(batch_count),
            '--logs-dir', str(logs_dir)
        ]
        
        # Add profile only if specified and not using randomize profiles
        if profile and not randomize_profiles:
            cmd.extend(['--patient_profile', str(profile)])
        
        # Add flags as needed
        if randomize_profiles:
            cmd.append('--randomize-profiles')
        
        if refresh_cache:
            cmd.append('--refresh_cache')
            
        # Log the full command for debugging
        print(f"Executing batch command: {' '.join(cmd)}")
        
        # Create a file to capture output
        output_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        print(f"Output will be written to: {output_file.name}")
        
        # Create a status file to track progress
        status_file = os.path.join(logs_dir, 'batch_status.json')
        batch_status = {
            "status": "starting",
            "total_conversations": batch_count,
            "completed_conversations": 0,
            "start_time": datetime.datetime.now().isoformat(),
            "results": []
        }
        
        with open(status_file, 'w') as f:
            json.dump(batch_status, f)
        
        # Execute the command in a separate process
        try:
            process = subprocess.Popen(
                cmd,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=project_root  # Set working directory to project root
            )
            print(f"Batch process started with PID: {process.pid}")
        except Exception as e:
            error_msg = f"Failed to start batch process: {str(e)}"
            print(error_msg)
            return jsonify({"error": error_msg}), 500
        
        # Store process information
        active_batches[batch_id] = {
            'process': process,
            'output_file': output_file,
            'status_file': status_file,
            'questionnaire': questionnaire,
            'profile': profile,
            'assistant_model': assistant_model,
            'patient_model': patient_model,
            'randomize_profiles': randomize_profiles,
            'batch_count': batch_count,
            'logs_dir': logs_dir,
            'status': 'in_progress',
            'start_time': time.time()
        }
        
        return jsonify({
            "batch_id": batch_id,
            "total_conversations": batch_count,
            "message": "Batch generation started"
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in start_batch(): {str(e)}")
        print(error_trace)
        return jsonify({
            "error": str(e),
            "traceback": error_trace
        }), 500

@app.route('/api/batches/<batch_id>/status', methods=['GET'])
def get_batch_status(batch_id):
    """Get the status of a batch generation."""
    global active_batches
    
    # Add detailed debugging for this request
    print(f"Batch status request received for batch_id: {batch_id}")
    batch_dir = os.path.join(chat_log_manager.get_logs_directory(), batch_id)
    print(f"Looking for batch directory: {batch_dir}")
    
    if batch_id not in active_batches:
        # Check if directory exists even if not in active_batches
        if os.path.isdir(batch_dir):
            print(f"Batch directory found but not in active_batches, attempting to recover status")
            status_file = os.path.join(batch_dir, 'batch_status.json')
            
            if os.path.isfile(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                    
                    # Batch exists but not in memory, reconstruct as completed
                    return jsonify({
                        "status": "completed",
                        "total_conversations": status_data.get('total_conversations', 0),
                        "completed_conversations": status_data.get('completed_conversations', 0),
                        "results": status_data.get('results', []),
                        "in_progress_conversation": None,
                        "recovered": True
                    })
                except Exception as e:
                    print(f"Error recovering batch status: {e}")
        
        return jsonify({"error": "Batch not found"}), 404
    
    batch_data = active_batches[batch_id]
    print(f"Batch data found: {batch_id}")
    
    try:
        # Check if the process is still running
        process = batch_data['process']
        if process.poll() is not None:
            # Process has ended
            return_code = process.returncode
            print(f"Batch process has ended with return code: {return_code}")
            
            if return_code != 0:
                # Process ended with error
                with open(batch_data['output_file'].name, 'r') as f:
                    error_output = f.read()
                
                # Update status
                batch_data['status'] = 'error'
                
                return jsonify({
                    "status": "error",
                    "error": f"Batch process ended with return code {return_code}",
                    "output": error_output
                })
            
            # Process completed successfully
            batch_data['status'] = 'completed'
            print(f"Batch process completed successfully")
            
            # Read results from logs directory
            results = []
            logs_dir = batch_data['logs_dir']
            
            # Check for batch_results.json file first (preferred format)
            batch_results_path = os.path.join(logs_dir, 'batch_results.json')
            print(f"Looking for batch results at: {batch_results_path}")
            
            if os.path.exists(batch_results_path):
                try:
                    with open(batch_results_path, 'r') as f:
                        results_data = json.load(f)
                        if 'results' in results_data:
                            results = results_data['results']
                            print(f"Found {len(results)} results in batch_results.json")
                except Exception as e:
                    print(f"Error reading batch results: {str(e)}")
            
            # If no results found, scan for individual log files
            if not results:
                print(f"No results found in batch_results.json, scanning for individual log files")
                log_count = 0
                for filename in os.listdir(logs_dir):
                    if filename.endswith('.json') and filename != 'batch_status.json' and filename != 'batch_results.json':
                        log_count += 1
                        try:
                            filepath = os.path.join(logs_dir, filename)
                            with open(filepath, 'r') as f:
                                log_data = json.load(f)
                                
                                # Extract basic info from log
                                result = {
                                    'conversation_id': filename.replace('.json', ''),
                                    'profile': log_data.get('metadata', {}).get('patient_profile', 'Unknown'),
                                    'status': 'completed',
                                }
                                
                                # Add metadata if available
                                if 'metadata' in log_data and 'duration' in log_data['metadata']:
                                    result['duration'] = log_data['metadata']['duration']
                                
                                results.append(result)
                        except Exception as e:
                            print(f"Error processing log file {filename}: {str(e)}")
                
                print(f"Found {log_count} log files, processed {len(results)}")
            
            return jsonify({
                "status": "completed",
                "total_conversations": batch_data['batch_count'],
                "completed_conversations": len(results),
                "results": results,
                "in_progress_conversation": None
            })
        
        # Process is still running - check status file
        try:
            status_file = batch_data['status_file']
            print(f"Process still running, checking status file: {status_file}")
            
            if not os.path.exists(status_file):
                print(f"Status file does not exist: {status_file}")
                return jsonify({
                    "status": "in_progress",
                    "total_conversations": batch_data['batch_count'],
                    "completed_conversations": 0,
                    "results": [],
                    "in_progress_conversation": 0,
                    "status_file_missing": True
                })
            
            # Check status file age
            file_mod_time = os.path.getmtime(status_file)
            current_time = time.time()
            file_age = current_time - file_mod_time
            print(f"Status file age: {file_age:.2f} seconds")
            
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            
            # Extract the in-progress conversation index from the status data
            in_progress_conversation = status_data.get('in_progress_conversation')
            completed = status_data.get('completed_conversations', 0)
            total = status_data.get('total_conversations', batch_data['batch_count'])
            
            print(f"Status data: {completed}/{total} completed, in_progress={in_progress_conversation}")
            
            return jsonify({
                "status": "in_progress",
                "total_conversations": total,
                "completed_conversations": completed,
                "results": status_data.get('results', []),
                "in_progress_conversation": in_progress_conversation,
                "file_age": file_age
            })
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading status file: {e}")
            # Status file might be empty or not yet created
            return jsonify({
                "status": "in_progress",
                "total_conversations": batch_data['batch_count'],
                "completed_conversations": 0,
                "results": [],
                "in_progress_conversation": 0,  # Assume first conversation is in progress if unknown
                "error_reading_status": str(e)
            })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in get_batch_status: {e}")
        print(error_trace)
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": error_trace
        })

def create_app(logs_dir=None):
    """Create and configure the Flask application."""
    global chat_log_manager, chat_evaluator
    
    # Make sure time is imported
    import time
    
    chat_log_manager = ChatLogManager(logs_dir)
    chat_evaluator = ChatLogEvaluator(
        logs_dir=chat_log_manager.get_logs_directory(),
        ollama_url="http://localhost:11434",
        model="qwen2.5:3b"
    )
    
    print(f"Using logs directory: {chat_log_manager.get_logs_directory()}")
    
    return app

def main():
    """Run the application."""
    parser = argparse.ArgumentParser(description="Chat Log Viewer")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the web server on")
    parser.add_argument('--logs-dir', type=str, help="Directory containing chat logs")
    args = parser.parse_args()
    
    app = create_app(args.logs_dir)
    
    print(f"Starting chat viewer on http://127.0.0.1:{args.port}")
    app.run(debug=True, port=args.port)

if __name__ == '__main__':
    main()
