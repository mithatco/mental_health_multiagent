"""
Chat Log Viewer Flask Application

A simple web interface for viewing mental health assessment chat logs.
"""
import os
import argparse
import time
from flask import Flask, render_template, jsonify, request, send_file, Response, send_from_directory
from .chat_log_manager import ChatLogManager
import io
import json
import datetime
import csv
import re
import threading

# Update the import to use our ChatLogEvaluator directly
from utils.chat_evaluator import ChatLogEvaluator

app = Flask(__name__)

# Create singletons
chat_log_manager = None
chat_evaluator = None

# Track ongoing evaluations with timestamps for cleanup
ongoing_evaluations = {}

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

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

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
    
    # Start evaluation in a background thread
    def run_evaluation():
        try:
            # Run evaluation
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
    
    # Mark evaluation as in progress with start time
    ongoing_evaluations[log_id] = {
        'status': 'in_progress', 
        'start_time': time.time(),
        'progress': 0
    }
    
    thread = threading.Thread(target=run_evaluation)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Evaluation started'
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
        print(f"Found ongoing evaluation: {evaluation_data}")
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
