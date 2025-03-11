"""
Chat Log Viewer Flask Application

A simple web interface for viewing mental health assessment chat logs.
"""
import os
import argparse
from flask import Flask, render_template, jsonify, request, send_file, Response
from .chat_log_manager import ChatLogManager
import io

app = Flask(__name__)

# Create a singleton chat log manager
chat_log_manager = None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/logs')
def api_logs():
    """API endpoint to get all logs."""
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    logs = chat_log_manager.list_logs(refresh=refresh)
    profiles = chat_log_manager.get_unique_profiles()
    return jsonify({
        'logs': logs,
        'profiles': profiles
    })

@app.route('/api/logs/<log_id>')
def api_log(log_id):
    """API endpoint to get a specific log."""
    log_data = chat_log_manager.get_log(log_id)
    if log_data:
        return jsonify(log_data)
    else:
        return jsonify({'error': 'Log not found'}), 404

@app.route('/api/logs/<log_id>/export')
def api_export_log(log_id):
    """API endpoint to export a log as a text file."""
    text_content = chat_log_manager.export_log_as_text(log_id)
    
    if text_content:
        log_data = chat_log_manager.get_log(log_id)
        filename = f"export_{log_data.get('questionnaire', 'chat')}.txt"
        
        # Return as downloadable text file
        return Response(
            text_content,
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )
    else:
        return jsonify({'error': 'Log not found'}), 404

def create_app(logs_dir=None):
    """Create and configure the Flask application."""
    global chat_log_manager
    chat_log_manager = ChatLogManager(logs_dir)
    
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
