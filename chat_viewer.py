#!/usr/bin/env python3
"""
Chat Log Viewer Launcher

Simple script to launch the chat log viewer.
"""
import os
import sys
import argparse
from interface.app import create_app

def main():
    parser = argparse.ArgumentParser(description="Chat Log Viewer")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the web server on")
    parser.add_argument('--logs-dir', type=str, help="Directory containing chat logs")
    args = parser.parse_args()
    
    # Get logs directory - default to chat_logs in the current directory if not specified
    logs_dir = args.logs_dir
    if not logs_dir:
        logs_dir = os.path.join(os.getcwd(), "chat_logs")
    
    # Create the directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    # Check if the directory exists and is accessible
    if not os.path.isdir(logs_dir):
        print(f"Error: {logs_dir} is not a directory")
        sys.exit(1)

    # Create Flask app
    app = create_app(logs_dir)
    
    # Run the app
    print(f"Starting chat viewer at http://127.0.0.1:{args.port}")
    print(f"Using logs directory: {logs_dir}")
    print("Press Ctrl+C to quit")
    
    app.run(debug=True, port=args.port)

if __name__ == "__main__":
    main()
