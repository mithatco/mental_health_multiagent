#!/usr/bin/env python3
"""
Chat Log Viewer Launcher

Simple script to launch the chat log viewer.
"""
import os
import sys
import argparse
import traceback

def main():
    # Check if Flask is installed
    try:
        import flask
        print("Flask is installed.")
    except ImportError:
        print("ERROR: Flask is not installed. Please install it with: pip install flask")
        print("You may also need to install other dependencies: pip install requests numpy")
        sys.exit(1)
    
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
    
    # Try to import the app module and create the app
    try:
        # Verify directory structure
        interface_dir = os.path.join(os.path.dirname(__file__), "interface")
            
        # Check for app.py in interface directory
        app_file = os.path.join(interface_dir, "app.py")
        
        # Ensure interface directory is in the Python path
        if interface_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(__file__))
        
        # Try importing the module
        try:
            from interface.app import create_app
        except ImportError as e:
            print(f"ERROR importing interface.app: {e}")
            print("Traceback:", traceback.format_exc())
            sys.exit(1)
    
        # Create Flask app
        print("Creating Flask app...")
        app = create_app(logs_dir)
        
        # Run the app
        print(f"Starting chat viewer at http://127.0.0.1:{args.port}")
        print(f"Using logs directory: {logs_dir}")
        print("Press Ctrl+C to quit")
        
        app.run(debug=True, port=args.port, use_reloader=True)
    except Exception as e:
        print(f"ERROR: Failed to start the app: {e}")
        print("Traceback:", traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
