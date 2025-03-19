"""
Debug logging utility for the Mental Health Multi-Agent System.
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class DebugLogger:
    """Debug logger for tracking file operations and other diagnostics."""
    
    def __init__(self, name="DebugLogger", debug_dir=None):
        """
        Initialize the debug logger.
        
        Args:
            name: Logger name
            debug_dir: Directory to save debug logs (None for no file logging)
        """
        self.logger = logging.getLogger(name)
        self.debug_dir = debug_dir
        
        # Create debug directory if specified
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            
            # Add file handler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(debug_dir, f"{name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
    
    def log_file_operation(self, operation, filepath, result=None, error=None):
        """
        Log a file operation with detailed information.
        
        Args:
            operation: Type of operation (e.g., 'read', 'write', 'delete')
            filepath: Path to the file
            result: Result of the operation (if any)
            error: Error message (if any)
        """
        # Get file status
        file_exists = os.path.exists(filepath)
        file_size = os.path.getsize(filepath) if file_exists else 0
        file_info = {
            'exists': file_exists,
            'size': file_size,
            'is_file': os.path.isfile(filepath) if file_exists else False,
            'is_dir': os.path.isdir(filepath) if file_exists else False,
            'abs_path': os.path.abspath(filepath),
            'parent_exists': os.path.exists(os.path.dirname(filepath)),
        }
        
        # Format the log message
        message = f"File {operation}: {filepath}"
        if error:
            message += f" - ERROR: {error}"
            self.logger.error(message)
            self.logger.error(f"File info: {json.dumps(file_info, indent=2)}")
        else:
            message += f" - Result: {result}"
            self.logger.info(message)
            self.logger.debug(f"File info: {json.dumps(file_info, indent=2)}")

# Create global instance
debug_logger = DebugLogger()
