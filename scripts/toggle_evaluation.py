#!/usr/bin/env python3
"""
Utility script to enable or disable RAG evaluation.

This script modifies the conversation_handler.py file to toggle whether
RAG evaluation is performed during conversations.
"""

import os
import re
import sys
from pathlib import Path

def main():
    # Find the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Find the conversation_handler.py file
    handler_path = project_root / "utils" / "conversation_handler.py"
    
    if not handler_path.exists():
        print(f"Error: Could not find conversation_handler.py at {handler_path}")
        return 1
    
    # Read the current file content
    with open(handler_path, 'r') as f:
        content = f.read()
    
    # Check if evaluation section exists in the file
    evaluation_pattern = r'# Add RAG evaluation if deepeval is available and we have enough context\s+if self\.has_evaluator'
    
    if not re.search(evaluation_pattern, content):
        print("Could not find the RAG evaluation section in the conversation handler.")
        return 1
    
    # Check current state and get user choice
    if len(sys.argv) > 1 and sys.argv[1] in ['enable', 'disable']:
        choice = sys.argv[1]
    else:
        # Determine if evaluation is currently enabled or disabled
        if 'self.has_evaluator = False  # Evaluation disabled' in content:
            current_state = 'disabled'
        else:
            current_state = 'enabled'
            
        print(f"RAG evaluation is currently {current_state}.")
        choice = input("Would you like to 'enable' or 'disable' RAG evaluation? [enable/disable]: ").lower()
    
    # Implement the chosen option
    if choice == 'disable':
        # Force evaluation to be disabled by setting has_evaluator to False
        updated_content = re.sub(
            r'(self\.has_evaluator = )([^\n]+)',
            r'\1False  # Evaluation disabled',
            content
        )
        
        # Skip the initialization code
        updated_content = re.sub(
            r'(# Initialize RAG evaluator.*?\n)(\s+try:.*?)(\s+# Add rag_evaluation_results)',
            r'\1        self.has_evaluator = False  # Evaluation disabled\3',
            updated_content, 
            flags=re.DOTALL
        )
        
        # Write the updated content
        with open(handler_path, 'w') as f:
            f.write(updated_content)
            
        print("RAG evaluation has been DISABLED.")
        
    elif choice == 'enable':
        # Enable evaluation by removing any forced disabling
        updated_content = re.sub(
            r'self\.has_evaluator = False  # Evaluation disabled',
            r'self.has_evaluator = self.rag_evaluator.is_available',
            content
        )
        
        # Restore the initialization code if needed
        if 'try:' not in content or 'except ImportError:' not in content:
            # This is a more complex change that would need to restore the full try-except block
            print("Warning: The initialization code appears to be missing. Manual restoration may be needed.")
        
        # Write the updated content
        with open(handler_path, 'w') as f:
            f.write(updated_content)
            
        print("RAG evaluation has been ENABLED.")
        
    else:
        print("Invalid choice. Please specify 'enable' or 'disable'.")
        return 1
    
    print("\nChanges will take effect the next time you run the application.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
