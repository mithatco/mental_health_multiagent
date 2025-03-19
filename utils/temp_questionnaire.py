"""
Utility for creating temporary questionnaires for batch processing.
"""
import os
import json
import tempfile
from typing import List, Dict, Any, Optional

class TempQuestionnaire:
    """Helper class for creating temporary questionnaires for batch processing."""
    
    @staticmethod
    def create_temp_json(questions: List[str]) -> str:
        """
        Create a temporary JSON file containing questions.
        
        Args:
            questions: List of questions to include
            
        Returns:
            Path to the created temporary file
        """
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w+') as temp_file:
            json.dump({"questions": questions}, temp_file)
            temp_file_path = temp_file.name
        
        return temp_file_path
    
    @staticmethod
    def cleanup(file_path: str) -> bool:
        """
        Clean up temporary file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if clean up was successful, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
            return True
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")
            return False
