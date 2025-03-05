import os
import json
import datetime
from typing import List, Dict, Any, Optional

class ChatLogger:
    """Utility for logging chat conversations and diagnoses."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the chat logger.
        
        Args:
            log_dir (str, optional): Directory to save chat logs
        """
        if log_dir:
            self.log_dir = log_dir
        else:
            # Default to 'chat_logs' directory in project root
            self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_logs")
        
        # Create the log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def save_chat(self, 
                 conversation: List[Dict[str, str]], 
                 diagnosis: str, 
                 questionnaire_name: str = "unknown",
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a chat conversation to file.
        
        Args:
            conversation: List of conversation messages (dicts with 'role' and 'content')
            diagnosis: Final diagnosis text
            questionnaire_name: Name of the questionnaire used
            metadata: Additional metadata to save
            
        Returns:
            Path to the saved chat log file
        """
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}_{questionnaire_name.replace('.pdf', '')}.json"
        file_path = os.path.join(self.log_dir, filename)
        
        # Prepare data to save
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "questionnaire": questionnaire_name,
            "conversation": conversation,
            "diagnosis": diagnosis,
            "metadata": metadata or {}
        }
        
        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save a plain text version for easier reading
        txt_file_path = os.path.join(self.log_dir, filename.replace('.json', '.txt'))
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Conversation with {questionnaire_name}\n")
            f.write(f"Time: {datetime.datetime.now().isoformat()}\n\n")
            
            for msg in conversation:
                role = msg['role'].upper()
                if role == "SYSTEM":
                    continue  # Skip system messages in the readable version
                f.write(f"{role}: {msg['content']}\n\n")
                
            f.write("\n==== DIAGNOSIS ====\n\n")
            f.write(diagnosis)
            f.write("\n")
        
        return file_path
    
    def list_chat_logs(self) -> List[str]:
        """List all available chat logs."""
        return [f for f in os.listdir(self.log_dir) 
                if f.endswith('.json') or f.endswith('.txt')]
    
    def get_log_path(self, filename: str) -> str:
        """Get full path to a log file."""
        return os.path.join(self.log_dir, filename)
    
    def load_chat(self, filename: str) -> Dict[str, Any]:
        """Load a chat log file."""
        if not filename.endswith('.json'):
            filename += '.json'
        
        file_path = os.path.join(self.log_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
