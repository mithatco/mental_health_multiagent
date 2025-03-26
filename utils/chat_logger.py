import os
import json
import datetime
from typing import List, Dict, Any, Optional

class ChatLogger:
    """Class for handling chat log saving and loading."""
    
    def __init__(self, log_dir: str = "chat_logs"):
        """
        Initialize with a directory to save logs.
        
        Args:
            log_dir: Directory to save logs (default: "chat_logs")
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def save_chat(self, 
                  conversation: List[Dict[str, str]], 
                  diagnosis: str,
                  questionnaire_name: str = "unknown",
                  metadata: Optional[Dict[str, Any]] = None,
                  log_path: Optional[str] = None) -> str:
        """
        Save a chat log to a JSON file.
        
        Args:
            conversation: List of conversation messages
            diagnosis: Final diagnosis text
            questionnaire_name: Name of the questionnaire used
            metadata: Additional metadata to include
            log_path: Optional explicit path for the log file
            
        Returns:
            Path to the saved log file
        """
        # Generate a timestamp for the filename if not provided
        timestamp = datetime.datetime.now().isoformat()
        
        # If log_path is provided, use it directly without creating subdirectories
        if log_path:
            file_path = log_path
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        else:
            # Generate a unique filename based on timestamp
            filename = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(self.log_dir, filename)
        
        # Prepare the log data
        log_data = {
            "timestamp": timestamp,
            "questionnaire": questionnaire_name,
            "conversation": conversation,
            "diagnosis": diagnosis,
        }
        
        # Add metadata if provided
        if metadata:
            log_data["metadata"] = metadata
        
        # Add RAG summary information
        log_data["rag_summary"] = self._generate_rag_summary(conversation)
        
        # Save to file atomically
        temp_path = file_path + '.tmp'
        try:
            # First write to a temporary file
            with open(temp_path, 'w') as f:
                json.dump(log_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Then rename to final path (atomic operation)
            os.replace(temp_path, file_path)
            
            print(f"Chat log saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving chat log: {str(e)}")
            # Try to clean up the temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise
    
    def load_chat(self, log_id: str) -> Dict[str, Any]:
        """
        Load a chat log from file.
        
        Args:
            log_id: ID of the log to load
            
        Returns:
            Dict containing the log data
        """
        # Ensure log_id has .json extension
        if not log_id.endswith('.json'):
            log_id += '.json'
        
        # Look for the file in the logs directory
        file_path = os.path.join(self.log_dir, log_id)
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat log: {str(e)}")
            raise

    def _generate_rag_summary(self, conversation_log):
        """Generate a summary of RAG usage from the conversation log."""
        rag_summary = {
            "total_rag_queries": 0,
            "total_documents_accessed": 0,
            "documents_accessed": {}
        }
        
        # Process all messages looking for RAG usage
        for message in conversation_log:
            if "rag_usage" in message:
                rag_usage = message.get("rag_usage", {})
                
                # Count this as a RAG query if documents were accessed
                documents = rag_usage.get("documents", rag_usage.get("accessed_documents", []))
                if documents:
                    rag_summary["total_rag_queries"] += 1
                    
                    # Add document counts
                    for doc in documents:
                        title = doc.get("title", "Unknown")
                        score = doc.get("score", 0.0)
                        rag_summary["total_documents_accessed"] += 1
                        
                        if title in rag_summary["documents_accessed"]:
                            # Update access count
                            rag_summary["documents_accessed"][title]["access_count"] += 1
                            
                            # Track highest score
                            if score > rag_summary["documents_accessed"][title]["highest_score"]:
                                rag_summary["documents_accessed"][title]["highest_score"] = score
                                
                            # Track average score
                            current_count = rag_summary["documents_accessed"][title]["access_count"]
                            current_avg = rag_summary["documents_accessed"][title]["average_score"]
                            new_avg = ((current_avg * (current_count - 1)) + score) / current_count
                            rag_summary["documents_accessed"][title]["average_score"] = round(new_avg, 4)
                        else:
                            rag_summary["documents_accessed"][title] = {
                                "access_count": 1,
                                "highest_score": score,
                                "average_score": score,
                                "example_excerpt": doc.get("excerpt", doc.get("highlight", "No excerpt available"))
                            }
        
        return rag_summary
