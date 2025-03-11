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
        
        # Debug: print conversation messages with RAG info
        print("\n[DEBUG] Checking conversation for RAG usage...")
        rag_found = False
        for i, msg in enumerate(conversation):
            if msg.get("role") == "assistant" and "rag_usage" in msg:
                rag_found = True
                print(f"[DEBUG] Message {i} from assistant has RAG info: {msg['rag_usage']['count']} documents")
        
        if not rag_found:
            print("[DEBUG] No RAG usage information found in any messages")
        
        # Add RAG summary information
        rag_summary = self._summarize_rag_usage(conversation)
        if rag_summary:
            print(f"[DEBUG] RAG summary created: {rag_summary['total_rag_queries']} queries, {rag_summary['total_documents_accessed']} docs")
            data["rag_summary"] = rag_summary
        else:
            print("[DEBUG] No RAG summary created")
        
        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save a plain text version for easier reading
        txt_file_path = os.path.join(self.log_dir, filename.replace('.json', '.txt'))
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Conversation with {questionnaire_name}\n")
            f.write(f"Time: {datetime.datetime.now().isoformat()}\n\n")
            
            # Include RAG summary in the text file if available
            if rag_summary:
                f.write("=== RAG USAGE SUMMARY ===\n")
                f.write(f"Total RAG queries: {rag_summary['total_rag_queries']}\n")
                f.write(f"Total documents accessed: {rag_summary['total_documents_accessed']}\n")
                f.write("Documents accessed:\n")
                for doc_title, doc_info in rag_summary['documents_accessed'].items():
                    f.write(f"- {doc_title} (accessed {doc_info['access_count']} times)\n")
                    f.write(f"  Example excerpt: {doc_info['example_excerpt']}\n")
                f.write("\n=== CONVERSATION ===\n\n")
            
            for msg in conversation:
                role = msg['role'].upper()
                if role == "SYSTEM":
                    continue  # Skip system messages in the readable version
                f.write(f"{role}: {msg['content']}\n")
                
                # Include RAG info in the readable version
                if role == "ASSISTANT" and "rag_usage" in msg:
                    rag_info = msg["rag_usage"]
                    f.write(f"[RAG: {rag_info['count']} documents accessed]\n")
                    for i, doc in enumerate(rag_info['accessed_documents']):
                        f.write(f"  - Doc {i+1}: {doc.get('title', 'Unknown')}\n")
                
                f.write("\n")
                
            f.write("\n==== DIAGNOSIS ====\n\n")
            f.write(diagnosis)
            f.write("\n")
        
        return file_path
    
    def _summarize_rag_usage(self, conversation_log):
        """Extract and summarize RAG usage from conversation log."""
        rag_summary = {
            "total_rag_queries": 0,
            "total_documents_accessed": 0,
            "documents_accessed": {}
        }
        
        # Debug print for investigation
        print(f"[DEBUG] Summarizing RAG usage from {len(conversation_log)} messages")
        
        for i, message in enumerate(conversation_log):
            if message.get("role") == "assistant" and "rag_usage" in message:
                print(f"[DEBUG] Found RAG usage in message {i}")
                rag_usage = message["rag_usage"]
                rag_summary["total_rag_queries"] += 1
                
                if "count" in rag_usage:
                    rag_summary["total_documents_accessed"] += rag_usage["count"]
                    print(f"[DEBUG]   Documents accessed: {rag_usage['count']}")
                    
                if "accessed_documents" in rag_usage:
                    for doc in rag_usage["accessed_documents"]:
                        doc_title = doc.get("title", "Unknown")
                        print(f"[DEBUG]   Document: {doc_title}")
                        
                        if doc_title in rag_summary["documents_accessed"]:
                            rag_summary["documents_accessed"][doc_title]["access_count"] += 1
                        else:
                            rag_summary["documents_accessed"][doc_title] = {
                                "access_count": 1,
                                "example_excerpt": doc.get("excerpt", "")
                            }
        
        if rag_summary["total_rag_queries"] > 0:
            return rag_summary
        else:
            return None

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
