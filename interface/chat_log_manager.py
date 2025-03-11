"""
Chat Log Manager

Manages loading and accessing chat logs.
"""
import os
import json
import datetime
from pathlib import Path

class ChatLogManager:
    """Class to manage chat logs"""
    
    def __init__(self, logs_dir=None):
        """Initialize with the logs directory."""
        if logs_dir is None:
            # Default to chat_logs in the project directory
            self.logs_dir = os.path.join(os.getcwd(), "chat_logs")
        else:
            self.logs_dir = os.path.abspath(logs_dir)
        
        # Create the directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.logs_cache = None
    
    def get_logs_directory(self):
        """Return the logs directory path."""
        return self.logs_dir
        
    def list_logs(self, refresh=False):
        """List all available chat logs."""
        # Return cached logs if available and refresh not requested
        if self.logs_cache is not None and not refresh:
            return self.logs_cache
            
        logs = []
        
        try:
            # Get all .json files in the logs directory
            for filename in os.listdir(self.logs_dir):
                if filename.endswith(".json"):
                    try:
                        log_path = os.path.join(self.logs_dir, filename)
                        with open(log_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Extract basic metadata
                        timestamp = data.get('timestamp', '')
                        try:
                            date_obj = datetime.datetime.fromisoformat(timestamp)
                            formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                            formatted_date = "Unknown date"
                            
                        logs.append({
                            'id': filename,
                            'filename': filename,
                            'questionnaire': data.get('questionnaire', 'Unknown'),
                            'profile': data.get('metadata', {}).get('patient_profile', 'Unknown'),
                            'timestamp': timestamp,
                            'formatted_date': formatted_date
                        })
                    except Exception as e:
                        print(f"Error loading log {filename}: {str(e)}")
        except Exception as e:
            print(f"Error listing logs directory: {str(e)}")
            
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Cache the results
        self.logs_cache = logs
        
        return logs
    
    def get_log(self, log_id):
        """Get a specific log by ID (filename)."""
        log_path = os.path.join(self.logs_dir, log_id)
        
        if not os.path.exists(log_path):
            return None
            
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                return log_data
        except Exception as e:
            print(f"Error loading log {log_id}: {str(e)}")
            return None
    
    def get_unique_profiles(self):
        """Get a list of unique patient profiles from logs."""
        logs = self.list_logs()
        profiles = set()
        
        for log in logs:
            profile = log.get('profile')
            if profile and profile != 'Unknown':
                profiles.add(profile)
                
        return sorted(list(profiles))
    
    def export_log_as_text(self, log_id):
        """Export a log as formatted text."""
        log_data = self.get_log(log_id)
        
        if not log_data:
            return None
            
        # Generate text content
        output = []
        output.append(f"Conversation: {log_data.get('questionnaire', 'Unknown')}")
        output.append(f"Date: {log_data.get('timestamp', 'Unknown')}")
        output.append(f"Patient Profile: {log_data.get('metadata', {}).get('patient_profile', 'Unknown')}")
        output.append("=" * 50)
        output.append("")
        
        # Add conversation
        for msg in log_data.get('conversation', []):
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == "system":
                continue
                
            output.append(f"{role.upper()}:")
            output.append(content)
            output.append("")
        
        # Add diagnosis
        output.append("=" * 50)
        output.append("DIAGNOSIS:")
        output.append("=" * 50)
        output.append(log_data.get('diagnosis', 'No diagnosis available'))
        
        # Join with newlines
        return "\n".join(output)
