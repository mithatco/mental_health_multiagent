import os
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

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
    
    def validate_conversation_log(self, conversation_log):
        """
        Validate and fix the conversation log if needed to ensure it's properly formatted.
        
        Args:
            conversation_log: List of conversation entries
            
        Returns:
            Validated and fixed conversation log
        """
        validated_log = []
        
        # If conversation_log is None or empty, return empty list
        if not conversation_log:
            print("[DEBUG] Empty conversation log received")
            return []
        
        # Log the type and size of conversation log for debugging
        print(f"[DEBUG] Validating conversation log of type {type(conversation_log)} with {len(conversation_log) if hasattr(conversation_log, '__len__') else 'unknown'} entries")
        
        # Handle case where the entire conversation is accidentally a single string
        if isinstance(conversation_log, str):
            print("[DEBUG] Entire conversation log is a string, attempting to extract")
            conversation_log = [conversation_log]
            
        # First check if all entries are properly formatted objects
        if isinstance(conversation_log, list):
            all_objects = all(
                isinstance(entry, dict) and 
                "role" in entry and 
                "content" in entry 
                for entry in conversation_log
            )
            
            if all_objects:
                print("[DEBUG] Conversation log already properly formatted as list of dicts with role and content")
                return conversation_log
        else:
            print(f"[DEBUG] Conversation log is not a list but a {type(conversation_log)}")
            # Try to convert to list if it's some other iterable
            try:
                conversation_log = list(conversation_log)
                print(f"[DEBUG] Converted conversation log to list with {len(conversation_log)} entries")
            except:
                print("[DEBUG] Could not convert conversation log to list")
                # Return empty list as fallback
                return []
            
        # Process entries to fix any string representations
        for i, entry in enumerate(conversation_log):
            # If entry is a string, try to parse it
            if isinstance(entry, str):
                import json
                import re
                
                print(f"[DEBUG] Found string entry at position {i}, attempting to parse")
                
                # Clean the string to handle common formatting issues
                cleaned = entry.strip()
                
                # Remove code blocks markers
                cleaned = re.sub(r'^```json\s*', '', cleaned)
                cleaned = re.sub(r'^```\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
                
                # Special handling for Groq-like outputs: if the string is very long,
                # it's likely the entire conversation in one string
                if len(cleaned) > 1000:  # Arbitrary threshold to identify long strings
                    print(f"[DEBUG] Found long string entry ({len(cleaned)} chars), may be entire conversation")
                    
                    # First, try to find and extract a JSON array
                    json_pattern = re.compile(r'\[(.*)\]', re.DOTALL)
                    json_match = json_pattern.search(cleaned)
                    
                    if json_match:
                        json_text = f"[{json_match.group(1)}]"
                        try:
                            parsed_json = json.loads(json_text)
                            print(f"[DEBUG] Successfully extracted JSON array with {len(parsed_json)} items")
                            
                            # Process each entry in the parsed JSON
                            for parsed_entry in parsed_json:
                                if isinstance(parsed_entry, dict) and 'role' in parsed_entry and 'content' in parsed_entry:
                                    validated_log.append({
                                        'role': parsed_entry['role'],
                                        'content': parsed_entry['content']
                                    })
                            
                            if validated_log:
                                print(f"[DEBUG] Added {len(validated_log)} entries from extracted JSON array")
                                continue  # Skip to next entry
                        except json.JSONDecodeError:
                            print("[DEBUG] JSON array extraction failed, trying turn-based parsing")
                    
                    # If JSON parsing failed, try to parse conversation turns
                    # Look for patterns like "Assistant: message" and "Patient: message"
                    turn_pattern = re.compile(r'(assistant|patient|clinician|therapist|doctor|user):\s*(.*?)(?=(?:\n\s*|\r\n\s*|\r\s*)(?:assistant|patient|clinician|therapist|doctor|user):|$)', re.IGNORECASE | re.DOTALL)
                    turn_matches = turn_pattern.findall(cleaned)
                    
                    if turn_matches:
                        print(f"[DEBUG] Found {len(turn_matches)} conversation turns")
                        for role, content in turn_matches:
                            if role.lower() in ["assistant", "clinician", "therapist", "doctor"]:
                                std_role = "assistant"
                            else:
                                std_role = "patient"
                                
                            content = content.strip()
                            if content:  # Skip empty content
                                validated_log.append({
                                    'role': std_role,
                                    'content': content
                                })
                        
                        print(f"[DEBUG] Added {len(validated_log)} structured messages from turn parsing")
                        continue  # Skip to next entry
                
                # Attempt to extract a JSON array from shorter text
                matches = re.findall(r'\[(.*?)\]', cleaned, re.DOTALL)
                if matches:
                    # Get the largest match (most likely to be our conversation array)
                    largest_match = max(matches, key=len)
                    json_array_text = f"[{largest_match}]"
                    
                    try:
                        # Try to parse the JSON array
                        parsed_entries = json.loads(json_array_text)
                        
                        if isinstance(parsed_entries, list) and parsed_entries:
                            print(f"[DEBUG] Successfully parsed entry as JSON array with {len(parsed_entries)} items")
                            
                            # Process each entry in the parsed JSON
                            for parsed_entry in parsed_entries:
                                if isinstance(parsed_entry, dict) and 'role' in parsed_entry and 'content' in parsed_entry:
                                    validated_log.append({
                                        'role': parsed_entry['role'],
                                        'content': parsed_entry['content']
                                    })
                                else:
                                    print(f"[DEBUG] Skipping invalid entry in parsed JSON: {parsed_entry}")
                            
                            continue  # Skip to the next entry in the outer loop
                    except json.JSONDecodeError as e:
                        print(f"[DEBUG] Failed to parse as JSON array: {e}")
                        
                # If array extraction failed, try to look for patterns like "Role: Content"
                if not validated_log:
                    print("[DEBUG] Trying to extract conversation from text using patterns")
                    
                    # Look for patterns like "Assistant: [message]" and "Patient: [message]"
                    pattern = re.compile(r'(assistant|patient|clinician|therapist|user):\s*([^\n]+)(?:\n|$)', re.IGNORECASE)
                    matches = pattern.findall(cleaned)
                    
                    for role, content in matches:
                        # Standardize roles
                        if role.lower() in ["assistant", "clinician", "therapist"]:
                            std_role = "assistant"
                        else:
                            std_role = "patient"
                            
                        validated_log.append({
                            'role': std_role,
                            'content': content.strip()
                        })
                    
                    if matches:
                        print(f"[DEBUG] Extracted {len(matches)} messages using pattern matching")
                        continue
                
                # If all parsing attempts failed for this string, just include it as is
                print(f"[DEBUG] Unable to parse string entry, using it as assistant message")
                validated_log.append({
                    'role': 'assistant', 
                    'content': entry
                })
            elif isinstance(entry, dict):
                # Handle dictionary entries
                if 'role' in entry and 'content' in entry:
                    # Entry is already correctly formatted
                    validated_log.append(entry)
                elif 'text' in entry:
                    # Some models use 'text' instead of 'content'
                    role = entry.get('role', 'assistant')
                    validated_log.append({
                        'role': role,
                        'content': entry['text']
                    })
                else:
                    # Cannot determine what this entry should be
                    print(f"[DEBUG] Skipping invalid dict entry: {entry}")
            else:
                # Entry is neither string nor dict
                print(f"[DEBUG] Skipping entry of unsupported type: {type(entry)}")
        
        # If we didn't manage to validate anything but had input, log a warning
        if not validated_log and conversation_log:
            print(f"[WARNING] Failed to validate any entries from a log with {len(conversation_log)} entries")
            
            # Last resort: try to parse the entire conversation_log as one unit
            if len(conversation_log) == 1 and isinstance(conversation_log[0], str):
                print("[DEBUG] Attempting to parse single string as entire conversation")
                # Try to extract turn-by-turn conversation from the entire string
                text = conversation_log[0]
                
                # Look for alternating assistant/patient patterns
                turn_pattern = re.compile(r'(assistant|patient|clinician|therapist|doctor|user):\s*(.*?)(?=(?:\n\s*|\r\n\s*|\r\s*)(?:assistant|patient|clinician|therapist|doctor|user):|$)', re.IGNORECASE | re.DOTALL)
                turn_matches = turn_pattern.findall(text)
                
                if turn_matches:
                    print(f"[DEBUG] Found {len(turn_matches)} conversation turns in full text")
                    for role, content in turn_matches:
                        if role.lower() in ["assistant", "clinician", "therapist", "doctor"]:
                            std_role = "assistant"
                        else:
                            std_role = "patient"
                            
                        content = content.strip()
                        if content:  # Skip empty content
                            validated_log.append({
                                'role': std_role,
                                'content': content
                            })
            
        # Apply a final validation pass to ensure all entries have role and content
        final_validated = []
        for entry in validated_log:
            if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                # Ensure role is standardized
                role = entry['role'].lower()
                if role in ["assistant", "clinician", "therapist", "doctor"]:
                    entry['role'] = "assistant"
                elif role in ["patient", "user", "client"]:
                    entry['role'] = "patient"
                
                # Ensure content is a string
                if not isinstance(entry['content'], str):
                    entry['content'] = str(entry['content'])
                    
                final_validated.append(entry)
        
        if len(final_validated) != len(validated_log):
            print(f"[DEBUG] Final validation removed {len(validated_log) - len(final_validated)} entries")
            
        if final_validated:
            print(f"[DEBUG] Successfully validated {len(final_validated)} conversation entries")
            return final_validated
        
        # If still empty, return the original log as a last resort
        print("[DEBUG] Validation produced no valid entries, returning original log")
        return conversation_log

    def clean_diagnosis(self, diagnosis: str) -> str:
        """
        Clean diagnosis text by removing special tags like <think></think>.
        
        Args:
            diagnosis: The diagnosis text to clean
            
        Returns:
            Cleaned diagnosis text
        """
        import re
        
        # Remove <think> blocks
        diagnosis = re.sub(r'<think>.*?</think>', '', diagnosis, flags=re.DOTALL)
        
        # Remove any potential markdown code blocks
        diagnosis = re.sub(r'```(json|python)?\s*', '', diagnosis)
        diagnosis = re.sub(r'\s*```', '', diagnosis)
        
        return diagnosis.strip()

    def save_chat(self, 
                  conversation: List[Dict[str, str]], 
                  diagnosis: str,
                  questionnaire_name: str = "unknown",
                  metadata: Optional[Dict[str, Any]] = None,
                  log_path: Optional[str] = None,
                  timing_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Save a chat log to a JSON file.
        
        Args:
            conversation: List of conversation messages
            diagnosis: Final diagnosis text
            questionnaire_name: Name of the questionnaire used
            metadata: Additional metadata to include
            log_path: Optional explicit path for the log file
            timing_metrics: Optional timing metrics for performance analysis
            
        Returns:
            Path to the saved log file
        """
        # Validate the conversation log
        validated_conversation = self.validate_conversation_log(conversation)
        if len(validated_conversation) != len(conversation):
            print(f"[WARNING] Conversation log was fixed from {len(conversation)} to {len(validated_conversation)} entries")
        
        # Clean the diagnosis text to remove <think> tags and other special tags
        cleaned_diagnosis = self.clean_diagnosis(diagnosis)
        
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
            "conversation": validated_conversation,
            "diagnosis": cleaned_diagnosis,
        }
        
        # Add metadata if provided
        if metadata:
            log_data["metadata"] = metadata
        
        # Add timing metrics if provided
        if timing_metrics:
            log_data["timing_metrics"] = timing_metrics
        
        # Add RAG summary information
        log_data["rag_summary"] = self._generate_rag_summary(validated_conversation)
        
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
            "documents_accessed": {},
            "rag_disabled": False,
            "evaluation_metrics": {
                "contextual_relevancy": {"total_score": 0, "count": 0, "reasons": []},
                "faithfulness": {"total_score": 0, "count": 0, "reasons": []},
                "answer_relevancy": {"total_score": 0, "count": 0, "reasons": []},
                "overall": {"total_score": 0, "count": 0}
            }
        }
        
        # Process all messages looking for RAG usage
        for message in conversation_log:
            if "rag_usage" in message:
                rag_usage = message.get("rag_usage")
                
                # Check if RAG usage is None (disabled)
                if rag_usage is None:
                    rag_summary["rag_disabled"] = True
                    continue
                
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
                            # Get relevance explanation if available
                            relevance_explanation = doc.get("relevance_explanation", "")
                            
                            rag_summary["documents_accessed"][title] = {
                                "access_count": 1,
                                "highest_score": score,
                                "average_score": score,
                                "example_excerpt": doc.get("excerpt", doc.get("highlight", "No excerpt available")),
                                "relevance_explanation": relevance_explanation
                            }
                    
                    # Track evaluation metrics if available
                    if "evaluation" in rag_usage:
                        eval_data = rag_usage["evaluation"]
                        
                        # Track individual metrics
                        for metric_name in ["contextual_relevancy", "faithfulness", "answer_relevancy"]:
                            if metric_name in eval_data and "score" in eval_data[metric_name]:
                                metric_data = eval_data[metric_name]
                                score = metric_data.get("score", 0)
                                reason = metric_data.get("reason", "No reason provided")
                                
                                # Add score to total
                                rag_summary["evaluation_metrics"][metric_name]["total_score"] += score
                                rag_summary["evaluation_metrics"][metric_name]["count"] += 1
                                
                                # Store reason with score for context
                                reason_entry = {
                                    "score": score,
                                    "reason": reason,
                                    "passed": metric_data.get("passed", False)
                                }
                                rag_summary["evaluation_metrics"][metric_name]["reasons"].append(reason_entry)
                        
                        # Track overall score
                        if "average_score" in eval_data:
                            rag_summary["evaluation_metrics"]["overall"]["total_score"] += eval_data["average_score"]
                            rag_summary["evaluation_metrics"]["overall"]["count"] += 1
        
        # Calculate averages for all metrics
        for metric, data in rag_summary["evaluation_metrics"].items():
            if data["count"] > 0:
                data["average_score"] = round(data["total_score"] / data["count"], 4)
                
                # Limit to most recent 5 reasons to avoid huge logs
                if "reasons" in data:
                    data["reasons"] = data["reasons"][-5:]
            else:
                # Remove metrics with no data
                data.pop("total_score", None)
                data.pop("count", None)
                data["average_score"] = None
                if "reasons" in data:
                    data.pop("reasons", None)
        
        return rag_summary
    
    def list_chat_logs(self) -> List[str]:
        """
        List all available chat log files in the logs directory.
        
        Returns:
            List of log file names
        """
        if not os.path.exists(self.log_dir):
            return []
        
        try:
            files = os.listdir(self.log_dir)
            # Filter for JSON files only
            json_files = [f for f in files if f.endswith('.json')]
            return json_files
        except Exception as e:
            print(f"Error listing chat logs: {str(e)}")
            return []