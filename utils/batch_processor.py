"""
Batch processing utility for generating multiple conversations.
"""

import os
import time
import random
import csv
import json
from datetime import datetime
from pathlib import Path

from agents.mental_health_assistant import MentalHealthAssistant
from agents.patient import Patient
from utils.conversation_handler import ConversationHandler
from utils.chat_logger import ChatLogger


class BatchProcessor:
    """Process multiple conversations in batch mode."""
    
    def __init__(self, 
                ollama_url: str, 
                assistant_model: str, 
                patient_model: str, 
                rag_engine=None, 
                logs_dir=None):
        """
        Initialize the batch processor.
        
        Args:
            ollama_url: URL for the Ollama API
            assistant_model: Model to use for the assistant agent
            patient_model: Model to use for the patient agent
            rag_engine: RAG engine for document retrieval
            logs_dir: Directory to save logs
        """
        self.ollama_url = ollama_url
        self.assistant_model = assistant_model
        self.patient_model = patient_model
        self.rag_engine = rag_engine
        
        # Set up chat logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if logs_dir:
            self.logs_dir = os.path.join(logs_dir, f"batch_{timestamp}")
        else:
            self.logs_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "chat_logs", 
                f"batch_{timestamp}"
            )
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.chat_logger = ChatLogger(self.logs_dir)
        
        # Get available profiles
        self.available_profiles = Patient.list_available_profiles()
        
    def process_batch(self, questions, count=5, profile=None, randomize_profiles=False):
        """
        Process a batch of conversations.
        
        Args:
            questions: List of questions for the assistant to ask
            count: Number of conversations to generate
            profile: Specific profile to use (None for default)
            randomize_profiles: Whether to randomize profiles for each conversation
            
        Returns:
            List of result summaries
        """
        results = []
        
        print(f"Starting batch processing of {count} conversations")
        print(f"Saving results to: {self.logs_dir}")
        
        # Create a CSV file for the summary
        summary_path = os.path.join(self.logs_dir, "batch_summary.csv")
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Conversation', 'Profile', 'Questions', 'Duration (s)', 'Log File'])
        
        for i in range(1, count + 1):
            print(f"\nProcessing conversation {i} of {count}")
            
            # Select profile for this conversation
            current_profile = self._select_profile(profile, randomize_profiles, i)
            
            # Run the conversation
            start_time = time.time()
            
            # Initialize agents
            assistant = MentalHealthAssistant(self.ollama_url, self.assistant_model, questions.copy(), self.rag_engine)
            patient = Patient(self.ollama_url, self.patient_model, current_profile)
            
            # Set up conversation handler
            conversation = ConversationHandler(assistant, patient)
            
            # Run the conversation
            print(f"Running conversation with profile: {current_profile}")
            diagnosis = conversation.run()
            
            duration = time.time() - start_time
            
            # Save conversation
            metadata = {
                "batch_id": i,
                "assistant_model": self.assistant_model,
                "patient_model": self.patient_model,
                "patient_profile": current_profile or "default",
                "question_count": len(questions),
                "duration": duration
            }
            
            log_path = self.chat_logger.save_chat(
                conversation.get_conversation_log(),
                diagnosis,
                questionnaire_name=f"batch_{i}",
                metadata=metadata
            )
            
            # Add to results
            result = {
                "conversation_id": i,
                "profile": current_profile,
                "question_count": len(questions),
                "duration": duration,
                "log_path": log_path,
                "diagnosis": diagnosis[:200] + "..." if len(diagnosis) > 200 else diagnosis  # Truncated diagnosis
            }
            results.append(result)
            
            # Add to CSV summary
            with open(summary_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i, current_profile, len(questions), f"{duration:.2f}", os.path.basename(log_path)])
            
            print(f"Conversation {i} completed in {duration:.2f} seconds")
            
            # Add a small delay between conversations to avoid rate limiting
            if i < count:
                time.sleep(1)
        
        # Save full results as JSON
        with open(os.path.join(self.logs_dir, "batch_results.json"), 'w') as f:
            json.dump({"results": results}, f, indent=2)
        
        print(f"\nBatch processing complete. Generated {count} conversations.")
        print(f"Results saved to: {self.logs_dir}")
        
        return results
    
    def _select_profile(self, specified_profile, randomize, conversation_index):
        """Select a profile based on the specified options."""
        if randomize:
            # Randomize profile selection
            if self.available_profiles:
                return random.choice(self.available_profiles)
            else:
                return None
        else:
            # Use the specified profile or default
            return specified_profile
