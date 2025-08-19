"""
Batch runner for mental health assessment conversations.
"""
import os
import json
import time
import random
import csv
import datetime
import logging
import traceback
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BatchRunner")

class BatchRunner:
    """Handles running multiple conversation simulations in batch mode."""
    
    def __init__(self, conversation_runner, logs_dir: str, status_file: Optional[str] = None):
        """
        Initialize with a conversation runner and logs directory.
        
        Args:
            conversation_runner: The conversation runner function or class
            logs_dir: Directory to save logs to
            status_file: Optional JSON file to write status updates to
        """
        self.conversation_runner = conversation_runner
        self.logs_dir = logs_dir
        self.status_file = status_file
        self.results = []
        self.current_conversation = None  # Track which conversation is currently running
        
        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)
        logger.info(f"BatchRunner initialized with logs directory: {logs_dir}")
        
        if status_file:
            logger.info(f"Will write status updates to: {status_file}")
            # Initialize status file with starting state
            self._update_status(0, 0, in_progress_idx=None, force_write=True)
    
    def run_batch(self, batch_size: int, pdf_path: str, patient_profile: Optional[str] = None, 
                 randomize_profiles: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """
        Run a batch of conversations.
        
        Args:
            batch_size: Number of conversations to run
            pdf_path: Path to PDF to use for questions
            patient_profile: Patient profile to use (or None for default)
            randomize_profiles: Whether to randomize profiles for each conversation
            **kwargs: Additional arguments to pass to conversation runner
            
        Returns:
            List of results with metadata for each conversation
        """
        logger.info(f"Starting batch of {batch_size} conversations")
        logger.info(f"PDF path: {pdf_path}")
        logger.info(f"Patient profile: {patient_profile if not randomize_profiles else 'randomized'}")
        logger.info(f"Logs directory: {self.logs_dir}")
        
        # Log the kwargs for debugging
        logger.info(f"Conversation runner kwargs: {kwargs}")
        
        # List available profiles for randomization
        available_profiles = None
        if randomize_profiles:
            try:
                from agents.patient import Patient
                available_profiles = Patient.list_available_profiles()
                logger.info(f"Available profiles for randomization: {available_profiles}")
            except Exception as e:
                logger.error(f"Error listing available profiles: {e}")
                randomize_profiles = False
        
        start_time = time.time()
        self.results = []
        completed_count = 0
        
        # Initialize or update status file
        self._update_status(batch_size, completed_count, in_progress_idx=0, force_write=True)
            
        # Run the batch
        for i in range(batch_size):
            logger.info(f"Starting conversation {i+1}/{batch_size}")
            self.current_conversation = i
            
            # Update status to show which conversation is in progress
            self._update_status(batch_size, completed_count, in_progress_idx=i, force_write=True)
            
            # Randomize profile if requested
            current_profile = patient_profile
            if randomize_profiles and available_profiles:
                current_profile = random.choice(available_profiles)
                logger.info(f"Randomized profile for conversation {i+1}: {current_profile}")
            
            # Generate a unique conversation ID for this batch item
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            conversation_id = f"chat_{timestamp}_batch_{i+1}"
            
            # Create the explicit log filename in the batch directory
            log_filename = os.path.join(self.logs_dir, f"{conversation_id}.json")
            logger.info(f"Log will be saved to: {log_filename}")
            
            # Run conversation
            conversation_start = time.time()
            try:
                # Make a copy of kwargs to avoid modifying the original
                run_kwargs = kwargs.copy()
                
                # Set logs_dir to None to prevent automatic subdirectory creation
                # Instead we'll use explicit log_filename
                run_kwargs['logs_dir'] = None
                
                # Set disable_output=True to suppress console output
                run_kwargs['disable_output'] = True
                
                # Log the exact arguments being passed to the conversation runner
                logger.info(f"Running conversation {i+1}/{batch_size} with exact parameters:")
                logger.info(f"  - Profile: {current_profile}")
                logger.info(f"  - Log file: {log_filename}")
                logger.info(f"  - Full conversation mode: {run_kwargs.get('full_conversation', False)}")
                logger.info(f"  - RAG evaluation disabled: {run_kwargs.get('disable_rag_evaluation', False)}")
                logger.info(f"  - Assistant provider: {run_kwargs.get('assistant_provider', 'ollama')}")
                logger.info(f"  - Patient provider: {run_kwargs.get('patient_provider', 'ollama')}")
                
                # Log API key presence (but not the actual keys)
                if 'groq_api_key' in run_kwargs and run_kwargs['groq_api_key']:
                    logger.info(f"  - Using Groq API key: Yes")
                
                if 'openai_api_key' in run_kwargs and run_kwargs['openai_api_key']:
                    logger.info(f"  - Using OpenAI API key: Yes")
                
                # Run the conversation with the specified log filename
                conversation_result = self.conversation_runner(
                    pdf_path=pdf_path,
                    patient_profile=current_profile,
                    log_filename=log_filename,  # Use explicit filename
                    **run_kwargs
                )
                
                duration = time.time() - conversation_start
                
                # Verify the log file exists
                if os.path.exists(log_filename):
                    logger.info(f"Verified log file was saved: {log_filename}")
                else:
                    logger.warning(f"Log file was not found at expected location: {log_filename}")
                    # Try to create the directory again and force retry the save
                    log_dir = os.path.dirname(log_filename)
                    os.makedirs(log_dir, exist_ok=True)
                    logger.info(f"Created log directory: {log_dir}")
                
                # Collect result metadata
                result = {
                    'conversation_id': conversation_id,
                    'profile': current_profile,
                    'duration': duration,
                    'status': 'completed',
                    'question_count': conversation_result.get('question_count', 0),
                    'log_path': log_filename  # Store log path for easy access
                }
                
                # Check if diagnosis was generated
                if 'diagnosis' in conversation_result and conversation_result['diagnosis']:
                    logger.info(f"Conversation {i+1} completed successfully with diagnosis")
                else:
                    logger.warning(f"Conversation {i+1} completed but no diagnosis was generated")
                    result['status'] = 'completed_no_diagnosis'
                
                logger.info(f"Conversation {i+1} completed in {duration:.2f} seconds")
                
            except Exception as e:
                error_text = traceback.format_exc()
                logger.error(f"Error in conversation {i+1}: {e}")
                logger.error(f"Traceback: {error_text}")
                
                duration = time.time() - conversation_start
                # Record the error
                result = {
                    'conversation_id': conversation_id,
                    'profile': current_profile,
                    'duration': duration,
                    'status': 'error',
                    'error': str(e)
                }
            
            # Add result to results list
            self.results.append(result)
            completed_count += 1
            
            # Update status file
            if i < batch_size - 1:  # If not the last conversation
                # Next conversation will be in progress
                self._update_status(batch_size, completed_count, in_progress_idx=i+1, force_write=True)
            else:
                # No more conversations in progress
                self._update_status(batch_size, completed_count, in_progress_idx=None, force_write=True)
        
        # Calculate statistics
        total_duration = time.time() - start_time
        avg_duration = sum([r['duration'] for r in self.results]) / len(self.results) if self.results else 0
        
        logger.info(f"Batch completed: {len(self.results)} conversations")
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        logger.info(f"Average conversation duration: {avg_duration:.2f} seconds")
        
        # Save batch results
        batch_results_path = os.path.join(self.logs_dir, 'batch_results.json')
        try:
            batch_results = {
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'batch_size': batch_size,
                'profiles': [r['profile'] for r in self.results],
                'randomize_profiles': randomize_profiles,
                'timestamp': datetime.datetime.now().isoformat(),
                'results': self.results
            }
            
            # Write to a temporary file first, then rename for atomicity
            temp_path = batch_results_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(batch_results, f, indent=2)
                # Ensure data is flushed to disk
                f.flush()
                os.fsync(f.fileno())
            
            # Now rename the file (atomic operation)
            os.replace(temp_path, batch_results_path)
            logger.info(f"Batch results saved to {batch_results_path}")
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
            logger.error(traceback.format_exc())
        
        # Generate CSV summary
        try:
            self._generate_csv_summary()
        except Exception as e:
            logger.error(f"Error generating CSV summary: {e}")
            logger.error(traceback.format_exc())
        
        return self.results
    
    def _update_status(self, total: int, completed: int, in_progress_idx: Optional[int] = None, force_write: bool = False):
        """
        Update the status file with progress.
        
        Args:
            total: Total number of conversations
            completed: Number of completed conversations
            in_progress_idx: Index of conversation currently in progress (None if none in progress)
            force_write: Force writing the file even if the status hasn't changed
        """
        if not self.status_file:
            return
            
        try:
            status = {
                "status": "in_progress" if completed < total else "completed",
                "total_conversations": total,
                "completed_conversations": completed,
                "in_progress_conversation": in_progress_idx,
                "start_time": datetime.datetime.now().isoformat(),
                "results": self.results
            }
            
            # Write to a temporary file first, then rename for atomicity
            temp_path = self.status_file + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(status, f)
                # Ensure data is flushed to disk
                f.flush()
                os.fsync(f.fileno())
            
            # Now rename the file (atomic operation)
            os.replace(temp_path, self.status_file)
            
            logger.info(f"Updated status file: {completed}/{total} conversations completed, {in_progress_idx} in progress")
        except Exception as e:
            logger.error(f"Error updating status file: {e}")
            logger.error(traceback.format_exc())
    
    def _generate_csv_summary(self):
        """Generate a CSV summary of the batch results."""
        summary_path = os.path.join(self.logs_dir, 'batch_summary.csv')
        
        try:
            with open(summary_path, 'w', newline='') as csvfile:
                fieldnames = ['Conversation ID', 'Profile', 'Duration (s)', 'Status', 'Questions']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.results:
                    writer.writerow({
                        'Conversation ID': result['conversation_id'],
                        'Profile': result['profile'] or 'Default',
                        'Duration (s)': f"{result['duration']:.2f}",
                        'Status': result['status'],
                        'Questions': result.get('question_count', 'N/A')
                    })
            
            logger.info(f"Batch summary CSV saved to {summary_path}")
        except Exception as e:
            logger.error(f"Error generating CSV summary: {e}")
            logger.error(traceback.format_exc())
