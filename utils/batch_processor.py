"""
Batch processing utility for mental health conversations.
"""
import os
import time
import datetime
from typing import List, Optional, Dict, Any
import logging
from utils.batch_runner import BatchRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BatchProcessor")

class BatchProcessor:
    """Process batches of conversations using the specified models."""
    
    def __init__(self, ollama_url, assistant_model, patient_model, rag_engine, logs_dir=None):
        """
        Initialize the batch processor.
        
        Args:
            ollama_url: URL to the Ollama API
            assistant_model: Model to use for the assistant
            patient_model: Model to use for the patient
            rag_engine: RAG engine to use for document retrieval
            logs_dir: Directory to save logs to
        """
        self.ollama_url = ollama_url
        self.assistant_model = assistant_model
        self.patient_model = patient_model
        self.rag_engine = rag_engine
        self.logs_dir = logs_dir or "chat_logs"
        
        # Ensure logs directory exists
        os.makedirs(self.logs_dir, exist_ok=True)
        
        logger.info(f"BatchProcessor initialized with models: {assistant_model} (assistant), {patient_model} (patient)")
        logger.info(f"Logs directory: {self.logs_dir}")
    
    def _get_conversation_runner(self):
        """Get a function to run a single conversation."""
        # Import here to avoid circular imports
        from main import run_conversation
        
        return run_conversation
    
    def process_batch(self, questions: List[str], count: int = 5, profile: Optional[str] = None, 
                      randomize_profiles: bool = False) -> Dict[str, Any]:
        """
        Process a batch of conversations.
        
        Args:
            questions: List of questions to ask
            count: Number of conversations to generate
            profile: Profile to use for the patient (or None to use default)
            randomize_profiles: Whether to randomize profiles for each conversation
            
        Returns:
            Dict with batch results
        """
        # Get temp file name to save questions
        from tempfile import NamedTemporaryFile
        import json
        import traceback
        
        logger.info(f"Starting batch of {count} conversations with {len(questions)} questions")
        
        # Create batch directory if not exists - use the provided logs_dir directly
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_id = f"batch_{timestamp}"
        batch_dir = self.logs_dir
        
        # Ensure batch directory exists
        os.makedirs(batch_dir, exist_ok=True)
        
        # Create a status file to track progress
        status_file = os.path.join(batch_dir, 'batch_status.json')
        logger.info(f"Status will be tracked in: {status_file}")
        
        # Write questions to a temporary file
        temp_file_path = None
        try:
            with NamedTemporaryFile(suffix='.json', delete=False, mode='w+') as temp_file:
                # Format for easy debugging
                json_data = {
                    "questions": questions,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "count": len(questions)
                }
                json.dump(json_data, temp_file)
                temp_file_path = temp_file.name
                
            logger.info(f"Created temporary questions file: {temp_file_path}")
            
            # Verify the file was written correctly
            try:
                with open(temp_file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Successfully loaded temporary file with {len(data['questions'])} questions")
            except Exception as e:
                logger.error(f"Error verifying temporary file: {e}")
                raise ValueError(f"Failed to create valid temporary questions file: {e}")
            
            # Get conversation runner
            conversation_runner = self._get_conversation_runner()
            logger.info("Obtained conversation runner")
            
            # Initialize batch runner
            batch_runner = BatchRunner(conversation_runner, batch_dir, status_file)
            logger.info("Initialized batch runner")
            
            # Run the batch
            start_time = time.time()
            logger.info(f"Running batch with {count} conversations...")
            
            # Add a try/except block to capture errors during batch processing
            try:
                results = batch_runner.run_batch(
                    batch_size=count,
                    pdf_path=temp_file_path,
                    patient_profile=profile,
                    randomize_profiles=randomize_profiles,
                    assistant_model=self.assistant_model,
                    patient_model=self.patient_model,
                    logs_dir=batch_dir,
                    disable_output=True
                )
                
                # Log results
                duration = time.time() - start_time
                logger.info(f"Batch completed in {duration:.2f} seconds")
                logger.info(f"Generated {len(results)} conversations")
                
                # Return results
                return {
                    "batch_id": batch_id,
                    "count": len(results),
                    "duration": duration,
                    "results": results
                }
            except Exception as e:
                logger.error(f"Error during batch processing: {e}")
                logger.error(traceback.format_exc())
                
                # Save error details to a file for debugging
                error_file = os.path.join(batch_dir, "batch_error.log")
                try:
                    with open(error_file, 'w') as f:
                        f.write(f"Error during batch processing: {e}\n\n")
                        f.write(traceback.format_exc())
                except Exception as write_error:
                    logger.error(f"Failed to write error log: {write_error}")
                
                raise
        finally:
            # Clean up temp file
            try:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.info(f"Removed temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
