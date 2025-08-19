import os
import argparse
import sys
import json
import time
from pathlib import Path
from utils.document_processor import extract_questions_from_text, DocumentProcessor
from utils.rag_engine import RAGEngine
from agents.mental_health_assistant import MentalHealthAssistant
from agents.patient import Patient
from agents.full_conversation_agent import FullConversationAgent
from utils.conversation_handler import ConversationHandler
from utils.full_conversation_handler import FullConversationHandler
from utils.chat_logger import ChatLogger
from utils.batch_processor import BatchProcessor

# Define default paths
DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "documents"
)
DEFAULT_QUESTIONNAIRES_DIR = os.path.join(DEFAULT_DOCS_DIR, "questionnaires")
DEFAULT_LOGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chat_logs"
)

# Add global debug log function
def debug_log(message):
    """Print debug message with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {message}")

def load_questions_from_json(file_path):
    """
    Load questions from a JSON file (used for batch processing).
    
    Args:
        file_path: Path to the JSON file containing questions
        
    Returns:
        List of questions or None if file couldn't be loaded
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if it has a 'questions' field with a list
        if 'questions' in data and isinstance(data['questions'], list):
            return data['questions']
        
        return None
    except Exception as e:
        print(f"Error loading questions from JSON: {e}")
        return None

def run_conversation(pdf_path, patient_profile=None, assistant_provider="ollama", assistant_model="qwen3:4b", 
                    patient_provider="ollama", patient_model="qwen3:4b", full_conversation=False, disable_output=False, logs_dir=None,
                    log_filename=None, refresh_cache=False, no_save=False, state_file=None, disable_rag=False, disable_rag_evaluation=False,
                    groq_api_key=None, openai_api_key=None):
    """
    Run a simulated mental health assessment conversation between an AI assistant and an AI patient.
    
    Args:
        pdf_path: Path to PDF document containing mental health assessment questions
        patient_profile: Name of the patient profile to use
        assistant_provider: Provider for the assistant LLM ("ollama", "groq", or "openai")
        assistant_model: Name of the model to use for the assistant
        patient_provider: Provider for the patient LLM ("ollama", "groq", or "openai")
        patient_model: Name of the model to use for the patient
        full_conversation: Whether to generate the entire conversation in a single LLM call
        disable_output: Whether to disable console output
        logs_dir: Directory to save logs to
        log_filename: Specific filename to use for the log (overrides default naming)
        refresh_cache: Whether to refresh the document cache
        no_save: Whether to disable saving logs
        state_file: Path to file for updating state during conversation
        disable_rag: Whether to disable RAG document retrieval completely
        disable_rag_evaluation: Whether to disable RAG evaluation but keep document retrieval
        groq_api_key: API key for Groq (if using Groq provider)
        openai_api_key: API key for OpenAI (if using OpenAI provider)
        
    Returns:
        Dict containing conversation results
    """
    # Import required modules
    import time
    from datetime import datetime
    from utils.rag_engine import RAGEngine
    from agents.mental_health_assistant import MentalHealthAssistant
    from agents.patient import Patient
    from utils.conversation_handler import ConversationHandler
    from utils.chat_logger import ChatLogger
    from utils.debug_logger import debug_logger
    
    debug_log("Starting run_conversation function")
    start_time = time.time()
    
    # Add detailed debugging for file paths
    debug_logger.log_file_operation('input_check', pdf_path, 
                                   result=f"PDF path exists: {os.path.exists(pdf_path)}")
    if logs_dir:
        debug_logger.log_file_operation('input_check', logs_dir, 
                                       result=f"Logs dir exists: {os.path.exists(logs_dir)}")
    if log_filename:
        debug_logger.log_file_operation('input_check', log_filename, 
                                      result=f"Log file parent exists: {os.path.exists(os.path.dirname(log_filename))}")
    
    # If no logs directory specified but log_filename is, extract dir from there
    if not logs_dir and log_filename:
        logs_dir = os.path.dirname(log_filename)
        debug_logger.log_file_operation('derived', logs_dir, 
                                      result=f"Derived logs dir from filename: {logs_dir}")
    
    # Initialize RAG engine
    if not disable_output:
        print("Initializing RAG engine...")
    
    debug_log("About to initialize RAG engine")
    # Determine project root and set up RAG engine
    project_root = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(project_root, "documents")
    
    try:
        if disable_rag:
            debug_log("RAG engine disabled by user setting")
            rag_engine = None
        else:
            debug_log(f"Creating RAG engine with docs_dir={docs_dir}")
            rag_engine = RAGEngine(docs_dir, refresh_cache=refresh_cache)
            debug_log("RAG engine initialized successfully")
    except Exception as e:
        debug_log(f"ERROR initializing RAG engine: {str(e)}")
        raise
    
    # If the file is a JSON file, we're in batch mode with a temporary questions file
    if pdf_path.endswith('.json'):
        debug_log("Processing JSON questions file")
        # Try to load questions directly from JSON
        questions = load_questions_from_json(pdf_path)
        questionnaire_name = "batch_questionnaire"
        
        if not questions:
            print("Error: Failed to load questions from JSON file")
            return {"error": "Failed to load questions from JSON file"}
    else:
        # Regular PDF processing
        if not disable_output:
            print(f"Processing PDF: {pdf_path}")
        
        debug_log(f"Extracting questions from PDF: {pdf_path}")
        try:
            questions = rag_engine.get_questions_from_file(pdf_path)
            questionnaire_name = os.path.basename(pdf_path)
            debug_log(f"Successfully extracted {len(questions)} questions")
        except Exception as e:
            debug_log(f"ERROR extracting questions: {str(e)}")
            print(f"Error extracting questions from PDF: {str(e)}")
            return {"error": f"Failed to extract questions: {str(e)}"}
    
    if not questions:
        print("No questions extracted from the PDF")
        return {"error": "No questions found in the PDF"}
    
    if not disable_output:
        print(f"Extracted {len(questions)} questions")
    
    # Set up provider options for the assistant
    assistant_provider_options = {}
    if assistant_provider == "ollama":
        assistant_provider_options["base_url"] = "http://localhost:11434"
    elif assistant_provider == "groq":
        assistant_provider_options["api_key"] = groq_api_key
    elif assistant_provider == "openai":
        assistant_provider_options["api_key"] = openai_api_key
    
    # Set up provider options for the patient
    patient_provider_options = {}
    if patient_provider == "ollama":
        patient_provider_options["base_url"] = "http://localhost:11434"
    elif patient_provider == "groq":
        patient_provider_options["api_key"] = groq_api_key
    elif patient_provider == "openai":
        patient_provider_options["api_key"] = openai_api_key
    
    # Initialize agents
    debug_log("Initializing assistant agent")
    try:
        assistant = MentalHealthAssistant(
            provider=assistant_provider,
            provider_options=assistant_provider_options,
            model=assistant_model, 
            questions=questions, 
            rag_engine=rag_engine,
            questionnaire_name=questionnaire_name
        )
        debug_log("Assistant agent initialized successfully")
    except Exception as e:
        debug_log(f"ERROR initializing assistant: {str(e)}")
        raise
    
    debug_log("Initializing patient agent")
    try:
        patient = Patient(
            provider=patient_provider,
            provider_options=patient_provider_options,
            model=patient_model, 
            profile_name=patient_profile
        )
        debug_log("Patient agent initialized successfully")
    except Exception as e:
        debug_log(f"ERROR initializing patient: {str(e)}")
        raise
    
    # Set up conversation handler with state tracking for API mode
    debug_log("Initializing conversation handler")
    
    # Use FullConversationHandler if full_conversation flag is set
    if full_conversation:
        from utils.full_conversation_handler import FullConversationHandler
        from agents.full_conversation_agent import FullConversationAgent
        
        debug_log("Initializing full conversation agent")
        try:
            agent = FullConversationAgent(
                provider=assistant_provider,
                provider_options=assistant_provider_options,
                model=assistant_model,
                patient_profile=patient_profile,
                questions=questions,
                rag_engine=rag_engine,
                questionnaire_name=questionnaire_name,
                disable_rag_evaluation=disable_rag_evaluation
            )
            debug_log("Full conversation agent initialized successfully")
            conversation = FullConversationHandler(agent, state_file=state_file, disable_rag_evaluation=disable_rag_evaluation)
        except Exception as e:
            debug_log(f"ERROR initializing full conversation agent: {str(e)}")
            raise
    else:
        conversation = ConversationHandler(assistant, patient, state_file=state_file, disable_rag_evaluation=disable_rag_evaluation)
    
    # Run the conversation
    if not disable_output:
        print("\nStarting conversation...\n")
    
    debug_log("Starting conversation.run()")
    try:
        diagnosis = conversation.run(disable_output=disable_output)
        debug_log("Conversation completed successfully")
    except Exception as e:
        debug_log(f"ERROR during conversation: {str(e)}")
        import traceback
        debug_log(f"Traceback: {traceback.format_exc()}")
        raise
    
    if not disable_output:
        print("\n=== Final Diagnosis ===")
        print(diagnosis)
    
    # If in API mode, update the state file with the diagnosis
    if state_file:
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            state['status'] = 'completed'
            state['diagnosis'] = diagnosis
            
            with open(state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Error updating state file: {str(e)}")
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Save the conversation if requested
    if not no_save:
        # Initialize the chat logger
        chat_logger = ChatLogger(logs_dir)
        
        # Create metadata
        metadata = {
            "assistant_model": assistant_model,
            "patient_model": patient_model,
            "patient_profile": patient_profile or "default",
            "question_count": len(questions),
            "duration": duration,
            "full_conversation": full_conversation
        }
        
        # Extract timing metrics if available (FullConversationHandler)
        timing_metrics = None
        if hasattr(conversation, 'timing_metrics'):
            timing_metrics = conversation.timing_metrics
        
        # Save using specified log_filename if provided
        try:
            if log_filename:
                log_path = chat_logger.save_chat(
                    conversation.get_conversation_log(),
                    diagnosis,
                    questionnaire_name=questionnaire_name,
                    metadata=metadata,
                    timing_metrics=timing_metrics,
                    log_path=log_filename
                )
                if not disable_output:
                    print(f"\nConversation saved to: {log_path}")
            else:
                log_path = chat_logger.save_chat(
                    conversation.get_conversation_log(),
                    diagnosis,
                    questionnaire_name=questionnaire_name,
                    metadata=metadata,
                    timing_metrics=timing_metrics
                )
                if not disable_output:
                    print(f"\nConversation saved to: {log_path}")
        except Exception as e:
            print(f"Error saving conversation log: {str(e)}")
    
    # Return result
    return {
        "conversation_id": f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "question_count": len(questions),
        "diagnosis": diagnosis,
        "duration": duration
    }

def process_batch_with_optimization(batch_size, questions, questionnaire_name, patient_profile=None, assistant_provider="ollama", assistant_model="qwen3:4b", 
                  patient_provider="ollama", patient_model="qwen3:4b", full_conversation=False, disable_rag=False, disable_rag_evaluation=False,
                  groq_api_key=None, openai_api_key=None, logs_dir=None, randomize_profiles=False):
    """
    Optimized batch processor that pre-initializes shared resources once instead of per conversation.
    
    Args:
        batch_size: Number of conversations to run
        questions: Pre-loaded list of questions to use (avoids reloading for each conversation)
        questionnaire_name: Name of the questionnaire
        patient_profile: Profile to use for the patient
        assistant_provider: Provider for the assistant LLM
        assistant_model: Model name for the assistant
        patient_provider: Provider for the patient LLM
        patient_model: Model name for the patient
        full_conversation: Whether to use full conversation mode
        disable_rag: Whether to disable RAG completely
        disable_rag_evaluation: Whether to disable just RAG evaluation
        groq_api_key: API key for Groq
        openai_api_key: API key for OpenAI
        logs_dir: Directory to save logs
        randomize_profiles: Whether to randomize patient profiles
        
    Returns:
        Dictionary with batch results
    """
    print(f"Starting optimized batch generation of {batch_size} conversations")
    
    # Import required modules
    import os
    import time
    import datetime
    from utils.batch_runner import BatchRunner
    from agents.patient import Patient
    from utils.debug_logger import debug_logger
    
    debug_log("Starting optimized batch processing")
    start_time = time.time()
    
    # Pre-initialize RAG engine once for all conversations
    rag_engine = None
    if not disable_rag:
        from utils.rag_engine import RAGEngine
        debug_log("Initializing shared RAG engine for batch")
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            docs_dir = os.path.join(project_root, "documents")
            rag_engine = RAGEngine(docs_dir)
            debug_log("Shared RAG engine initialized successfully")
        except Exception as e:
            debug_log(f"ERROR initializing shared RAG engine: {str(e)}")
            raise
    
    # Set up shared provider options for the assistant
    assistant_provider_options = {}
    if assistant_provider == "ollama":
        assistant_provider_options["base_url"] = "http://localhost:11434"
    elif assistant_provider == "groq":
        assistant_provider_options["api_key"] = groq_api_key
    elif assistant_provider == "openai":
        assistant_provider_options["api_key"] = openai_api_key
    
    # Set up shared provider options for the patient
    patient_provider_options = {}
    if patient_provider == "ollama":
        patient_provider_options["base_url"] = "http://localhost:11434"
    elif patient_provider == "groq":
        patient_provider_options["api_key"] = groq_api_key
    elif patient_provider == "openai":
        patient_provider_options["api_key"] = openai_api_key
    
    # Define optimized conversation runner that uses pre-loaded resources
    def optimized_conversation_runner(pdf_path, patient_profile=None, log_filename=None, **kwargs):
        """Optimized runner that skips redundant loading"""
        debug_log(f"Running optimized conversation with pre-loaded resources")
        
        # Initialize agents - we still need to do this per conversation
        try:
            if full_conversation:
                from agents.full_conversation_agent import FullConversationAgent
                from utils.full_conversation_handler import FullConversationHandler
                
                debug_log("Initializing full conversation agent with pre-loaded resources")
                agent = FullConversationAgent(
                    provider=assistant_provider,
                    provider_options=assistant_provider_options,
                    model=assistant_model,
                    patient_profile=patient_profile,
                    questions=questions,
                    rag_engine=rag_engine,
                    questionnaire_name=questionnaire_name,
                    disable_rag_evaluation=disable_rag_evaluation
                )
                conversation = FullConversationHandler(agent, disable_rag_evaluation=disable_rag_evaluation)
            else:
                from agents.mental_health_assistant import MentalHealthAssistant
                from agents.patient import Patient
                from utils.conversation_handler import ConversationHandler
                
                debug_log("Initializing assistant and patient agents with pre-loaded resources")
                assistant = MentalHealthAssistant(
                    provider=assistant_provider,
                    provider_options=assistant_provider_options,
                    model=assistant_model, 
                    questions=questions, 
                    rag_engine=rag_engine,
                    questionnaire_name=questionnaire_name
                )
                
                patient = Patient(
                    provider=patient_provider,
                    provider_options=patient_provider_options,
                    model=patient_model, 
                    profile_name=patient_profile
                )
                
                conversation = ConversationHandler(assistant, patient, disable_rag_evaluation=disable_rag_evaluation)
            
            # Run the conversation
            debug_log("Starting conversation.run() with optimized resources")
            diagnosis = conversation.run(disable_output=True)
            
            # Save the conversation log
            if log_filename:
                from utils.chat_logger import ChatLogger
                from datetime import datetime
                
                # Extract the directory from the log filename
                log_dir = os.path.dirname(log_filename)
                chat_logger = ChatLogger(log_dir)
                
                # Create metadata
                metadata = {
                    "assistant_model": assistant_model,
                    "patient_model": patient_model,
                    "patient_profile": patient_profile or "default",
                    "question_count": len(questions),
                    "duration": time.time() - start_time,
                    "full_conversation": full_conversation
                }
                
                # Extract timing metrics if available
                timing_metrics = None
                if hasattr(conversation, 'timing_metrics'):
                    timing_metrics = conversation.timing_metrics
                
                # Save the log
                log_path = chat_logger.save_chat(
                    conversation.get_conversation_log(),
                    diagnosis,
                    questionnaire_name=questionnaire_name,
                    metadata=metadata,
                    timing_metrics=timing_metrics,
                    log_path=log_filename
                )
                debug_log(f"Conversation saved to: {log_path}")
            
            # Return result
            return {
                "conversation_id": f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "question_count": len(questions),
                "diagnosis": diagnosis,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            debug_log(f"ERROR in optimized conversation runner: {str(e)}")
            import traceback
            debug_log(f"Traceback: {traceback.format_exc()}")
            raise
    
    # Create batch directory with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_id = f"batch_{timestamp}"
    
    # If logs_dir is provided, create batch_TIMESTAMP subdirectory inside it
    # Otherwise use the default chat_logs location
    if logs_dir:
        batch_dir = os.path.join(logs_dir, batch_id)
    else:
        batch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs", batch_id)
    
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create status file path
    status_file = os.path.join(batch_dir, 'batch_status.json')
    
    # Initialize batch runner with our optimized conversation runner
    batch_runner = BatchRunner(optimized_conversation_runner, batch_dir, status_file)
    debug_log("Initialized batch runner with optimized conversation runner")
    
    # Run the batch using a dummy PDF path since we've already loaded the questions
    dummy_pdf_path = "dummy_path.json"  # We won't actually load this
    try:
        debug_log(f"Running batch with optimized runner")
        results = batch_runner.run_batch(
            batch_size=batch_size,
            pdf_path=dummy_pdf_path,  # Dummy path, not used in our optimized runner
            patient_profile=patient_profile,
            randomize_profiles=randomize_profiles,
            assistant_provider=assistant_provider,
            assistant_model=assistant_model,
            patient_provider=patient_provider,
            patient_model=patient_model,
            disable_output=True,
            groq_api_key=groq_api_key,
            openai_api_key=openai_api_key,
            full_conversation=full_conversation,
            disable_rag_evaluation=disable_rag_evaluation
        )
        
        # Calculate duration
        duration = time.time() - start_time
        debug_log(f"Optimized batch completed in {duration:.2f} seconds")
        
        # Return results
        return {
            "batch_id": batch_id,
            "count": len(results),
            "duration": duration,
            "results": results,
            "batch_dir": batch_dir
        }
    except Exception as e:
        debug_log(f"ERROR during optimized batch processing: {str(e)}")
        import traceback
        debug_log(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Multi-agent Mental Health Assistant Application")
    parser.add_argument('--pdf_path', type=str, help="Path to a specific questionnaire PDF")
    parser.add_argument('--docs_dir', type=str, default=DEFAULT_DOCS_DIR, 
                        help=f"Directory containing reference documents (default: {DEFAULT_DOCS_DIR})")
    parser.add_argument('--questionnaires_dir', type=str, default=DEFAULT_QUESTIONNAIRES_DIR,
                        help=f"Directory containing questionnaires (default: {DEFAULT_QUESTIONNAIRES_DIR})")
    
    # LLM provider options
    parser.add_argument('--assistant_provider', type=str, default="ollama", choices=["ollama", "groq", "openai"],
                        help="LLM provider to use for the assistant (default: ollama)")
    parser.add_argument('--patient_provider', type=str, default="ollama", choices=["ollama", "groq", "openai"],
                        help="LLM provider to use for the patient (default: ollama)")
    parser.add_argument('--ollama_url', type=str, default="http://localhost:11434", 
                        help="URL for the Ollama API (default: http://localhost:11434)")
    parser.add_argument('--groq_api_key', type=str, help="API key for Groq (can also be set via GROQ_API_KEY env var)")
    parser.add_argument('--openai_api_key', type=str, help="API key for OpenAI (can also be set via OPENAI_API_KEY env var)")
    
    # Model options
    parser.add_argument('--assistant_model', type=str, default="qwen3:4b", 
                        help="Model to use for the assistant (default: qwen3:4b)")
    parser.add_argument('--patient_model', type=str, default="qwen3:4b", 
                        help="Model to use for the patient (default: qwen3:4b)")
    
    # Other options remain the same
    parser.add_argument('--patient_profile', type=str, help="Profile to use for the patient")
    parser.add_argument('--refresh_cache', action='store_true', help="Refresh the document cache")
    parser.add_argument('--no-save', action='store_true', help="Don't save conversation logs")
    parser.add_argument('--logs-dir', type=str, default=DEFAULT_LOGS_DIR,
                       help=f"Directory to save conversation logs (default: {DEFAULT_LOGS_DIR})")
    
    # Add batch processing arguments
    parser.add_argument('--batch', '-n', type=int, help="Number of conversations to generate in batch mode")
    parser.add_argument('--randomize-profiles', action='store_true', 
                        help="Randomize patient profiles for each conversation in batch mode")
    
    # Add state file argument for API mode
    parser.add_argument('--state-file', type=str, help="Path to a state file for API mode")
    
    # Add option to skip sentence transformer loading
    parser.add_argument('--skip-transformers', action='store_true', 
                       help="Skip loading SentenceTransformer models (faster startup)")
    
    # Add option to generate full conversation in a single LLM call
    parser.add_argument('--full_conversation', action='store_true', 
                       help="Generate the entire conversation in a single LLM call")
    
    # Add interactive mode option
    parser.add_argument('--interactive', type=str, choices=['true', 'false'], default='false',
                       help="Whether the chat is in interactive mode with a real user")
    
    # Add option to disable RAG
    parser.add_argument('--disable-rag', action='store_true',
                       help="Disable the use of RAG for retrieving information")
    
    # Add option to disable just RAG evaluation
    parser.add_argument('--disable-rag-evaluation', action='store_true',
                       help="Disable only the RAG evaluation after responses, but keep RAG document retrieval")
    
    args = parser.parse_args()
    debug_log("Starting main function with args: " + str(vars(args)))
    
    # Set environment variable to skip transformers if requested
    if args.skip_transformers:
        os.environ['SKIP_TRANSFORMERS'] = '1'
        debug_log("Setting SKIP_TRANSFORMERS=1 in environment")
    
    # Create documents and questionnaires directories if they don't exist
    os.makedirs(args.docs_dir, exist_ok=True)
    os.makedirs(args.questionnaires_dir, exist_ok=True)
    
    # Check if there are any files in the documents directory
    doc_files = [f for f in os.listdir(args.docs_dir) if os.path.isfile(os.path.join(args.docs_dir, f))]
    
    # Check specifically for questionnaire files
    questionnaire_files = []
    if os.path.exists(args.questionnaires_dir):
        questionnaire_files = [f for f in os.listdir(args.questionnaires_dir) 
                              if os.path.isfile(os.path.join(args.questionnaires_dir, f))]
    
    if not questionnaire_files:
        print(f"No questionnaire files found in {args.questionnaires_dir}.")
    else:
        print(f"Found {len(questionnaire_files)} questionnaire files: {', '.join(questionnaire_files)}")
    
    # Initialize RAG engine with both directories
    print("Initializing RAG engine and processing documents...")
    debug_log("About to initialize main RAG engine")
    try:
        rag_engine = RAGEngine(args.docs_dir, questionnaire_dir=args.questionnaires_dir)
        debug_log("Main RAG engine initialized successfully")
    except Exception as e:
        debug_log(f"ERROR initializing main RAG engine: {str(e)}")
        import traceback
        debug_log(f"Traceback: {traceback.format_exc()}")
        print(f"Error initializing RAG engine: {e}")
        sys.exit(1)
    
    # Get questionnaires
    debug_log("Getting questionnaires from RAG engine")
    questionnaires = rag_engine.get_questionnaires()
    debug_log(f"Found {len(questionnaires)} questionnaires")
    
    # Define questions variable before using it
    questions = []
    selected_name = None
    
    if not questionnaires:
        # If no questionnaires found, but PDF was specified, try to process it directly
        if args.pdf_path and os.path.exists(args.pdf_path):
            print(f"No questionnaires extracted from directory. Trying direct PDF: {args.pdf_path}")
            document = DocumentProcessor.load_document(args.pdf_path)
            if document:
                questions = extract_questions_from_text(document.content)
                if questions:
                    print(f"Successfully extracted {len(questions)} questions from {args.pdf_path}")
                    # Create a manual questionnaire
                    filename = os.path.basename(args.pdf_path)
                    selected_name = filename
                    questionnaires = {filename: questions}
        
        # If still no questionnaires, exit
        if not questionnaires:
            print("No questionnaires found. Exiting application.")
            sys.exit(1)
    else:
        # If we have questionnaires, let user choose one or use the specified one
        if args.pdf_path:
            # If specific PDF specified, use that
            filename = os.path.basename(args.pdf_path)
            if filename in questionnaires:
                selected_name = filename
                questions = questionnaires[selected_name]
                print(f"Using specified PDF: {args.pdf_path} with {len(questions)} questions")
            else:
                # Try to load it directly
                document = DocumentProcessor.load_document(args.pdf_path)
                if document:
                    questions = extract_questions_from_text(document.content)
                    if questions:
                        print(f"Using specified PDF: {args.pdf_path} with {len(questions)} questions")
                        selected_name = os.path.basename(args.pdf_path)
                    else:
                        print(f"No questions found in specified PDF: {args.pdf_path}")
                        sys.exit(1)
        else:
            # API mode or batch mode - just use first questionnaire if we're not in interactive mode
            if args.state_file or (args.batch and args.batch > 0):
                selected_name = list(questionnaires.keys())[0]
                questions = questionnaires[selected_name]
                print(f"Auto-selected: {selected_name} ({len(questions)} questions)")
            else:
                # Let user choose from available questionnaires in interactive mode
                print("\nAvailable questionnaires:")
                questionnaire_names = list(questionnaires.keys())
                
                for i, name in enumerate(questionnaire_names, 1):
                    print(f"{i}. {name} ({len(questionnaires[name])} questions)")
                
                try:
                    choice = int(input("\nSelect a questionnaire (number): "))
                    if 1 <= choice <= len(questionnaire_names):
                        selected_name = questionnaire_names[choice-1]
                        questions = questionnaires[selected_name]
                        print(f"Selected: {selected_name} ({len(questions)} questions)\n")
                    else:
                        print("Invalid selection. Using first questionnaire.")
                        selected_name = questionnaire_names[0]
                        questions = questionnaires[selected_name]
                except (ValueError, IndexError):
                    print("Invalid selection. Using first questionnaire.")
                    selected_name = questionnaire_names[0]
                    questions = questionnaires[selected_name]
    
    if not questions:
        print("No questions found in the selected questionnaire.")
        sys.exit(1)
    
    print(f"Loaded {len(questions)} questions")
    
    # Let user choose a patient profile if not specified
    patient_profile = args.patient_profile
    available_profiles = Patient.list_available_profiles()
    
    # Skip profile selection if we're in batch mode with randomized profiles
    # or if we're in API mode (state_file is specified)
    if not (args.batch and args.batch > 0 and args.randomize_profiles) and not args.state_file:
        if not patient_profile and available_profiles:
            print("\nAvailable patient profiles:")
            for i, profile in enumerate(sorted(available_profiles), 1):
                print(f"{i}. {profile}")
            
            try:
                choice = int(input("\nSelect a patient profile (number): "))
                if 1 <= choice <= len(available_profiles):
                    patient_profile = sorted(available_profiles)[choice-1]
                    print(f"Selected patient profile: {patient_profile}\n")
                else:
                    print("Invalid selection. Using default profile.")
            except (ValueError, IndexError):
                print("Invalid selection. Using default profile.")
    
    # Check if we're in batch mode
    if args.batch and args.batch > 0:
        # Initialize RAG engine
        print("Initializing RAG engine and processing documents...")
        rag_engine = None if args.disable_rag else RAGEngine(args.docs_dir)
        
        # Print a status message about loading questions
        print(f"Loading questions for batch processing...")
        
        # If we already have questions loaded, use them directly
        if 'questions' in locals() and questions:
            print(f"Using {len(questions)} pre-loaded questions from {selected_name}")
        else:
            # Otherwise we need to extract them from the specified questionnaire
            if args.pdf_path:
                print(f"Extracting questions from {args.pdf_path}...")
                # If using a specific PDF, load questions from it
                try:
                    document = DocumentProcessor.load_document(args.pdf_path)
                    if document:
                        questions = extract_questions_from_text(document.content)
                        selected_name = os.path.basename(args.pdf_path)
                except Exception as e:
                    print(f"Error loading document: {e}")
                    print("Falling back to first available questionnaire...")
                    selected_name = list(questionnaires.keys())[0]
                    questions = questionnaires[selected_name]
            else:
                # Otherwise use the first available questionnaire
                print(f"Using first available questionnaire...")
                selected_name = list(questionnaires.keys())[0]
                questions = questionnaires[selected_name]
        
        print(f"Using {len(questions)} questions for batch processing")
        
        # Set up provider options for the assistant
        assistant_provider_options = {}
        if args.assistant_provider == "ollama":
            assistant_provider_options["base_url"] = args.ollama_url
        elif args.assistant_provider == "groq":
            assistant_provider_options["api_key"] = args.groq_api_key or os.environ.get("GROQ_API_KEY")
            if not assistant_provider_options["api_key"]:
                print("Error: Groq API key required. Set with --groq_api_key or GROQ_API_KEY environment variable.")
                sys.exit(1)
        elif args.assistant_provider == "openai":
            assistant_provider_options["api_key"] = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not assistant_provider_options["api_key"]:
                print("Error: OpenAI API key required. Set with --openai_api_key or OPENAI_API_KEY environment variable.")
                sys.exit(1)
        
        # Set up provider options for the patient
        patient_provider_options = {}
        if args.patient_provider == "ollama":
            patient_provider_options["base_url"] = args.ollama_url
        elif args.patient_provider == "groq":
            patient_provider_options["api_key"] = args.groq_api_key or os.environ.get("GROQ_API_KEY")
            if not patient_provider_options["api_key"]:
                print("Error: Groq API key required. Set with --groq_api_key or GROQ_API_KEY environment variable.")
                sys.exit(1)
        elif args.patient_provider == "openai":
            patient_provider_options["api_key"] = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not patient_provider_options["api_key"]:
                print("Error: OpenAI API key required. Set with --openai_api_key or OPENAI_API_KEY environment variable.")
                sys.exit(1)
        
        print(f"Starting batch generation of {args.batch} conversations")
        if args.randomize_profiles:
            print("Using randomized patient profiles for each conversation")
        if args.full_conversation:
            print("Using full conversation mode for batch generation")
        if args.disable_rag_evaluation:
            print("RAG evaluation disabled")
        
        # Use the new optimized batch processing
        try:
            batch_result = process_batch_with_optimization(
                batch_size=args.batch,
                questions=questions,
                questionnaire_name=selected_name,
                patient_profile=args.patient_profile,
                assistant_provider=args.assistant_provider,
                assistant_model=args.assistant_model,
                patient_provider=args.patient_provider,
                patient_model=args.patient_model,
                full_conversation=args.full_conversation,
                disable_rag=args.disable_rag,
                disable_rag_evaluation=args.disable_rag_evaluation,
                groq_api_key=args.groq_api_key or os.environ.get("GROQ_API_KEY"),
                openai_api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                logs_dir=args.logs_dir,
                randomize_profiles=args.randomize_profiles
            )
            
            print(f"Batch processing completed: {batch_result['count']} conversations")
            print(f"Total duration: {batch_result['duration']:.2f} seconds")
            print(f"Average duration per conversation: {batch_result['duration'] / batch_result['count']:.2f} seconds")
            print(f"Results saved to: {batch_result['results']}")
            
        except Exception as e:
            print(f"Error during batch processing: {e}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
        
        return
    
    # Check if we're in API mode (state file is specified)
    state_file = args.state_file
    state = {"conversation": [], "status": "starting"}
    interactive_mode = False  # Default to non-interactive mode
    human_user = False  # Default to AI patient
    
    # Parse interactive mode flag
    if args.interactive.lower() == 'true':
        interactive_mode = True
    
    # If a state file is provided, read it to check for interactive mode
    if state_file:
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                # Check if we're in interactive mode (user is the patient)
                interactive_mode = state.get('interactive_mode', False)
                human_user = state.get('human_user', False)
                
                if interactive_mode:
                    debug_log("Running in interactive mode with real user as patient")
                
                if human_user:
                    debug_log("Human user is acting as the patient")
        except Exception as e:
            debug_log(f"Error reading state file: {str(e)}")
            print(f"Warning: Could not read state file: {str(e)}")

    # Initialize agents
    assistant_provider_options = {}
    if args.assistant_provider == "ollama":
        assistant_provider_options["base_url"] = args.ollama_url
    elif args.assistant_provider == "groq":
        assistant_provider_options["api_key"] = args.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not assistant_provider_options["api_key"]:
            print("Error: Groq API key required. Set with --groq_api_key or GROQ_API_KEY environment variable.")
            sys.exit(1)
    elif args.assistant_provider == "openai":
        assistant_provider_options["api_key"] = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not assistant_provider_options["api_key"]:
            print("Error: OpenAI API key required. Set with --openai_api_key or OPENAI_API_KEY environment variable.")
            sys.exit(1)
    
    patient_provider_options = {}
    if args.patient_provider == "ollama":
        patient_provider_options["base_url"] = args.ollama_url
    elif args.patient_provider == "groq":
        patient_provider_options["api_key"] = args.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not patient_provider_options["api_key"]:
            print("Error: Groq API key required. Set with --groq_api_key or GROQ_API_KEY environment variable.")
            sys.exit(1)
    elif args.patient_provider == "openai":
        patient_provider_options["api_key"] = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not patient_provider_options["api_key"]:
            print("Error: OpenAI API key required. Set with --openai_api_key or OPENAI_API_KEY environment variable.")
            sys.exit(1)
    
    if args.full_conversation:
        from agents.full_conversation_agent import FullConversationAgent
        agent = FullConversationAgent(
            provider=args.assistant_provider,
            provider_options=assistant_provider_options,
            model=args.assistant_model,
            patient_profile=patient_profile,
            questions=questions,
            rag_engine=None if args.disable_rag else rag_engine,  # Pass None if RAG is disabled
            questionnaire_name=selected_name
        )
    else:
        # Initialize agents
        assistant = MentalHealthAssistant(
            provider=args.assistant_provider,
            provider_options=assistant_provider_options,
            model=args.assistant_model, 
            questions=questions, 
            rag_engine=None if args.disable_rag else rag_engine,  # Pass None if RAG is disabled
            questionnaire_name=selected_name
        )
        patient = Patient(
            provider=args.patient_provider,
            provider_options=patient_provider_options,
            model=args.patient_model, 
            profile_name=patient_profile
        )
    
    # Initialize the chat logger
    if not args.no_save:
        # Ensure logs directory exists
        os.makedirs(args.logs_dir, exist_ok=True)
        debug_log(f"Using logs directory: {args.logs_dir}")
        chat_logger = ChatLogger(args.logs_dir)
    
    # Set up conversation handler with state tracking for API mode
    if not args.full_conversation:
        conversation = ConversationHandler(assistant, patient, state_file=state_file, disable_rag_evaluation=args.disable_rag_evaluation)
    else:
        conversation = FullConversationHandler(agent, state_file=state_file, disable_rag_evaluation=args.disable_rag_evaluation)
    
    # Run the conversation
    try:
        diagnosis = conversation.run()
        
        # Check if the diagnosis is a special signal for interactive mode
        if interactive_mode and human_user and diagnosis == "INTERACTIVE_MODE_RUNNING":
            print("\nContinuing interactive conversation through web interface...")
            # For interactive mode with human user, we just exit the script here
            # as the conversation will continue through the web interface
            return
        
        print("\n=== Final Diagnosis ===")
        print(diagnosis)
        
        # If in API mode, update the state file with the diagnosis
        if state_file:
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                state['status'] = 'completed'
                state['diagnosis'] = diagnosis
                
                with open(state_file, 'w') as f:
                    json.dump(state, f)
            except Exception as e:
                print(f"Error updating state file: {str(e)}")
        
        # Save the conversation if requested
        if not args.no_save:
            # Fix the dict_keys not being subscriptable error
            if isinstance(questionnaires, dict) and questionnaires:
                # If selected_name is not set, use the first questionnaire
                if not selected_name:
                    selected_name = list(questionnaires.keys())[0]
            else:
                selected_name = "default_questionnaire"
            
            metadata = {
                "assistant_model": args.assistant_model,
                "patient_model": args.patient_model,
                "patient_profile": patient_profile or "default",
                "question_count": len(questions),
                "full_conversation": args.full_conversation
            }
            
            # Extract timing metrics if available (FullConversationHandler)
            timing_metrics = None
            if hasattr(conversation, 'timing_metrics'):
                timing_metrics = conversation.timing_metrics
            
            log_path = chat_logger.save_chat(
                conversation.get_conversation_log(),
                diagnosis,
                questionnaire_name=selected_name,
                metadata=metadata,
                timing_metrics=timing_metrics
            )
            print(f"\nConversation saved to: {log_path}")
            print(f"You can find all conversation logs in: {chat_logger.log_dir}")
    except KeyboardInterrupt:
        print("\nConversation interrupted by user.")
        if state_file:
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                state['status'] = 'interrupted'
                
                with open(state_file, 'w') as f:
                    json.dump(state, f)
            except Exception as e:
                print(f"Error updating state file: {str(e)}")
    except Exception as e:
        print(f"\nError during conversation: {str(e)}")
        if state_file:
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                state['status'] = 'error'
                state['error'] = str(e)
                
                with open(state_file, 'w') as f:
                    json.dump(state, f)
            except Exception as err:
                print(f"Error updating state file: {str(err)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        debug_log(f"FATAL ERROR: {str(e)}")
        import traceback
        debug_log(f"Traceback: {traceback.format_exc()}")
        print(f"Error: {e}")
        sys.exit(1)
