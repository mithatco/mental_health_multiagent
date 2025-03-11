import os
import argparse
import sys
from pathlib import Path
from utils.document_processor import extract_questions_from_text, DocumentProcessor
from utils.rag_engine import RAGEngine
from agents.mental_health_assistant import MentalHealthAssistant
from agents.patient import Patient
from utils.conversation_handler import ConversationHandler
from utils.chat_logger import ChatLogger
from utils.batch_processor import BatchProcessor

# Define default paths
DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "documents"
)
DEFAULT_QUESTIONNAIRES_DIR = os.path.join(DEFAULT_DOCS_DIR, "questionnaires")

def main():
    parser = argparse.ArgumentParser(description="Multi-agent Mental Health Assistant Application")
    parser.add_argument('--pdf_path', type=str, help="Path to a specific questionnaire PDF")
    parser.add_argument('--docs_dir', type=str, default=DEFAULT_DOCS_DIR, 
                        help=f"Directory containing reference documents (default: {DEFAULT_DOCS_DIR})")
    parser.add_argument('--questionnaires_dir', type=str, default=DEFAULT_QUESTIONNAIRES_DIR,
                        help=f"Directory containing questionnaires (default: {DEFAULT_QUESTIONNAIRES_DIR})")
    parser.add_argument('--ollama_url', type=str, default="http://localhost:11434", 
                        help="URL for the Ollama API (default: http://localhost:11434)")
    parser.add_argument('--assistant_model', type=str, default="qwen2.5:3b", 
                        help="Ollama model to use for the assistant (default: qwen2.5:3b)")
    parser.add_argument('--patient_model', type=str, default="qwen2.5:3b", 
                        help="Ollama model to use for the patient (default: qwen2.5:3b)")
    parser.add_argument('--patient_profile', type=str, help="Profile to use for the patient")
    parser.add_argument('--refresh_cache', action='store_true', help="Refresh the document cache")
    parser.add_argument('--no-save', action='store_true', help="Don't save conversation logs")
    parser.add_argument('--logs-dir', type=str, help="Directory to save conversation logs")
    
    # Add batch processing arguments
    parser.add_argument('--batch', '-n', type=int, help="Number of conversations to generate in batch mode")
    parser.add_argument('--randomize-profiles', action='store_true', 
                        help="Randomize patient profiles for each conversation in batch mode")
    
    args = parser.parse_args()
    
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
    rag_engine = RAGEngine(args.docs_dir, questionnaire_dir=args.questionnaires_dir)
    
    # Get questionnaires
    questionnaires = rag_engine.get_questionnaires()
    
    # Define questions variable before using it
    questions = []
    
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
                    questionnaires = {filename: questions}
        
        # If still no questionnaires, create a default one
        if not questionnaires:
            print("No questionnaires found. Exiting application.")
            sys.exit(1)
            # print("No questionnaires found. Creating a default questionnaire.")
            # default_questions = [
            #     "How have you been feeling emotionally over the past two weeks?",
            #     "Have you been experiencing difficulty sleeping? If so, please describe.",
            #     "Have you noticed any changes in your appetite or weight recently?",
            #     "How would you rate your energy levels throughout the day?",
            #     "Do you find yourself feeling sad, down, or hopeless?",
            #     "How is your concentration when trying to perform tasks or activities?",
            #     "Do you find yourself feeling anxious, nervous, or on edge?",
            #     "Have you experienced any unusual thoughts or perceptions?",
            #     "Have your symptoms affected your daily activities or work?",
            #     "Do you have thoughts of harming yourself or others?"
            # ]
            # questions = default_questions
            # questionnaires = {"default_questionnaire.pdf": questions}
    else:
        # If we have questionnaires, let user choose one
        if args.pdf_path:
            # If specific PDF specified, use that
            document = DocumentProcessor.load_document(args.pdf_path)
            if document:
                questions = extract_questions_from_text(document.content)
                print(f"Using specified PDF: {args.pdf_path} with {len(questions)} questions")
        else:
            # Let user choose from available questionnaires
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
        rag_engine = RAGEngine(args.docs_dir)
        
        # Process using batch mode
        batch_processor = BatchProcessor(
            args.ollama_url,
            args.assistant_model,
            args.patient_model,
            rag_engine,
            args.logs_dir
        )
        
        print(f"Starting batch generation of {args.batch} conversations")
        if args.randomize_profiles:
            print("Using randomized patient profiles for each conversation")
        
        batch_processor.process_batch(
            questions,
            count=args.batch,
            profile=args.patient_profile,
            randomize_profiles=args.randomize_profiles
        )
        
        return
    
    # Initialize agents
    assistant = MentalHealthAssistant(args.ollama_url, args.assistant_model, questions, rag_engine)
    patient = Patient(args.ollama_url, args.patient_model, patient_profile)
    
    # Initialize the chat logger
    if not args.no_save:
        chat_logger = ChatLogger(args.logs_dir)
    
    # Set up conversation handler
    conversation = ConversationHandler(assistant, patient)
    
    # Run the conversation
    diagnosis = conversation.run()
    
    print("\n=== Final Diagnosis ===")
    print(diagnosis)
    
    # Save the conversation if requested
    if not args.no_save:
        # Fix the dict_keys not being subscriptable error
        if isinstance(questionnaires, dict) and questionnaires:
            # Convert dict_keys to list first, then access by index
            selected_name = list(questionnaires.keys())[0]
        else:
            selected_name = "default_questionnaire"
        
        metadata = {
            "assistant_model": args.assistant_model,
            "patient_model": args.patient_model,
            "patient_profile": patient_profile or "default",
            "question_count": len(questions)
        }
        log_path = chat_logger.save_chat(
            conversation.get_conversation_log(),
            diagnosis,
            questionnaire_name=selected_name,
            metadata=metadata
        )
        print(f"\nConversation saved to: {log_path}")
        print(f"You can find all conversation logs in: {chat_logger.log_dir}")

if __name__ == "__main__":
    main()
