import os
import argparse
import sys
from pathlib import Path
from utils.document_processor import extract_questions_from_text, DocumentProcessor
from utils.rag_engine import RAGEngine
from agents.mental_health_assistant import MentalHealthAssistant
from agents.patient import Patient
from utils.conversation_handler import ConversationHandler

# Define default paths
DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "documents"
)

def main():
    parser = argparse.ArgumentParser(description="Multi-agent Mental Health Assistant Application")
    parser.add_argument('--pdf_path', type=str, help="Path to a specific questionnaire PDF")
    parser.add_argument('--docs_dir', type=str, default=DEFAULT_DOCS_DIR, 
                        help=f"Directory containing documents (default: {DEFAULT_DOCS_DIR})")
    parser.add_argument('--ollama_url', type=str, default="http://localhost:11434", 
                        help="URL for the Ollama API (default: http://localhost:11434)")
    parser.add_argument('--assistant_model', type=str, default="qwen2.5:3b", 
                        help="Ollama model to use for the assistant (default: qwen2.5:3b)")
    parser.add_argument('--patient_model', type=str, default="qwen2.5:3b", 
                        help="Ollama model to use for the patient (default: qwen2.5:3b)")
    parser.add_argument('--refresh_cache', action='store_true', help="Refresh the document cache")
    args = parser.parse_args()
    
    # Create documents directory if it doesn't exist
    os.makedirs(args.docs_dir, exist_ok=True)
    
    # Check if there are any files in the directory
    files = [f for f in os.listdir(args.docs_dir) if os.path.isfile(os.path.join(args.docs_dir, f))]
    if not files:
        print(f"No files found in {args.docs_dir}. Please add some documents.")
        sys.exit(1)
        
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {args.docs_dir}. Found these files: {', '.join(files)}")
        print("The application works best with PDF questionnaires.")
    else:
        print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    
    # Initialize RAG engine
    print("Initializing RAG engine and processing documents...")
    rag_engine = RAGEngine(args.docs_dir)
    
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
            print("No questionnaires found. Creating a default questionnaire.")
            default_questions = [
                "How have you been feeling emotionally over the past two weeks?",
                "Have you been experiencing difficulty sleeping? If so, please describe.",
                "Have you noticed any changes in your appetite or weight recently?",
                "How would you rate your energy levels throughout the day?",
                "Do you find yourself feeling sad, down, or hopeless?",
                "How is your concentration when trying to perform tasks or activities?",
                "Do you find yourself feeling anxious, nervous, or on edge?",
                "Have you experienced any unusual thoughts or perceptions?",
                "Have your symptoms affected your daily activities or work?",
                "Do you have thoughts of harming yourself or others?"
            ]
            questions = default_questions
            questionnaires = {"default_questionnaire.pdf": questions}
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
    
    # Initialize agents
    assistant = MentalHealthAssistant(args.ollama_url, args.assistant_model, questions, rag_engine)
    patient = Patient(args.ollama_url, args.patient_model)
    
    # Set up conversation handler
    conversation = ConversationHandler(assistant, patient)
    
    # Run the conversation
    diagnosis = conversation.run()
    
    print("\n=== Final Diagnosis ===")
    print(diagnosis)

if __name__ == "__main__":
    main()
