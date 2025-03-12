import sys
import os
from pathlib import Path
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.document_processor import DocumentProcessor, extract_questions_from_text

def debug_extraction(file_path):
    """Load a document and print the extracted questions for debugging."""
    document = DocumentProcessor.load_document(file_path)
    
    if not document:
        print(f"Error: Could not load document from {file_path}")
        return
        
    print(f"\nDocument: {file_path}")
    print(f"Content length: {len(document.content)} characters")
    print("First 200 characters of content:")
    print(document.content[:200])
    
    questions = extract_questions_from_text(document.content)
    
    print(f"\nExtracted {len(questions)} questions:")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    if not questions:
        print("No questions were extracted.")
        print("\nHere's the full document content for inspection:")
        print("="*80)
        print(document.content)
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Debug question extraction from documents")
    parser.add_argument("file_path", help="Path to the document file to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} does not exist")
        return
        
    debug_extraction(args.file_path)

if __name__ == "__main__":
    main()
