import PyPDF2
import re
import os
from pathlib import Path

def extract_questions_from_pdf(pdf_path):
    """
    Extract questions from a PDF file containing a mental health questionnaire.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of questions extracted from the PDF
    """
    questions = []
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                # Simple parsing - assuming one question per line that ends with a question mark
                # This might need adjustment based on the actual PDF format
                for line in text.split('\n'):
                    line = line.strip()
                    if line.endswith('?'):
                        questions.append(line)
                    
                    # Alternative: extract questions using regex
                    # matches = re.findall(r'\d+\.\s+(.*?\?)', text)
                    # questions.extend(matches)
    
    except Exception as e:
        print(f"Error extracting questions from PDF: {str(e)}")
        
    return questions

def scan_pdf_directory(directory_path):
    """
    Scan a directory for PDF files.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        list: List of PDF file paths
    """
    pdf_files = []
    
    try:
        for file in os.listdir(directory_path):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(directory_path, file))
    except Exception as e:
        print(f"Error scanning directory: {str(e)}")
    
    return pdf_files

def get_available_questionnaires(pdf_directory):
    """
    Get a list of available questionnaire PDFs with their names.
    
    Args:
        pdf_directory (str): Path to the PDF directory
        
    Returns:
        dict: Dictionary mapping questionnaire names to file paths
    """
    questionnaires = {}
    
    pdf_files = scan_pdf_directory(pdf_directory)
    for pdf_path in pdf_files:
        # Get the file name without extension as the questionnaire name
        name = Path(pdf_path).stem
        questionnaires[name] = pdf_path
    
    return questionnaires
