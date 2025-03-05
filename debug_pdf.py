#!/usr/bin/env python3
"""
PDF Debugging Tool

This script helps diagnose issues with PDF processing in the mental health multi-agent system.
It will analyze a PDF file and show extracted text and questions.
"""

import os
import sys
import argparse
from pathlib import Path
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file with page numbers."""
    text_by_page = {}
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                text_by_page[page_num + 1] = text
                
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return {}
    
    return text_by_page

def find_questions(text):
    """Find potential questions in text using various methods."""
    results = {
        "question_marks": [],
        "numbered_items": [],
        "potential_questions": []
    }
    
    # Method 1: Lines ending with question mark
    for line in text.split('\n'):
        line = line.strip()
        if line.endswith('?'):
            results["question_marks"].append(line)
    
    # Method 2: Numbered items
    import re
    numbered_pattern = re.compile(r'(\d+\.\s*[^.!?\n]+)')
    results["numbered_items"] = [item.strip() for item in numbered_pattern.findall(text)]
    
    # Method 3: Any sentence with a question mark
    question_pattern = re.compile(r'([^.!?\n]+\?)')
    results["potential_questions"] = [q.strip() for q in question_pattern.findall(text)]
    
    return results

def main():
    parser = argparse.ArgumentParser(description="PDF Debugging Tool")
    parser.add_argument('pdf_path', help="Path to the PDF file to analyze")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)
        
    if not args.pdf_path.lower().endswith('.pdf'):
        print(f"Warning: File does not appear to be a PDF: {args.pdf_path}")
    
    print(f"Analyzing PDF: {args.pdf_path}")
    print("-" * 50)
    
    # Extract text by page
    text_by_page = extract_text_from_pdf(args.pdf_path)
    
    if not text_by_page:
        print("Failed to extract any text from the PDF.")
        print("This might indicate the PDF is encrypted, damaged, or contains only images.")
        print("Try using OCR software to convert it to a text-searchable PDF.")
        sys.exit(1)
    
    # Print basic stats
    print(f"Successfully extracted text from {len(text_by_page)} pages.")
    total_length = sum(len(text) for text in text_by_page.values())
    print(f"Total text length: {total_length} characters")
    
    # Analyze each page
    all_text = ""
    for page_num, text in text_by_page.items():
        all_text += text
        print(f"\nPage {page_num} - {len(text)} characters:")
        print("-" * 30)
        print(f"Preview: {text[:200]}..." if len(text) > 200 else text)
        
        # Find questions on this page
        questions = find_questions(text)
        if questions["question_marks"]:
            print(f"\n  Questions ending with '?' on page {page_num}:")
            for q in questions["question_marks"]:
                print(f"  - {q}")
                
        if questions["numbered_items"]:
            print(f"\n  Numbered items on page {page_num}:")
            for q in questions["numbered_items"]:
                print(f"  - {q}")
    
    # Overall analysis
    print("\n" + "=" * 50)
    print("OVERALL ANALYSIS")
    print("=" * 50)
    
    all_questions = find_questions(all_text)
    
    # Print all potential questions
    all_potential = set(all_questions["question_marks"] + 
                        all_questions["potential_questions"])
    
    print(f"\nFound {len(all_potential)} potential questions in the document:")
    for i, q in enumerate(all_potential, 1):
        print(f"{i}. {q}")
    
    # Print recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    if not all_potential:
        print("No questions were found in this PDF. Possible solutions:")
        print("1. The PDF might be image-based. Try using OCR software.")
        print("2. Questions might not end with question marks. Check the numbered items.")
        print("3. Try creating a text file with your questions instead.")
    else:
        print(f"Found {len(all_potential)} questions. Copy them to a text file if needed.")
    
    # Check if PDF might be scanned/image-based
    if total_length < 100 * len(text_by_page):  # Very little text per page
        print("\nThis PDF appears to contain very little text and might be image-based.")
        print("Consider using OCR software to convert it to a text-searchable PDF.")

if __name__ == "__main__":
    main()
