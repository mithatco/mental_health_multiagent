import os
from pathlib import Path
import PyPDF2
import docx
import re
import json
from typing import List, Dict, Any, Optional, Union

class Document:
    """Represents a document with content and metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(metadata={self.metadata}, content={self.content[:50]}...)"

class DocumentProcessor:
    """Process and extract text from various document types."""
    
    @staticmethod
    def load_document(file_path: str) -> Optional[Document]:
        """
        Load a document from a file path.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document object or None if file type is not supported
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Get basic metadata
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "extension": file_ext,
            "created": os.path.getctime(file_path),
            "modified": os.path.getmtime(file_path),
        }
        
        # Process different file types
        if file_ext == '.pdf':
            content = DocumentProcessor._extract_from_pdf(file_path)
        elif file_ext == '.txt':
            content = DocumentProcessor._extract_from_txt(file_path)
        elif file_ext in ('.docx', '.doc'):
            content = DocumentProcessor._extract_from_docx(file_path)
        elif file_ext == '.json':
            content, meta = DocumentProcessor._extract_from_json(file_path)
            metadata.update(meta)
        else:
            print(f"Unsupported file type: {file_ext}")
            return None
        
        if not content:
            print(f"Warning: No content extracted from {file_path}")
            
        return Document(content, metadata)
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
        
        return text
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Extract text from a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error extracting text from TXT {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = docx.Document(file_path)
            content = [paragraph.text for paragraph in doc.paragraphs]
            return '\n'.join(content)
        except Exception as e:
            print(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def _extract_from_json(file_path: str) -> tuple:
        """Extract text and metadata from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Assuming JSON has 'content' field, or convert the whole JSON to string
            content = data.get('content', json.dumps(data))
            
            # Extract any additional metadata
            metadata = {k: v for k, v in data.items() if k != 'content'}
            
            return content, metadata
        except Exception as e:
            print(f"Error extracting text from JSON {file_path}: {str(e)}")
            return "", {}

class Chunker:
    """Split documents into smaller chunks for processing."""
    
    @staticmethod
    def chunk_document(document: Document, 
                      chunk_size: int = 1000, 
                      overlap: int = 200) -> List[Document]:
        """
        Split a document into overlapping chunks.
        
        Args:
            document: Document object to chunk
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of Document objects (chunks)
        """
        content = document.content
        chunks = []
        
        # For very short documents, don't chunk
        if len(content) <= chunk_size:
            return [document]
        
        # Split into chunks
        start = 0
        chunk_id = 0
        
        while start < len(content):
            # Calculate end position
            end = min(start + chunk_size, len(content))
            
            # If not the first chunk, include overlap
            if start > 0:
                start = max(0, start - overlap)
            
            # Try to end chunks at sentence boundaries when possible
            if end < len(content):
                # Look for sentence end within the last 100 chars of the chunk
                search_zone = content[max(end-100, 0):end]
                sentence_ends = [m.end() + max(end-100, 0) for m in re.finditer(r'[.!?]\s+', search_zone)]
                
                if sentence_ends:
                    # Use the last sentence end as the chunk end
                    end = sentence_ends[-1]
            
            # Create chunk with same metadata + chunk-specific metadata
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_start": start,
                "chunk_end": end,
                "is_chunk": True
            })
            
            chunk_document = Document(content[start:end], chunk_metadata)
            chunks.append(chunk_document)
            
            # Move to next chunk
            start = end
            chunk_id += 1
        
        return chunks

def extract_questions_from_text(text: str) -> List[str]:
    """
    Extract questions from text content.
    
    Args:
        text: Text content to extract questions from
        
    Returns:
        List of questions found in the text
    """
    questions = []
    
    # Split text into lines
    for line in text.split('\n'):
        line = line.strip()
        # Find lines that end with question marks
        if line.endswith('?'):
            questions.append(line)
    
    # If no questions found with simple method, try more aggressive regex approach
    if not questions:
        # Use regex to find questions (any text ending with a question mark)
        question_pattern = re.compile(r'([^.!?\n]+\?)')
        additional_questions = question_pattern.findall(text)
        questions.extend([q.strip() for q in additional_questions if q.strip()])
    
    # If still no questions, try to split by numbers (common in questionnaires)
    if not questions:
        # Look for numbered items which might be questions
        numbered_pattern = re.compile(r'(\d+\.\s*[^.!?\n]+)')
        numbered_items = numbered_pattern.findall(text)
        questions.extend([item.strip() for item in numbered_items if item.strip()])
    
    # If still nothing, just split by newlines and take non-empty lines
    # This is a fallback so we at least have something to work with
    if not questions:
        questions = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 20]
        # Limit to 20 items maximum in this case
        questions = questions[:20]
    
    return questions

def process_documents_directory(directory_path: str) -> Dict[str, List[Document]]:
    """
    Process all documents in a directory.
    
    Args:
        directory_path: Path to the directory containing documents
        
    Returns:
        Dictionary mapping file types to lists of Document objects
    """
    document_map = {}
    print(f"Scanning directory: {directory_path}")
    
    try:
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f))]
        
        print(f"Found {len(files)} files in directory")
        
        for file_path in files:
            print(f"Processing file: {file_path}")
            document = DocumentProcessor.load_document(file_path)
            if document:
                ext = Path(file_path).suffix.lower()
                if ext not in document_map:
                    document_map[ext] = []
                document_map[ext].append(document)
                print(f"Added document with ext {ext}, content length: {len(document.content)}")
            else:
                print(f"Failed to load document: {file_path}")
    
    except Exception as e:
        print(f"Error processing directory {directory_path}: {str(e)}")
    
    return document_map

def get_questionnaires_from_documents(documents: List[Document]) -> Dict[str, List[str]]:
    """
    Extract questionnaires from documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Dictionary mapping document names to lists of questions
    """
    questionnaire_map = {}
    
    for document in documents:
        questions = extract_questions_from_text(document.content)
        if questions:
            name = document.metadata.get("filename", "Unknown")
            questionnaire_map[name] = questions
    
    return questionnaire_map
