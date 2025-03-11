import os
from typing import List, Dict, Any, Optional
from .document_processor import Document, DocumentProcessor, process_documents_directory, extract_questions_from_text
from .vector_store import SimpleVectorStore

class RAGEngine:
    """Retrieval-Augmented Generation engine for the mental health assistant."""
    
    def __init__(self, documents_dir: str, questionnaire_dir: str = None, cache_dir: str = None):
        """
        Initialize the RAG engine.
        
        Args:
            documents_dir: Directory containing reference documents
            questionnaire_dir: Directory containing questionnaires (if None, uses documents_dir)
            cache_dir: Directory to cache embeddings
        """
        self.documents_dir = documents_dir
        self.questionnaire_dir = questionnaire_dir or documents_dir
        
        # Set default cache directory if not provided
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        
        # Create vector store
        self.vector_store = SimpleVectorStore(cache_dir=cache_dir)
        
        # Load and process all documents for RAG
        self.document_map = self._load_documents()
        
        # Load questionnaires separately
        self.questionnaire_map = self._load_questionnaires()
    
    def _load_documents(self) -> Dict[str, List[Document]]:
        """Load reference documents from the documents directory."""
        print(f"Loading reference documents from {self.documents_dir}")
        document_map = process_documents_directory(self.documents_dir)
        
        # Add all documents to vector store
        all_docs = []
        for docs in document_map.values():
            all_docs.extend(docs)
        
        self.vector_store.add_documents(all_docs)
        
        return document_map
    
    def _load_questionnaires(self) -> Dict[str, List[str]]:
        """Load questionnaires from the questionnaire directory."""
        questionnaire_map = {}
        
        print(f"Loading questionnaires from {self.questionnaire_dir}")
        
        if self.questionnaire_dir == self.documents_dir:
            # If using the same directory, we've already processed these documents
            # Just extract questions from what we have
            if '.pdf' in self.document_map:
                for document in self.document_map['.pdf']:
                    questions = extract_questions_from_text(document.content)
                    filename = document.metadata.get('filename', 'Unknown')
                    print(f"  - {filename}: Found {len(questions)} questions")
                    if questions:
                        questionnaire_map[filename] = questions
        else:
            # Process the questionnaire directory separately
            questionnaire_docs = process_documents_directory(self.questionnaire_dir)
            
            # Extract questions from all document types
            for ext, docs in questionnaire_docs.items():
                for document in docs:
                    questions = extract_questions_from_text(document.content)
                    filename = document.metadata.get('filename', 'Unknown')
                    print(f"  - {filename}: Found {len(questions)} questions")
                    if questions:
                        questionnaire_map[filename] = questions
        
        # If no questionnaires found but we have documents, try to create placeholder questions
        if not questionnaire_map and self.document_map:
            for ext, docs in self.document_map.items():
                if docs:
                    # Get first document
                    doc = docs[0]
                    filename = doc.metadata.get('filename', 'Unknown')
                    
                    # Create some default questions if we can't extract them
                    print(f"No questions detected in {filename}. Creating default questions.")
                    
                    # Split content into chunks and turn them into basic questions
                    content = doc.content
                    chunks = [content[i:i+200].replace("\n", " ").strip() 
                             for i in range(0, min(len(content), 2000), 200)]
                    
                    questions = []
                    for i, chunk in enumerate(chunks, 1):
                        if chunk:
                            # Create a question from the chunk
                            questions.append(f"Question {i}: Based on this text: '{chunk}', how do you feel?")
                    
                    if questions:
                        questionnaire_map[filename] = questions
                        break
        
        return questionnaire_map
    
    def get_questionnaires(self) -> Dict[str, List[str]]:
        """
        Get available questionnaires.
        
        Returns:
            Dictionary mapping questionnaire names to lists of questions
        """
        return self.questionnaire_map
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of relevant document contents
        """
        results = self.vector_store.search(query, top_k=top_k)
        return [doc.content for doc, _ in results]
    
    def get_context_for_question(self, question: str) -> str:
        """
        Get context for a specific question.
        
        Args:
            question: Question to get context for
            
        Returns:
            Context string
        """
        context_docs = self.retrieve(question)
        return "\n\n".join(context_docs)
    
    def refresh_documents(self) -> None:
        """Refresh documents from the documents directory."""
        self.document_map = self._load_documents()
        self.questionnaire_map = self._load_questionnaires()
