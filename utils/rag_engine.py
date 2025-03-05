import os
from typing import List, Dict, Any, Optional
from .document_processor import Document, DocumentProcessor, process_documents_directory, extract_questions_from_text
from .vector_store import SimpleVectorStore

class RAGEngine:
    """Retrieval-Augmented Generation engine for the mental health assistant."""
    
    def __init__(self, documents_dir: str, cache_dir: str = None):
        """
        Initialize the RAG engine.
        
        Args:
            documents_dir: Directory containing documents
            cache_dir: Directory to cache embeddings
        """
        self.documents_dir = documents_dir
        
        # Set default cache directory if not provided
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        
        # Create vector store
        self.vector_store = SimpleVectorStore(cache_dir=cache_dir)
        
        # Load and process documents
        self.document_map = self._load_documents()
        
        # Create a mapping of questionnaire names to questions
        self.questionnaire_map = self._extract_questionnaires()
    
    def _load_documents(self) -> Dict[str, List[Document]]:
        """Load documents from the documents directory."""
        document_map = process_documents_directory(self.documents_dir)
        
        # Add all documents to vector store
        all_docs = []
        for docs in document_map.values():
            all_docs.extend(docs)
        
        self.vector_store.add_documents(all_docs)
        
        return document_map
    
    def _extract_questionnaires(self) -> Dict[str, List[str]]:
        """Extract questionnaires from the documents."""
        questionnaire_map = {}
        
        # First check all document types, not just PDFs
        for ext, docs in self.document_map.items():
            print(f"Processing documents with extension: {ext}, count: {len(docs)}")
            
            for document in docs:
                questions = extract_questions_from_text(document.content)
                filename = document.metadata.get('filename', 'Unknown')
                print(f"  - {filename}: Found {len(questions)} questions")
                
                if questions:
                    questionnaire_map[filename] = questions
        
        # If no questionnaires found, but we have documents, use the first one
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
        self.questionnaire_map = self._extract_questionnaires()
