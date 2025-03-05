import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import requests

from .document_processor import Document, Chunker

class SimpleVectorStore:
    """A simple vector store implementation that doesn't require external dependencies."""
    
    def __init__(self, embedding_dim: int = 384, cache_dir: str = None):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            cache_dir: Directory to cache embeddings
        """
        self.documents = []
        self.embeddings = []
        self.embedding_dim = embedding_dim
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def add_documents(self, documents: List[Document], chunk_size: int = 1000) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            chunk_size: Size of chunks to split documents into
        """
        # Chunk documents
        chunker = Chunker()
        all_chunks = []
        
        for doc in documents:
            chunks = chunker.chunk_document(doc, chunk_size=chunk_size)
            all_chunks.extend(chunks)
        
        # Compute embeddings for chunks
        for chunk in all_chunks:
            embedding = self._get_embedding(chunk.content)
            
            if embedding is not None:
                self.documents.append(chunk)
                self.embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.documents:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        if query_embedding is None:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, similarity in similarities[:top_k]:
            results.append((self.documents[i], similarity))
        
        return results
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text using Ollama API.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector or None if embedding failed
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{hash(text)}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass  # If loading from cache fails, compute embedding
        
        # If no cache, compute embedding with Ollama
        try:
            # Try to use the embedding model
            embedding_model = "nomic-embed-text:latest"
            
            # First check if model exists
            try:
                check_response = requests.get("http://localhost:11434/api/tags")
                if check_response.status_code == 200:
                    available_models = [model["name"] for model in check_response.json()["models"]]
                    
                    # If the embedding model isn't available, use a fallback model
                    if embedding_model not in available_models:
                        print(f"Warning: '{embedding_model}' not found. Checking for alternative embedding models...")
                        
                        # Check for any embedding model variants
                        for model in available_models:
                            if "embed" in model.lower():
                                embedding_model = model
                                print(f"Using alternative embedding model: {embedding_model}")
                                break
                        else:
                            # If no embedding model is available, use any available model
                            if available_models:
                                embedding_model = available_models[0]
                                print(f"No embedding models found. Using general model: {embedding_model}")
                            else:
                                raise ValueError("No models available in Ollama")
            except Exception as e:
                print(f"Error checking available models: {str(e)}")
                # Continue with the default model and let it fail if necessary
            
            # Use the selected model to generate embedding
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": embedding_model, "prompt": text}
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json()["embedding"])
                
                # Cache the embedding
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
                
                return embedding
            else:
                print(f"Error getting embedding: {response.text}")
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
        
        # Fallback to random embedding if all else fails
        print("Using fallback random embedding")
        fallback = np.random.rand(self.embedding_dim)
        fallback = fallback / np.linalg.norm(fallback)  # Normalize
        return fallback
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def save(self, file_path: str) -> None:
        """Save the vector store to a file."""
        data = {
            'documents': self.documents,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'embedding_dim': self.embedding_dim
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'SimpleVectorStore':
        """Load a vector store from a file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        store = cls(embedding_dim=data['embedding_dim'])
        store.documents = data['documents']
        store.embeddings = [np.array(emb) for emb in data['embeddings']]
        
        return store
