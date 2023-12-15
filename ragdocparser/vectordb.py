"""
Vector database integration for RAG systems.
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import asdict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .chunker import TextChunk
from .config import Config

class VectorDBInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def create_collection(self, name: str, **kwargs):
        """Create a new collection."""
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk], collection_name: str = None):
        """Add text chunks to the database."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str):
        """Delete a collection."""
        pass

class ChromaDBManager(VectorDBInterface):
    """ChromaDB implementation for vector storage."""
    
    def __init__(self, config: Config):
        """Initialize ChromaDB manager.
        
        Args:
            config: Configuration object
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
            
        self.config = config
        self.db_path = config.get("vector_db.db_path", "./chroma_db")
        self.embedding_model = config.get("vector_db.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.default_collection = config.get("vector_db.collection_name", "ragdocparser")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self._collections = {}
    
    def create_collection(self, name: str, **kwargs):
        """Create a new ChromaDB collection."""
        try:
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            self._collections[name] = collection
            return collection
        except Exception as e:
            # Collection might already exist
            collection = self.client.get_collection(name)
            self._collections[name] = collection
            return collection
    
    def get_collection(self, name: str = None):
        """Get or create collection."""
        collection_name = name or self.default_collection
        
        if collection_name not in self._collections:
            try:
                self._collections[collection_name] = self.client.get_collection(collection_name)
            except:
                self._collections[collection_name] = self.create_collection(collection_name)
        
        return self._collections[collection_name]
    
    def add_chunks(self, chunks: List[TextChunk], collection_name: str = None):
        """Add text chunks to ChromaDB."""
        if not chunks:
            return
        
        collection = self.get_collection(collection_name)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(chunk.content)
            
            # Use existing embedding or generate placeholder
            if chunk.embedding:
                embeddings.append(chunk.embedding)
            else:
                # Generate a placeholder embedding if none exists
                # In practice, embeddings should be generated before storing
                                 # Skip chunks without embeddings in production
                 continue
            
            # Prepare metadata
            metadata = chunk.metadata.copy()
            metadata.update({
                "token_count": chunk.token_count,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index
            })
            metadatas.append(metadata)
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(self, 
               query: str, 
               k: int = 5, 
               collection_name: str = None,
               embedding: List[float] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks in ChromaDB."""
        collection = self.get_collection(collection_name)
        
        if embedding:
            # Search using provided embedding
            results = collection.query(
                query_embeddings=[embedding],
                n_results=k
            )
        else:
            # Search using query text (ChromaDB will generate embedding)
            results = collection.query(
                query_texts=[query],
                n_results=k
            )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def delete_collection(self, name: str):
        """Delete a ChromaDB collection."""
        try:
            self.client.delete_collection(name)
            if name in self._collections:
                del self._collections[name]
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")
    
    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """Get statistics about a collection."""
        collection = self.get_collection(collection_name)
        
        try:
            count = collection.count()
            return {
                "document_count": count,
                "collection_name": collection_name or self.default_collection
            }
        except Exception as e:
            return {"error": str(e)}

class VectorDBManager:
    """Main vector database manager that can work with different backends."""
    
    def __init__(self, config: Config):
        """Initialize vector DB manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        provider = config.get("vector_db.provider", "chromadb").lower()
        
        if provider == "chromadb":
            self.db = ChromaDBManager(config)
        else:
            raise ValueError(f"Unsupported vector DB provider: {provider}")
    
    def add_documents(self, 
                     chunks: List[TextChunk], 
                     collection_name: str = None,
                     generate_embeddings: bool = True):
        """Add documents to vector database.
        
        Args:
            chunks: List of text chunks to add
            collection_name: Name of collection to add to
            generate_embeddings: Whether to generate embeddings if missing
        """
        if generate_embeddings:
            # Check if any chunks are missing embeddings
            missing_embeddings = [c for c in chunks if c.embedding is None]
            if missing_embeddings:
                from .chunker import TextChunker
                chunker = TextChunker()
                chunker.generate_embeddings(missing_embeddings, self.config.get("vector_db.embedding_model"))
        
        self.db.add_chunks(chunks, collection_name)
    
    def search_similar(self, 
                      query: str, 
                      k: int = 5, 
                      collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            collection_name: Collection to search in
            
        Returns:
            List of similar documents with metadata
        """
        return self.db.search(query, k, collection_name)
    
    def create_collection(self, name: str, **kwargs):
        """Create a new collection."""
        return self.db.create_collection(name, **kwargs)
    
    def delete_collection(self, name: str):
        """Delete a collection."""
        self.db.delete_collection(name)
    
    def get_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """Get collection statistics."""
        if hasattr(self.db, 'get_collection_stats'):
            return self.db.get_collection_stats(collection_name)
        return {"error": "Stats not available for this provider"} 