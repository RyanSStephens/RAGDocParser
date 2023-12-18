"""
Vector database integration for storing and retrieving document embeddings.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .chunker import Chunk
from .config import config

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_texts([query])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAI embeddings")
        
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        self.model = model
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise


class VectorDatabase:
    """Vector database interface using ChromaDB."""
    
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: Optional[str] = None,
                 embedding_provider: Optional[EmbeddingProvider] = None):
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb package is required for vector database functionality")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory or config.chroma_persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding provider
        self.embedding_provider = embedding_provider
        if self.embedding_provider is None and OPENAI_AVAILABLE and config.openai_api_key:
            try:
                self.embedding_provider = OpenAIEmbeddingProvider()
                logger.info("Using OpenAI embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            if self.embedding_provider:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Document chunks with embeddings"}
                )
            else:
                # Use ChromaDB's default embedding function
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Document chunks with default embeddings"}
                )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector database."""
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for chunk in chunks:
            # Generate unique ID if not present
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            
            documents.append(chunk.content)
            metadatas.append(chunk.metadata or {})
            ids.append(chunk_id)
        
        try:
            # Generate embeddings if provider is available
            if self.embedding_provider:
                logger.info(f"Generating embeddings for {len(documents)} chunks...")
                embeddings = self.embedding_provider.embed_texts(documents)
                
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                # Let ChromaDB handle embeddings
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector database: {e}")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search for similar chunks."""
        try:
            # Generate query embedding if provider is available
            if self.embedding_provider:
                query_embedding = self.embedding_provider.embed_query(query)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            else:
                # Use ChromaDB's default embedding
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'persist_directory': self.persist_directory,
                'embedding_provider': type(self.embedding_provider).__name__ if self.embedding_provider else 'ChromaDB Default'
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all data)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with embeddings"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise


class DocumentVectorStore:
    """High-level interface for document vector storage."""
    
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: Optional[str] = None,
                 embedding_provider: Optional[EmbeddingProvider] = None):
        
        self.vector_db = VectorDatabase(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_provider=embedding_provider
        )
    
    def add_document(self, document: Dict[str, Any], chunker) -> List[Chunk]:
        """Add a document to the vector store."""
        # Chunk the document
        chunks = chunker.chunk_document(document)
        
        if not chunks:
            logger.warning(f"No chunks generated for document: {document.get('file_path', 'unknown')}")
            return []
        
        # Add chunks to vector database
        self.vector_db.add_chunks(chunks)
        
        logger.info(f"Added document with {len(chunks)} chunks: {document.get('file_path', 'unknown')}")
        return chunks
    
    def search_documents(self, 
                        query: str, 
                        n_results: int = 5,
                        file_type: Optional[str] = None,
                        file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        where_filter = {}
        
        if file_type:
            where_filter['file_type'] = file_type
        
        if file_path:
            where_filter['file_path'] = file_path
        
        where = where_filter if where_filter else None
        
        results = self.vector_db.search(
            query=query,
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results.get('documents'):
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'distance': results['distances'][0][i] if results.get('distances') else None,
                    'id': results['ids'][0][i] if results.get('ids') else None,
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_db.get_collection_info()
