"""
Integration service with fixed imports.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """High-level document processing service."""
    
    def __init__(self,
                 use_ocr: bool = True,
                 chunk_strategy: str = "sentence",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 collection_name: str = "documents"):
        
        # Lazy imports to avoid circular dependencies
        from .parser import DocumentParser
        from .chunker import TextChunker
        from .vectordb import DocumentVectorStore, OpenAIEmbeddingProvider
        from .config import config
        
        # Initialize components
        self.parser = DocumentParser(use_ocr=use_ocr)
        self.chunker = TextChunker(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        # Initialize embedding provider
        embedding_provider = None
        if config.openai_api_key:
            try:
                embedding_provider = OpenAIEmbeddingProvider()
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI embeddings: {e}")
        
        self.vector_store = DocumentVectorStore(
            collection_name=collection_name,
            embedding_provider=embedding_provider
        )
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single document through the complete pipeline."""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Parse document
            document = self.parser.parse(file_path)
            
            # Generate chunks
            chunks = self.chunker.chunk_document(document)
            
            # Add to vector store
            self.vector_store.add_document(document, self.chunker)
            
            result = {
                'document': document,
                'chunks': len(chunks),
                'status': 'success'
            }
            
            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {
                'document': None,
                'chunks': 0,
                'status': 'error',
                'error': str(e)
            }
