"""
Integration service for combining parsing, chunking, and vector storage.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .parser import DocumentParser
from .chunker import TextChunker
from .vectordb import DocumentVectorStore, OpenAIEmbeddingProvider
from .llm_providers import LLMManager
from .config import config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """High-level document processing service."""
    
    def __init__(self,
                 use_ocr: bool = True,
                 chunk_strategy: str = "sentence",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 collection_name: str = "documents"):
        
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
        
        # Initialize LLM manager
        self.llm_manager = LLMManager()
    
    def process_document(self, file_path: Union[str, Path], generate_summary: bool = True) -> Dict[str, Any]:
        """Process a single document through the complete pipeline."""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Parse document
            document = self.parser.parse(file_path)
            
            # Generate chunks
            chunks = self.chunker.chunk_document(document)
            
            # Add to vector store
            self.vector_store.add_document(document, self.chunker)
            
            # Generate summary if requested
            summary = None
            if generate_summary and self.llm_manager.providers:
                try:
                    summary = self.llm_manager.generate_document_summary(document)
                except Exception as e:
                    logger.warning(f"Could not generate summary: {e}")
            
            result = {
                'document': document,
                'chunks': len(chunks),
                'summary': summary,
                'status': 'success'
            }
            
            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {
                'document': None,
                'chunks': 0,
                'summary': None,
                'status': 'error',
                'error': str(e)
            }
    
    def process_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
        """Process all documents in a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        supported_formats = self.parser.supported_formats()
        files_to_process = [
            f for f in directory_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_formats
        ]
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        results = {
            'total_files': len(files_to_process),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'files': []
        }
        
        for file_path in files_to_process:
            result = self.process_document(file_path)
            results['files'].append({
                'path': str(file_path),
                'status': result['status'],
                'chunks': result['chunks'],
                'error': result.get('error')
            })
            
            if result['status'] == 'success':
                results['successful'] += 1
                results['total_chunks'] += result['chunks']
            else:
                results['failed'] += 1
        
        logger.info(f"Directory processing complete: {results['successful']}/{results['total_files']} files successful")
        return results
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search processed documents."""
        return self.vector_store.search_documents(query, n_results)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            'vector_store': vector_stats,
            'parser_formats': self.parser.supported_formats(),
            'chunker_strategy': self.chunker.strategy,
            'llm_providers': list(self.llm_manager.providers.keys()),
        }
