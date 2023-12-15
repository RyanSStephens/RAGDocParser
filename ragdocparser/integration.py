"""
Integration module for RAGDocParser.
Provides high-level interfaces to combine all components for complete document processing workflows.
"""

import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import RAGConfig
from .parser import DocumentParser
from .chunker import TextChunker, TextChunk
from .vectordb import ChromaDBManager
from .llm_providers import RAGManager, OpenAIProvider, AnthropicProvider, CohereProvider
from .ocr import OCRProcessor
from .scraper import DocumentationScraper

logger = logging.getLogger(__name__)

class RAGDocumentProcessor:
    """Complete RAG document processing pipeline."""
    
    def __init__(self, config: Union[str, Dict, RAGConfig] = None):
        """Initialize the RAG document processor.
        
        Args:
            config: Configuration file path, dict, or RAGConfig object
        """
        if isinstance(config, str):
            self.config = RAGConfig.from_file(config)
        elif isinstance(config, dict):
            self.config = RAGConfig.from_dict(config)
        elif isinstance(config, RAGConfig):
            self.config = config
        else:
            self.config = RAGConfig()
        
        # Initialize components
        self.document_parser = DocumentParser(self.config)
        self.text_chunker = TextChunker(self.config)
        self.vectordb = ChromaDBManager(self.config)
        self.ocr_processor = None
        self.scraper = None
        self.rag_manager = None
        
        # Initialize optional components based on config
        self._init_optional_components()
    
    def _init_optional_components(self):
        """Initialize optional components based on configuration."""
        
        # Initialize OCR if configured
        if getattr(self.config, 'ocr_enabled', False):
            try:
                self.ocr_processor = OCRProcessor(
                    use_gpu=getattr(self.config, 'ocr_use_gpu', False),
                    languages=getattr(self.config, 'ocr_languages', ['en'])
                )
                logger.info("OCR processor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OCR processor: {e}")
        
        # Initialize scraper if configured
        if getattr(self.config, 'scraper_enabled', False):
            try:
                self.scraper = DocumentationScraper(
                    max_concurrent=getattr(self.config, 'scraper_max_concurrent', 5)
                )
                logger.info("Documentation scraper initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize scraper: {e}")
        
        # Initialize LLM provider and RAG manager
        self._init_llm_provider()
    
    def _init_llm_provider(self):
        """Initialize LLM provider based on configuration."""
        provider_type = getattr(self.config, 'llm_provider', 'openai').lower()
        
        try:
            if provider_type == 'openai':
                provider = OpenAIProvider(
                    api_key=getattr(self.config, 'openai_api_key', None),
                    model=getattr(self.config, 'openai_model', 'gpt-3.5-turbo')
                )
            elif provider_type == 'anthropic':
                provider = AnthropicProvider(
                    api_key=getattr(self.config, 'anthropic_api_key', None),
                    model=getattr(self.config, 'anthropic_model', 'claude-2.1')
                )
            elif provider_type == 'cohere':
                provider = CohereProvider(
                    api_key=getattr(self.config, 'cohere_api_key', None),
                    model=getattr(self.config, 'cohere_model', 'command')
                )
            else:
                logger.warning(f"Unknown LLM provider: {provider_type}")
                return
            
            self.rag_manager = RAGManager(provider, self.vectordb)
            logger.info(f"RAG manager initialized with {provider_type} provider")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider {provider_type}: {e}")
    
    def process_documents(self, 
                         source_paths: List[Union[str, Path]],
                         collection_name: str = "documents",
                         use_ocr: bool = None) -> Dict[str, Any]:
        """Process multiple documents and store in vector database.
        
        Args:
            source_paths: List of file/directory paths to process
            collection_name: Name of the vector database collection
            use_ocr: Whether to use OCR for image-based documents
            
        Returns:
            Processing results summary
        """
        if use_ocr is None:
            use_ocr = getattr(self.config, 'ocr_enabled', False) and self.ocr_processor is not None
        
        results = {
            "processed_files": [],
            "failed_files": [],
            "total_chunks": 0,
            "total_documents": 0
        }
        
        # Collect all files to process
        all_files = []
        for source_path in source_paths:
            source_path = Path(source_path)
            if source_path.is_file():
                all_files.append(source_path)
            elif source_path.is_dir():
                # Get all supported files in directory
                all_files.extend(self.document_parser.get_supported_files(source_path))
        
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process files
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_document, file_path, use_ocr): file_path
                for file_path in all_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    if chunks:
                        # Store in vector database
                        self.vectordb.add_chunks(chunks, collection_name)
                        results["processed_files"].append(str(file_path))
                        results["total_chunks"] += len(chunks)
                        results["total_documents"] += 1
                        logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                    else:
                        results["failed_files"].append(str(file_path))
                        logger.warning(f"No content extracted from {file_path}")
                        
                except Exception as e:
                    results["failed_files"].append(str(file_path))
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Processing complete: {results['total_documents']} documents, "
                   f"{results['total_chunks']} chunks")
        
        return results
    
    def _process_single_document(self, file_path: Path, use_ocr: bool) -> List[TextChunk]:
        """Process a single document and return chunks."""
        
        # Check if PDF is image-based and OCR is available
        if (file_path.suffix.lower() == '.pdf' and use_ocr and 
            self.ocr_processor and self.ocr_processor.is_image_based_pdf(file_path)):
            
            # Use OCR for image-based PDF
            ocr_results = self.ocr_processor.extract_text_from_pdf_images(file_path)
            
            # Combine OCR results
            full_text = ""
            metadata = {"source": str(file_path), "ocr_processed": True, "pages": []}
            
            for page_result in ocr_results:
                if page_result.get("text"):
                    full_text += f"\n\nPage {page_result['page']}:\n{page_result['text']}"
                    metadata["pages"].append({
                        "page": page_result["page"],
                        "confidence": page_result.get("confidence", 0),
                        "word_count": page_result.get("word_count", 0)
                    })
            
            if not full_text.strip():
                return []
            
            # Create document info
            from .parser import DocumentInfo
            doc_info = DocumentInfo(
                file_path=str(file_path),
                content=full_text,
                metadata=metadata,
                file_type="pdf",
                file_size=file_path.stat().st_size
            )
            
        else:
            # Use regular document parsing
            doc_info = self.document_parser.parse_document(file_path)
            if not doc_info or not doc_info.content:
                return []
        
        # Chunk the content
        chunks = self.text_chunker.chunk_document(doc_info)
        return chunks
    
    def process_urls(self, 
                    urls: List[str],
                    collection_name: str = "web_documents") -> Dict[str, Any]:
        """Process documents from URLs using the scraper.
        
        Args:
            urls: List of URLs to scrape
            collection_name: Vector database collection name
            
        Returns:
            Processing results summary
        """
        if not self.scraper:
            raise RuntimeError("Scraper not initialized. Check configuration.")
        
        # Scrape URLs
        scraped_docs = []
        for url in urls:
            try:
                result = self.scraper.scrape_single_url(url)
                if result and result.get('content'):
                    scraped_docs.append(result)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
        
        if not scraped_docs:
            return {"processed_urls": [], "failed_urls": urls, "total_chunks": 0}
        
        # Process scraped content
        all_chunks = []
        processed_urls = []
        
        for doc in scraped_docs:
            try:
                # Create document info
                from .parser import DocumentInfo
                doc_info = DocumentInfo(
                    file_path=doc['url'],
                    content=doc['content'],
                    metadata=doc.get('metadata', {}),
                    file_type="html",
                    file_size=len(doc['content'])
                )
                
                # Chunk document
                chunks = self.text_chunker.chunk_document(doc_info)
                all_chunks.extend(chunks)
                processed_urls.append(doc['url'])
                
            except Exception as e:
                logger.error(f"Failed to process scraped content from {doc.get('url', 'unknown')}: {e}")
        
        # Store in vector database
        if all_chunks:
            self.vectordb.add_chunks(all_chunks, collection_name)
        
        failed_urls = [url for url in urls if url not in processed_urls]
        
        return {
            "processed_urls": processed_urls,
            "failed_urls": failed_urls,
            "total_chunks": len(all_chunks)
        }
    
    def ask_question(self, 
                    question: str,
                    collection_name: str = "documents",
                    k: int = 5) -> Dict[str, Any]:
        """Ask a question against the processed documents.
        
        Args:
            question: Question to ask
            collection_name: Collection to search in
            k: Number of relevant chunks to retrieve
            
        Returns:
            Answer with sources and metadata
        """
        if not self.rag_manager:
            raise RuntimeError("RAG manager not initialized. Check LLM provider configuration.")
        
        return self.rag_manager.ask_question(question, collection_name, k)
    
    def get_collections_info(self) -> List[Dict[str, Any]]:
        """Get information about all collections in the vector database."""
        return self.vectordb.list_collections()
    
    def export_collection(self, collection_name: str, output_path: str):
        """Export a collection to a file.
        
        Args:
            collection_name: Name of collection to export
            output_path: Path to save the export
        """
        # Get all documents from collection
        stats = self.vectordb.get_collection_stats(collection_name)
        if stats['document_count'] == 0:
            logger.warning(f"Collection '{collection_name}' is empty")
            return
        
        # Retrieve all chunks (this might be memory intensive for large collections)
        all_results = self.vectordb.search_similar(
            query="", # Empty query to get all
            k=stats['document_count'],
            collection_name=collection_name
        )
        
        # Export to JSON
        import json
        export_data = {
            "collection_name": collection_name,
            "export_timestamp": datetime.now().isoformat(),
            "total_chunks": len(all_results),
            "chunks": all_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(all_results)} chunks to {output_path}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.vectordb:
            # Close vector database connections if needed
            pass
        logger.info("RAG document processor cleanup completed")


def create_processor(config_path: str = None) -> RAGDocumentProcessor:
    """Factory function to create a RAG document processor.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Configured RAGDocumentProcessor instance
    """
    return RAGDocumentProcessor(config_path) 