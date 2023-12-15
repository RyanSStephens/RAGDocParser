#!/usr/bin/env python3
"""
Advanced RAG Pipeline Example for RAGDocParser

This example demonstrates a complete RAG workflow including:
- Document parsing with OCR support
- Web scraping for additional content
- Vector database storage and retrieval
- LLM-powered question answering
- Batch processing and performance monitoring
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

from ragdocparser.integration import RAGDocumentProcessor
from ragdocparser.config import RAGConfig
from ragdocparser.utils import ProgressBar, timing_decorator

def create_advanced_config() -> RAGConfig:
    """Create an advanced configuration for the RAG pipeline."""
    config = RAGConfig()
    
    # Enable advanced features
    config.ocr_enabled = True
    config.ocr_use_gpu = False  # Set to True if you have GPU support
    config.ocr_languages = ['en']
    
    config.scraper_enabled = True
    config.scraper_max_concurrent = 3
    
    # Configure chunking for better retrieval
    config.chunk_size = 800
    config.chunk_overlap = 150
    config.min_chunk_size = 50
    
    # Performance settings
    config.max_workers = 4
    
    # LLM configuration (set your preferred provider)
    config.llm_provider = "openai"  # or "anthropic", "cohere"
    config.openai_model = "gpt-3.5-turbo"
    
    return config

@timing_decorator
def process_document_library(processor: RAGDocumentProcessor, 
                           documents_dir: str,
                           collection_name: str = "document_library") -> Dict[str, Any]:
    """Process a library of documents with various formats."""
    print(f"\n📚 Processing document library: {documents_dir}")
    
    documents_path = Path(documents_dir)
    if not documents_path.exists():
        print(f"❌ Directory not found: {documents_dir}")
        return {"error": "Directory not found"}
    
    # Find all documents
    all_files = []
    supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.html'}
    
    for ext in supported_extensions:
        all_files.extend(documents_path.rglob(f"*{ext}"))
    
    print(f"Found {len(all_files)} documents to process")
    
    if not all_files:
        print("No supported documents found")
        return {"processed_files": [], "total_chunks": 0}
    
    # Process documents
    results = processor.process_documents(all_files, collection_name, use_ocr=True)
    
    print(f"✅ Processed {results['total_documents']} documents")
    print(f"📄 Created {results['total_chunks']} text chunks")
    
    if results['failed_files']:
        print(f"⚠️  Failed to process {len(results['failed_files'])} files")
        
    return results

@timing_decorator  
def scrape_documentation_sites(processor: RAGDocumentProcessor,
                              urls: List[str],
                              collection_name: str = "web_docs") -> Dict[str, Any]:
    """Scrape documentation from multiple websites."""
    print(f"\n🌐 Scraping {len(urls)} documentation sites")
    
    all_results = {"processed_urls": [], "failed_urls": [], "total_chunks": 0}
    
    for url in urls:
        print(f"Scraping: {url}")
        try:
            results = processor.process_urls([url], collection_name)
            all_results["processed_urls"].extend(results["processed_urls"])
            all_results["failed_urls"].extend(results["failed_urls"])
            all_results["total_chunks"] += results["total_chunks"]
            
            print(f"  ✅ Added {results['total_chunks']} chunks from {url}")
            
        except Exception as e:
            print(f"  ❌ Failed to scrape {url}: {e}")
            all_results["failed_urls"].append(url)
    
    print(f"🌐 Scraped {len(all_results['processed_urls'])} URLs successfully")
    print(f"📄 Total web chunks: {all_results['total_chunks']}")
    
    return all_results

def interactive_qa_session(processor: RAGDocumentProcessor,
                          collections: List[str]):
    """Run an interactive Q&A session."""
    print(f"\n💬 Interactive Q&A Session")
    print(f"Available collections: {', '.join(collections)}")
    print("Type 'quit' to exit, 'collections' to list collections")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'collections':
                collection_info = processor.get_collections_info()
                print("\n📊 Collection Information:")
                for info in collection_info:
                    print(f"  - {info['name']}: {info.get('count', 'unknown')} items")
                continue
            
            if not question:
                continue
            
            # Ask question across all collections
            best_answer = None
            best_confidence = 0
            
            for collection in collections:
                try:
                    result = processor.ask_question(question, collection, k=5)
                    
                    if result.get('confidence', 0) > best_confidence:
                        best_answer = result
                        best_confidence = result.get('confidence', 0)
                        best_answer['source_collection'] = collection
                        
                except Exception as e:
                    print(f"⚠️  Error querying {collection}: {e}")
            
            if best_answer:
                print(f"\n🤖 Answer (from {best_answer.get('source_collection', 'unknown')}):")
                print(f"📊 Confidence: {best_answer.get('confidence', 0):.2f}")
                print("-" * 40)
                print(best_answer['answer'])
                
                # Show sources
                if best_answer.get('sources'):
                    print(f"\n📚 Sources ({len(best_answer['sources'])}):")
                    for i, source in enumerate(best_answer['sources'][:3], 1):
                        print(f"{i}. {source.get('metadata', {}).get('source', 'Unknown')}")
                        if source.get('similarity'):
                            print(f"   Similarity: {source['similarity']:.3f}")
                        print(f"   Preview: {source['content'][:100]}...")
            else:
                print("❌ No relevant information found in any collection.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n🚀 RAGDocParser Advanced Pipeline Demo")
    print("=" * 50)
    
    # Create configuration
    config = create_advanced_config()
    
    # Initialize processor
    print("🔧 Initializing RAG Document Processor...")
    processor = RAGDocumentProcessor(config)
    
    # Define collections
    collections_to_process = []
    
    # 1. Process local documents if directory exists
    docs_dir = "./sample_documents"
    if Path(docs_dir).exists():
        print(f"\n📁 Found local documents directory: {docs_dir}")
        process_document_library(processor, docs_dir, "local_docs")
        collections_to_process.append("local_docs")
    else:
        print(f"\n📁 Local documents directory not found: {docs_dir}")
        print("   Create this directory and add PDF, DOCX, TXT, or MD files to test document processing")
    
    # 2. Process some sample web documentation
    sample_urls = [
        "https://docs.python.org/3/tutorial/",
        "https://fastapi.tiangolo.com/tutorial/",
    ]
    
    if processor.scraper:
        print("\n🌐 Testing web scraping functionality...")
        try:
            scrape_documentation_sites(processor, sample_urls, "web_docs")
            collections_to_process.append("web_docs")
        except Exception as e:
            print(f"⚠️  Web scraping failed: {e}")
    else:
        print("🌐 Web scraping not enabled")
    
    # 3. Show collection statistics
    print("\n📊 Collection Statistics:")
    collections_info = processor.get_collections_info()
    for info in collections_info:
        print(f"  - {info['name']}: {info.get('count', 'unknown')} items")
    
    # 4. Demonstrate Q&A if we have collections
    if collections_to_process and processor.rag_manager:
        print("\n💡 Testing Q&A functionality...")
        
        # Test some sample questions
        sample_questions = [
            "What is Python?",
            "How do I create a FastAPI application?",
            "What are the main features?",
        ]
        
        for question in sample_questions:
            print(f"\n❓ Question: {question}")
            
            best_answer = None
            best_confidence = 0
            
            for collection in collections_to_process:
                try:
                    result = processor.ask_question(question, collection, k=3)
                    if result.get('confidence', 0) > best_confidence:
                        best_answer = result
                        best_confidence = result.get('confidence', 0)
                except Exception as e:
                    print(f"   ⚠️  Error with {collection}: {e}")
            
            if best_answer:
                print(f"🤖 Answer (confidence: {best_confidence:.2f}):")
                print(f"   {best_answer['answer'][:200]}...")
            else:
                print("❌ No answer found")
        
        # Start interactive session
        print("\n" + "=" * 50)
        interactive_qa_session(processor, collections_to_process)
        
    elif not processor.rag_manager:
        print("\n⚠️  LLM provider not configured. Set API keys to enable Q&A functionality.")
        print("   Available providers: OpenAI, Anthropic, Cohere")
    else:
        print("\n⚠️  No collections available for Q&A.")
    
    # Cleanup
    processor.cleanup()
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    # Check for API keys
    api_keys_present = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"), 
        os.getenv("COHERE_API_KEY")
    ])
    
    if not api_keys_present:
        print("⚠️  No LLM API keys found in environment variables.")
        print("   Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or COHERE_API_KEY to enable Q&A features.")
        print("   Document processing and vector storage will still work.\n")
    
    demonstrate_batch_processing() 