#!/usr/bin/env python3
"""
Basic usage examples for RAGDocParser.
"""

import os
from ragdocparser import DocumentParser, Config

def basic_document_parsing():
    """Basic document parsing example."""
    print("=== Basic Document Parsing ===")
    
    # Initialize parser with default configuration
    parser = DocumentParser()
    
    # Parse a single file
    document = parser.parse_file("sample_document.pdf")
    if document:
        print(f"Parsed: {document.filename}")
        print(f"Content length: {len(document.content)} characters")
        print(f"First 200 characters: {document.content[:200]}...")
    
    # Parse a directory
    documents = parser.parse_directory("./docs", recursive=True)
    print(f"\nParsed {len(documents)} documents from directory")
    
    # Process and store in vector database
    if documents:
        chunks_count = parser.save_to_vector_db(documents)
        print(f"Created {chunks_count} chunks and stored in vector database")

def custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration ===")
    
    # Create custom configuration
    config = Config()
    config.set("chunking.chunk_size", 500)
    config.set("chunking.chunk_overlap", 100)
    config.set("vector_db.collection_name", "my_documents")
    
    # Initialize parser with custom config
    parser = DocumentParser(config)
    
    # Parse and process documents
    documents = parser.parse_directory("./docs")
    if documents:
        chunks_count = parser.save_to_vector_db(documents, "my_collection")
        print(f"Processed {len(documents)} documents into {chunks_count} chunks")

def search_example():
    """Example of searching the vector database."""
    print("\n=== Search Example ===")
    
    parser = DocumentParser()
    
    # Search for similar content
    results = parser.vectordb.search_similar("machine learning algorithms", k=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['content'][:100]}...")
        if 'distance' in result:
            print(f"   Similarity score: {1 - result['distance']:.3f}")

def web_scraping_example():
    """Example of web scraping integration."""
    print("\n=== Web Scraping Example ===")
    
    from ragdocparser.scraper import WebScraper
    
    # Initialize scraper
    scraper = WebScraper(delay=1.0, max_pages=10)
    
    # Scrape a documentation site
    pages = scraper.scrape_documentation_site("https://docs.python.org")
    print(f"Scraped {len(pages)} pages")
    
    # Convert to document format and process
    if pages:
        parser = DocumentParser()
        documents = []
        
        for page in pages:
            # Create pseudo-document from scraped page
            doc_info = type('DocumentInfo', (), {
                'filename': f"scraped_{hash(page.url)}.html",
                'filepath': page.url,
                'content': page.content,
                'metadata': {'title': page.title, 'url': page.url},
                'parse_time': 0.0,
                'file_size': len(page.content),
                'content_hash': hash(page.content)
            })()
            documents.append(doc_info)
        
        chunks_count = parser.save_to_vector_db(documents, "scraped_docs")
        print(f"Processed scraped content into {chunks_count} chunks")

if __name__ == "__main__":
    # Run examples (comment out sections that require actual files)
    print("RAGDocParser Usage Examples")
    print("=" * 40)
    
    try:
        # basic_document_parsing()  # Uncomment if you have sample documents
        custom_configuration()
        # search_example()  # Uncomment if you have data in vector DB
        # web_scraping_example()  # Uncomment to test web scraping
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have sample documents or modify the examples accordingly.") 