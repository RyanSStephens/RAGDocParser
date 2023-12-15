#!/usr/bin/env python3
"""
Command line interface for RAGDocParser.
"""

import argparse
import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any

from ragdocparser import DocumentParser, Config
from ragdocparser.scraper import WebScraper

def parse_directory_command(args):
    """Parse a directory of documents."""
    config = Config(args.config) if args.config else Config()
    parser = DocumentParser(config)
    
    print(f"Parsing directory: {args.directory}")
    print(f"Recursive: {args.recursive}")
    print(f"Collection: {args.collection}")
    
    # Parse documents
    documents = parser.parse_directory(args.directory, recursive=args.recursive)
    
    if not documents:
        print("No documents found or parsed.")
        return
    
    print(f"Parsed {len(documents)} documents")
    
    # Save to vector database if requested
    if not args.no_vectorize:
        print("Processing and storing in vector database...")
        chunks_count = parser.save_to_vector_db(documents, args.collection)
        print(f"Created {chunks_count} chunks and stored in vector database")
    
    # Save raw documents if requested
    if args.output:
        print(f"Saving documents to {args.output}")
        os.makedirs(args.output, exist_ok=True)
        
        for i, doc in enumerate(documents):
            filename = f"doc_{i:03d}_{Path(doc.filename).stem}.txt"
            filepath = os.path.join(args.output, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {doc.filepath}\n")
                f.write(f"Size: {doc.file_size} bytes\n")
                f.write(f"Parse time: {doc.parse_time:.3f}s\n")
                f.write("-" * 50 + "\n")
                f.write(doc.content)
        
        # Save metadata
        metadata = []
        for doc in documents:
            metadata.append({
                'filename': doc.filename,
                'filepath': doc.filepath,
                'file_size': doc.file_size,
                'parse_time': doc.parse_time,
                'content_hash': doc.content_hash,
                'metadata': doc.metadata
            })
        
        with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(documents)} documents to {args.output}")

def parse_file_command(args):
    """Parse a single file."""
    config = Config(args.config) if args.config else Config()
    parser = DocumentParser(config)
    
    print(f"Parsing file: {args.file}")
    
    document = parser.parse_file(args.file)
    
    if not document:
        print("Failed to parse file.")
        return
    
    print(f"Parsed {document.filename}")
    print(f"Content length: {len(document.content)} characters")
    print(f"Parse time: {document.parse_time:.3f}s")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(document.content)
        print(f"Content saved to {args.output}")
    else:
        # Print first 500 characters
        print("\nContent preview:")
        print("-" * 50)
        print(document.content[:500])
        if len(document.content) > 500:
            print("...")

def scrape_command(args):
    """Scrape a website."""
    print(f"Scraping website: {args.url}")
    
    scraper = WebScraper(
        delay=args.delay,
        max_pages=args.max_pages
    )
    
    if args.docs_only:
        pages = scraper.scrape_documentation_site(args.url)
    else:
        pages = scraper.scrape_site(args.url, max_depth=args.depth)
    
    print(f"Scraped {len(pages)} pages")
    
    if args.output:
        scraper.save_scraped_data(args.output)
        print(f"Saved scraped data to {args.output}")
    
    # Process with RAGDocParser if requested
    if args.vectorize:
        print("Processing scraped content...")
        config = Config(args.config) if args.config else Config()
        parser = DocumentParser(config)
        
        # Convert to document format
        documents = []
        for page in pages:
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
        
        chunks_count = parser.save_to_vector_db(documents, args.collection)
        print(f"Created {chunks_count} chunks and stored in vector database")

def search_command(args):
    """Search the vector database."""
    config = Config(args.config) if args.config else Config()
    parser = DocumentParser(config)
    
    print(f"Searching for: {args.query}")
    
    results = parser.vectordb.search_similar(
        args.query, 
        k=args.limit, 
        collection_name=args.collection
    )
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('metadata', {}).get('source', 'Unknown source')}")
        if 'distance' in result:
            print(f"   Similarity: {1 - result['distance']:.3f}")
        print(f"   Content: {result['content'][:200]}...")
        print()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAGDocParser - Document parser optimized for RAG systems"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Parse directory command
    parse_dir_parser = subparsers.add_parser('parse-dir', help='Parse a directory of documents')
    parse_dir_parser.add_argument('directory', help='Directory to parse')
    parse_dir_parser.add_argument('--recursive', '-r', action='store_true', help='Parse subdirectories')
    parse_dir_parser.add_argument('--output', '-o', help='Output directory for parsed documents')
    parse_dir_parser.add_argument('--collection', '-c', help='Vector database collection name')
    parse_dir_parser.add_argument('--no-vectorize', action='store_true', help='Skip vector database storage')
    
    # Parse file command
    parse_file_parser = subparsers.add_parser('parse-file', help='Parse a single file')
    parse_file_parser.add_argument('file', help='File to parse')
    parse_file_parser.add_argument('--output', '-o', help='Output file for parsed content')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape a website')
    scrape_parser.add_argument('url', help='URL to scrape')
    scrape_parser.add_argument('--output', '-o', help='Output directory for scraped content')
    scrape_parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    scrape_parser.add_argument('--max-pages', type=int, default=50, help='Maximum pages to scrape')
    scrape_parser.add_argument('--depth', type=int, default=3, help='Maximum crawl depth')
    scrape_parser.add_argument('--docs-only', action='store_true', help='Only scrape documentation pages')
    scrape_parser.add_argument('--vectorize', action='store_true', help='Store in vector database')
    scrape_parser.add_argument('--collection', '-c', help='Vector database collection name')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search the vector database')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', '-l', type=int, default=5, help='Number of results to return')
    search_parser.add_argument('--collection', '-c', help='Vector database collection name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'parse-dir':
            parse_directory_command(args)
        elif args.command == 'parse-file':
            parse_file_command(args)
        elif args.command == 'scrape':
            scrape_command(args)
        elif args.command == 'search':
            search_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 