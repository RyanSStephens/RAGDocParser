#!/usr/bin/env python3
"""
Basic usage examples for RAG Document Parser.
"""

import os
from pathlib import Path

from ragdocparser import DocumentParser, TextChunker, DocumentVectorStore


def example_parse_document():
    """Example: Parse a single document."""
    print("=== Document Parsing Example ===")
    
    # Initialize parser
    parser = DocumentParser()
    
    # Show supported formats
    print(f"Supported formats: {parser.supported_formats()}")
    
    # Parse a document (you'll need to provide a real file)
    # document = parser.parse("path/to/your/document.pdf")
    # print(f"Parsed document: {document['file_path']}")
    # print(f"Pages: {document['metadata'].get('pages', 1)}")
    # print(f"Content preview: {document['content'][0]['content'][:200]}...")


def example_chunk_text():
    """Example: Chunk text using different strategies."""
    print("\n=== Text Chunking Example ===")
    
    sample_text = """
    This is a sample document for testing the chunking functionality.
    It contains multiple sentences and paragraphs to demonstrate different chunking strategies.
    
    The first paragraph talks about the importance of text chunking in RAG applications.
    Proper chunking ensures that relevant information is retrieved effectively.
    
    The second paragraph discusses different chunking strategies.
    Fixed-size chunking splits text into equal-sized chunks.
    Sentence-based chunking respects sentence boundaries.
    Paragraph-based chunking preserves paragraph structure.
    """
    
    # Test different chunking strategies
    strategies = ['fixed', 'sentence', 'paragraph']
    
    for strategy in strategies:
        print(f"\n--- {strategy.title()} Chunking ---")
        chunker = TextChunker(strategy=strategy, chunk_size=200, overlap=50)
        
        # Create a mock document structure
        mock_document = {
            'content': [{'page': 1, 'content': sample_text}],
            'metadata': {'file_path': 'sample.txt', 'file_type': 'txt'},
            'file_path': 'sample.txt',
            'file_type': 'txt'
        }
        
        chunks = chunker.chunk_document(mock_document)
        stats = chunker.get_chunk_statistics(chunks)
        
        print(f"Generated {len(chunks)} chunks")
        print(f"Average chunk length: {stats['avg_chunk_length']:.1f}")
        
        # Show first chunk
        if chunks:
            print(f"First chunk: {chunks[0].content[:100]}...")


def example_vector_database():
    """Example: Use vector database for storage and search."""
    print("\n=== Vector Database Example ===")
    
    try:
        # Initialize vector store
        vector_store = DocumentVectorStore(collection_name="example_docs")
        
        # Create sample chunks
        from ragdocparser.chunker import Chunk
        
        sample_chunks = [
            Chunk(
                content="Machine learning is a subset of artificial intelligence.",
                start_index=0,
                end_index=59,
                chunk_id="chunk_1",
                metadata={"topic": "AI", "source": "textbook"}
            ),
            Chunk(
                content="Natural language processing helps computers understand human language.",
                start_index=60,
                end_index=128,
                chunk_id="chunk_2",
                metadata={"topic": "NLP", "source": "textbook"}
            ),
            Chunk(
                content="Vector databases enable efficient similarity search over embeddings.",
                start_index=129,
                end_index=195,
                chunk_id="chunk_3",
                metadata={"topic": "Vector DB", "source": "documentation"}
            )
        ]
        
        # Add chunks to vector database
        vector_store.vector_db.add_chunks(sample_chunks)
        print(f"Added {len(sample_chunks)} chunks to vector database")
        
        # Search for similar content
        query = "What is machine learning?"
        results = vector_store.search_documents(query, n_results=2)
        
        print(f"\nSearch results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['content']}")
            print(f"   Metadata: {result['metadata']}")
        
        # Show database stats
        stats = vector_store.get_stats()
        print(f"\nDatabase stats: {stats}")
        
    except Exception as e:
        print(f"Vector database example failed (this is expected without proper setup): {e}")
        print("To use vector database features, ensure you have:")
        print("1. ChromaDB installed: pip install chromadb")
        print("2. OpenAI API key set (optional): export OPENAI_API_KEY=your_key")


def main():
    """Run all examples."""
    print("RAG Document Parser - Basic Usage Examples")
    print("=" * 50)
    
    example_parse_document()
    example_chunk_text()
    example_vector_database()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Install required dependencies: pip install -r requirements.txt")
    print("2. Set up environment variables (see .env.example)")
    print("3. Try the CLI: python cli.py --help")


if __name__ == "__main__":
    main()
