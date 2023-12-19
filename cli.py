#!/usr/bin/env python3
"""
Command-line interface for RAG Document Parser.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional

from ragdocparser.parser import DocumentParser
from ragdocparser.chunker import TextChunker
from ragdocparser.vectordb import DocumentVectorStore, OpenAIEmbeddingProvider
from ragdocparser.config import config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ragdocparser.log')
        ]
    )


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, log_level):
    """RAG Document Parser CLI."""
    ctx.ensure_object(dict)
    setup_logging(log_level)
    ctx.obj['log_level'] = log_level


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file for parsed content')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'text', 'yaml']), 
              help='Output format')
def parse(file_path, output, output_format):
    """Parse a document and extract text content."""
    try:
        parser = DocumentParser()
        result = parser.parse(file_path)
        
        if output_format == 'json':
            import json
            output_content = json.dumps(result, indent=2, default=str)
        elif output_format == 'yaml':
            import yaml
            output_content = yaml.dump(result, default_flow_style=False)
        else:  # text
            content_parts = []
            for page in result.get('content', []):
                content_parts.append(f"Page {page.get('page', 1)}:")
                content_parts.append(page.get('content', ''))
                content_parts.append('')
            output_content = '\n'.join(content_parts)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            click.echo(f"Parsed content saved to: {output}")
        else:
            click.echo(output_content)
            
    except Exception as e:
        click.echo(f"Error parsing document: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--strategy', default='sentence', 
              type=click.Choice(['fixed', 'sentence', 'paragraph']),
              help='Chunking strategy')
@click.option('--chunk-size', default=1000, help='Chunk size in characters')
@click.option('--overlap', default=200, help='Overlap between chunks')
@click.option('--output', '-o', help='Output file for chunks')
def chunk(file_path, strategy, chunk_size, overlap, output):
    """Parse and chunk a document."""
    try:
        # Parse document
        parser = DocumentParser()
        document = parser.parse(file_path)
        
        # Chunk document
        chunker = TextChunker(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap
        )
        chunks = chunker.chunk_document(document)
        
        # Prepare output
        chunk_data = {
            'file_path': file_path,
            'strategy': strategy,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'total_chunks': len(chunks),
            'chunks': [
                {
                    'id': chunk.chunk_id,
                    'content': chunk.content,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'metadata': chunk.metadata
                }
                for chunk in chunks
            ]
        }
        
        # Get statistics
        stats = chunker.get_chunk_statistics(chunks)
        chunk_data['statistics'] = stats
        
        import json
        output_content = json.dumps(chunk_data, indent=2, default=str)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            click.echo(f"Chunks saved to: {output}")
        else:
            click.echo(output_content)
        
        # Print summary
        click.echo(f"\nSummary:", err=True)
        click.echo(f"  Total chunks: {len(chunks)}", err=True)
        click.echo(f"  Average chunk length: {stats.get('avg_chunk_length', 0):.1f} characters", err=True)
        click.echo(f"  Strategy used: {strategy}", err=True)
            
    except Exception as e:
        click.echo(f"Error chunking document: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--collection', default='documents', help='Vector database collection name')
@click.option('--strategy', default='sentence', 
              type=click.Choice(['fixed', 'sentence', 'paragraph']),
              help='Chunking strategy')
@click.option('--chunk-size', default=1000, help='Chunk size in characters')
@click.option('--overlap', default=200, help='Overlap between chunks')
@click.option('--recursive', is_flag=True, help='Process directories recursively')
def index(input_path, collection, strategy, chunk_size, overlap, recursive):
    """Index documents into vector database."""
    try:
        # Initialize components
        parser = DocumentParser()
        chunker = TextChunker(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        # Initialize vector store with OpenAI embeddings if available
        embedding_provider = None
        if config.openai_api_key:
            try:
                embedding_provider = OpenAIEmbeddingProvider()
                click.echo("Using OpenAI embeddings")
            except Exception as e:
                click.echo(f"Warning: Could not initialize OpenAI embeddings: {e}")
        
        vector_store = DocumentVectorStore(
            collection_name=collection,
            embedding_provider=embedding_provider
        )
        
        input_path = Path(input_path)
        
        if input_path.is_file():
            # Process single file
            files_to_process = [input_path]
        else:
            # Process directory
            pattern = "**/*" if recursive else "*"
            supported_formats = parser.supported_formats()
            files_to_process = [
                f for f in input_path.glob(pattern)
                if f.is_file() and f.suffix.lower() in supported_formats
            ]
        
        if not files_to_process:
            click.echo("No supported files found to process")
            return
        
        click.echo(f"Processing {len(files_to_process)} files...")
        
        total_chunks = 0
        successful_files = 0
        
        for file_path in files_to_process:
            try:
                click.echo(f"Processing: {file_path}")
                
                # Parse document
                document = parser.parse(file_path)
                
                # Add to vector store
                chunks = vector_store.add_document(document, chunker)
                total_chunks += len(chunks)
                successful_files += 1
                
                click.echo(f"  Added {len(chunks)} chunks")
                
            except Exception as e:
                click.echo(f"  Error processing {file_path}: {e}", err=True)
                continue
        
        # Print summary
        click.echo(f"\nIndexing complete:")
        click.echo(f"  Files processed: {successful_files}/{len(files_to_process)}")
        click.echo(f"  Total chunks added: {total_chunks}")
        click.echo(f"  Collection: {collection}")
        
        # Show vector store stats
        stats = vector_store.get_stats()
        click.echo(f"  Vector store total documents: {stats.get('count', 0)}")
            
    except Exception as e:
        click.echo(f"Error indexing documents: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--collection', default='documents', help='Vector database collection name')
@click.option('--limit', default=5, help='Number of results to return')
@click.option('--file-type', help='Filter by file type')
@click.option('--file-path', help='Filter by file path')
def search(query, collection, limit, file_type, file_path):
    """Search indexed documents."""
    try:
        # Initialize vector store
        embedding_provider = None
        if config.openai_api_key:
            try:
                embedding_provider = OpenAIEmbeddingProvider()
            except Exception:
                pass
        
        vector_store = DocumentVectorStore(
            collection_name=collection,
            embedding_provider=embedding_provider
        )
        
        # Perform search
        results = vector_store.search_documents(
            query=query,
            n_results=limit,
            file_type=file_type,
            file_path=file_path
        )
        
        if not results:
            click.echo("No results found")
            return
        
        click.echo(f"Found {len(results)} results for query: '{query}'\n")
        
        for i, result in enumerate(results, 1):
            click.echo(f"Result {i}:")
            click.echo(f"  File: {result['metadata'].get('file_path', 'unknown')}")
            click.echo(f"  Page: {result['metadata'].get('page_number', 'unknown')}")
            click.echo(f"  Chunk ID: {result.get('id', 'unknown')}")
            if result.get('distance') is not None:
                click.echo(f"  Distance: {result['distance']:.4f}")
            click.echo(f"  Content: {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
            click.echo()
            
    except Exception as e:
        click.echo(f"Error searching documents: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--collection', default='documents', help='Vector database collection name')
def info(collection):
    """Show information about the vector database."""
    try:
        vector_store = DocumentVectorStore(collection_name=collection)
        stats = vector_store.get_stats()
        
        click.echo(f"Vector Database Information:")
        click.echo(f"  Collection: {stats.get('name', 'unknown')}")
        click.echo(f"  Total documents: {stats.get('count', 0)}")
        click.echo(f"  Persist directory: {stats.get('persist_directory', 'unknown')}")
        click.echo(f"  Embedding provider: {stats.get('embedding_provider', 'unknown')}")
        
    except Exception as e:
        click.echo(f"Error getting database info: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
