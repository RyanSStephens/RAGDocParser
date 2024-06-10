"""
Basic tests for RAG Document Parser.
"""

import pytest
import tempfile
from pathlib import Path

from ragdocparser import DocumentParser, TextChunker, Chunk


class TestDocumentParser:
    """Test document parser functionality."""
    
    def test_supported_formats(self):
        """Test that parser reports supported formats."""
        parser = DocumentParser()
        formats = parser.supported_formats()
        
        assert isinstance(formats, list)
        assert '.txt' in formats
        assert '.pdf' in formats
    
    def test_parse_text_file(self):
        """Test parsing a text file."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
            temp_path = f.name
        
        try:
            parser = DocumentParser()
            result = parser.parse(temp_path)
            
            assert result['file_type'] == 'txt'
            assert len(result['content']) == 1
            assert 'This is a test document' in result['content'][0]['content']
            assert 'metadata' in result
            assert result['metadata']['extraction_method'] == 'direct_read'
            
        finally:
            Path(temp_path).unlink()
    
    def test_parse_nonexistent_file(self):
        """Test parsing a nonexistent file raises error."""
        parser = DocumentParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent_file.txt")
    
    def test_parse_unsupported_format(self):
        """Test parsing unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            parser = DocumentParser()
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                parser.parse(temp_path)
                
        finally:
            Path(temp_path).unlink()


class TestTextChunker:
    """Test text chunking functionality."""
    
    def test_fixed_chunker(self):
        """Test fixed-size chunking."""
        text = "This is a test document. " * 100  # Long text
        chunker = TextChunker(strategy='fixed', chunk_size=100, overlap=20)
        
        # Create mock document
        document = {
            'content': [{'page': 1, 'content': text}],
            'metadata': {'file_path': 'test.txt'},
            'file_path': 'test.txt',
            'file_type': 'txt'
        }
        
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.content) <= 120 for chunk in chunks)  # Allow some flexibility
    
    def test_sentence_chunker(self):
        """Test sentence-based chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = TextChunker(strategy='sentence', chunk_size=50, overlap=10)
        
        document = {
            'content': [{'page': 1, 'content': text}],
            'metadata': {'file_path': 'test.txt'},
            'file_path': 'test.txt',
            'file_type': 'txt'
        }
        
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_empty_text_chunking(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        
        document = {
            'content': [{'page': 1, 'content': ''}],
            'metadata': {'file_path': 'test.txt'},
            'file_path': 'test.txt',
            'file_type': 'txt'
        }
        
        chunks = chunker.chunk_document(document)
        assert len(chunks) == 0
    
    def test_chunk_statistics(self):
        """Test chunk statistics calculation."""
        text = "This is a test. " * 50
        chunker = TextChunker(strategy='fixed', chunk_size=100)
        
        document = {
            'content': [{'page': 1, 'content': text}],
            'metadata': {'file_path': 'test.txt'},
            'file_path': 'test.txt',
            'file_type': 'txt'
        }
        
        chunks = chunker.chunk_document(document)
        stats = chunker.get_chunk_statistics(chunks)
        
        assert 'total_chunks' in stats
        assert 'total_characters' in stats
        assert 'avg_chunk_length' in stats
        assert stats['total_chunks'] == len(chunks)


class TestChunk:
    """Test Chunk class functionality."""
    
    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            content="Test content",
            start_index=0,
            end_index=12,
            chunk_id="test_chunk"
        )
        
        assert chunk.content == "Test content"
        assert chunk.start_index == 0
        assert chunk.end_index == 12
        assert chunk.chunk_id == "test_chunk"
        assert 'length' in chunk.metadata
        assert 'word_count' in chunk.metadata
    
    def test_chunk_metadata_calculation(self):
        """Test automatic metadata calculation."""
        content = "This is a test chunk with multiple words."
        chunk = Chunk(
            content=content,
            start_index=0,
            end_index=len(content),
            chunk_id="test"
        )
        
        assert chunk.metadata['length'] == len(content)
        assert chunk.metadata['word_count'] == len(content.split())
        assert chunk.metadata['char_count'] == len(content)


if __name__ == "__main__":
    pytest.main([__file__])
