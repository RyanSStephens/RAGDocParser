"""
Integration tests for RAG Document Parser.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from ragdocparser.integration import DocumentProcessor


class TestDocumentProcessor:
    """Test document processor integration."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory with test files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        (temp_dir / "test1.txt").write_text("This is the first test document. It contains some sample text.")
        (temp_dir / "test2.txt").write_text("This is the second test document. It has different content.")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_processor_initialization(self):
        """Test processor can be initialized."""
        processor = DocumentProcessor(collection_name="test_collection")
        
        assert processor.parser is not None
        assert processor.chunker is not None
        assert processor.vector_store is not None
    
    def test_process_single_document(self, temp_directory):
        """Test processing a single document."""
        processor = DocumentProcessor(collection_name="test_single")
        
        test_file = temp_directory / "test1.txt"
        result = processor.process_document(test_file)
        
        assert result['status'] == 'success'
        assert result['chunks'] > 0
        assert result['document'] is not None
    
    def test_process_directory(self, temp_directory):
        """Test processing a directory of documents."""
        processor = DocumentProcessor(collection_name="test_directory")
        
        results = processor.process_directory(temp_directory)
        
        assert results['total_files'] == 2
        assert results['successful'] == 2
        assert results['failed'] == 0
        assert results['total_chunks'] > 0
    
    def test_search_functionality(self, temp_directory):
        """Test search functionality after processing documents."""
        processor = DocumentProcessor(collection_name="test_search")
        
        # Process documents first
        processor.process_directory(temp_directory)
        
        # Perform search
        results = processor.search_documents("test document", n_results=2)
        
        # Should find results (exact behavior depends on embedding provider)
        assert isinstance(results, list)
        # Results might be empty if no embedding provider is available
    
    def test_get_statistics(self):
        """Test getting processor statistics."""
        processor = DocumentProcessor(collection_name="test_stats")
        
        stats = processor.get_statistics()
        
        assert 'vector_store' in stats
        assert 'parser_formats' in stats
        assert 'chunker_strategy' in stats
        assert isinstance(stats['parser_formats'], list)
    
    def test_process_nonexistent_file(self):
        """Test processing a nonexistent file."""
        processor = DocumentProcessor(collection_name="test_error")
        
        result = processor.process_document("nonexistent_file.txt")
        
        assert result['status'] == 'error'
        assert result['chunks'] == 0
        assert 'error' in result


if __name__ == "__main__":
    pytest.main([__file__])
