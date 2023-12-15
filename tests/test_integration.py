"""
Integration tests for RAGDocParser.
Tests the complete pipeline from document processing to question answering.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from ragdocparser.integration import RAGDocumentProcessor
from ragdocparser.config import RAGConfig
from ragdocparser.utils import clean_text, calculate_content_hash


class TestRAGIntegration(unittest.TestCase):
    """Test the complete RAG pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RAGConfig()
        self.config.chunk_size = 500
        self.config.chunk_overlap = 50
        self.config.max_workers = 2
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_document_processor_initialization(self):
        """Test that the document processor initializes correctly."""
        processor = RAGDocumentProcessor(self.config)
        
        self.assertIsNotNone(processor.document_parser)
        self.assertIsNotNone(processor.text_chunker)
        self.assertIsNotNone(processor.vectordb)
        
    def test_process_text_documents(self):
        """Test processing of text documents."""
        # Create test documents
        test_docs = {
            'doc1.txt': 'This is a test document about machine learning. It contains information about algorithms and data processing.',
            'doc2.md': '# Test Document\n\nThis is a markdown document with information about natural language processing and text analysis.',
            'doc3.txt': 'A short document about RAG systems and their applications in modern AI.'
        }
        
        # Save test documents
        doc_paths = []
        for filename, content in test_docs.items():
            file_path = Path(self.temp_dir) / filename
            file_path.write_text(content, encoding='utf-8')
            doc_paths.append(file_path)
        
        # Process documents
        processor = RAGDocumentProcessor(self.config)
        results = processor.process_documents(doc_paths, "test_collection")
        
        self.assertEqual(len(results['processed_files']), 3)
        self.assertEqual(results['total_documents'], 3)
        self.assertGreater(results['total_chunks'], 0)
        self.assertEqual(len(results['failed_files']), 0)
    
    def test_empty_directory_processing(self):
        """Test processing of empty directory."""
        processor = RAGDocumentProcessor(self.config)
        results = processor.process_documents([self.temp_dir], "empty_collection")
        
        self.assertEqual(len(results['processed_files']), 0)
        self.assertEqual(results['total_documents'], 0)
        self.assertEqual(results['total_chunks'], 0)
    
    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        processor = RAGDocumentProcessor(self.config)
        fake_file = Path(self.temp_dir) / "nonexistent.txt"
        
        results = processor.process_documents([fake_file], "test_collection")
        
        self.assertEqual(len(results['processed_files']), 0)
        self.assertEqual(len(results['failed_files']), 1)
    
    def test_collections_info(self):
        """Test getting collections information."""
        processor = RAGDocumentProcessor(self.config)
        
        # Initially should have no collections or empty collections
        collections = processor.get_collections_info()
        self.assertIsInstance(collections, list)
    
    @patch('ragdocparser.llm_providers.OpenAIProvider')
    def test_question_answering_without_llm(self, mock_provider):
        """Test question answering when LLM provider is not available."""
        processor = RAGDocumentProcessor(self.config)
        processor.rag_manager = None  # Simulate no LLM provider
        
        with self.assertRaises(RuntimeError):
            processor.ask_question("What is machine learning?", "test_collection")
    
    def test_processor_cleanup(self):
        """Test processor cleanup."""
        processor = RAGDocumentProcessor(self.config)
        
        # Should not raise any errors
        processor.cleanup()


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        messy_text = "This   is  a\r\n  messy    text\n\n\nwith   extra   spaces."
        cleaned = clean_text(messy_text)
        
        self.assertNotIn('\r', cleaned)
        self.assertNotIn('  ', cleaned)  # No double spaces
        self.assertNotIn('\n\n\n', cleaned)  # No triple newlines
    
    def test_clean_text_empty(self):
        """Test cleaning empty or None text."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
    
    def test_calculate_content_hash(self):
        """Test content hash calculation."""
        text1 = "This is a test document."
        text2 = "This is a test document."
        text3 = "This is a different document."
        
        hash1 = calculate_content_hash(text1)
        hash2 = calculate_content_hash(text2)
        hash3 = calculate_content_hash(text3)
        
        self.assertEqual(hash1, hash2)  # Same content, same hash
        self.assertNotEqual(hash1, hash3)  # Different content, different hash
        self.assertEqual(len(hash1), 64)  # SHA-256 produces 64-character hex string


class TestConfigurationHandling(unittest.TestCase):
    """Test configuration handling."""
    
    def test_config_with_dict(self):
        """Test creating processor with dictionary configuration."""
        config_dict = {
            'chunk_size': 800,
            'chunk_overlap': 100,
            'max_workers': 3
        }
        
        processor = RAGDocumentProcessor(config_dict)
        self.assertEqual(processor.config.chunk_size, 800)
        self.assertEqual(processor.config.chunk_overlap, 100)
        self.assertEqual(processor.config.max_workers, 3)
    
    def test_config_with_file(self):
        """Test creating processor with file configuration."""
        config_dict = {
            'chunk_size': 600,
            'chunk_overlap': 75
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_dict, f)
            config_file = f.name
        
        try:
            processor = RAGDocumentProcessor(config_file)
            self.assertEqual(processor.config.chunk_size, 600)
            self.assertEqual(processor.config.chunk_overlap, 75)
        finally:
            os.unlink(config_file)
    
    def test_default_config(self):
        """Test processor with default configuration."""
        processor = RAGDocumentProcessor()
        
        # Should have reasonable defaults
        self.assertIsInstance(processor.config.chunk_size, int)
        self.assertIsInstance(processor.config.chunk_overlap, int)
        self.assertGreater(processor.config.chunk_size, 0)
        self.assertGreaterEqual(processor.config.chunk_overlap, 0)


if __name__ == '__main__':
    unittest.main() 