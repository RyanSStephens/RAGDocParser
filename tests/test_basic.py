"""
Basic tests for RAGDocParser.
"""

import unittest
import tempfile
import os
from pathlib import Path

from ragdocparser import DocumentParser, Config, TextChunker

class TestDocumentParser(unittest.TestCase):
    """Test DocumentParser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.parser = DocumentParser(self.config)
        
        # Create temporary test file
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, 'w') as f:
            f.write("This is a test document for RAGDocParser. " * 50)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_parse_text_file(self):
        """Test parsing a text file."""
        document = self.parser.parse_file(self.test_file)
        self.assertIsNotNone(document)
        self.assertEqual(document.filename, "test.txt")
        self.assertGreater(len(document.content), 0)
        self.assertIn("test document", document.content)
    
    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        document = self.parser.parse_file("nonexistent.txt")
        self.assertIsNone(document)

class TestTextChunker(unittest.TestCase):
    """Test TextChunker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = "This is a test document. " * 20
        chunks = self.chunker.chunk_text(text)
        
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 150)  # Allowing some variance
            self.assertGreater(len(chunk.content), 0)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = self.chunker.chunk_text("")
        self.assertEqual(len(chunks), 0)

class TestConfig(unittest.TestCase):
    """Test Config functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        chunk_size = config.get("chunking.chunk_size")
        self.assertEqual(chunk_size, 1000)
    
    def test_set_config(self):
        """Test setting configuration values."""
        config = Config()
        config.set("chunking.chunk_size", 500)
        self.assertEqual(config.get("chunking.chunk_size"), 500)
    
    def test_nested_config(self):
        """Test nested configuration access."""
        config = Config()
        provider = config.get("vector_db.provider")
        self.assertEqual(provider, "chromadb")

if __name__ == '__main__':
    unittest.main() 