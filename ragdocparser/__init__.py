"""
RAG Document Parser - A comprehensive document processing system for RAG applications.
"""

__version__ = "0.3.0"
__author__ = "Ryan Stephens"
__email__ = "ryan@example.com"

from .parser import DocumentParser
from .chunker import TextChunker, Chunk
from .vectordb import DocumentVectorStore, VectorDatabase, OpenAIEmbeddingProvider
from .config import Config, config
from .integration import DocumentProcessor

# Optional imports (may not be available depending on dependencies)
try:
    from .ocr import OCRProcessor, ImageDocumentParser
except ImportError:
    pass

try:
    from .scraper import WebScraper, URLDocumentParser
except ImportError:
    pass

try:
    from .llm_providers import LLMManager, OpenAIProvider, AnthropicProvider
except ImportError:
    pass

__all__ = [
    "DocumentParser", 
    "TextChunker", 
    "Chunk",
    "DocumentVectorStore", 
    "VectorDatabase", 
    "OpenAIEmbeddingProvider",
    "Config", 
    "config",
    "DocumentProcessor"
]
