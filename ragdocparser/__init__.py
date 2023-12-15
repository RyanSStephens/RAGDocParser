"""
RAGDocParser - Document parser optimized for RAG systems.

A comprehensive document processing library that handles multiple formats,
performs OCR on images, and integrates with vector databases for optimal
RAG (Retrieval-Augmented Generation) performance.
"""

__version__ = "1.5.0"
__author__ = "Ryan Stephens"
__email__ = "ryan@example.com"

from .parser import DocumentParser
from .chunker import TextChunker
from .vectordb import VectorDBManager
from .config import Config

__all__ = [
    "DocumentParser",
    "TextChunker", 
    "VectorDBManager",
    "Config"
] 