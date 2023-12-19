"""
RAG Document Parser - A comprehensive document processing system for RAG applications.
"""

__version__ = "0.2.0"
__author__ = "Ryan Stephens"
__email__ = "ryan@example.com"

from .parser import DocumentParser
from .chunker import TextChunker, Chunk
from .vectordb import DocumentVectorStore, VectorDatabase, OpenAIEmbeddingProvider
from .config import Config, config

__all__ = [
    "DocumentParser", 
    "TextChunker", 
    "Chunk",
    "DocumentVectorStore", 
    "VectorDatabase", 
    "OpenAIEmbeddingProvider",
    "Config", 
    "config"
]
