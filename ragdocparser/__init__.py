"""
RAG Document Parser - A comprehensive document processing system for RAG applications.
"""

__version__ = "0.1.0"
__author__ = "Ryan Stephens"
__email__ = "ryan@example.com"

from .parser import DocumentParser
from .chunker import TextChunker
from .config import Config

__all__ = ["DocumentParser", "TextChunker", "Config"]
