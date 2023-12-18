"""
Text chunking utilities for document processing.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate basic statistics
        self.metadata.update({
            'length': len(self.content),
            'word_count': len(self.content.split()),
            'char_count': len(self.content),
        })


class BaseChunker(ABC):
    """Base class for text chunkers."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunker with overlap."""
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Look for the last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk = Chunk(
                    content=chunk_content,
                    start_index=start,
                    end_index=end,
                    chunk_id=f"chunk_{chunk_id}",
                    metadata=metadata.copy() if metadata else {}
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = max(end - self.overlap, start + 1)
            if start >= len(text):
                break
        
        return chunks


class SentenceChunker(BaseChunker):
    """Chunker that respects sentence boundaries."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__(chunk_size, overlap)
        # Simple sentence boundary detection
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split text into chunks respecting sentence boundaries."""
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = self.sentence_pattern.split(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        start_index = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        start_index=start_index,
                        end_index=start_index + len(current_chunk),
                        chunk_id=f"chunk_{chunk_id}",
                        metadata=metadata.copy() if metadata else {}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk with overlap handling
                if self.overlap > 0 and chunks:
                    # Try to include some overlap from previous chunk
                    prev_words = chunks[-1].content.split()
                    overlap_words = prev_words[-min(self.overlap//10, len(prev_words)):]
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
                
                start_index = start_index + len(current_chunk) - len(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                chunk_id=f"chunk_{chunk_id}",
                metadata=metadata.copy() if metadata else {}
            )
            chunks.append(chunk)
        
        return chunks


class ParagraphChunker(BaseChunker):
    """Chunker that respects paragraph boundaries."""
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split text into chunks respecting paragraph boundaries."""
        if not text.strip():
            return []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        start_index = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        start_index=start_index,
                        end_index=start_index + len(current_chunk),
                        chunk_id=f"chunk_{chunk_id}",
                        metadata=metadata.copy() if metadata else {}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk
                current_chunk = paragraph
                start_index = start_index + len(current_chunk)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                chunk_id=f"chunk_{chunk_id}",
                metadata=metadata.copy() if metadata else {}
            )
            chunks.append(chunk)
        
        return chunks


class TextChunker:
    """Main text chunker with multiple strategies."""
    
    def __init__(self, strategy: str = "sentence", chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        
        self.chunkers = {
            'fixed': FixedSizeChunker(chunk_size, overlap),
            'sentence': SentenceChunker(chunk_size, overlap),
            'paragraph': ParagraphChunker(chunk_size, overlap),
        }
        
        if strategy not in self.chunkers:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Chunk text using the specified strategy."""
        chunker = self.chunkers[self.strategy]
        return chunker.chunk(text, metadata)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk a parsed document."""
        all_chunks = []
        
        for page_data in document.get('content', []):
            page_text = page_data.get('content', '')
            page_num = page_data.get('page', 1)
            
            # Create metadata for this page
            page_metadata = document.get('metadata', {}).copy()
            page_metadata.update({
                'page_number': page_num,
                'file_path': document.get('file_path', ''),
                'file_type': document.get('file_type', ''),
            })
            
            # Chunk the page text
            page_chunks = self.chunk_text(page_text, page_metadata)
            
            # Update chunk IDs to include page information
            for i, chunk in enumerate(page_chunks):
                chunk.chunk_id = f"page_{page_num}_chunk_{i}"
                chunk.metadata['chunk_index'] = i
            
            all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Calculate statistics for a list of chunks."""
        if not chunks:
            return {}
        
        lengths = [len(chunk.content) for chunk in chunks]
        word_counts = [chunk.metadata.get('word_count', 0) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(lengths),
            'total_words': sum(word_counts),
            'avg_chunk_length': sum(lengths) / len(lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_chunk_length': min(lengths),
            'max_chunk_length': max(lengths),
            'strategy_used': self.strategy,
            'chunk_size_setting': self.chunk_size,
            'overlap_setting': self.overlap,
        }
