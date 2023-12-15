"""
Text chunking module optimized for RAG systems.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
from sentence_transformers import SentenceTransformer

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    start_index: int
    end_index: int
    token_count: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class TextChunker:
    """Advanced text chunker optimized for RAG performance."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 encoding_name: str = "cl100k_base"):
        """Initialize text chunker.
        
        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to avoid tiny chunks
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to a simpler tokenizer
            self.encoding = None
            
        self.sentence_model = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with spaCy/NLTK
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_text(self, 
                   text: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Chunk text using hierarchical approach.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
            
        metadata = metadata or {}
        chunks = []
        
        # First try paragraph-based chunking
        paragraphs = self._split_by_paragraphs(text)
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self._count_tokens(paragraph)
            current_tokens = self._count_tokens(current_chunk)
            
            # If adding this paragraph would exceed chunk size
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = self._create_chunk(
                    current_chunk, 
                    current_start, 
                    current_start + len(current_chunk),
                    metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + paragraph
                current_start = current_start + overlap_start
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                metadata
            )
            chunks.append(chunk)
        
        # Handle oversized chunks by sentence splitting
        final_chunks = []
        for chunk in chunks:
            if chunk.token_count > self.chunk_size * 1.5:
                final_chunks.extend(self._split_large_chunk(chunk))
            else:
                final_chunks.append(chunk)
        
        # Filter out tiny chunks
        final_chunks = [c for c in final_chunks if c.token_count >= self.min_chunk_size]
        
        return final_chunks
    
    def _create_chunk(self, 
                      content: str, 
                      start: int, 
                      end: int, 
                      metadata: Dict[str, Any]) -> TextChunk:
        """Create a TextChunk object."""
        return TextChunk(
            content=content.strip(),
            start_index=start,
            end_index=end,
            token_count=self._count_tokens(content),
            metadata=metadata.copy()
        )
    
    def _split_large_chunk(self, chunk: TextChunk) -> List[TextChunk]:
        """Split an oversized chunk into smaller ones."""
        sentences = self._split_by_sentences(chunk.content)
        sub_chunks = []
        current_text = ""
        current_start = chunk.start_index
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            current_tokens = self._count_tokens(current_text)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_text:
                # Create sub-chunk
                sub_chunk = self._create_chunk(
                    current_text,
                    current_start,
                    current_start + len(current_text),
                    chunk.metadata
                )
                sub_chunks.append(sub_chunk)
                
                # Start new with overlap
                overlap_start = max(0, len(current_text) - self.chunk_overlap)
                current_text = current_text[overlap_start:] + " " + sentence
                current_start = current_start + overlap_start
            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence
        
        # Add final sub-chunk
        if current_text.strip():
            sub_chunk = self._create_chunk(
                current_text,
                current_start,
                current_start + len(current_text),
                chunk.metadata
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def generate_embeddings(self, chunks: List[TextChunk], model_name: str = None):
        """Generate embeddings for chunks.
        
        Args:
            chunks: List of TextChunk objects
            model_name: Name of sentence transformer model
        """
        if not model_name:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
        if not self.sentence_model or self.sentence_model.model_name != model_name:
            self.sentence_model = SentenceTransformer(model_name)
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()
    
    def chunk_documents(self, 
                       documents: List[Dict[str, Any]]) -> List[TextChunk]:
        """Chunk multiple documents.
        
        Args:
            documents: List of documents with 'content' and optional metadata
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.copy()
            metadata.pop('content', None)
            
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks 