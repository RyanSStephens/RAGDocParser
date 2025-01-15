"""
Advanced AI integration for semantic analysis and intelligent processing.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .config import config
from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SemanticAnalysis:
    """Results of semantic analysis."""
    summary: str
    key_topics: List[str]
    sentiment_score: float
    complexity_score: float
    readability_score: float
    entities: List[Dict[str, Any]]


class SemanticAnalyzer:
    """Advanced semantic analysis using AI models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
    
    def analyze_document(self, document: Dict[str, Any]) -> SemanticAnalysis:
        """Perform comprehensive semantic analysis of a document."""
        # Combine all content
        content_parts = []
        for page in document.get('content', []):
            content_parts.append(page.get('content', ''))
        
        full_text = ' '.join(content_parts)
        
        # Perform various analyses
        summary = self._generate_extractive_summary(full_text)
        topics = self._extract_key_topics(full_text)
        sentiment = self._analyze_sentiment(full_text)
        complexity = self._calculate_complexity(full_text)
        readability = self._calculate_readability(full_text)
        entities = self._extract_entities(full_text)
        
        return SemanticAnalysis(
            summary=summary,
            key_topics=topics,
            sentiment_score=sentiment,
            complexity_score=complexity,
            readability_score=readability,
            entities=entities
        )
    
    def _generate_extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate extractive summary by selecting key sentences."""
        sentences = text.split('. ')
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple extractive summarization - select sentences with high keyword density
        # In a real implementation, you'd use more sophisticated methods
        sentence_scores = []
        
        for sentence in sentences:
            # Simple scoring based on sentence length and common words
            words = sentence.split()
            score = len(words) * 0.1  # Prefer longer sentences
            
            # Boost score for sentences with important indicators
            important_words = ['important', 'significant', 'key', 'main', 'primary', 'conclusion']
            for word in words:
                if word.lower() in important_words:
                    score += 1
            
            sentence_scores.append((sentence, score))
        
        # Select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'
    
    def _extract_key_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Extract key topics using simple keyword analysis."""
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words as topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        topics = [word for word, freq in sorted_words[:max_topics]]
        
        return topics
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment (simplified implementation)."""
        # Simple sentiment analysis based on positive/negative word counts
        positive_words = {'good', 'great', 'excellent', 'positive', 'success', 'effective', 'beneficial', 'advantage', 'improvement', 'better'}
        negative_words = {'bad', 'poor', 'negative', 'failure', 'problem', 'issue', 'difficulty', 'challenge', 'worse', 'disadvantage'}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        # Return score from -1 (very negative) to 1 (very positive)
        return (positive_count - negative_count) / total_sentiment_words
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Normalize to 0-1 scale (higher = more complex)
        complexity = min(1.0, (avg_sentence_length / 20) * 0.5 + (avg_word_length / 10) * 0.5)
        
        return complexity
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        sentences = len(text.split('.'))
        words = len(text.split())
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        
        # Normalize to 0-1 scale (higher = more readable)
        return max(0.0, min(1.0, score / 100))
    
    def _count_syllables(self, text: str) -> int:
        """Simple syllable counting."""
        words = text.lower().split()
        syllable_count = 0
        
        for word in words:
            # Simple vowel counting method
            vowels = 'aeiouy'
            word_syllables = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        word_syllables += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Every word has at least one syllable
            if word_syllables == 0:
                word_syllables = 1
            
            syllable_count += word_syllables
        
        return syllable_count
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities (simplified implementation)."""
        # Simple pattern-based entity extraction
        import re
        
        entities = []
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({'type': 'EMAIL', 'text': email, 'confidence': 0.9})
        
        # Phone numbers (simple pattern)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            entities.append({'type': 'PHONE', 'text': phone, 'confidence': 0.8})
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        for url in urls:
            entities.append({'type': 'URL', 'text': url, 'confidence': 0.95})
        
        # Dates (simple pattern)
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
        dates = re.findall(date_pattern, text)
        for date in dates:
            entities.append({'type': 'DATE', 'text': date, 'confidence': 0.7})
        
        return entities


class IntelligentChunker:
    """Intelligent chunking using semantic similarity."""
    
    def __init__(self, semantic_analyzer: Optional[SemanticAnalyzer] = None):
        self.analyzer = semantic_analyzer or SemanticAnalyzer()
    
    def chunk_by_semantic_similarity(self, text: str, target_chunk_size: int = 1000, similarity_threshold: float = 0.7) -> List[Chunk]:
        """Chunk text based on semantic similarity between sentences."""
        if not self.analyzer.model:
            # Fallback to simple sentence-based chunking
            return self._fallback_chunking(text, target_chunk_size)
        
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return [Chunk(content=text, start_index=0, end_index=len(text), chunk_id="chunk_0")]
        
        # Generate embeddings for sentences
        try:
            embeddings = self.analyzer.model.encode(sentences)
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            return self._fallback_chunking(text, target_chunk_size)
        
        # Group sentences by semantic similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embeddings = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            chunk_embedding = np.mean(current_chunk_embeddings, axis=0)
            similarity = np.dot(chunk_embedding, embeddings[i]) / (
                np.linalg.norm(chunk_embedding) * np.linalg.norm(embeddings[i])
            )
            
            # Check if sentence should be added to current chunk
            current_chunk_text = '. '.join(current_chunk_sentences)
            would_exceed_size = len(current_chunk_text) + len(sentences[i]) > target_chunk_size
            
            if similarity >= similarity_threshold and not would_exceed_size:
                current_chunk_sentences.append(sentences[i])
                current_chunk_embeddings.append(embeddings[i])
            else:
                # Finalize current chunk and start new one
                chunk_text = '. '.join(current_chunk_sentences)
                start_idx = text.find(current_chunk_sentences[0])
                end_idx = start_idx + len(chunk_text)
                
                chunk = Chunk(
                    content=chunk_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    chunk_id=f"semantic_chunk_{len(chunks)}",
                    metadata={'chunking_method': 'semantic_similarity', 'avg_similarity': float(np.mean([similarity]))}
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_chunk_embeddings = [embeddings[i]]
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = '. '.join(current_chunk_sentences)
            start_idx = text.find(current_chunk_sentences[0])
            end_idx = start_idx + len(chunk_text)
            
            chunk = Chunk(
                content=chunk_text,
                start_index=start_idx,
                end_index=end_idx,
                chunk_id=f"semantic_chunk_{len(chunks)}",
                metadata={'chunking_method': 'semantic_similarity'}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_chunking(self, text: str, target_size: int) -> List[Chunk]:
        """Fallback to simple chunking when semantic analysis fails."""
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= target_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    start_idx = text.find(current_chunk.strip())
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        start_index=start_idx,
                        end_index=start_idx + len(current_chunk.strip()),
                        chunk_id=f"fallback_chunk_{chunk_id}",
                        metadata={'chunking_method': 'fallback_sentence'}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                current_chunk = sentence + ". "
        
        # Add final chunk
        if current_chunk:
            start_idx = text.find(current_chunk.strip())
            chunk = Chunk(
                content=current_chunk.strip(),
                start_index=start_idx,
                end_index=start_idx + len(current_chunk.strip()),
                chunk_id=f"fallback_chunk_{chunk_id}",
                metadata={'chunking_method': 'fallback_sentence'}
            )
            chunks.append(chunk)
        
        return chunks
