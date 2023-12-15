"""
Multi-language support for RAGDocParser.
Handles text processing, tokenization, and embeddings for multiple languages.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """Supported languages for document processing."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    DUTCH = "nl"
    POLISH = "pl"
    TURKISH = "tr"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"

@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: SupportedLanguage
    confidence: float
    detected_text_sample: str
    alternative_languages: List[Tuple[SupportedLanguage, float]]

class LanguageDetector:
    """Detect document language using multiple methods."""
    
    def __init__(self):
        """Initialize language detector."""
        # Language-specific character patterns
        self.language_patterns = {
            SupportedLanguage.CHINESE: re.compile(r'[\u4e00-\u9fff]'),
            SupportedLanguage.JAPANESE: re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]'),
            SupportedLanguage.KOREAN: re.compile(r'[\uac00-\ud7af]'),
            SupportedLanguage.ARABIC: re.compile(r'[\u0600-\u06ff]'),
            SupportedLanguage.RUSSIAN: re.compile(r'[\u0400-\u04ff]'),
        }
        
        # Common words for basic detection
        self.language_stopwords = {
            SupportedLanguage.ENGLISH: ['the', 'and', 'in', 'to', 'of', 'a', 'is', 'that', 'for', 'with'],
            SupportedLanguage.SPANISH: ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se'],
            SupportedLanguage.FRENCH: ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
            SupportedLanguage.GERMAN: ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            SupportedLanguage.ITALIAN: ['il', 'di', 'che', 'e', 'la', 'a', 'per', 'in', 'un', 'è'],
            SupportedLanguage.PORTUGUESE: ['o', 'de', 'a', 'e', 'que', 'do', 'da', 'em', 'um', 'para'],
            SupportedLanguage.DUTCH: ['de', 'van', 'het', 'en', 'in', 'een', 'te', 'dat', 'op', 'voor'],
            SupportedLanguage.POLISH: ['w', 'na', 'i', 'z', 'do', 'o', 'że', 'a', 'się', 'co'],
            SupportedLanguage.TURKISH: ['bir', 've', 'bu', 'da', 'de', 'o', 'ile', 'için', 'var', 'daha'],
            SupportedLanguage.SWEDISH: ['och', 'i', 'att', 'det', 'av', 'är', 'för', 'den', 'till', 'en'],
            SupportedLanguage.NORWEGIAN: ['og', 'i', 'det', 'av', 'en', 'til', 'å', 'være', 'som', 'på'],
            SupportedLanguage.DANISH: ['og', 'i', 'det', 'at', 'en', 'til', 'af', 'er', 'som', 'på'],
            SupportedLanguage.FINNISH: ['on', 'ja', 'ei', 'se', 'että', 'oli', 'tai', 'kun', 'hän', 'niin']
        }
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect the language of the given text."""
        if not text or len(text.strip()) < 10:
            return LanguageDetectionResult(
                language=SupportedLanguage.ENGLISH,
                confidence=0.0,
                detected_text_sample=text[:100],
                alternative_languages=[]
            )
        
        # Normalize text for analysis
        clean_text = text.lower().strip()
        words = re.findall(r'\b\w+\b', clean_text)
        
        if not words:
            return LanguageDetectionResult(
                language=SupportedLanguage.ENGLISH,
                confidence=0.0,
                detected_text_sample=text[:100],
                alternative_languages=[]
            )
        
        # Check character-based patterns first (high confidence)
        for lang, pattern in self.language_patterns.items():
            if pattern.search(text):
                char_matches = len(pattern.findall(text))
                confidence = min(char_matches / len(text), 1.0)
                if confidence > 0.1:  # At least 10% characters match
                    return LanguageDetectionResult(
                        language=lang,
                        confidence=min(confidence * 2, 1.0),
                        detected_text_sample=text[:100],
                        alternative_languages=[]
                    )
        
        # Word-based detection for Latin script languages
        language_scores = {}
        
        for lang, stopwords in self.language_stopwords.items():
            matches = sum(1 for word in words if word in stopwords)
            score = matches / len(words) if words else 0
            language_scores[lang] = score
        
        # Sort by score
        sorted_scores = sorted(language_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_scores and sorted_scores[0][1] > 0:
            detected_lang = sorted_scores[0][0]
            confidence = sorted_scores[0][1]
            
            # Get alternative languages
            alternatives = [
                (lang, score) for lang, score in sorted_scores[1:4] 
                if score > 0.01
            ]
            
            return LanguageDetectionResult(
                language=detected_lang,
                confidence=min(confidence * 3, 1.0),  # Boost confidence
                detected_text_sample=text[:100],
                alternative_languages=alternatives
            )
        
        # Default to English if no clear detection
        return LanguageDetectionResult(
            language=SupportedLanguage.ENGLISH,
            confidence=0.1,
            detected_text_sample=text[:100],
            alternative_languages=[]
        )

class MultiLanguageProcessor:
    """Process documents in multiple languages."""
    
    def __init__(self):
        """Initialize multi-language processor."""
        self.detector = LanguageDetector()
        
        # Language-specific sentence separators
        self.sentence_separators = {
            SupportedLanguage.CHINESE: r'[。！？]',
            SupportedLanguage.JAPANESE: r'[。！？]',
            SupportedLanguage.KOREAN: r'[。！？]',
            SupportedLanguage.ARABIC: r'[.!?؟]',
            'default': r'[.!?]'
        }
        
        # Language-specific chunk size adjustments (words)
        self.language_chunk_adjustments = {
            SupportedLanguage.CHINESE: 0.5,  # Chinese characters are denser
            SupportedLanguage.JAPANESE: 0.6,
            SupportedLanguage.KOREAN: 0.7,
            SupportedLanguage.ARABIC: 0.8,
            SupportedLanguage.GERMAN: 1.2,   # German has longer compound words
            SupportedLanguage.FINNISH: 1.1,
            'default': 1.0
        }
    
    def process_text(
        self, 
        text: str,
        detect_language: bool = True,
        target_language: Optional[SupportedLanguage] = None
    ) -> Dict[str, any]:
        """Process text with language-aware techniques."""
        
        # Detect language if not specified
        if detect_language and not target_language:
            detection_result = self.detector.detect_language(text)
            language = detection_result.language
            detection_confidence = detection_result.confidence
        else:
            language = target_language or SupportedLanguage.ENGLISH
            detection_confidence = 1.0 if target_language else 0.5
        
        # Clean and normalize text based on language
        cleaned_text = self._clean_text_by_language(text, language)
        
        # Extract sentences using language-specific patterns
        sentences = self._extract_sentences(cleaned_text, language)
        
        # Adjust chunk size based on language
        adjustment_factor = self.language_chunk_adjustments.get(
            language, 
            self.language_chunk_adjustments['default']
        )
        
        return {
            'detected_language': language.value,
            'language_confidence': detection_confidence,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'sentence_count': len(sentences),
            'chunk_size_adjustment': adjustment_factor,
            'word_count': len(cleaned_text.split()),
            'character_count': len(cleaned_text)
        }
    
    def _clean_text_by_language(self, text: str, language: SupportedLanguage) -> str:
        """Clean text using language-specific rules."""
        # Basic cleaning for all languages
        cleaned = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        cleaned = cleaned.strip()
        
        # Language-specific cleaning
        if language in [SupportedLanguage.ARABIC]:
            # Remove Arabic diacritics for better processing
            cleaned = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', cleaned)
        
        elif language in [SupportedLanguage.CHINESE, SupportedLanguage.JAPANESE]:
            # Remove excessive punctuation repetition
            cleaned = re.sub(r'[。！？]{2,}', '。', cleaned)
        
        elif language == SupportedLanguage.KOREAN:
            # Basic Korean text normalization
            cleaned = re.sub(r'[ㅋㅎ]{2,}', '', cleaned)  # Remove repeated ㅋ, ㅎ
        
        return cleaned
    
    def _extract_sentences(self, text: str, language: SupportedLanguage) -> List[str]:
        """Extract sentences using language-specific patterns."""
        
        # Get appropriate sentence separator pattern
        pattern = self.sentence_separators.get(
            language, 
            self.sentence_separators['default']
        )
        
        # Split by sentence separators
        sentences = re.split(pattern, text)
        
        # Clean and filter sentences
        sentences = [
            sent.strip() 
            for sent in sentences 
            if sent.strip() and len(sent.strip()) > 3
        ]
        
        return sentences
    
    def get_optimal_chunk_size(
        self, 
        base_chunk_size: int, 
        language: SupportedLanguage
    ) -> int:
        """Get optimal chunk size for the specified language."""
        adjustment = self.language_chunk_adjustments.get(
            language,
            self.language_chunk_adjustments['default']
        )
        
        return int(base_chunk_size * adjustment)
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported."""
        try:
            SupportedLanguage(language_code)
            return True
        except ValueError:
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]
    
    def estimate_embedding_efficiency(self, language: SupportedLanguage) -> float:
        """Estimate embedding efficiency for different languages."""
        # Some languages may have better/worse embedding representations
        efficiency_scores = {
            SupportedLanguage.ENGLISH: 1.0,
            SupportedLanguage.SPANISH: 0.95,
            SupportedLanguage.FRENCH: 0.95,
            SupportedLanguage.GERMAN: 0.90,
            SupportedLanguage.ITALIAN: 0.93,
            SupportedLanguage.PORTUGUESE: 0.92,
            SupportedLanguage.RUSSIAN: 0.85,
            SupportedLanguage.CHINESE: 0.80,
            SupportedLanguage.JAPANESE: 0.75,
            SupportedLanguage.KOREAN: 0.80,
            SupportedLanguage.ARABIC: 0.70,
            SupportedLanguage.DUTCH: 0.88,
            SupportedLanguage.POLISH: 0.85,
            SupportedLanguage.TURKISH: 0.82,
            SupportedLanguage.SWEDISH: 0.87,
            SupportedLanguage.NORWEGIAN: 0.87,
            SupportedLanguage.DANISH: 0.87,
            SupportedLanguage.FINNISH: 0.83
        }
        
        return efficiency_scores.get(language, 0.75)

# Global processor instance
_global_processor: Optional[MultiLanguageProcessor] = None

def get_multilang_processor() -> MultiLanguageProcessor:
    """Get or create the global multi-language processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = MultiLanguageProcessor()
    return _global_processor

def detect_text_language(text: str) -> LanguageDetectionResult:
    """Convenience function to detect text language."""
    processor = get_multilang_processor()
    return processor.detector.detect_language(text)

def process_multilingual_text(
    text: str,
    target_language: Optional[str] = None
) -> Dict[str, any]:
    """Convenience function to process multilingual text."""
    processor = get_multilang_processor()
    
    lang_enum = None
    if target_language:
        try:
            lang_enum = SupportedLanguage(target_language)
        except ValueError:
            logger.warning(f"Unsupported language code: {target_language}")
    
    return processor.process_text(text, target_language=lang_enum) 