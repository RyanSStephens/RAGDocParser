"""
LLM provider integrations for text generation and analysis.
"""

import logging
import openai
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .config import config

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        self.model = model
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text using OpenAI GPT."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary using OpenAI."""
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters:

{text}

Summary:"""
        
        return self.generate_text(prompt, max_tokens=max_length//3)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-2"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for Claude integration")
        
        self.api_key = api_key or config.anthropic_api_key
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text using Anthropic Claude."""
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                max_tokens_to_sample=max_tokens,
                temperature=temperature
            )
            return response.completion.strip()
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            raise
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary using Claude."""
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters:

{text}"""
        
        return self.generate_text(prompt, max_tokens=max_length//3)


class LLMManager:
    """Manager for multiple LLM providers."""
    
    def __init__(self):
        self.providers = {}
        
        # Initialize available providers
        if config.openai_api_key:
            try:
                self.providers['openai'] = OpenAIProvider()
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        if config.anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                self.providers['anthropic'] = AnthropicProvider()
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {e}")
    
    def get_provider(self, provider_name: str) -> BaseLLMProvider:
        """Get a specific provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available. Available: {list(self.providers.keys())}")
        return self.providers[provider_name]
    
    def generate_document_summary(self, document: Dict[str, Any], provider: str = None) -> str:
        """Generate a summary for a document."""
        if not self.providers:
            raise RuntimeError("No LLM providers available")
        
        # Use specified provider or default to first available
        provider_name = provider or list(self.providers.keys())[0]
        llm_provider = self.get_provider(provider_name)
        
        # Combine document content
        content_parts = []
        for page in document.get('content', []):
            content_parts.append(page.get('content', ''))
        
        full_text = ' '.join(content_parts)
        
        # Truncate if too long (keep first 4000 chars for context)
        if len(full_text) > 4000:
            full_text = full_text[:4000] + "..."
        
        try:
            summary = llm_provider.generate_summary(full_text)
            return summary
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            raise
