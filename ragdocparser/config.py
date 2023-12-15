"""
Configuration management for RAGDocParser.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for RAG document parser."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_config_file()
            
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100
            },
            "vector_db": {
                "provider": "chromadb",
                "collection_name": "ragdocparser",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "llm": {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "cohere_api_key": os.getenv("COHERE_API_KEY"),
                "default_model": "openai"
            },
            "ocr": {
                "tesseract_path": None,
                "language": "eng",
                "dpi": 300
            },
            "parsing": {
                "supported_formats": [".pdf", ".docx", ".txt", ".html", ".md"],
                "max_file_size_mb": 50,
                "timeout_seconds": 300
            }
        }
    
    def _load_config_file(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)
                self._merge_config(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing."""
        for key, value in new_config.items():
            if key in self._config and isinstance(self._config[key], dict):
                self._config[key].update(value)
            else:
                self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'chunking.chunk_size')."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key path."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config.copy() 