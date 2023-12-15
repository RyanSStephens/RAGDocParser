"""
Configuration management for RAG Document Parser.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """Application configuration."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Vector Database
    chroma_persist_directory: str = Field("./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    # OCR Settings
    tesseract_path: Optional[str] = Field(None, env="TESSERACT_PATH")
    
    # Chunking
    default_chunk_size: int = Field(1000, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(200, env="DEFAULT_CHUNK_OVERLAP")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
config = Config()
