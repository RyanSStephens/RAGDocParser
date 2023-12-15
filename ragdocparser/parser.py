"""
Main document parser module with multi-format support and OCR.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import requests
from PIL import Image
import pytesseract

# Internal imports
from .config import Config
from .chunker import TextChunker, TextChunk
from .vectordb import VectorDBManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentInfo:
    """Information about a parsed document."""
    filename: str
    filepath: str
    content: str
    metadata: Dict[str, Any]
    parse_time: float
    file_size: int
    content_hash: str

class DocumentParser:
    """Main document parser with multi-format support."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize document parser."""
        self.config = config or Config()
        self.chunker = TextChunker(
            chunk_size=self.config.get("chunking.chunk_size", 1000),
            chunk_overlap=self.config.get("chunking.chunk_overlap", 200),
            min_chunk_size=self.config.get("chunking.min_chunk_size", 100)
        )
        self.vectordb = VectorDBManager(self.config)
        
        # Supported file extensions
        self.supported_formats = self.config.get(
            "parsing.supported_formats", 
            [".pdf", ".docx", ".txt", ".html", ".md"]
        )
        
        # OCR configuration
        self.ocr_enabled = True
        tesseract_path = self.config.get("ocr.tesseract_path")
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def _extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        
                return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF {filepath}: {e}")
            return ""
    
    def _extract_text_from_docx(self, filepath: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = DocxDocument(filepath)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading DOCX {filepath}: {e}")
            return ""
    
    def _extract_text_from_html(self, filepath: str) -> str:
        """Extract text from HTML file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text and clean it up
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text
        except Exception as e:
            logger.error(f"Error reading HTML {filepath}: {e}")
            return ""
    
    def _extract_text_from_txt(self, filepath: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {filepath}: {e}")
            return ""
    
    def parse_file(self, filepath: str) -> Optional[DocumentInfo]:
        """Parse a single file and extract text."""
        start_time = time.time()
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        path = Path(filepath)
        extension = path.suffix.lower()
        
        # Extract text based on file type
        content = ""
        if extension == ".pdf":
            content = self._extract_text_from_pdf(filepath)
        elif extension == ".docx":
            content = self._extract_text_from_docx(filepath)
        elif extension in [".html", ".htm"]:
            content = self._extract_text_from_html(filepath)
        elif extension in [".txt", ".md"]:
            content = self._extract_text_from_txt(filepath)
        else:
            content = self._extract_text_from_txt(filepath)
        
        if not content.strip():
            logger.warning(f"No content extracted from {filepath}")
            return None
        
        # Get metadata
        metadata = {
            "filename": path.name,
            "file_extension": extension,
            "file_size": path.stat().st_size,
            "directory": str(path.parent)
        }
        
        parse_time = time.time() - start_time
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        
        return DocumentInfo(
            filename=path.name,
            filepath=str(path.absolute()),
            content=content,
            metadata=metadata,
            parse_time=parse_time,
            file_size=path.stat().st_size,
            content_hash=content_hash
        )
    
    def parse_directory(self, directory: str, recursive: bool = True, max_workers: int = 4) -> List[DocumentInfo]:
        """Parse all supported files in a directory with parallel processing."""
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Collect all files to parse
        files_to_parse = []
        path = Path(directory)
        
        if recursive:
            for ext in self.supported_formats:
                files_to_parse.extend(path.rglob(f"*{ext}"))
        else:
            for file in path.iterdir():
                if file.is_file() and file.suffix.lower() in self.supported_formats:
                    files_to_parse.append(file)
        
        logger.info(f"Found {len(files_to_parse)} files to parse")
        
        # Parse files in parallel for better performance
        documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.parse_file, str(file)): file 
                for file in files_to_parse
            }
            
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    doc_info = future.result()
                    if doc_info:
                        documents.append(doc_info)
                        logger.info(f"Parsed {doc_info.filename}")
                except Exception as e:
                    logger.error(f"Error parsing {file}: {e}")
        
        return documents
    
    def save_to_vector_db(self, documents: List[DocumentInfo], collection_name: str = None):
        """Process documents and save to vector database."""
        all_chunks = []
        
        for doc in documents:
            doc_metadata = doc.metadata.copy()
            doc_metadata.update({
                "content_hash": doc.content_hash,
                "source": doc.filepath
            })
            
            chunks = self.chunker.chunk_text(doc.content, doc_metadata)
            all_chunks.extend(chunks)
        
        if all_chunks:
            self.chunker.generate_embeddings(all_chunks)
            self.vectordb.add_documents(all_chunks, collection_name)
        
        return len(all_chunks) 