"""
Document parser for various file formats.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """Parse a document and return extracted content."""
        pass


class PDFParser(BaseParser):
    """Parser for PDF documents."""
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """Parse PDF document."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        text_content = []
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata.update({
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', ''),
                    'producer': pdf_reader.metadata.get('/Producer', ''),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                    'modification_date': pdf_reader.metadata.get('/ModDate', ''),
                    'pages': len(pdf_reader.pages)
                })
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append({
                                'page': page_num + 1,
                                'content': page_text.strip()
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise
        
        return {
            'content': text_content,
            'metadata': metadata,
            'file_path': str(file_path),
            'file_type': 'pdf'
        }


class TXTParser(BaseParser):
    """Parser for plain text documents."""
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """Parse text document."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Unable to decode file {file_path}")
        
        metadata = {
            'file_size': file_path.stat().st_size,
            'creation_date': file_path.stat().st_ctime,
            'modification_date': file_path.stat().st_mtime,
        }
        
        return {
            'content': [{'page': 1, 'content': content}],
            'metadata': metadata,
            'file_path': str(file_path),
            'file_type': 'txt'
        }


class DocumentParser:
    """Main document parser that handles multiple file formats."""
    
    def __init__(self):
        self.parsers = {
            '.pdf': PDFParser(),
            '.txt': TXTParser(),
        }
        
        if DOCX_AVAILABLE:
            self.parsers['.docx'] = DocxParser()
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """Parse a document based on its file extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.parsers:
            raise ValueError(f"Unsupported file format: {extension}")
        
        parser = self.parsers[extension]
        return parser.parse(file_path)
    
    def supported_formats(self) -> List[str]:
        """Return list of supported file formats."""
        return list(self.parsers.keys())
