"""
OCR (Optical Character Recognition) functionality for image-based documents.
"""

import logging
import io
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

from .config import config

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor using Tesseract."""
    
    def __init__(self, tesseract_path: Optional[str] = None, language: str = 'eng'):
        self.tesseract_path = tesseract_path or config.tesseract_path
        self.language = language
        
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy."""
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL Image
        return Image.fromarray(thresh)
    
    def extract_text_from_image(self, image: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Extract text from a single image."""
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(processed_image, lang=self.language)
            
            # Get detailed data including confidence scores
            data = pytesseract.image_to_data(processed_image, lang=self.language, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'word_count': len(text.split()),
                'char_count': len(text),
                'language': self.language,
                'preprocessing_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            raise
    
    def extract_text_from_pdf_images(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Extract text from PDF by converting pages to images first."""
        pdf_path = Path(pdf_path)
        
        try:
            # Convert PDF pages to images
            pages = convert_from_path(pdf_path, dpi=300)
            
            results = []
            for page_num, page_image in enumerate(pages, 1):
                logger.info(f"Processing page {page_num} of {pdf_path}")
                
                # Extract text from page image
                page_result = self.extract_text_from_image(page_image)
                page_result.update({
                    'page': page_num,
                    'source_file': str(pdf_path),
                    'extraction_method': 'OCR'
                })
                
                results.append(page_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {e}")
            raise


class ImageDocumentParser:
    """Parser for image-based documents."""
    
    def __init__(self, ocr_processor: Optional[OCRProcessor] = None):
        self.ocr_processor = ocr_processor or OCRProcessor()
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'}
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse an image document using OCR."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {file_path.suffix}")
        
        try:
            # Extract text using OCR
            ocr_result = self.ocr_processor.extract_text_from_image(file_path)
            
            # Get file metadata
            stat = file_path.stat()
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
            
            metadata = {
                'file_name': file_path.name,
                'file_size': stat.st_size,
                'creation_date': stat.st_ctime,
                'modification_date': stat.st_mtime,
                'image_width': width,
                'image_height': height,
                'image_mode': mode,
                'ocr_confidence': ocr_result.get('confidence', 0),
                'ocr_language': ocr_result.get('language', 'unknown'),
                'word_count': ocr_result.get('word_count', 0),
                'char_count': ocr_result.get('char_count', 0),
            }
            
            return {
                'content': [{'page': 1, 'content': ocr_result['text']}],
                'metadata': metadata,
                'file_path': str(file_path),
                'file_type': 'image',
                'extraction_method': 'OCR'
            }
            
        except Exception as e:
            logger.error(f"Error parsing image document {file_path}: {e}")
            raise
