"""
OCR (Optical Character Recognition) module for RAGDocParser.
Handles extracting text from images and scanned PDFs.
"""

import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR processor using Tesseract and EasyOCR for text extraction."""
    
    def __init__(self, use_gpu: bool = False, languages: List[str] = None):
        """Initialize OCR processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration (EasyOCR only)
            languages: List of language codes for recognition
        """
        self.use_gpu = use_gpu
        self.languages = languages or ['en']
        self.tesseract_available = self._check_tesseract()
        self.easyocr_available = self._check_easyocr()
        
        if self.easyocr_available:
            self._init_easyocr()
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            # Test tesseract
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, pytesseract.TesseractNotFoundError):
            logger.warning("Tesseract not found. Install tesseract-ocr and pytesseract")
            return False
    
    def _check_easyocr(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            return True
        except ImportError:
            logger.warning("EasyOCR not found. Install with: pip install easyocr")
            return False
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu
            )
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_available = False
    
    def extract_text_from_image(self, 
                               image_path: Union[str, Path],
                               method: str = "auto") -> Dict[str, Any]:
        """Extract text from image file.
        
        Args:
            image_path: Path to image file
            method: OCR method ('tesseract', 'easyocr', 'auto')
            
        Returns:
            Dictionary with extracted text and metadata
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Determine OCR method
        if method == "auto":
            if self.easyocr_available:
                method = "easyocr"
            elif self.tesseract_available:
                method = "tesseract"
            else:
                raise RuntimeError("No OCR engine available")
        
        try:
            if method == "easyocr" and self.easyocr_available:
                return self._extract_with_easyocr(image_path)
            elif method == "tesseract" and self.tesseract_available:
                return self._extract_with_tesseract(image_path)
            else:
                raise RuntimeError(f"OCR method '{method}' not available")
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "method": method,
                "error": str(e)
            }
    
    def _extract_with_tesseract(self, image_path: Path) -> Dict[str, Any]:
        """Extract text using Tesseract."""
        import pytesseract
        from PIL import Image
        
        # Open and process image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text with confidence
        data = pytesseract.image_to_data(
            image, 
            output_type=pytesseract.Output.DICT,
            lang='+'.join(self.languages)
        )
        
        # Filter out low confidence detections
        text_parts = []
        confidences = []
        
        for i, conf in enumerate(data['conf']):
            if int(conf) > 30:  # Confidence threshold
                text = data['text'][i].strip()
                if text:
                    text_parts.append(text)
                    confidences.append(int(conf))
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "text": full_text,
            "confidence": avg_confidence / 100.0,  # Normalize to 0-1
            "method": "tesseract",
            "word_count": len(text_parts),
            "raw_data": data
        }
    
    def _extract_with_easyocr(self, image_path: Path) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        # Read image with EasyOCR
        results = self.easyocr_reader.readtext(str(image_path))
        
        # Process results
        text_parts = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Confidence threshold
                text_parts.append(text)
                confidences.append(confidence)
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "text": full_text,
            "confidence": avg_confidence,
            "method": "easyocr",
            "word_count": len(text_parts),
            "raw_results": results
        }
    
    def extract_text_from_pdf_images(self, 
                                   pdf_path: Union[str, Path],
                                   method: str = "auto") -> List[Dict[str, Any]]:
        """Extract text from PDF by converting pages to images.
        
        Args:
            pdf_path: Path to PDF file
            method: OCR method to use
            
        Returns:
            List of dictionaries with text for each page
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            import pdf2image
        except ImportError:
            raise ImportError("pdf2image not installed. Install with: pip install pdf2image")
        
        # Convert PDF pages to images
        try:
            images = pdf2image.convert_from_path(pdf_path)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []
        
        results = []
        
        for page_num, image in enumerate(images, 1):
            try:
                # Save image to memory
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Save temporary image file (cross-platform)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                image.save(temp_path)
                
                # Extract text
                result = self.extract_text_from_image(temp_path, method)
                result['page'] = page_num
                result['source'] = str(pdf_path)
                
                results.append(result)
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                results.append({
                    "text": "",
                    "confidence": 0.0,
                    "page": page_num,
                    "error": str(e)
                })
        
        return results
    
    def is_image_based_pdf(self, pdf_path: Union[str, Path]) -> bool:
        """Check if PDF is image-based (scanned) by analyzing text content.
        
        Returns True if PDF appears to be image-based.
        """
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check first few pages
                total_text_length = 0
                pages_checked = min(3, len(reader.pages))
                
                for i in range(pages_checked):
                    try:
                        page_text = reader.pages[i].extract_text()
                        total_text_length += len(page_text.strip())
                    except:
                        continue
                
                # If very little text found, likely image-based
                avg_text_per_page = total_text_length / pages_checked
                return avg_text_per_page < 100  # Threshold
                
        except Exception as e:
            logger.error(f"Error checking PDF type: {e}")
            return False
    
    def batch_process_images(self, 
                           image_paths: List[Union[str, Path]],
                           method: str = "auto") -> List[Dict[str, Any]]:
        """Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            method: OCR method to use
            
        Returns:
            List of extraction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.extract_text_from_image(image_path, method)
                result['source'] = str(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    "text": "",
                    "confidence": 0.0,
                    "source": str(image_path),
                    "error": str(e)
                })
        
        return results


def create_ocr_processor(use_gpu: bool = False, 
                        languages: List[str] = None) -> OCRProcessor:
    """Factory function to create OCR processor."""
    return OCRProcessor(use_gpu=use_gpu, languages=languages) 