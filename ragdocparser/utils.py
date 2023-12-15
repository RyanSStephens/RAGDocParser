"""
Utility functions for RAGDocParser.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters except newlines and tabs
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

def extract_metadata_from_filename(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from filename and path."""
    metadata = {
        'filename': file_path.name,
        'directory': str(file_path.parent),
        'extension': file_path.suffix.lower(),
        'size': file_path.stat().st_size if file_path.exists() else 0,
        'stem': file_path.stem
    }
    
    # Try to extract date from filename
    date_pattern = r'(\d{4}[-_]\d{2}[-_]\d{2})'
    date_match = re.search(date_pattern, file_path.name)
    if date_match:
        metadata['date_in_filename'] = date_match.group(1)
    
    # Extract version from filename
    version_pattern = r'[vV]?(\d+\.\d+(?:\.\d+)?)'
    version_match = re.search(version_pattern, file_path.name)
    if version_match:
        metadata['version_in_filename'] = version_match.group(1)
    
    return metadata

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters."""
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    safe_name = re.sub(r'[\x00-\x1f\x7f]', '', safe_name)
    
    # Limit length
    if len(safe_name) > 255:
        name, ext = Path(safe_name).stem, Path(safe_name).suffix
        max_name_len = 255 - len(ext)
        safe_name = name[:max_name_len] + ext
    
    return safe_name

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate token count for text.
    
    This is a rough estimation. For exact counts, use tiktoken.
    """
    if not text:
        return 0
    
    # Rough estimation: ~4 characters per token for English
    if model in ["gpt-3.5-turbo", "gpt-4"]:
        return len(text) // 4
    elif "claude" in model.lower():
        return len(text) // 4  # Similar to GPT
    else:
        return len(text) // 4  # Default estimation

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

class ProgressBar:
    """Simple progress bar for CLI operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress bar."""
        self.current += increment
        self._display()
    
    def _display(self):
        """Display the progress bar."""
        if self.total == 0:
            return
        
        progress = self.current / self.total
        bar_length = 50
        filled_length = int(bar_length * progress)
        
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed_time * (self.total - self.current) / self.current
            eta_str = f" ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        print(f'\r{self.description}: |{bar}| {self.current}/{self.total} '
              f'({progress:.1%}){eta_str}', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                 f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

def merge_metadata(base_metadata: Dict[str, Any], 
                  additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two metadata dictionaries, with additional taking precedence."""
    merged = base_metadata.copy()
    merged.update(additional_metadata)
    return merged

def is_text_file(file_path: Path) -> bool:
    """Check if a file is likely a text file based on extension and content."""
    text_extensions = {'.txt', '.md', '.rst', '.py', '.js', '.html', '.css', 
                      '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf',
                      '.log', '.csv', '.tsv'}
    
    if file_path.suffix.lower() in text_extensions:
        return True
    
    # Check first few bytes for binary indicators
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(512)
            if b'\x00' in chunk:  # Null bytes indicate binary
                return False
            
            # Check if most bytes are printable
            text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
            return bool(chunk.translate(None, text_chars))
    except:
        return False 