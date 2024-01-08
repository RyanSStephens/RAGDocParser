"""
Web scraping functionality with improved error handling and retry logic.
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
from pathlib import Path
import random

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper with retry logic and improved error handling."""
    
    def __init__(self, 
                 user_agent: str = "RAGDocParser/1.0",
                 timeout: int = 30,
                 delay: float = 1.0,
                 max_retries: int = 3):
        self.user_agent = user_agent
        self.timeout = timeout
        self.delay = delay
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def extract_content_with_requests(self, url: str) -> Dict[str, Any]:
        """Extract content with retry logic."""
        if not BS4_AVAILABLE:
            raise ImportError("beautifulsoup4 is required for web scraping")
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = self.delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retry attempt {attempt + 1}, waiting {delay:.2f}s")
                    time.sleep(delay)
                
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract metadata
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ''
                
                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc.get('content', '') if meta_desc else ''
                
                # Extract main content
                content_selectors = [
                    'main', 'article', '.content', '#content',
                    '.post-content', '.entry-content', '.article-content'
                ]
                
                main_content = None
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        main_content = element
                        break
                
                if not main_content:
                    main_content = soup.find('body')
                
                # Remove unwanted elements
                if main_content:
                    for unwanted in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                        unwanted.decompose()
                    text_content = main_content.get_text(separator=' ', strip=True)
                else:
                    text_content = soup.get_text(separator=' ', strip=True)
                
                # Extract links
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text().strip()
                    if href and link_text:
                        absolute_url = urljoin(url, href)
                        links.append({'url': absolute_url, 'text': link_text})
                
                metadata = {
                    'url': url,
                    'title': title_text,
                    'description': description,
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', ''),
                    'word_count': len(text_content.split()),
                    'char_count': len(text_content),
                    'extraction_method': 'requests_beautifulsoup',
                    'attempts': attempt + 1
                }
                
                return {
                    'content': text_content,
                    'metadata': metadata,
                    'links': links[:50],
                }
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                continue
        
        # All attempts failed
        logger.error(f"All {self.max_retries} attempts failed for {url}")
        raise last_exception
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL with improved error handling."""
        logger.info(f"Scraping URL: {url}")
        
        # Add delay to be respectful
        time.sleep(self.delay)
        
        return self.extract_content_with_requests(url)
