"""
Web scraper module for extracting content from documentation sites.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from typing import List, Dict, Any, Set, Optional
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class ScrapedPage:
    """Represents a scraped web page."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    links: List[str]

class WebScraper:
    """Web scraper optimized for documentation sites."""
    
    def __init__(self, 
                 delay: float = 1.0,
                 max_pages: int = 100,
                 user_agent: str = None):
        """Initialize web scraper.
        
        Args:
            delay: Delay between requests in seconds
            max_pages: Maximum pages to scrape per domain
            user_agent: Custom user agent string
        """
        self.delay = delay
        self.max_pages = max_pages
        self.session = requests.Session()
        
        # Set user agent
        if user_agent:
            self.session.headers.update({'User-Agent': user_agent})
        else:
            self.session.headers.update({
                'User-Agent': 'RAGDocParser/1.0 (Document Parser Bot)'
            })
        
        self.visited_urls: Set[str] = set()
        self.scraped_pages: List[ScrapedPage] = []
    
    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL."""
        # Remove fragment
        if '#' in url:
            url = url.split('#')[0]
        
        # Remove common tracking parameters
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Remove tracking parameters
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']
        for param in tracking_params:
            query_params.pop(param, None)
        
        # Rebuild URL
        if query_params:
            query_string = '&'.join([f"{k}={v[0]}" for k, v in query_params.items()])
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query_string}"
        else:
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        return url
    
    def _extract_text_content(self, soup: BeautifulSoup, base_url: str) -> tuple[str, str, List[str]]:
        """Extract text content, title, and links from soup."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Also try h1 if no title
        if not title:
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text().strip()
        
        # Extract main content
        content_selectors = [
            'main', 'article', '.content', '.documentation', 
            '.docs', '.main-content', '#content', '#main'
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            content_element = soup.find('body')
        
        if content_element:
            # Get text content
            content = content_element.get_text(separator='\n', strip=True)
            
            # Clean up content
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 3:  # Filter out very short lines
                    cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)
        else:
            content = ""
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            # Filter internal links only
            if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                cleaned_url = self._clean_url(absolute_url)
                if cleaned_url not in links:
                    links.append(cleaned_url)
        
        return content, title, links
    
    def _should_scrape_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped."""
        parsed = urlparse(url)
        
        # Must be same domain
        if parsed.netloc != base_domain:
            return False
        
        # Skip common non-content URLs
        skip_patterns = [
            r'/search', r'/login', r'/register', r'/contact',
            r'/api/', r'/admin/', r'\.pdf$', r'\.jpg$', r'\.png$',
            r'\.css$', r'\.js$', r'/download/', r'/edit/'
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url.lower()):
                return False
        
        return True
    
    def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single page."""
        try:
            logger.info(f"Scraping: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content, title, links = self._extract_text_content(soup, url)
            
            if not content.strip():
                logger.warning(f"No content extracted from {url}")
                return None
            
            # Create metadata
            metadata = {
                'url': url,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(content),
                'scraped_at': time.time()
            }
            
            page = ScrapedPage(
                url=url,
                title=title or f"Page from {urlparse(url).netloc}",
                content=content,
                metadata=metadata,
                links=links
            )
            
            return page
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_site(self, 
                   start_url: str, 
                   max_depth: int = 3,
                   url_patterns: List[str] = None) -> List[ScrapedPage]:
        """Scrape an entire site starting from a URL.
        
        Args:
            start_url: Starting URL to scrape
            max_depth: Maximum depth to crawl
            url_patterns: Optional regex patterns for URLs to include
            
        Returns:
            List of scraped pages
        """
        self.visited_urls.clear()
        self.scraped_pages.clear()
        
        base_domain = urlparse(start_url).netloc
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        
        while urls_to_visit and len(self.scraped_pages) < self.max_pages:
            current_url, depth = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            if depth > max_depth:
                continue
            
            self.visited_urls.add(current_url)
            
            # Add delay between requests
            if len(self.visited_urls) > 1:
                time.sleep(self.delay)
            
            # Scrape the page
            page = self.scrape_page(current_url)
            if page:
                self.scraped_pages.append(page)
                
                # Add new URLs to visit
                for link in page.links:
                    if (link not in self.visited_urls and 
                        self._should_scrape_url(link, base_domain)):
                        
                        # Check URL patterns if provided
                        if url_patterns:
                            if any(re.search(pattern, link) for pattern in url_patterns):
                                urls_to_visit.append((link, depth + 1))
                        else:
                            urls_to_visit.append((link, depth + 1))
        
        logger.info(f"Scraped {len(self.scraped_pages)} pages from {base_domain}")
        return self.scraped_pages
    
    def scrape_documentation_site(self, 
                                base_url: str,
                                docs_path: str = "/docs") -> List[ScrapedPage]:
        """Scrape a documentation site with common patterns.
        
        Args:
            base_url: Base URL of the site
            docs_path: Path to documentation section
            
        Returns:
            List of scraped pages
        """
        start_url = urljoin(base_url, docs_path)
        
        # Common documentation URL patterns
        doc_patterns = [
            r'/docs/', r'/documentation/', r'/guide/', r'/tutorial/',
            r'/api/', r'/reference/', r'/manual/'
        ]
        
        return self.scrape_site(
            start_url, 
            max_depth=4, 
            url_patterns=doc_patterns
        )
    
    def save_scraped_data(self, output_dir: str):
        """Save scraped data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save individual pages
        for i, page in enumerate(self.scraped_pages):
            filename = f"page_{i:03d}_{urlparse(page.url).path.replace('/', '_')}.txt"
            # Clean filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                f.write(f"URL: {page.url}\n")
                f.write(f"Title: {page.title}\n")
                f.write(f"Scraped at: {time.ctime(page.metadata['scraped_at'])}\n")
                f.write("-" * 50 + "\n")
                f.write(page.content)
        
        # Save metadata
        metadata = []
        for page in self.scraped_pages:
            metadata.append({
                'url': page.url,
                'title': page.title,
                'content_length': len(page.content),
                'scraped_at': page.metadata['scraped_at']
            })
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(self.scraped_pages)} pages to {output_dir}")
    
    def convert_to_documents(self) -> List[Dict[str, Any]]:
        """Convert scraped pages to document format for processing."""
        documents = []
        
        for page in self.scraped_pages:
            doc = {
                'content': page.content,
                'title': page.title,
                'url': page.url,
                'source_type': 'web_scrape',
                'scraped_at': page.metadata['scraped_at']
            }
            documents.append(doc)
        
        return documents 