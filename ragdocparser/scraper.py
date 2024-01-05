"""
Web scraping functionality for extracting content from URLs.
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin, urlparse
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import selenium
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper for extracting content from web pages."""
    
    def __init__(self, 
                 user_agent: str = "RAGDocParser/1.0",
                 timeout: int = 30,
                 delay: float = 1.0):
        self.user_agent = user_agent
        self.timeout = timeout
        self.delay = delay
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def extract_content_with_requests(self, url: str) -> Dict[str, Any]:
        """Extract content using requests and BeautifulSoup."""
        if not BS4_AVAILABLE:
            raise ImportError("beautifulsoup4 is required for web scraping")
        
        try:
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
            # Try to find main content areas
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
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body')
            
            # Remove unwanted elements
            if main_content:
                for unwanted in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    unwanted.decompose()
                
                # Extract text
                text_content = main_content.get_text(separator=' ', strip=True)
            else:
                text_content = soup.get_text(separator=' ', strip=True)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                link_text = link.get_text().strip()
                if href and link_text:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(url, href)
                    links.append({
                        'url': absolute_url,
                        'text': link_text
                    })
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                src = img['src']
                alt = img.get('alt', '')
                absolute_url = urljoin(url, src)
                images.append({
                    'url': absolute_url,
                    'alt': alt
                })
            
            metadata = {
                'url': url,
                'title': title_text,
                'description': description,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.content),
                'encoding': response.encoding,
                'links_count': len(links),
                'images_count': len(images),
                'word_count': len(text_content.split()),
                'char_count': len(text_content),
                'extraction_method': 'requests_beautifulsoup'
            }
            
            return {
                'content': text_content,
                'metadata': metadata,
                'links': links[:50],  # Limit links to avoid huge data
                'images': images[:20],  # Limit images
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            raise
    
    def extract_content_with_selenium(self, url: str, wait_time: int = 10) -> Dict[str, Any]:
        """Extract content using Selenium for JavaScript-heavy pages."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("selenium is required for JavaScript-enabled scraping")
        
        driver = None
        try:
            # Setup Chrome driver with headless option
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'--user-agent={self.user_agent}')
            
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            
            # Load page
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            # Extract content
            title = driver.title
            
            # Try to get main content
            content_selectors = [
                'main', 'article', '[role="main"]',
                '.content', '#content', '.post-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                try:
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    main_content = element.text
                    break
                except:
                    continue
            
            # If no main content found, use body
            if not main_content:
                body = driver.find_element(By.TAG_NAME, "body")
                main_content = body.text
            
            # Extract links
            links = []
            try:
                link_elements = driver.find_elements(By.TAG_NAME, "a")
                for link in link_elements[:50]:  # Limit to avoid huge data
                    href = link.get_attribute('href')
                    text = link.text.strip()
                    if href and text:
                        links.append({'url': href, 'text': text})
            except:
                pass
            
            metadata = {
                'url': url,
                'title': title,
                'word_count': len(main_content.split()),
                'char_count': len(main_content),
                'links_count': len(links),
                'extraction_method': 'selenium_webdriver'
            }
            
            return {
                'content': main_content,
                'metadata': metadata,
                'links': links,
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL with Selenium {url}: {e}")
            raise
        finally:
            if driver:
                driver.quit()
    
    def scrape_url(self, url: str, use_selenium: bool = False) -> Dict[str, Any]:
        """Scrape content from a URL."""
        logger.info(f"Scraping URL: {url}")
        
        # Add delay to be respectful
        time.sleep(self.delay)
        
        if use_selenium:
            result = self.extract_content_with_selenium(url)
        else:
            result = self.extract_content_with_requests(url)
        
        return result
    
    def scrape_multiple_urls(self, urls: List[str], use_selenium: bool = False) -> List[Dict[str, Any]]:
        """Scrape content from multiple URLs."""
        results = []
        
        for i, url in enumerate(urls):
            try:
                logger.info(f"Scraping URL {i+1}/{len(urls)}: {url}")
                result = self.scrape_url(url, use_selenium=use_selenium)
                results.append(result)
                
                # Add delay between requests
                if i < len(urls) - 1:
                    time.sleep(self.delay)
                    
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                continue
        
        return results


class URLDocumentParser:
    """Parser for URL-based documents."""
    
    def __init__(self, scraper: Optional[WebScraper] = None):
        self.scraper = scraper or WebScraper()
    
    def parse(self, url: str, use_selenium: bool = False) -> Dict[str, Any]:
        """Parse content from a URL."""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {url}")
            
            # Scrape content
            scraped_data = self.scraper.scrape_url(url, use_selenium=use_selenium)
            
            # Format as document structure
            content = scraped_data.get('content', '')
            metadata = scraped_data.get('metadata', {})
            
            # Add additional metadata
            metadata.update({
                'source_type': 'web_page',
                'domain': parsed_url.netloc,
                'scheme': parsed_url.scheme,
                'path': parsed_url.path,
            })
            
            return {
                'content': [{'page': 1, 'content': content}],
                'metadata': metadata,
                'file_path': url,
                'file_type': 'url',
                'links': scraped_data.get('links', []),
                'images': scraped_data.get('images', []),
            }
            
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            raise
