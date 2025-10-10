from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Set
import asyncio
import re

class WebScraper:
    """Web scraper using Playwright for JavaScript-heavy sites"""

    def __init__(
        self,
        max_pages: int = 50,
        max_depth: int = 3,
        timeout: int = 30000,
        headless: bool = True
    ):
        """
        Initialize web scraper

        Args:
            max_pages: Maximum number of pages to scrape per domain
            max_depth: Maximum depth to follow links
            timeout: Page load timeout in milliseconds
            headless: Run browser in headless mode
        """
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout
        self.headless = headless

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _is_same_domain(self, url: str, base_domain: str) -> bool:
        """Check if URL belongs to the same domain"""
        return url.startswith(base_domain)

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for scraping (exclude media, documents, etc.)"""
        excluded_extensions = [
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.css', '.js', '.xml', '.json', '.zip', '.exe', '.doc',
            '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
        ]

        # Check for excluded extensions
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False

        # Check for excluded patterns
        excluded_patterns = [
            r'/wp-admin/', r'/wp-content/', r'/wp-includes/',
            r'/login', r'/signup', r'/register', r'/cart',
            r'/checkout', r'/account', r'#'
        ]

        if any(re.search(pattern, url.lower()) for pattern in excluded_patterns):
            return False

        return True

    def _clean_text(self, html: str) -> str:
        """Clean HTML and extract text"""
        soup = BeautifulSoup(html, 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML"""
        soup = BeautifulSoup(html, 'lxml')
        links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            # Remove fragment identifier
            absolute_url = absolute_url.split('#')[0]
            links.append(absolute_url)

        return list(set(links))  # Remove duplicates

    async def _scrape_page(self, page: Page, url: str) -> Dict[str, Any]:
        """Scrape a single page"""
        try:
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            await asyncio.sleep(1)  # Wait for dynamic content

            # Get HTML content
            html = await page.content()

            # Extract text
            text = self._clean_text(html)

            # Extract links
            links = self._extract_links(html, url)

            # Get page title
            title = await page.title()

            return {
                "url": url,
                "title": title,
                "text": text,
                "links": links,
                "success": True
            }

        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "success": False
            }

    async def scrape_domain(self, start_url: str) -> List[Dict[str, Any]]:
        """
        Scrape entire domain starting from a URL

        Args:
            start_url: Starting URL to scrape

        Returns:
            List of scraped pages with text and metadata
        """
        base_domain = self._get_domain(start_url)
        visited: Set[str] = set()
        to_visit: List[tuple[str, int]] = [(start_url, 0)]  # (url, depth)
        scraped_pages = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()

            while to_visit and len(scraped_pages) < self.max_pages:
                current_url, depth = to_visit.pop(0)

                # Skip if already visited
                if current_url in visited:
                    continue

                # Skip if max depth exceeded
                if depth > self.max_depth:
                    continue

                # Skip if not valid URL
                if not self._is_valid_url(current_url):
                    continue

                visited.add(current_url)

                # Scrape page
                result = await self._scrape_page(page, current_url)

                if result["success"]:
                    scraped_pages.append({
                        "url": result["url"],
                        "title": result["title"],
                        "text": result["text"],
                        "depth": depth
                    })

                    # Add new links to visit
                    for link in result.get("links", []):
                        if (
                            link not in visited and
                            self._is_same_domain(link, base_domain) and
                            self._is_valid_url(link)
                        ):
                            to_visit.append((link, depth + 1))

                # Small delay to be respectful
                await asyncio.sleep(0.5)

            await browser.close()

        return scraped_pages

# Global scraper instance
web_scraper = WebScraper()
