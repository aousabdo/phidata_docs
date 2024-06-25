import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import os

class PhidataDocsScraper:
    def __init__(self, start_url):
        self.start_url = start_url
        self.base_url = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
        self.visited_urls = set()
        self.to_visit = [start_url]
        self.scraped_data = []
        self.total_pages = 1  # Start with 1 (the start_url)
        self.pages_scraped = 0

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def is_same_domain(self, url):
        return urlparse(url).netloc == urlparse(self.start_url).netloc

    def get_page_content(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_links(self, html, base_url):
        soup = BeautifulSoup(html, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            if self.is_valid_url(full_url) and self.is_same_domain(full_url):
                yield full_url

    def scrape_page(self, url):
        self.pages_scraped += 1
        print(f"\nScraping page {self.pages_scraped} of {self.total_pages} found so far")
        print(f"URL: {url}")
        
        html = self.get_page_content(url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string if soup.title else "No title"
            
            self.scraped_data.append({
                "url": url,
                "title": title,
                "content": html  # Storing the full HTML content
            })
            
            print(f"Title: {title}")

            new_links = 0
            for link in self.extract_links(html, url):
                if link not in self.visited_urls and link not in self.to_visit:
                    self.to_visit.append(link)
                    self.total_pages += 1
                    new_links += 1
            
            print(f"Found {new_links} new links on this page")
            print(f"Total pages to scrape: {self.total_pages}")

    def save_results(self):
        os.makedirs("phidata_docs", exist_ok=True)
        
        for page in self.scraped_data:
            filename = urlparse(page["url"]).path
            if filename.endswith('/'):
                filename += 'index.html'
            elif not filename.endswith('.html'):
                filename += '.html'
            filename = filename.lstrip('/')
            filepath = os.path.join("phidata_docs", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page["content"])
        
        index = [{
            "url": page["url"],
            "title": page["title"],
            "local_path": urlparse(page["url"]).path.lstrip('/')
        } for page in self.scraped_data]
        
        with open('phidata_docs_index.json', 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=4)
        
        print(f"\nScraped data saved to phidata_docs directory")
        print(f"Index saved to phidata_docs_index.json")
        print(f"Total pages scraped: {self.pages_scraped}")

    def run(self):
        while self.to_visit:
            url = self.to_visit.pop(0)
            if url not in self.visited_urls:
                self.scrape_page(url)
                self.visited_urls.add(url)
                time.sleep(1)  # Be polite, wait a second between requests
        self.save_results()

if __name__ == "__main__":
    start_url = "https://docs.phidata.com/"
    scraper = PhidataDocsScraper(start_url)
    scraper.run()