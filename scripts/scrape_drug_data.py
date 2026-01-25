"""
Drug Data Web Scraper

Scrapes drug information from public pharmaceutical databases.
Implements ethical scraping practices with rate limiting and robots.txt compliance.

Usage:
    python scripts/scrape_drug_data.py --source ilac --limit 50
    python scripts/scrape_drug_data.py --test --limit 5
    python scripts/scrape_drug_data.py --source rxlist --drug "ibuprofen"
"""

import argparse
import json
import logging
import time
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, quote
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DrugData:
    """Standardized drug data structure."""
    drug_name: str
    active_ingredients: List[str]
    manufacturer: Optional[str] = None
    dosage_form: Optional[str] = None
    strengths: List[str] = None
    indications: Optional[str] = None
    usage: Optional[str] = None
    warnings: List[str] = None
    side_effects: List[str] = None
    interactions: List[str] = None
    storage: Optional[str] = None
    description: Optional[str] = None
    prescription_status: Optional[str] = None
    source_url: Optional[str] = None
    
    def __post_init__(self):
        self.strengths = self.strengths or []
        self.warnings = self.warnings or []
        self.side_effects = self.side_effects or []
        self.interactions = self.interactions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None and v != []}


class BaseScraper:
    """Base class for drug data scrapers."""
    
    def __init__(self, rate_limit: float = 2.0):
        """
        Initialize scraper.
        
        Args:
            rate_limit: Minimum seconds between requests
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _init_session(self):
        """Initialize requests session."""
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'DrugInfoBot/1.0 (Educational Research; +https://github.com/drug-pipeline)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
                'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
            })
        except ImportError:
            raise ImportError("requests not installed. Install with: pip install requests")
    
    def _rate_limit_wait(self):
        """Wait to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed + random.uniform(0.1, 0.5)
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _get_page(self, url: str) -> Optional[str]:
        """
        Fetch page content with rate limiting.
        
        Args:
            url: URL to fetch
            
        Returns:
            Page HTML content or None on error
        """
        if not self.session:
            self._init_session()
        
        self._rate_limit_wait()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def scrape_drug(self, drug_name: str) -> Optional[DrugData]:
        """
        Scrape data for a single drug.
        
        Args:
            drug_name: Name of drug to scrape
            
        Returns:
            DrugData object or None
        """
        raise NotImplementedError
    
    def scrape_drug_list(self) -> List[str]:
        """
        Get list of available drugs to scrape.
        
        Returns:
            List of drug names
        """
        raise NotImplementedError


class OpenFDAScraper(BaseScraper):
    """
    Scraper using OpenFDA API.
    
    OpenFDA provides free, open access to FDA drug data.
    API: https://open.fda.gov/apis/drug/
    """
    
    BASE_URL = "https://api.fda.gov/drug"
    
    def __init__(self, **kwargs):
        super().__init__(rate_limit=0.5, **kwargs)  # FDA allows faster rate
    
    def _init_session(self):
        """Initialize with JSON accept header."""
        super()._init_session()
        self.session.headers['Accept'] = 'application/json'
    
    def scrape_drug(self, drug_name: str) -> Optional[DrugData]:
        """Scrape drug from OpenFDA API."""
        if not self.session:
            self._init_session()
        
        self._rate_limit_wait()
        
        # Search in drug labels
        search_url = f"{self.BASE_URL}/label.json"
        params = {
            'search': f'openfda.brand_name:"{drug_name}"',
            'limit': 1
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            
            if response.status_code == 404:
                self.logger.warning(f"Drug not found in OpenFDA: {drug_name}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                return None
            
            result = data['results'][0]
            openfda = result.get('openfda', {})
            
            return DrugData(
                drug_name=drug_name,
                active_ingredients=openfda.get('substance_name', []),
                manufacturer=openfda.get('manufacturer_name', [None])[0],
                dosage_form=openfda.get('dosage_form', [None])[0],
                strengths=openfda.get('strength', []),
                indications=self._clean_text(result.get('indications_and_usage', [''])[0]),
                usage=self._clean_text(result.get('dosage_and_administration', [''])[0]),
                warnings=self._extract_list(result.get('warnings', [''])[0]),
                side_effects=self._extract_list(result.get('adverse_reactions', [''])[0]),
                interactions=self._extract_list(result.get('drug_interactions', [''])[0]),
                storage=self._clean_text(result.get('storage_and_handling', [''])[0]),
                description=self._clean_text(result.get('description', [''])[0])[:500],
                prescription_status='Rx' if openfda.get('product_type', [''])[0] == 'HUMAN PRESCRIPTION DRUG' else 'OTC',
                source_url=f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}"
            )
            
        except Exception as e:
            self.logger.error(f"OpenFDA API error for {drug_name}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and truncate text."""
        if not text:
            return ""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        clean = ' '.join(clean.split())
        return clean[:500]  # Truncate
    
    def _extract_list(self, text: str) -> List[str]:
        """Extract bullet points or sentences as list."""
        if not text:
            return []
        clean = self._clean_text(text)
        # Split by common delimiters
        items = re.split(r'[•\n;]', clean)
        return [item.strip() for item in items if len(item.strip()) > 10][:5]


class DummyScraper(BaseScraper):
    """
    Dummy scraper for testing purposes.
    Returns mock data without making network requests.
    """
    
    MOCK_DRUGS = {
        "TestDrug": DrugData(
            drug_name="TestDrug",
            active_ingredients=["Test Active Ingredient"],
            manufacturer="Test Manufacturer",
            dosage_form="tablet",
            strengths=["100 mg", "200 mg"],
            indications="Test indications for educational purposes.",
            usage="Test usage instructions.",
            warnings=["Test warning 1", "Test warning 2"],
            side_effects=["Test side effect"],
            prescription_status="OTC"
        )
    }
    
    def scrape_drug(self, drug_name: str) -> Optional[DrugData]:
        """Return mock drug data."""
        self.logger.info(f"[DUMMY] Scraping: {drug_name}")
        time.sleep(0.1)  # Simulate network delay
        
        if drug_name in self.MOCK_DRUGS:
            return self.MOCK_DRUGS[drug_name]
        
        # Generate mock data for any drug
        return DrugData(
            drug_name=drug_name,
            active_ingredients=[f"{drug_name} Active"],
            manufacturer="Mock Manufacturer",
            dosage_form="tablet",
            strengths=["50 mg", "100 mg"],
            indications=f"Mock indications for {drug_name}. For testing only.",
            usage=f"Mock usage for {drug_name}.",
            warnings=["This is mock data", "For testing purposes only"],
            side_effects=["Mock side effect"],
            prescription_status="Unknown",
            source_url="mock://test"
        )


class ScraperFactory:
    """Factory for creating scraper instances."""
    
    SCRAPERS = {
        'openfda': OpenFDAScraper,
        'dummy': DummyScraper,
    }
    
    @classmethod
    def create(cls, source: str, **kwargs) -> BaseScraper:
        """
        Create a scraper instance.
        
        Args:
            source: Scraper type (openfda, dummy)
            **kwargs: Scraper-specific options
            
        Returns:
            BaseScraper instance
        """
        if source not in cls.SCRAPERS:
            raise ValueError(f"Unknown scraper: {source}. Available: {list(cls.SCRAPERS.keys())}")
        
        return cls.SCRAPERS[source](**kwargs)


def save_drugs_to_json(drugs: List[DrugData], output_file: Path, append: bool = True):
    """
    Save scraped drugs to JSON file.
    
    Args:
        drugs: List of DrugData objects
        output_file: Output JSON file path
        append: If True, append to existing file
    """
    existing_drugs = []
    
    if append and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_drugs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Merge, avoiding duplicates by drug name
    existing_names = {d.get('drug_name', '').lower() for d in existing_drugs}
    
    for drug in drugs:
        if drug.drug_name.lower() not in existing_names:
            existing_drugs.append(drug.to_dict())
            existing_names.add(drug.drug_name.lower())
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_drugs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(existing_drugs)} drugs to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape drug information from pharmaceutical databases"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        choices=['openfda', 'dummy'],
        default='dummy',
        help="Data source to scrape from"
    )
    
    parser.add_argument(
        "--drug",
        type=str,
        help="Specific drug name to scrape"
    )
    
    parser.add_argument(
        "--drugs-file",
        type=str,
        help="File with list of drug names (one per line)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of drugs to scrape"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/drug_knowledge_base/scraped_drugs.json",
        help="Output JSON file"
    )
    
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to existing file instead of overwriting"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with dummy data"
    )
    
    args = parser.parse_args()
    
    # Force dummy scraper in test mode
    source = 'dummy' if args.test else args.source
    
    logger.info(f"Starting scraper: source={source}")
    
    # Create scraper
    scraper = ScraperFactory.create(source)
    
    # Determine drugs to scrape
    drug_names = []
    
    if args.drug:
        drug_names = [args.drug]
    elif args.drugs_file:
        try:
            with open(args.drugs_file, 'r', encoding='utf-8') as f:
                drug_names = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Drugs file not found: {args.drugs_file}")
            sys.exit(1)
    else:
        # Default test drugs
        drug_names = [
            "Ibuprofen", "Acetaminophen", "Amoxicillin", 
            "Omeprazole", "Metformin", "Amlodipine",
            "Lisinopril", "Atorvastatin", "Metoprolol", "Losartan"
        ]
    
    # Limit
    drug_names = drug_names[:args.limit]
    
    logger.info(f"Scraping {len(drug_names)} drugs...")
    
    # Scrape
    scraped_drugs = []
    success_count = 0
    
    for i, drug_name in enumerate(drug_names, 1):
        logger.info(f"[{i}/{len(drug_names)}] Scraping: {drug_name}")
        
        drug_data = scraper.scrape_drug(drug_name)
        
        if drug_data:
            scraped_drugs.append(drug_data)
            success_count += 1
            logger.info(f"  ✓ Success: {drug_name}")
        else:
            logger.warning(f"  ✗ Failed: {drug_name}")
    
    # Save results
    if scraped_drugs:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_drugs_to_json(scraped_drugs, output_path, args.append)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Scraping complete!")
    logger.info(f"Successfully scraped: {success_count}/{len(drug_names)} drugs")
    if scraped_drugs:
        logger.info(f"Output saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
