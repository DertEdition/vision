"""
Drug Information Web Scraper for ilacabak.com

Bu modül ilacabak.com sitesinden ilaç bilgilerini (yan etkiler, uyarılar, 
kullanım bilgileri vb.) çekerek drugs.json dosyasını zenginleştirir.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from urllib.parse import quote, urljoin
import unicodedata

import requests
from bs4 import BeautifulSoup

# PDF işleme için
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("UYARI: PyMuPDF (fitz) kurulu değil. PDF işleme devre dışı.")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DrugInfo:
    """İlaç bilgilerini tutan veri sınıfı"""
    drug_name: str
    indications: str = ""
    usage: str = ""
    warnings: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    storage: str = ""
    description: str = ""
    source_url: str = ""
    pdf_url: str = ""


class IlacabakScraper:
    """ilacabak.com web scraper sınıfı"""
    
    BASE_URL = "https://www.ilacabak.com"
    SEARCH_URL = "https://www.ilacabak.com/aralist.php"
    
    # Rate limiting
    REQUEST_DELAY = 1.5  # saniye
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
        })
        self.cache_dir = cache_dir or Path("./scraper_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.pdf_cache_dir = self.cache_dir / "pdfs"
        self.pdf_cache_dir.mkdir(exist_ok=True)
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Rate limiting uygula"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _normalize_drug_name(self, name: str) -> str:
        """İlaç ismini normalize et"""
        # Türkçe karakterleri koru, ama temizle
        name = name.strip().upper()
        # Parantez içindeki kısımları kaldır
        name = re.sub(r'\([^)]*\)', '', name)
        # Özel karakterleri temizle
        name = re.sub(r'[^\w\s]', ' ', name)
        # Çoklu boşlukları tek boşluğa çevir
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def _create_search_slug(self, drug_name: str) -> str:
        """Arama için URL-safe slug oluştur"""
        # Türkçe karakterleri dönüştür
        tr_map = {
            'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
            'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
        }
        slug = drug_name.lower()
        for tr_char, en_char in tr_map.items():
            slug = slug.replace(tr_char, en_char)
        # Alfanumerik olmayan karakterleri tire ile değiştir
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug
    
    def search_drug(self, drug_name: str) -> Optional[str]:
        """
        İlaç ismini sitede ara ve detay sayfası URL'sini döndür.
        Önce ilaç listesi sayfasında arar, sonra en yakın eşleşmeyi bulur.
        """
        # İlk harfe göre liste sayfasına git
        first_letter = drug_name.strip()[0].upper()
        if not first_letter.isalpha():
            first_letter = 'A'  # Sayı ile başlıyorsa A'dan başla
        
        list_url = f"{self.SEARCH_URL}?Id={first_letter}"
        
        self._rate_limit()
        try:
            response = self.session.get(list_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Liste sayfası alınamadı ({first_letter}): {e}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tüm ilaç linklerini bul
        normalized_search = self._normalize_drug_name(drug_name)
        search_words = normalized_search.split()
        
        best_match = None
        best_score = 0
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True).upper()
            
            # ilacabak.com linklerini kontrol et
            if 'ilacabak.com/' not in href and not href.startswith('/'):
                continue
            if href.endswith('.php') or 'aralist' in href:
                continue
            
            # Eşleşme skoru hesapla
            normalized_text = self._normalize_drug_name(text)
            
            # Tam eşleşme kontrolü
            if normalized_search in normalized_text or normalized_text in normalized_search:
                score = 100
            else:
                # Kelime bazlı eşleşme
                text_words = normalized_text.split()
                matching_words = sum(1 for word in search_words if any(word in tw or tw in word for tw in text_words))
                score = (matching_words / len(search_words)) * 100 if search_words else 0
            
            if score > best_score:
                best_score = score
                if href.startswith('http'):
                    best_match = href
                else:
                    best_match = urljoin(self.BASE_URL, href)
        
        if best_score >= 50:  # En az %50 eşleşme
            logger.info(f"Eşleşme bulundu: {drug_name} -> {best_match} (skor: {best_score:.0f}%)")
            return best_match
        else:
            logger.warning(f"Eşleşme bulunamadı: {drug_name} (en iyi skor: {best_score:.0f}%)")
            return None
    
    def get_drug_page(self, url: str) -> Optional[BeautifulSoup]:
        """İlaç detay sayfasını al"""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Sayfa alınamadı ({url}): {e}")
            return None
    
    def extract_pdf_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Sayfadan PDF URL'sini çıkar"""
        # PDF linkini bul
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if '.pdf' in href.lower():
                if href.startswith('http'):
                    return href
                return urljoin(base_url, href)
        return None
    
    def download_pdf(self, pdf_url: str, drug_name: str) -> Optional[Path]:
        """PDF dosyasını indir"""
        if not PDF_SUPPORT:
            logger.warning("PDF desteği kurulu değil")
            return None
        
        # Cache kontrolü
        safe_name = self._create_search_slug(drug_name)
        pdf_path = self.pdf_cache_dir / f"{safe_name}.pdf"
        
        if pdf_path.exists():
            logger.info(f"PDF cache'den alınıyor: {pdf_path}")
            return pdf_path
        
        self._rate_limit()
        try:
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"PDF indirildi: {pdf_path}")
            return pdf_path
        except requests.RequestException as e:
            logger.error(f"PDF indirilemedi ({pdf_url}): {e}")
            return None
    
    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """PDF dosyasından bilgileri çıkar"""
        if not PDF_SUPPORT:
            return {}
        
        result = {
            'indications': '',
            'usage': '',
            'warnings': [],
            'side_effects': [],
            'interactions': [],
            'storage': '',
            'description': ''
        }
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            
            # Bölümleri ayır ve parse et
            result = self._parse_drug_text(full_text)
            
        except Exception as e:
            logger.error(f"PDF parse edilemedi ({pdf_path}): {e}")
        
        return result
    
    def _parse_drug_text(self, text: str) -> Dict[str, Any]:
        """Kullanma talimatı metnini parse et"""
        result = {
            'indications': '',
            'usage': '',
            'warnings': [],
            'side_effects': [],
            'interactions': [],
            'storage': '',
            'description': ''
        }
        
        # Metin bölümlerini tanımla
        sections = {
            'indications': [
                r'(?:nedir\s+ve\s+ne\s+için\s+kullanılır|endikasyonlar|kullanım\s+alanları)',
                r'(?:1\.\s*[A-ZÇĞİÖŞÜ].*?nedir.*?kullanılır)'
            ],
            'usage': [
                r'(?:nasıl\s+kullanılır|kullanım\s+şekli|dozaj|posology)',
                r'(?:3\.\s*.*?nasıl\s+kullanılır)'
            ],
            'warnings': [
                r'(?:kullanmadan\s+önce|uyarılar|kontrendikasyonlar|dikkat\s+edilmesi)',
                r'(?:2\.\s*.*?kullanmadan\s+önce)'
            ],
            'side_effects': [
                r'(?:yan\s+etkiler|istenmeyen\s+etkiler|olası\s+yan)',
                r'(?:4\.\s*.*?yan\s+etkileri)'
            ],
            'interactions': [
                r'(?:etkileşimler|diğer\s+ilaçlar|birlikte\s+kullanım)'
            ],
            'storage': [
                r'(?:saklama|muhafaza|nasıl\s+saklanır)',
                r'(?:5\.\s*.*?saklanır)'
            ]
        }
        
        text_lower = text.lower()
        
        for field, patterns in sections.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Bölümün başlangıç pozisyonunu bul
                    start_pos = match.end()
                    
                    # Sonraki bölümün başlangıcını bul
                    next_section_pos = len(text)
                    for other_patterns in sections.values():
                        for other_pattern in other_patterns:
                            other_match = re.search(other_pattern, text_lower[start_pos:])
                            if other_match:
                                pos = start_pos + other_match.start()
                                if pos < next_section_pos:
                                    next_section_pos = pos
                    
                    # Bölüm metnini çıkar
                    section_text = text[start_pos:next_section_pos].strip()
                    
                    # Temizle
                    section_text = re.sub(r'\s+', ' ', section_text)
                    section_text = section_text[:2000]  # Max karakter limiti
                    
                    if field in ['warnings', 'side_effects', 'interactions']:
                        # Liste olarak parse et
                        items = re.split(r'[•\-\*\n]', section_text)
                        result[field] = [item.strip() for item in items if item.strip() and len(item.strip()) > 5][:20]
                    else:
                        result[field] = section_text
                    
                    break
        
        return result
    
    def scrape_drug(self, drug_name: str) -> Optional[DrugInfo]:
        """Tek bir ilaç için tüm bilgileri topla"""
        logger.info(f"İlaç scrape ediliyor: {drug_name}")
        
        # 1. İlaç sayfasını bul
        drug_url = self.search_drug(drug_name)
        if not drug_url:
            return None
        
        # 2. Ana sayfayı al
        soup = self.get_drug_page(drug_url)
        if not soup:
            return None
        
        drug_info = DrugInfo(
            drug_name=drug_name,
            source_url=drug_url
        )
        
        # 3. Genel bilgiyi çıkar (sayfadan)
        page_text = soup.get_text(separator=' ')
        
        # Kısa açıklama
        desc_match = re.search(r'hakkında\s+kısa\s+bilgi[:\s]*(.{50,500})', page_text.lower())
        if desc_match:
            drug_info.description = desc_match.group(1).strip()[:500]
        
        # 4. Kullanma talimatı sayfasını kontrol et
        kt_url = drug_url + '/kullanma-talimati'
        kt_soup = self.get_drug_page(kt_url)
        
        if kt_soup:
            # PDF linkini bul
            pdf_url = self.extract_pdf_url(kt_soup, kt_url)
            if pdf_url:
                drug_info.pdf_url = pdf_url
                
                # PDF'i indir ve parse et
                pdf_path = self.download_pdf(pdf_url, drug_name)
                if pdf_path:
                    pdf_data = self.parse_pdf(pdf_path)
                    
                    drug_info.indications = pdf_data.get('indications', '')
                    drug_info.usage = pdf_data.get('usage', '')
                    drug_info.warnings = pdf_data.get('warnings', [])
                    drug_info.side_effects = pdf_data.get('side_effects', [])
                    drug_info.interactions = pdf_data.get('interactions', [])
                    drug_info.storage = pdf_data.get('storage', '')
        
        return drug_info


def enrich_drugs_subset(
    drugs_json_path: Path,
    output_path: Path,
    limit: int = 10,
    start_index: int = 0
) -> List[Dict]:
    """
    drugs.json'dan bir alt küme seçip zenginleştir
    """
    # drugs.json'ı yükle
    with open(drugs_json_path, 'r', encoding='utf-8') as f:
        drugs = json.load(f)
    
    # Alt küme seç
    subset = drugs[start_index:start_index + limit]
    
    scraper = IlacabakScraper()
    enriched = []
    
    for i, drug in enumerate(subset):
        drug_name = drug.get('drug_name', '')
        logger.info(f"[{i+1}/{len(subset)}] İşleniyor: {drug_name}")
        
        try:
            info = scraper.scrape_drug(drug_name)
            
            if info:
                # Mevcut drug kaydını güncelle
                updated_drug = drug.copy()
                
                if info.indications:
                    updated_drug['indications'] = info.indications
                if info.usage:
                    updated_drug['usage'] = info.usage
                if info.warnings:
                    updated_drug['warnings'] = info.warnings
                if info.side_effects:
                    updated_drug['side_effects'] = info.side_effects
                if info.interactions:
                    updated_drug['interactions'] = info.interactions
                if info.storage:
                    updated_drug['storage'] = info.storage
                if info.description:
                    updated_drug['description'] = info.description
                
                # Meta bilgi ekle
                updated_drug['_source_url'] = info.source_url
                updated_drug['_pdf_url'] = info.pdf_url
                
                enriched.append(updated_drug)
                logger.info(f"✓ Zenginleştirildi: {drug_name}")
            else:
                # Orijinal kaydı ekle
                enriched.append(drug)
                logger.warning(f"✗ Zenginleştirilemedi: {drug_name}")
                
        except Exception as e:
            logger.error(f"Hata ({drug_name}): {e}")
            enriched.append(drug)
    
    # Sonuçları kaydet
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Sonuçlar kaydedildi: {output_path}")
    return enriched


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='İlacabak.com Web Scraper')
    parser.add_argument('--input', '-i', type=str, required=True, help='drugs.json dosya yolu')
    parser.add_argument('--output', '-o', type=str, required=True, help='Çıktı dosya yolu')
    parser.add_argument('--limit', '-l', type=int, default=10, help='İşlenecek ilaç sayısı')
    parser.add_argument('--start', '-s', type=int, default=0, help='Başlangıç indeksi')
    
    args = parser.parse_args()
    
    enrich_drugs_subset(
        drugs_json_path=Path(args.input),
        output_path=Path(args.output),
        limit=args.limit,
        start_index=args.start
    )
