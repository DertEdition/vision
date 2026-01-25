"""
Interactive Drug Addition Tool

Provides an interactive command-line interface for adding new drugs
to the knowledge base. Supports both interactive and batch modes.

Usage:
    python scripts/add_drug_interactive.py
    python scripts/add_drug_interactive.py --quick "DrugName"
    python scripts/add_drug_interactive.py --validate drugs.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DRUGS_FILE = Path("data/drug_knowledge_base/drugs.json")
DEFAULT_CHROMA_DIR = Path("data/chroma_db")


@dataclass
class DrugEntry:
    """Drug entry data structure with validation."""
    drug_name: str
    active_ingredients: List[str] = field(default_factory=list)
    manufacturer: Optional[str] = None
    dosage_form: Optional[str] = None
    strengths: List[str] = field(default_factory=list)
    indications: Optional[str] = None
    usage: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    storage: Optional[str] = None
    description: Optional[str] = None
    prescription_status: Optional[str] = None
    
    def validate(self) -> List[str]:
        """
        Validate drug entry.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.drug_name or len(self.drug_name) < 2:
            errors.append("İlaç adı en az 2 karakter olmalı")
        
        if not self.active_ingredients:
            errors.append("En az bir etken madde gerekli")
        
        if self.dosage_form and self.dosage_form not in [
            'tablet', 'kapsül', 'şurup', 'süspansiyon', 'inhaler', 
            'enjeksiyon', 'krem', 'merhem', 'damla', 'saşe', 'efervesan tablet',
            'çiğneme tableti', 'enterik kaplı tablet', 'film tablet'
        ]:
            errors.append(f"Geçersiz dozaj formu: {self.dosage_form}")
        
        if self.prescription_status and self.prescription_status not in [
            'OTC', 'Rx', 'OTC/Rx', 'Unknown'
        ]:
            errors.append(f"Geçersiz reçete durumu: {self.prescription_status}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Remove None values and empty lists
        return {k: v for k, v in data.items() if v is not None and v != []}


class DrugDatabase:
    """Manager for drug JSON database."""
    
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.drugs: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load drugs from JSON file."""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.drugs = json.load(f)
                logger.info(f"Loaded {len(self.drugs)} drugs from {self.json_path}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error loading drugs: {e}")
                self.drugs = []
        else:
            self.drugs = []
    
    def _save(self):
        """Save drugs to JSON file."""
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.drugs, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.drugs)} drugs to {self.json_path}")
    
    def add(self, drug: DrugEntry) -> bool:
        """
        Add a drug to the database.
        
        Args:
            drug: DrugEntry to add
            
        Returns:
            True if added successfully
        """
        # Check for duplicates
        existing = self.find(drug.drug_name)
        if existing:
            logger.warning(f"Drug already exists: {drug.drug_name}")
            return False
        
        # Validate
        errors = drug.validate()
        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return False
        
        # Add
        self.drugs.append(drug.to_dict())
        self._save()
        logger.info(f"Added drug: {drug.drug_name}")
        return True
    
    def find(self, drug_name: str) -> Optional[Dict]:
        """Find drug by name (case-insensitive)."""
        drug_name_lower = drug_name.lower()
        for drug in self.drugs:
            if drug.get('drug_name', '').lower() == drug_name_lower:
                return drug
        return None
    
    def update(self, drug_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing drug."""
        drug = self.find(drug_name)
        if not drug:
            return False
        
        drug.update(updates)
        self._save()
        return True
    
    def remove(self, drug_name: str) -> bool:
        """Remove a drug by name."""
        drug = self.find(drug_name)
        if drug:
            self.drugs.remove(drug)
            self._save()
            return True
        return False
    
    def list_all(self) -> List[str]:
        """Get list of all drug names."""
        return [d.get('drug_name', 'Unknown') for d in self.drugs]
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all drugs in database."""
        results = {}
        for drug_dict in self.drugs:
            drug = DrugEntry(**drug_dict)
            errors = drug.validate()
            if errors:
                results[drug.drug_name] = errors
        return results


def prompt_input(prompt: str, required: bool = False, default: str = None) -> Optional[str]:
    """Get input from user with optional default."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        value = input(prompt).strip()
        
        if not value and default:
            return default
        
        if not value and required:
            print("Bu alan zorunlu!")
            continue
        
        return value if value else None


def prompt_list(prompt: str, min_items: int = 0) -> List[str]:
    """Get list of items from user."""
    print(f"{prompt} (her satıra bir tane, boş satır ile bitirin):")
    items = []
    while True:
        item = input(f"  {len(items) + 1}. ").strip()
        if not item:
            if len(items) >= min_items:
                break
            else:
                print(f"En az {min_items} öğe gerekli!")
                continue
        items.append(item)
    return items


def prompt_choice(prompt: str, choices: List[str], default: str = None) -> str:
    """Get choice from predefined options."""
    print(f"{prompt}:")
    for i, choice in enumerate(choices, 1):
        marker = " (varsayılan)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")
    
    while True:
        value = input("Seçiminiz (numara veya değer): ").strip()
        
        if not value and default:
            return default
        
        # Check if number
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        
        # Check if value matches
        if value in choices:
            return value
        
        print("Geçersiz seçim!")


def interactive_add_drug(db: DrugDatabase) -> Optional[DrugEntry]:
    """
    Interactively collect drug information.
    
    Returns:
        DrugEntry or None if cancelled
    """
    print("\n" + "=" * 60)
    print("YENİ İLAÇ EKLEME")
    print("=" * 60)
    print("(İptal etmek için Ctrl+C)")
    print()
    
    try:
        # Required fields
        drug_name = prompt_input("İlaç Adı", required=True)
        
        # Check if exists
        if db.find(drug_name):
            print(f"\n⚠️  '{drug_name}' zaten veritabanında mevcut!")
            overwrite = input("Üzerine yazmak ister misiniz? (e/h): ").lower()
            if overwrite != 'e':
                return None
            db.remove(drug_name)
        
        # Active ingredients
        active_ingredients = prompt_list("Etken Maddeler", min_items=1)
        
        # Manufacturer
        manufacturer = prompt_input("Üretici Firma")
        
        # Dosage form
        dosage_forms = ['tablet', 'kapsül', 'şurup', 'süspansiyon', 'inhaler', 
                       'enjeksiyon', 'krem', 'merhem', 'damla', 'saşe', 
                       'efervesan tablet', 'çiğneme tableti']
        dosage_form = prompt_choice("Dozaj Formu", dosage_forms, default='tablet')
        
        # Strengths
        print("\nDoz Miktarları (ör: 500 mg, 10 mg/ml):")
        strengths = prompt_list("Dozlar")
        
        # Indications
        indications = prompt_input("Kullanım Alanları (endikasyonlar)")
        
        # Usage
        usage = prompt_input("Kullanım Talimatları (doz, sıklık)")
        
        # Warnings
        print("\nUyarılar ve Önlemler:")
        warnings = prompt_list("Uyarılar")
        
        # Side effects
        print("\nYan Etkiler:")
        side_effects = prompt_list("Yan Etkiler")
        
        # Interactions
        print("\nİlaç Etkileşimleri:")
        interactions = prompt_list("Etkileşimler")
        
        # Storage
        storage = prompt_input("Saklama Koşulları")
        
        # Description
        description = prompt_input("Genel Açıklama")
        
        # Prescription status
        prescription_statuses = ['OTC', 'Rx', 'OTC/Rx']
        prescription_status = prompt_choice("Reçete Durumu", prescription_statuses, default='OTC')
        
        # Create entry
        drug = DrugEntry(
            drug_name=drug_name,
            active_ingredients=active_ingredients,
            manufacturer=manufacturer,
            dosage_form=dosage_form,
            strengths=strengths,
            indications=indications,
            usage=usage,
            warnings=warnings,
            side_effects=side_effects,
            interactions=interactions,
            storage=storage,
            description=description,
            prescription_status=prescription_status
        )
        
        # Validate
        errors = drug.validate()
        if errors:
            print("\n⚠️  Doğrulama Hataları:")
            for error in errors:
                print(f"  - {error}")
            return None
        
        # Preview
        print("\n" + "-" * 60)
        print("ÖNIZLEME:")
        print("-" * 60)
        print(json.dumps(drug.to_dict(), indent=2, ensure_ascii=False))
        print("-" * 60)
        
        confirm = input("\nKaydetmek ister misiniz? (e/h): ").lower()
        if confirm == 'e':
            return drug
        
        return None
        
    except KeyboardInterrupt:
        print("\n\nİptal edildi.")
        return None


def quick_add_drug(db: DrugDatabase, drug_name: str):
    """Quick add with minimal information."""
    print(f"\nHızlı Ekleme: {drug_name}")
    
    # Minimal required info
    active = input("Etken Madde: ").strip()
    if not active:
        print("Etken madde zorunlu!")
        return
    
    drug = DrugEntry(
        drug_name=drug_name,
        active_ingredients=[active],
        dosage_form='tablet',
        prescription_status='Unknown'
    )
    
    if db.add(drug):
        print(f"✓ '{drug_name}' eklendi (minimal bilgi)")
    else:
        print(f"✗ '{drug_name}' eklenemedi")


def validate_database(json_path: Path):
    """Validate all entries in database."""
    db = DrugDatabase(json_path)
    
    print(f"\nVeri Tabanı Doğrulaması: {json_path}")
    print("=" * 60)
    print(f"Toplam İlaç: {len(db.drugs)}")
    
    errors = db.validate_all()
    
    if errors:
        print(f"\n⚠️  Hatalı Kayıtlar: {len(errors)}")
        for drug_name, drug_errors in errors.items():
            print(f"\n  {drug_name}:")
            for error in drug_errors:
                print(f"    - {error}")
    else:
        print("\n✓ Tüm kayıtlar geçerli!")
    
    # Statistics
    print("\n" + "-" * 60)
    print("İSTATİSTİKLER:")
    print("-" * 60)
    
    manufacturers = set()
    dosage_forms = {}
    prescription_stats = {'OTC': 0, 'Rx': 0, 'OTC/Rx': 0, 'Unknown': 0}
    
    for drug in db.drugs:
        if mfr := drug.get('manufacturer'):
            manufacturers.add(mfr)
        
        df = drug.get('dosage_form', 'Bilinmiyor')
        dosage_forms[df] = dosage_forms.get(df, 0) + 1
        
        ps = drug.get('prescription_status', 'Unknown')
        if ps in prescription_stats:
            prescription_stats[ps] += 1
    
    print(f"Üretici Sayısı: {len(manufacturers)}")
    print(f"\nDozaj Formları:")
    for form, count in sorted(dosage_forms.items(), key=lambda x: -x[1]):
        print(f"  {form}: {count}")
    
    print(f"\nReçete Durumu:")
    for status, count in prescription_stats.items():
        if count > 0:
            print(f"  {status}: {count}")


def update_knowledge_base(json_path: Path, chroma_dir: Path):
    """Update ChromaDB knowledge base after changes."""
    print("\nKnowledge base güncelleniyor...")
    
    try:
        # Import populate script functionality
        from populate_knowledge_base import populate_knowledge_base
        
        populate_knowledge_base(
            json_file=json_path,
            persist_directory=str(chroma_dir),
            clear_existing=True
        )
        print("✓ Knowledge base güncellendi!")
        
    except ImportError:
        print("⚠️  populate_knowledge_base.py bulunamadı")
        print("Manuel olarak çalıştırın:")
        print(f"  python scripts/populate_knowledge_base.py --clear --file {json_path}")
    except Exception as e:
        print(f"✗ Güncelleme hatası: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="İnteraktif ilaç ekleme aracı"
    )
    
    parser.add_argument(
        "--quick",
        type=str,
        metavar="DRUG_NAME",
        help="Hızlı ekleme modu - sadece temel bilgiler"
    )
    
    parser.add_argument(
        "--validate",
        type=str,
        metavar="JSON_FILE",
        help="Mevcut veritabanını doğrula"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Tüm ilaçları listele"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default=str(DEFAULT_DRUGS_FILE),
        help="İlaç veritabanı dosyası"
    )
    
    parser.add_argument(
        "--update-kb",
        action="store_true",
        help="Ekleme sonrası knowledge base'i güncelle"
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.file)
    
    # Validation mode
    if args.validate:
        validate_database(Path(args.validate))
        return
    
    # Initialize database
    db = DrugDatabase(json_path)
    
    # List mode
    if args.list:
        print(f"\nKayıtlı İlaçlar ({len(db.drugs)}):")
        for drug_name in sorted(db.list_all()):
            print(f"  • {drug_name}")
        return
    
    # Quick add mode
    if args.quick:
        quick_add_drug(db, args.quick)
        if args.update_kb:
            update_knowledge_base(json_path, DEFAULT_CHROMA_DIR)
        return
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("İLAÇ VERİTABANI YÖNETİMİ")
    print("=" * 60)
    print(f"Veritabanı: {json_path}")
    print(f"Mevcut İlaç Sayısı: {len(db.drugs)}")
    print()
    
    while True:
        print("\nSeçenekler:")
        print("  1. Yeni ilaç ekle")
        print("  2. İlaçları listele")
        print("  3. İlaç ara")
        print("  4. Veritabanını doğrula")
        print("  5. Knowledge base'i güncelle")
        print("  6. Çıkış")
        print()
        
        choice = input("Seçiminiz: ").strip()
        
        if choice == '1':
            drug = interactive_add_drug(db)
            if drug and db.add(drug):
                print(f"\n✓ '{drug.drug_name}' başarıyla eklendi!")
                
                if args.update_kb:
                    update_knowledge_base(json_path, DEFAULT_CHROMA_DIR)
        
        elif choice == '2':
            print(f"\nKayıtlı İlaçlar ({len(db.drugs)}):")
            for drug_name in sorted(db.list_all()):
                print(f"  • {drug_name}")
        
        elif choice == '3':
            search = input("İlaç adı: ").strip()
            drug = db.find(search)
            if drug:
                print("\n" + json.dumps(drug, indent=2, ensure_ascii=False))
            else:
                print(f"'{search}' bulunamadı.")
        
        elif choice == '4':
            validate_database(json_path)
        
        elif choice == '5':
            update_knowledge_base(json_path, DEFAULT_CHROMA_DIR)
        
        elif choice == '6':
            print("\nÇıkılıyor...")
            break
        
        else:
            print("Geçersiz seçim!")


if __name__ == "__main__":
    main()
