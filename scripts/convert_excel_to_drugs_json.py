#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel'den drugs.json formatına dönüşüm scripti.
TİTCK'den çekilen Ruhsatlı Beşeri Tıbbi Ürünler listesini
mevcut drugs.json formatına dönüştürür.
"""

import pandas as pd
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_drug_base_name(full_name: str) -> str:
    """İlaç isminden temel adı çıkarır (doz bilgilerini ve miktarları çıkarır)"""
    if pd.isna(full_name):
        return ""
    
    name = str(full_name).strip()
    
    # Parantez içindekileri daha sonra analiz için tut ama ana isimden çıkar
    # Örn: "PAROL 500 MG TABLET, 20 TABLET" -> "PAROL"
    
    # Önce virgülden sonrasını kaldır (genellikle miktar bilgisi)
    name = name.split(',')[0].strip()
    
    # Dozaj ve form bilgilerini kaldır
    # "500 MG", "100 MCG", "0.75 MG" gibi
    name = re.sub(r'\s+[\d,\.]+\s*(MG|MCG|G|ML|%|IU|GRAM|MILIGRAM|MIKROGRAM)\b', '', name, flags=re.IGNORECASE)
    
    # Yüzde işaretli ifadeleri kaldır (%0.9, %5, vb)
    name = re.sub(r'%\s*[\d,\.]+', '', name)
    
    # Formları kaldır - bunları dosage_form olarak ayrıca tutalım
    forms = [
        r'\bTABLET\b', r'\bFİLM TABLET\b', r'\bFILM TABLET\b', r'\bKAPSÜL\b', r'\bKAPSUL\b',
        r'\bSÜSPANSİYON\b', r'\bSUSPANSIYON\b', r'\bŞURUP\b', r'\bSURUP\b', 
        r'\bAMPUL\b', r'\bFLAKON\b', r'\bENJEKSİYON\b', r'\bENJEKSIYON\b',
        r'\bİNFÜZYON\b', r'\bINFUZYON\b', r'\bİNHALER\b', r'\bINHALER\b',
        r'\bKREM\b', r'\bJEL\b', r'\bPOMAD\b', r'\bMERHEM\b', r'\bLOSYON\b',
        r'\bDAMLA\b', r'\bSPREY\b', r'\bEFERVESAN\b', r'\bGRANÜL\b', r'\bGRANUL\b',
        r'\bSAŞE\b', r'\bSASE\b', r'\bÇÖZELTİ\b', r'\bCOZELTI\b', r'\bÇİĞNEME\b', r'\bCIGNEME\b',
        r'\bÇİĞNENEBİLİR\b', r'\bCIGNENEBILIR\b', r'\bTOZ\b', r'\bPELLET\b',
        r'\bI\.V\.\b', r'\bI\.M\.\b', r'\bS\.C\.\b', r'\bÖZELTİ\b', r'\bOZELTI\b',
        r'\bİÇİN\b', r'\bICIN\b', r'\bENTERİK\b', r'\bENTERIK\b', r'\bKAPLI\b',
        r'\bUZATILMIŞ SALINIMLI\b', r'\bUZATILMIS SALINIMLI\b',
        r'\bSETLİ\b', r'\bSETLI\b', r'\bSETSİZ\b', r'\bSETSIZ\b',
        r'\bİRİGASYON\b', r'\bIRIGASYON\b', r'\bSOLÜSYON\b', r'\bSOLUSYON\b',
        r'\bKONSANTRE\b', r'\bPF\b', r'\bÖZELTİSİ\b', r'\bOZELTISI\b',
        r'\bİÇEREN\b', r'\bICEREN\b', r'\bİNDE\b', r'\bINDE\b',
    ]
    
    for form in forms:
        name = re.sub(form, '', name, flags=re.IGNORECASE)
    
    # Fazla boşlukları temizle
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def extract_dosage_form(full_name: str) -> str:
    """İlaç isminden dozaj formunu çıkarır"""
    if pd.isna(full_name):
        return "bilinmiyor"
    
    name = str(full_name).upper()
    
    forms_mapping = {
        'tablet': ['TABLET', 'FİLM TABLET', 'FILM TABLET', 'ÇİĞNEME TABLETİ', 'CIGNEME TABLETI', 'ÇİĞNENEBİLİR', 'ENTERIK KAPLI TABLET'],
        'kapsül': ['KAPSÜL', 'KAPSUL', 'MIKROPELLET KAPSUL', 'YUMUŞAK KAPSÜL'],
        'süspansiyon': ['SÜSPANSİYON', 'SUSPANSIYON', 'ORAL SÜSPANSİYON'],
        'şurup': ['ŞURUP', 'SURUP', 'ORAL SOLÜSİON'],
        'ampul': ['AMPUL', 'AMPÜL'],
        'flakon': ['FLAKON'],
        'enjeksiyonluk çözelti': ['ENJEKSİYON', 'ENJEKSIYON', 'ENJEKSIYONLUK', 'ENJEKSİYONLUK'],
        'infüzyon çözeltisi': ['İNFÜZYON', 'INFUZYON', 'I.V. İNFÜZYON'],
        'inhaler': ['İNHALER', 'INHALER', 'İNHALASYON'],
        'krem': ['KREM'],
        'jel': ['JEL'],
        'pomat': ['POMAT', 'POMAD'],
        'merhem': ['MERHEM'],
        'losyon': ['LOSYON'],
        'damla': ['DAMLA', 'GÖZ DAMLASI', 'KULAK DAMLASI'],
        'sprey': ['SPREY', 'SPRAY', 'NAZAL SPREY', 'ORAL SPREY'],
        'efervesan tablet': ['EFERVESAN'],
        'granül': ['GRANÜL', 'GRANUL'],
        'saşe': ['SAŞE', 'SASE'],
        'çözelti': ['ÇÖZELTİ', 'COZELTI', 'ÖZELTİ', 'OZELTI', 'SOLÜSİON', 'SOLUSYON'],
        'toz': ['TOZ', 'ORAL TOZ'],
        'supozituar': ['SUPOZİTUAR', 'SUPOZITUAR'],
        'implant': ['IMPLANT', 'İMPLANT'],
        'flaster': ['FLASTER', 'TRANSDERMAL'],
        'ovül': ['OVÜL', 'OVUL', 'VAJİNAL'],
        'irrigasyon çözeltisi': ['İRİGASYON', 'IRIGASYON'],
    }
    
    for form_name, keywords in forms_mapping.items():
        for keyword in keywords:
            if keyword in name:
                return form_name
    
    return 'bilinmiyor'


def extract_strength(full_name: str) -> list:
    """İlaç isminden dozaj miktarını çıkarır"""
    if pd.isna(full_name):
        return []
    
    name = str(full_name)
    strengths = []
    
    # Dozaj patternleri
    patterns = [
        r'([\d,\.]+)\s*(MG|MCG|G|ML|%|IU|GRAM|MILIGRAM|MIKROGRAM|MG/ML|MCG/DOZ|MG/DOZ)',
        r'%\s*([\d,\.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, name, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                strength = f"{match[0]} {match[1]}".strip()
            else:
                strength = f"%{match}"
            strengths.append(strength)
    
    # Tekrarları kaldır ve sırala
    return list(dict.fromkeys(strengths))


def clean_manufacturer_name(name: str) -> str:
    """Üretici/ruhsat sahibi ismini temizler"""
    if pd.isna(name):
        return "Bilinmiyor"
    
    name = str(name).strip()
    
    # Kısaltmaları çıkar
    name = re.sub(r'\s+(A\.Ş\.|A\.S\.|LTD\.|ŞTİ\.|STI\.|SAN\.|TİC\.|TIC\.|İLAÇ|ILAC|İTH\.|ITH\.|İHR\.|IHR\.)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def main():
    # Excel dosyasını oku
    excel_path = Path(__file__).parent.parent / "data" / "RuhsatlBeeriTbbirnlerListesi23.01.2026_eb655c6a-c7a9-4383-8a82-5ea258035a78.xlsx"
    
    print(f"Excel dosyası okunuyor: {excel_path}")
    df = pd.read_excel(excel_path, header=1)
    
    # Sütun isimlerini düzelt (Türkçe karakterler bozuk olabilir)
    # İndeksle erişeceğiz: 
    # 0: SIRA NO, 1: BARKOD, 2: ÜRÜN ADI, 3: ETKİN MADDE, 4: ATC KODU, 5: RUHSAT SAHİBİ
    
    print(f"Toplam kayıt: {len(df)}")
    
    # İlaçları grupla - temel isim + etkin madde bazında
    drugs_dict = defaultdict(lambda: {
        'drug_name': '',
        'active_ingredients': set(),
        'manufacturer': set(),
        'dosage_form': set(),
        'strengths': set(),
        'atc_codes': set(),
        'barcodes': set(),
        'original_names': set(),
    })
    
    for idx, row in df.iterrows():
        full_name = row.iloc[2]  # ÜRÜN ADI
        active_ingredient = row.iloc[3]  # ETKİN MADDE
        manufacturer = row.iloc[5]  # RUHSAT SAHİBİ
        atc_code = row.iloc[4]  # ATC KODU
        barcode = row.iloc[1]  # BARKOD
        
        if pd.isna(full_name):
            continue
            
        base_name = extract_drug_base_name(full_name)
        
        if not base_name or len(base_name) < 2:
            continue
        
        # Temel isim anahtarı olarak kullan
        key = base_name.upper()
        
        drugs_dict[key]['drug_name'] = base_name
        drugs_dict[key]['original_names'].add(str(full_name))
        
        if not pd.isna(active_ingredient):
            # Birden fazla etkin madde virgül ile ayrılmış olabilir
            ingredients = str(active_ingredient).split('+')
            for ing in ingredients:
                ing = ing.strip()
                if ing:
                    drugs_dict[key]['active_ingredients'].add(ing.title())
        
        if not pd.isna(manufacturer):
            drugs_dict[key]['manufacturer'].add(clean_manufacturer_name(manufacturer))
        
        dosage_form = extract_dosage_form(full_name)
        if dosage_form != 'bilinmiyor':
            drugs_dict[key]['dosage_form'].add(dosage_form)
        
        strengths = extract_strength(full_name)
        for s in strengths:
            drugs_dict[key]['strengths'].add(s)
        
        if not pd.isna(atc_code):
            drugs_dict[key]['atc_codes'].add(str(atc_code).strip())
        
        if not pd.isna(barcode):
            drugs_dict[key]['barcodes'].add(str(barcode).strip())
    
    print(f"Benzersiz ilaç sayısı (konsolide): {len(drugs_dict)}")
    
    # JSON formatına dönüştür
    drugs_list = []
    
    for key, drug_data in sorted(drugs_dict.items()):
        # Formları birleştir
        forms = list(drug_data['dosage_form'])
        if not forms:
            forms_str = 'bilinmiyor'
        elif len(forms) == 1:
            forms_str = forms[0]
        else:
            forms_str = ', '.join(sorted(forms))
        
        # Üreticileri seç (en kısa olanı tercih et - genellikle en temiz)
        manufacturers = list(drug_data['manufacturer'])
        if manufacturers:
            manufacturer = min(manufacturers, key=len)
        else:
            manufacturer = 'Bilinmiyor'
        
        # Güçleri birleştir
        strengths_list = sorted(list(drug_data['strengths']))
        
        # Aktif bileşenleri listele
        active_ingredients = sorted(list(drug_data['active_ingredients']))
        
        drug_entry = {
            "drug_name": drug_data['drug_name'],
            "active_ingredients": active_ingredients if active_ingredients else ["Bilinmiyor"],
            "manufacturer": manufacturer,
            "dosage_form": forms_str,
            "strengths": strengths_list if strengths_list else ["Bilinmiyor"],
            "indications": "",  # Excel'de bu bilgi yok
            "usage": "",  # Excel'de bu bilgi yok
            "warnings": [],  # Excel'de bu bilgi yok
            "side_effects": [],  # Excel'de bu bilgi yok
            "interactions": [],  # Excel'de bu bilgi yok
            "storage": "",  # Excel'de bu bilgi yok
            "description": "",  # Excel'de bu bilgi yok
            "prescription_status": "Rx",  # Varsayılan olarak reçeteli
            "atc_codes": sorted(list(drug_data['atc_codes'])) if drug_data['atc_codes'] else [],
        }
        
        drugs_list.append(drug_entry)
    
    # JSON dosyasına yaz
    output_path = Path(__file__).parent.parent / "data" / "drug_knowledge_base" / "drugs.json"
    
    # Önce yedek al
    if output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"Yedek oluşturuldu: {backup_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(drugs_list, f, ensure_ascii=False, indent=4)
    
    print(f"drugs.json dosyası oluşturuldu: {output_path}")
    print(f"Toplam ilaç sayısı: {len(drugs_list)}")
    
    # İstatistikler
    with_ingredients = sum(1 for d in drugs_list if d['active_ingredients'] and d['active_ingredients'][0] != 'Bilinmiyor')
    with_strengths = sum(1 for d in drugs_list if d['strengths'] and d['strengths'][0] != 'Bilinmiyor')
    
    print(f"\nİstatistikler:")
    print(f"- Etkin maddesi tanımlı: {with_ingredients}")
    print(f"- Dozaj bilgisi olan: {with_strengths}")


if __name__ == "__main__":
    main()
