# İlaç Veritabanına Katkı Rehberi

Bu rehber, ilaç knowledge base'ine nasıl katkıda bulunacağınızı açıklar.

## 🎯 Katkı Türleri

### 1. Yeni İlaç Ekleme

Yeni ilaç eklemek için iki yöntem var:

#### Yöntem A: İnteraktif Araç (Önerilen)

```bash
python scripts/add_drug_interactive.py
```

Bu araç, size tüm alanları adım adım sorar ve girdiyi doğrular.

#### Yöntem B: Hızlı Ekleme

```bash
python scripts/add_drug_interactive.py --quick "İlaç Adı"
```

Sadece temel bilgilerle hızlı ekleme yapar.

#### Yöntem C: Manuel JSON Düzenleme

`data/drug_knowledge_base/drugs.json` dosyasını düzenleyin:

```json
{
  "drug_name": "İlaç Adı",
  "active_ingredients": ["Etken Madde 1", "Etken Madde 2"],
  "manufacturer": "Üretici Firma",
  "dosage_form": "tablet",
  "strengths": ["100 mg", "200 mg"],
  "indications": "Kullanım alanları...",
  "usage": "Kullanım talimatları...",
  "warnings": ["Uyarı 1", "Uyarı 2"],
  "side_effects": ["Yan etki 1", "Yan etki 2"],
  "interactions": ["Etkileşim 1", "Etkileşim 2"],
  "storage": "Saklama koşulları",
  "description": "Genel açıklama",
  "prescription_status": "OTC"
}
```

### 2. Mevcut İlaç Güncelleme

1. `drugs.json` dosyasında ilacı bulun
2. Gerekli alanları güncelleyin
3. Knowledge base'i yenileyin:

```bash
python scripts/populate_knowledge_base.py --clear
```

### 3. Hata Bildirimi

Yanlış veya eksik bilgi bulursanız GitHub Issues kullanın.

## 📋 Veri Kalite Standartları

### Zorunlu Alanlar

| Alan | Açıklama |
|------|----------|
| `drug_name` | İlacın ticari adı (Türkçe karakterler korunmalı) |
| `active_ingredients` | En az bir etken madde |

### Önerilen Alanlar

| Alan | Açıklama |
|------|----------|
| `manufacturer` | Üretici firma adı |
| `dosage_form` | tablet, kapsül, şurup, inhaler, vb. |
| `strengths` | Doz miktarları (örn: ["500 mg", "1000 mg"]) |
| `indications` | Kullanım alanları ve endikasyonlar |
| `usage` | Doz talimatları |
| `warnings` | Uyarılar ve kontrendikasyonlar |
| `side_effects` | Bilinen yan etkiler |
| `prescription_status` | OTC, Rx veya OTC/Rx |

### Kabul Edilebilir Değerler

**dosage_form:**
- tablet, film tablet, çiğneme tableti, efervesan tablet
- kapsül, mikropelet kapsül
- şurup, süspansiyon
- damla, sprey
- inhaler
- krem, merhem, jel
- enjeksiyon, ampul
- saşe, toz

**prescription_status:**
- `OTC`: Reçetesiz satılır
- `Rx`: Reçete gerektirir
- `OTC/Rx`: Doza göre değişir

## 🔍 Güvenilir Kaynaklar

Bilgi eklerken güvenilir kaynaklar kullanın:

1. **TİTCK** - Türkiye İlaç ve Tıbbi Cihaz Kurumu
2. **İlaç Prospektüsleri** - Ürünün resmi prospektüsü
3. **İlaç.com** - Türkiye ilaç veritabanı
4. **SGK İlaç Listesi** - Geri ödeme listesi

## ⚠️ Önemli Uyarılar

1. **Tıbbi Tavsiye Değildir**: Bu bir eğitim projesidir
2. **Telif Hakları**: Prospektüsleri kelimesi kelimesine kopyalamayın
3. **Doğruluk**: Bilgilerin güncel ve doğru olduğundan emin olun
4. **Türkçe**: İlaç adları ve bilgiler Türkçe olmalı

## 🧪 Test Etme

Değişikliklerinizi test edin:

```bash
# Veritabanını doğrula
python scripts/add_drug_interactive.py --validate data/drug_knowledge_base/drugs.json

# Knowledge base'i güncelle
python scripts/populate_knowledge_base.py --clear

# Pipeline ile test
python main.py --dummy data/test_image.jpg
```

## 📝 Pull Request Süreci

1. Kendi branch'inizi oluşturun
2. Değişikliklerinizi yapın
3. Testleri çalıştırın
4. Önceki değişiklikleri anlamlı biçimde açıklayan commit mesajı yazın
5. Pull request açın

Teşekkürler! 🙏
