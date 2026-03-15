# Testler – CARE Test Plan Report (Bölüm 6.3 & 6.7)

Bu klasör, `CARE_Test_Plan_Report.docx` dokümanında tanımlanan **Image Analysis Service** ve **Medical Image Analysis Service** test case'lerini içermektedir.

## Testleri Çalıştırma

```bash
python -m pytest tests\test_image_analysis_service.py tests\test_medical_image_analysis_service.py -v -s -o "addopts="
```

> **Not:** `pyproject.toml` içindeki `--cov` flag'i `pytest-cov` paketi yüklü değilse hata verir. `-o "addopts="` parametresi bunu geçersiz kılar.

---

## Test Dosyaları

### `test_image_analysis_service.py` — Bölüm 6.3: Image Analysis Service

| Test ID | Test Metodu | Açıklama |
|---------|-------------|----------|
| TC-IMG-01 | `TestTCIMG01::test_successful_drug_analysis` | İlaç kutusu görüntüsü başarıyla analiz edilir; ilaç adı, etken madde ve doz bilgisi çıkarılır |
| TC-IMG-02 | `TestTCIMG02::test_non_pharma_image_detected` | İlaç olmayan görüntü tespit edilir; uygun hata mesajı döner |
| TC-IMG-03 | `TestTCIMG03::test_unsupported_tiff_format` | Desteklenmeyen TIFF formatı reddedilir (`InvalidImageError`) |
| TC-IMG-03 | `TestTCIMG03::test_unsupported_ext_txt` | `.txt` uzantılı dosya reddedilir |
| TC-IMG-04 | `TestTCIMG04::test_low_resolution_image` | Düşük çözünürlüklü görüntü reddedilir (`ImageQualityError`) |
| TC-IMG-05 | `TestTCIMG05::test_corrupted_image_bytes` | Bozuk/boş byte dizisi reddedilir (`InvalidImageError`) |
| TC-IMG-05 | `TestTCIMG05::test_corrupted_base64` | Boş base64 girişi reddedilir |
| TC-IMG-06 | `TestTCIMG06::test_ocr_text_extraction` | Okunaklı ilaç etiketi metninden OCR ile text bloklar çıkarılır |
| TC-IMG-07 | `TestTCIMG07::test_no_text_pipeline_continues` | OCR metin bulamazsa pipeline hata ile devam eder |
| TC-IMG-08 | `TestTCIMG08::test_entity_extraction` | İlaç adı, etken madde ve doz varlıkları doğru şekilde çıkarılır |
| TC-IMG-09 | `TestTCIMG09::test_no_drug_name_low_confidence` | İlaç adı bulunamazsa `DrugNameNotFoundError` ile yönetilir |
| TC-IMG-10 | `TestTCIMG10::test_knowledge_retrieval_success` | Tanınan ilaç için ChromaDB'den bilgi başarıyla getirilir |
| TC-IMG-11 | `TestTCIMG11::test_knowledge_db_failure` | ChromaDB bağlantı hatası yönetilir (`KnowledgeBaseConnectionError`) |
| TC-IMG-12 | `TestTCIMG12::test_explanation_generated` | LLM tarafından kullanıcı dostu açıklama üretilir |
| TC-IMG-13 | `TestTCIMG13::test_invalid_explanation_fallback` | Güvensiz LLM yanıtı reddedilir; ham ilaç verisi döner |
| TC-IMG-14 | `TestTCIMG14::test_pipeline_timeout` | Zaman aşımı halinde kısmi sonuç ve hata listesi döner |
| TC-IMG-15 | `TestTCIMG15::test_user_response_clarity` | `get_user_response()` çıktısı gerekli tüm alanları içerir |

---

### `test_medical_image_analysis_service.py` — Bölüm 6.7: Medical Image Analysis Service

| Test ID | Test Metodu | Açıklama |
|---------|-------------|----------|
| TC-MIA-01 | `TestTCMIA01::test_successful_dermatology_analysis` | Dermatoloji görüntüsü başarıyla analiz edilir; güven skoru, açıklama ve uyarı döner |
| TC-MIA-02 | `TestTCMIA02::test_successful_chest_xray_analysis` | Göğüs röntgeni görüntüsü başarıyla analiz edilir; bulgular ve açıklama döner |
| TC-MIA-03 | `TestTCMIA03::test_low_confidence_flagged` | Düşük güven skoru uyarı ile işaretlenir |
| TC-MIA-04 | `TestTCMIA04::test_invalid_diagnosis_type` | Geçersiz teşhis tipi (`brain_scan`) reddedilir |
| TC-MIA-05 | `TestTCMIA05::test_unsupported_gif_format` | Desteklenmeyen GIF formatı reddedilir |
| TC-MIA-06 | `TestTCMIA06::test_malformed_base64` | Bozuk base64 girişi reddedilir (`InvalidImageError`) |
| TC-MIA-07 | `TestTCMIA07::test_model_load_failure` | Model yüklenemezse `ModelLoadError` ile yönetilir |
| TC-MIA-08 | `TestTCMIA08::test_non_medical_image_disclaimer` | Tıbbi olmayan görüntüde zorunlu disclaimer her zaman mevcuttur |
| TC-MIA-09 | `TestTCMIA09::test_explanation_fallback` | LLM açıklama üretemezse CNN sınıflandırma sonucu korunur |
| TC-MIA-10 | `TestTCMIA10::test_pipeline_timeout_partial` | Uzun süren isteklerde kısmi sonuç zamanlama bilgisiyle döner |
| TC-MIA-11 | `TestTCMIA11::test_classification_failure` | Sınıflandırma hatası (`ClassificationFailed`) yönetilir |
| TC-MIA-12 | `TestTCMIA12::test_pipeline_timing` | Pipeline 30 saniye içinde tamamlanır |
| TC-MIA-13 | `TestTCMIA13::test_disclaimer_always_present` | Disclaimer tüm tıbbi analiz sonuçlarında her zaman bulunur |
| TC-MIA-14 | `TestTCMIA14::test_medical_mode_accessible` | Tıbbi analiz modu (dermatoloji + göğüs röntgeni) erişilebilir durumdadır |
| TC-MIA-15 | `TestTCMIA15::test_user_response_clarity` | `get_user_response()` çıktısı gerekli tüm alanları içerir |

---

## Sonuç

```
32 passed in ~1.20s
```

Tüm testler harici bağımlılık olmadan **mock objeler** kullanılarak çalışır — gerçek CNN modeli, OCR motoru, LLM veya ChromaDB gerekmez.
