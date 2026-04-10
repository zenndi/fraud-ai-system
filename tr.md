# 💳 AI Fraud Detection System — Versiyon 2.0

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern%20API-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

## 🌍 Belge

Lütfen tercih ettiğiniz dili seçin:

✅ Türkçe: **README.md**  
🇬🇧 İngilizce: [README.md](README.md)

---

## 🚀 Yeni Özellikler ve Güncellemeler

### 🖥️ İnteraktif Yapay Zeka Dashboard
**Streamlit** ile oluşturulmuş profesyonel düzeyde gerçek zamanlı izleme arayüzü.
- **Canlı İşlem Akışı:** WebSocket üzerinden gelen işlemlerin anlık takibi.
- **Dinamik Risk Skorlama:** Her işlem için dolandırıcılık olasılığının anlık görselleştirilmesi.
- **Özel Karanlık Tema:** Finansal izleme için tasarlanmış modern kullanıcı arayüzü.
- **Karar Kontrolü:** Ayarlanabilir risk eşik değerleri (threshold) ile dinamik kontrol.

### 🔌 RESTful API Entegrasyonu
Yüksek performanslı **FastAPI** backend ile güçlendirilmiştir.
- **Ölçeklenebilir Mimari:** Diğer finansal sistemlerle entegrasyon için ayrılmış API katmanı.
- **Gerçek Zamanlı Tahmin:** Optimize edilmiş uç noktalar üzerinden model tahminleri.
- **Otomatik Dokümantasyon:** İnteraktif Swagger arayüzü dahildir.

### ⚡ Yüksek Performanslı Önbellekleme
Ultra düşük gecikme süresi ve verimli veri yönetimi için **Redis** entegrasyonu.
- **Hızlı Veri Erişimi:** Sık sorgulanan verileri önbelleğe alarak veritabanı yükünü minimize eder.
- **Düşük Gecikmeli Tahmin:** Karar mekanizmasını ~50ms yanıt sürelerine ulaşacak şekilde optimize eder.

---

## 📸 Dashboard Ön İzlemesi

Arayüz, kapsamlı dolandırıcılık yönetimi için dört özel modüle ayrılmıştır:

* 📊 **Dashboard** — Gerçek zamanlı sistem istatistikleri, canlı işlem akışı ve küresel risk metrikleri.
* 🎯 **Prediction** — Tek işlem risk skorlaması ve yapay zeka önerileri için manuel derinlemesine analiz aracı.
* 🧠 **Model Info** — Yapay sinir ağı mimarisi, eğitim konfigürasyonları ve performans metriklerinin detaylı dökümü.
* ⚙️ **Settings** — Risk eşiği (threshold) ayarlamaları ve dışa aktarma seçeneklerini içeren gerçek zamanlı karar kontrolü.

#### 1️⃣ Gerçek Zamanlı İzleme
![Live Monitoring](assets/dashboard/live_monitor.png)
*Canlı işlem akışı ve sistem sağlığı takibi.*

#### 2️⃣ Gelişmiş Analitik ve Risk Trendleri
![Advanced Analytics](assets/dashboard/analytics.png)
*Risk skorlarının detaylı dökümü ve geçmiş hacim analizi.*

#### 3️⃣ Manuel Analiz Aracı
![Single Transaction Analysis](assets/dashboard/manual_predict.png)
*Yapay zeka önerileri ile belirli işlemler için derinlemesine analiz.*

---

Bu proje, dengesiz finansal veriler üzerinde eğitilmiş bir yapay sinir ağı modeli kullanarak kredi kartı dolandırıcılık işlemlerini tespit etmek amacıyla tasarlanmıştır.

Proje, uçtan uca bir makine öğrenimi hattını (pipeline) içerir:
- Veri keşfi (EDA)
- Veri ön işleme
- Sınıf dengesizliğiyle (class imbalance) mücadele
- Yapay sinir ağı eğitimi
- Model değerlendirme
- Eşik değeri (threshold) optimizasyonu
- Risk skorlama


---

## 🚨 Problem Tanımı

> Kredi kartı dolandırıcılığı tespiti, dolandırıcılık işlemlerinin toplam işlemlerin çok küçük bir kısmını oluşturduğu, yüksek derecede dengesiz bir sınıflandırma problemidir.

Projenin amacı:
- Dolandırıcılık işlemlerini yüksek doğrulukla tespit etmek
- Yanlış alarmları (false positives) en aza indirmek
- Duyarlılığı (recall) maksimize etmek
- Her işlem için bir risk skoru üretmek


## 📊 Veri Seti

Kullanılan veri seti: **Credit Card Fraud Detection Dataset**
Kaynak: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Veri seti özellikleri:**
| Özellik                 | Değer                 |
| ----------------------- | --------------------- |
| Toplam İşlem Sayısı     | 284,807               |
| Dolandırıcılık Sayısı   | 492                   |
| Dolandırıcılık Oranı    | %0.173                |
| Değişkenler             | 30 sayısal özellik    |

Çoğu değişken, **PCA dönüşümü** kullanılarak anonimleştirilmiştir.

### Veriye Erişim
Veri seti boyutu GitHub sınırlarını aştığı için buraya dahil edilmemiştir. Veriyi Kaggle'dan indirip `data/` klasörü içine yerleştiriniz: `data/creditcard.csv`

## 🧠 Model ve Performans

### 1️⃣ Ön İşleme
Veri yükleme, ölçeklendirme ve dengesiz veri yapısına uygun eğitim/test ayrımı işlemlerini içerir.

### 2️⃣ Eşik Değeri (Threshold) Optimizasyonu
Varsayılan (0.5) yerine en iyi dengeyi bulmak için farklı eşikler test edilmiştir:

| Eşik Değeri | Precision | Recall | F1       |
| ----------- | --------- | ------ | -------- |
| 0.10        | 0.37      | 0.84   | 0.51     |
| 0.50        | 0.55      | 0.82   | 0.66     |
| 0.90        | 0.73      | 0.81   | **0.77** |

**En İyi Eşik Değeri: 0.90**

### 📈 Final Sonuçlar
| Metrik    | Skor |
| --------- | ----- |
| Precision | 0.738 |
| Recall    | 0.806 |
| F1 Score  | 0.770 |
| ROC-AUC   | 0.958 |

### Önemli Alanlar
- `Time` — İlk işlemden geçen süre
- `Amount` — İşlem tutarı
- `V1–V28` — PCA ile korunmuş anonim özellikler
- `Class` — Hedef değişken (0: Normal, 1: Fraud)

### Veri İndirme
> ⚠️ **Not:** Veri seti GitHub limitini aştığı için repo'da bulunmaz.

1. [Kaggle'dan](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) veri setini indirin
2. `data/` klasörüne `creditcard.csv` olarak yerleştirin

---


## 📁 Proje Yapısı

```bash
fraud-ai-system/
│
├── api/                # FastAPI backend servisleri
│   └── main.py         # API giriş noktası ve uç noktalar
│
├── dashboard/          # Streamlit arayüz bileşenleri
│   └── styles.py       # Özel CSS ve yerleşim ayarları
│
├── data/               # Veri setleri klasörü
│   └── creditcard.csv  # Ham veri (Git tarafından yoksayılır)
│
├── src/                # Ana makine öğrenimi pipeline'ı
│   ├── data_loader.py  # Veri yükleme
│   ├── preprocessing.py# Temizleme ve ölçeklendirme
│   ├── visualization.py# Görselleştirme fonksiyonları
│   ├── model.py        # Yapay Sinir Ağı (ANN) mimarisi
│   ├── trainer.py      # Eğitim mantığı
│   └── evaluator.py    # Performans analizi
│
├── models/             # Kaydedilmiş model dosyaları
│   └── fraud_model.h5  # Eğitilmiş model ağırlıkları
│
├── utils/              # Yardımcı fonksiyonlar
│   └── metrics.py      # Özel skorlama mantığı
│
├── assets/             # Görsel dökümantasyon
│   ├── eda/            # Analiz grafikleri
│   ├── model/          # Hata matrisi, ROC eğrileri
│   └── dashboard/      # Arayüz ekran görüntüleri
│
├── main.py             # Tüm süreci başlatan dosya
├── dashboard.py        # Dashboard uygulamasını çalıştıran dosya
├── requirements.txt    # Gerekli bağımlılıklar
└── tr.md               # Türkçe dökümantasyon
```



## ⚙️ Kurulum

### 1. Repo'yu Klonla
```bash
git clone [https://github.com/caglaeren/fraud-ai-system.git](https://github.com/caglaeren/fraud-ai-system.git)
cd fraud-ai-system
```

### 2. Sanal Ortam Oluşturma (Önerilir):
```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
```

### 3. Bağımlılıkları Kur
```bash
pip install -r requirements.txt
```

### ▶️  4. Sistemi çalıştırın

### 🔬 Tam Pipeline (Eğitim + Değerlendirme)
```bash
python main.py
```


### 🎨 Streamlit Dashboard
```bash
streamlit run dashboard.py
```

### 🔌 FastAPI REST API
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoint'leri:**

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/health` | GET | Sistem sağlık kontrolü |
| `/predict` | POST | Tek işlem tahmini |
| `/predict/batch` | POST | Toplu tahmin (max 100) |
| `/model/info` | GET | Model bilgileri |
| `/model/threshold` | POST | Threshold güncelleme |

**API Örneği (Tek İşlem):**
```json
POST /predict
{
  "transaction_id": "TXN-784512",
  "amount": 125.50,
  "currency": "USD",
  "time": 43200,
  "v_features": [0.1, -0.2, 0.3, ..., 0.05], // 28 PCA özelliği
  "user_id": "USER-99",
  "merchant_id": "MERCH-404",
  "location": {
    "country": "Türkiye",
    "city": "Ankara",
    "ip_address": "192.168.1.1"
  },
  "device": {
    "device_type": "Mobil",
    "os": "iOS",
    "is_emulator": false
  },
  "payment_method": "Kredi Kartı"
}
```

**Yanıt:**
```json
{
  "transaction_id": "TXN-784512",
  "risk_score": 0.0395,
  "risk_level": "none",
  "is_fraud": false,
  "confidence": 1.0,
  "threshold": 0.5,
  "model_version": "v5.0.0",
  "timestamp": "2026-04-11T00:41:09.955Z",
  "status": "approved",
  "processing_time_ms": 54.54,
  "rules_triggered": [],
  "recommendations": ["Normal işleme devam et"]
}
```

---

## 📈 Model Performansı

| Metric    | Score  | Açıklama |
|-----------|--------|----------|
| Precision | 0.738  | Fraud tahminlerinin doğruluğu |
| Recall    | 0.806  | Gerçek fraud'ları yakalama oranı |
| F1 Score  | 0.770  | Precision-Recall dengesi |
| ROC-AUC   | 0.958  | Model ayrıştırma gücü |

**En İyi Threshold:** `0.90` (F1-Score: 0.771)

---

## 🔧 Konfigürasyon

Tüm hiperparametreler `config.yaml` dosyasında toplanmıştır:

```yaml
data:
  path: "data/creditcard.csv"
  test_size: 0.2
  random_state: 42

model:
  input_dim: 30
  hidden_layers: [128, 64, 32, 16]  # Mimari
  dropout_rate: 0.3                  # Overfitting önleme
  use_batch_norm: true              # BatchNorm aktif
  activation: "relu"

training:
  epochs: 50
  batch_size: 2048
  learning_rate: 0.001
  early_stopping:
    enabled: true
    patience: 5
  learning_rate_scheduler:
    enabled: true
    factor: 0.5
    patience: 3

sampling:
  method: "smote"  # oversample, undersample, smote

evaluation:
  default_threshold: 0.5
  threshold_range: [0.1, 0.9]
```


---
## 🧠 Sistem Akışı
Sistem, veri girişinden gerçek zamanlı karar mekanizmasına kadar otomatik bir iş akışı izler:
1. **Veri Yükleme ve Analiz:** Finansal veri setini otomatik olarak çeker ve işlem modellerini anlamak için derinlemesine keşifsel veri analizi (EDA) yapar.
2. **Akıllı Ön İşleme:** Özellik ölçeklendirme işlemlerini gerçekleştirir ve gelişmiş örnekleme (SMOTE) kullanarak sınıf dengesizliğini giderir.
3. **Model Eğitimi:** Yüksek genelleme yeteneği için Dropout ve Batch Normalization içeren Derin Yapay Sinir Ağı (ANN) modelini eğitir.
4. **Gerçek Zamanlı Karar Mekanizması:** API üzerinden gelen işlemler, `risk_score` değerine göre üç farklı uygulama seviyesine ayrılır:
5. **Canlı İzleme ve Optimizasyon:** **FastAPI** backend sistemi, yüksek hızlı önbellekleme için **Redis** kullanarak ultra düşük gecikme süresiyle (~50ms) tahmin üretir.

| Risk Seviyesi | API Durumu | Alınan Aksiyon |
| :--- | :--- | :--- |
| 🔴 **CRITICAL** | `blocked` | Yüksek risk. İşlem otomatik olarak **Reddedilir**. |
| 🟡 **MEDIUM** | `flagged` | Şüpheli model. **Manuel İnceleme** ve ek doğrulama için işaretlenir. |
| 🟢 **NONE** | `approved` | Düşük risk. İşlem **Sorunsuz** şekilde onaylanır. |

---

## 🧠 Anahtar Notlar

1. **Threshold Optimizasyonu:** Default 0.5 değil, **0.90** optimal F1 için
2. **Dengesiz Veri:** %0.173 fraud → Class weights ve SMOTE kullan
3. **Feature Scaling:** Sadece `Amount` ve `Time` ölçeklenir, V1-V28 zaten PCA
4. **Model Kaydetme:** En iyi model `models/fraud_model_*.keras`'e kaydedilir

---

## 📋 Geliştirici Notları

### Hızlı Mod Geliştirme
- Config file ekledim → Değişiklikler için tek dosya
- Multiple metrics → Sadece accuracy değil
- Early stopping → Eğitim hızlandı
- Checkpoint → En iyi model otomatik kaydedilir

### Gelecek İyileştirmeleri
- [ ] Model avantajı (XGBoost, LightGBM karşılaştırması)
- [ ] Real-time streaming prediction (Kafka)
- [ ] Model monitoring drift tespiti
- [ ] A/B testing framework
- [ ] Docker containerization
- [ ] CI/CD pipeline

---

## 👤 Author & License

**Versiyon 2.0 Geliştiricisi:**  
Zendi (Based on original by Tuğba Çağla EREN)

**Orijinal Proje:** [@caglaeren/fraud-ai-system](https://github.com/caglaeren/fraud-ai-system)

**License:** MIT

---

## 🙏 Teşekkürler

- [Kaggle](https://kaggle.com) — Veri seti
- [TensorFlow](https://tensorflow.org) — Derin öğrenme framework'ü
- [scikit-learn](https://scikit-learn.org) — ML araçları

---

<p align="center">
  Made with ❤️ for better fraud detection
</p>
