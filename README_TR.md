# 💳 AI Fraud Detection System — Versiyon 2.0

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Machine Learning](https://img.shields.io/badge/MachineLearning-FraudDetection-green)
![Imbalanced Data](https://img.shields.io/badge/Data-Imbalanced-red)

## 🌍 Belge

✅ Türkçe: **README.md**  
🇬🇧 İngilizce: [README.md](README.md)

---

Kredi kartı dolandırıcılık tespiti için tasarlanmış, TensorFlow tabanlı derin sinir ağı sistemi. Yüksek oranda dengesiz finansal veriler üzerinde eğitilmiş model ile gerçek zamanlı risk skoru hesaplama.

## 🚀 Versiyon 2.0 Yenilikleri

### 📦 Geliştirilmiş Model Mimarisi
- **Dropout** katmanları ile overfitting önleme
- **Batch Normalization** ile eğitim stabilizasyonu
- **Çoklu gizli katman**: 128 → 64 → 32 → 16 nöron
- **Çoklu metrik**: Accuracy, Precision, Recall, F1-Score, AUC
- **Class weights** desteği ile dengesiz veri yönetimi

### ⚙️ Merkezi Konfigürasyon
- YAML tabanlı `config.yaml` ile tüm hiperparametreler
- Kolay ayarlanabilir model mimarisi, eğitim parametreleri
- **Early Stopping**, **Learning Rate Scheduling**, **Model Checkpointing**

### 🐞 Hata Düzeltmeleri
- **Duplicate predict çağrısı** hatası düzeltildi (`evaluator.py`)
- Threshold optimizasyonu geliştirildi
- Batch prediction desteği eklendi

### 📊 Yeni Özellikler
1. **Streamlit Dashboard** — Kullanıcı dostu web arayüzü
   - Real-time risk monitoring
   - Interactive prediction form
   - Model performance charts
   
2. **FastAPI REST API** — Production-ready API endpoints
   - `/predict` — Tek işlem tahmini
   - `/predict/batch` — Toplu tahmin
   - `/model/info` — Model bilgileri
   - `/health` — Sistem sağlık kontrolü

3. **Gelişmiş Görselleştirme**
   - Threshold vs Metrics grafikleri
   - Risk skoru dağılım histogramı
   - ROC curve ile model karşılaştırması

---

## 📂 Proje Yapısı (Versiyon 2.0)

```
fraud-ai-system/
│
├── config.yaml              🆕 YAML merkezi konfigürasyon
├── main.py                  🔄 Geliştirilmiş ana pipeline
├── dashboard.py             🆕 Streamlit web dashboard
├── api.py                   🆕 FastAPI REST endpoints
├── requirements.txt         🔄 Güncellenmiş bağımlılıklar
│
├── src/
│   ├── model.py             🔄 Dropout + BatchNorm + Config
│   ├── evaluator.py         🔄 Bug fix + Batch prediction
│   ├── trainer.py           🔄 Callback'ler + Checkpoint
│   ├── preprocessing.py     🔄 SMOTE/ADASYN desteği
│   ├── visualization.py     📊 EDA görselleştirmeleri
│   └── data_loader.py       📥 Veri yükleme
│
├── utils/
│   └── metrics.py           📈 Metrik hesaplamaları
│
├── assets/
│   ├── eda/                 📊 EDA grafikleri
│   └── model/               📉 Model performans grafikleri
│
├── models/                  📁 Eğitilmiş modeller (otomatik oluşturulur)
├── logs/                    📁 TensorBoard logları (opsiyonel)
│
├── data/                    📂 Veri seti (kaggle'dan indirilmeli)
│   └── creditcard.csv
│
└── README.md                📘 Bu dosya
```

---

## 📊 Veri Seti

**Kredi Kartı Dolandırıcılık Tespiti Veri Seti**

Kaynak: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Veri Seti Özellikleri

| Özellik               | Değer                 |
| --------------------- | --------------------- |
| Toplam işlem          | 284,807               |
| Dolandırıcılık işlemi | 492                   |
| Dolandırıcılık oranı  | %0.173                |
| Özellik sayısı        | 30 (sayısal, PCA)     |

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

## ⚙️ Kurulum

### 1. Repo'yu Klonla
```bash
git clone https://github.com/caglaeren/fraud-ai-system.git
cd fraud-ai-system
```

### 2. Bağımlılıkları Kur
```bash
pip install -r requirements.txt
```

**requirements.txt içeriği:**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
imbalanced-learn
pyyaml
streamlit
fastapi
uvicorn
```

---

## ▶️ Çalıştırma

### 🔬 Tam Pipeline (Eğitim + Değerlendirme)
```bash
python main.py
```

Sistem şunları yapar:
1. ✅ Veri setini yükler
2. ✅ EDA grafikleri oluşturur
3. ✅ Ön işleme (scaling, balancing)
4. ✅ Modeli eğitir (Dropout + BatchNorm)
5. ✅ Early stopping + checkpointing
6. ✅ Threshold optimizasyonu
7. ✅ Gelişmiş değerlendirme metrikleri
8. ✅ Görselleştirmeler kaydeder
9. ✅ Modeli `models/` klasörüne kaydeder

### 🎨 Streamlit Dashboard
```bash
streamlit run dashboard.py
```

Dashboard özellikleri:
- 📊 **Dashboard** — Genel sistem istatistikleri
- 🎯 **Prediction** — Tek işlem risk skoru
- 🧠 **Model Info** — Mimari, konfigürasyon, metrikler
- ⚙️ **Settings** — Threshold ayarları, export

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
  "amount": 125.50,
  "time": 43200,
  "v_features": [0.1, -0.2, 0.3, ...]  // 28 değer
}
```

**Yanıt:**
```json
{
  "risk_score": 0.8743,
  "is_fraud": true,
  "confidence": 0.7486,
  "threshold": 0.5,
  "model_version": "v2.0",
  "timestamp": "2026-04-02T13:00:00"
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
