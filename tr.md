# 💳 AI Sahtecilik Tespit Sistemi (Fraud Detection System)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Machine Learning](https://img.shields.io/badge/MachineLearning-FraudDetection-green)
![Imbalanced Data](https://img.shields.io/badge/Data-Imbalanced-red)


Yüksek derecede dengesiz finansal veriler üzerinde eğitilen bir sinir ağı modeli kullanarak sahte kredi kartı işlemlerini tespit etmek için tasarlanmış bir makine öğrenmesi sistemi.

Bu proje aşağıdakileri içeren tam bir makine öğrenmesi boru hattını (pipeline) göstermektedir:
- Veri keşfi (Data exploration)
- Veri ön işleme (Data preprocessing)
- Aşırı sınıf dengesizliğini ele alma
- Sinir ağı eğitimi
- Model değerlendirme
- Eşik değeri optimizasyonu
- Sahtecilik risk puanlaması


## 🚨 Problem

> Kredi kartı sahtecilik tespiti, sahte işlemlerin tüm işlemlerin sadece çok küçük bir kısmını oluşturduğu yüksek derecede dengesiz bir sınıflandırma problemidir.

Bu projenin amacı, aşağıdaki yeteneklere sahip bir makine öğrenmesi sistemi inşa etmektir:
- Sahte işlemleri tespit etme
- Yanlış pozitifleri en aza indirme
- Sahtecilik tespiti için recall (geri çağırma) değerini maksimize etme
- Her işlem için bir sahtecilik risk puanı üretme


## 📊 Veri Seti (Dataset)

Kullanılan veri seti:

**Credit Card Fraud Detection Dataset (Kredi Kartı Sahtecilik Tespit Veri Seti)**

Kaynak: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


**Veri seti özellikleri:**
| Özellik                  | Değer                 |
| ----------------------- | --------------------- |
| Toplam işlem sayısı      | 284,807               |
| Sahte işlemler          | 492                   |
| Sahtecilik oranı        | 0.173%                |
| Özellikler (Features)   | 30 sayısal özellik    |

Çoğu değişken **PCA dönüşümü** kullanılarak anonimleştirilmiştir.

**Önemli alanlar:**
- `Time` (Zaman)
- `Amount` (Tutar)
- `V1 – V28 (PCA özellikleri)`
- `Class` (Sınıf)

**Burada:**
- `0` → Normal işlem
- `1` → Sahte işlem


### Veri Setine Erişim

Veri seti GitHub'ın dosya boyutu sınırını aştığı için bu depoya dahil edilmemiştir.

Kaggle'dan indirin ve `data/` klasörüne yerleştirin:

`data/creditcard.csv`



## 📊 Keşifsel Veri Analizi (EDA) & Model Değerlendirmesi
Proje, veri setini anlamak için birkaç görselleştirme içermektedir.

### Sınıf Dağılımı (Class Distribution)
![Sınıf Dağılımı](assets/eda/class_distribution.png)


### Sahte vs Normal İşlem Tutarı
![Sahte vs Normal İşlem Tutarı](assets/eda/fraud_vs_normal_amount.png)


### Korelasyon Isı Haritası (Correlation Heatmap)
![Korelasyon Isı Haritası](assets/eda/correlation_heatmap.png)


### ROC Eğrisi (ROC Curve)
![ROC Eğrisi](assets/model/roc_curve.png)


### Karışıklık Matrisi (Confusion Matrix)
![Karışıklık Matrisi](assets/model/confusion_matrix.png)



## 🧠 Model
Proje, tam bir ML (Makine Öğrenmesi) boru hattı uygular:

### 1️⃣ Veri Yükleme (Data Loading)

Veri seti yüklemeyi ve sınıf dağılımı incelemesini gerçekleştirir.

### 2️⃣ Ön İşleme (Preprocessing)

Şunları içerir:

- Train/Test ayrımı
- Özellik ölçeklendirme (Feature scaling)
- Dengesiz verileri ele alma


### 3️⃣ Model Eğitimi (Model Training)

Sahte işlemleri tespit etmek için bir sınıflandırma modeli eğitilir.

Sistem şunları değerlendirir:

- Precision (Kesinlik)
- Recall (Hassasiyet/Geri Çağırma)
- F1 Score (F1 Puanı)
- ROC-AUC

### 4️⃣ Eşik Değeri Optimizasyonu (Threshold Optimization)

Varsayılan eşik değerini (0.5) kullanmak yerine, en iyi dengeyi bulmak için birden fazla eşik değeri test edilir:

| Eşik (Threshold) | Precision | Recall | F1       |
| --------- | --------- | ------ | -------- |
| 0.10      | 0.37      | 0.84   | 0.51     |
| 0.50      | 0.55      | 0.82   | 0.66     |
| 0.90      | 0.73      | 0.81   | **0.77** |


### En İyi Eşik Değeri:

`Eşik Değeri: 0.90`
`En İyi F1 Puanı: 0.7707`




## 📈 Sonuçlar (Results)

Final model performansı:
| Metrik (Metric) | Skor (Score) |
| --------- | ----- |
| Precision | 0.738 |
| Recall    | 0.806 |
| F1 Score  | 0.770 |
| ROC-AUC   | 0.958 |

**Yorum (Interpretation)**

- Model, sahte işlemlerin %80'ini tespit eder

- Yanlış alarmları azaltmak için makul bir kesinlik (precision) sağlar

- Güçlü ayırma yeteneği gösterir (ROC-AUC = 0.958)



## 📁 Proje Yapısı (Project Structure)

```
fraud-ai-system/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model.py
│   ├── trainer.py
│   └── evaluator.py
│
├── utils/
│   └── metrics.py
│
├── assets/
│   ├── eda/
│   └── model/
│
├── main.py
├── requirements.txt
└── README.md
```


## ⚙️ Kurulum (Installation)
Depoyu klonlayın:

```bash
git clone https://github.com/caglaeren/fraud-ai-system.git
cd fraud-ai-system
```

**Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

## ▶️ Projeyi Çalıştırma (Run the Project)

Tam boru hattını (pipeline) çalıştırın:

```bash
python main.py
```

**Sistem şunları yapacak:**
- Veri setini yükleyecek
- Modeli eğitecek
- Performansı değerlendirecek
- Görselleştirmeler oluşturacak


## 🧠 Temel Çıkarımlar (Key Takeaways)

- Sahtecilik tespiti, yüksek derecede dengesiz bir sınıflandırma problemidir
- Sadece doğruluk (accuracy) anlamlı değildir
- Precision ve recall dengelenmelidir
- Eşik değeri optimizasyonu performansı önemli ölçüde artırır


## 👤 Yazar (Author):

**Tuğba Çağla EREN**
