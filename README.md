# Athlete Recovery AI

AI destekli sporcu toparlanma skoru tahmin uygulaması.

## Proje Hakkında

300 atletin 28 günlük biyometri ve antrenman verisi üzerinde eğitilmiş **GradientBoosting** modeli ile günlük toparlanma puanını (0-100) tahmin eder.

**Veri kaynağı:** [Athlete Recovery & Biometric Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/sarveshchhetri/athlete-recovery-and-biometric-performance-dataset)

## Özellikler

- Sporcu bilgileri, antrenman yükü, yaşam tarzı ve fizyolojik veriler üzerinden tahmin
- Gerçek zamanlı skor hesaplama ve görsel gauge
- Faktör analizi (antrenman yükü, kas ağrısı, uyku kalitesi, stres, enerji)
- Kişiselleştirilmiş toparlanma önerileri
- Tahmini toparlanma süresi

## Teknoloji

- **Backend:** Python, Flask, scikit-learn
- **Frontend:** HTML5, CSS3, Vanilla JS
- **ML Model:** GradientBoosting Regressor (joblib)
- **Deploy:** Render (gunicorn)

## Kurulum (Yerel)

```bash
pip install -r requirements.txt
python app.py
```

Tarayıcıda `http://localhost:5000` adresine gidin (splash). Tahmin formu: `http://localhost:5000/app`.

**Önemli:** Uygulama `athlete-recovery-ai` klasöründen çalıştırılmalı; aksi halde `models/` bulunamaz. Model `scikit-learn 1.8` + `NumPy 2.2` ile uyumludur; `requirements.txt` sürümlerini değiştirmeyin veya modeli yeniden eğitip kaydedin.

## Sorun giderme

- **`ValueError: MT19937 is not a known BitGenerator`:** Eski `numpy==1.26` + `scikit-learn==1.5` ile oluşur. `pip install -r requirements.txt` ile güncel pinleri kurun.
- **Port 5000 meşgul:** `set PORT=5001` (Windows) veya `PORT=5001 python app.py` deneyin.

## Yapı

```
athlete-recovery-ai/
├── app.py                  # Flask backend
├── requirements.txt        # Python bağımlılıkları
├── render.yaml             # Render deploy ayarları
├── models/
│   └── athlete_recovery_model.joblib
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── notebook/
│   ├── athlete_recovery_prediction.ipynb
│   └── build_athlete_recovery_notebook.py
└── athlete_recovery_synthetic.csv
```

## Canlı Demo

Render üzerinde deploy edilmiştir.
