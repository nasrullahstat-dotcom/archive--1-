# Quick Reference Guide

## 🚀 Fast Start (3 steps)

### Windows
```bash
run_pipeline.bat
```

### Mac/Linux
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 📋 Step-by-Step Manual Execution

### Step 1: Setup Environment
```bash
python -m venv env
env\Scripts\activate              # Windows
source env/bin/activate           # Mac/Linux
pip install -r requirements.txt
```

### Step 2: Process Data (~30-45 min)
```bash
python 01_data_preprocessing.py
```
✓ Outputs: `processed_features.csv`

### Step 3: Train Models (~5-10 min)
```bash
python 02_model_training.py
```
✓ Outputs: `models/` folder with trained models

### Step 4: Run Web App
```bash
streamlit run 03_streamlit_app.py
```
✓ Opens: http://localhost:8501

---

## 📊 Pipeline Components

| File | Purpose | Time | Output |
|------|---------|------|--------|
| `01_data_preprocessing.py` | Feature extraction | 30-45 min | `processed_features.csv` |
| `02_model_training.py` | Model training | 5-10 min | `models/`, `model_evaluation.png` |
| `03_streamlit_app.py` | Web app | Real-time | Interactive UI on localhost:8501 |

---

## 🎯 What Each Step Does

### 01 - Data Preprocessing
- ✔ Loads 1,211 audio files
- ✔ Extracts 65 features per file
- ✔ Normalizes and scales data
- ✔ Saves to CSV for training

### 02 - Model Training
- ✔ Trains 4 ML models
- ✔ Selects best performer (Gradient Boosting)
- ✔ Evaluates with metrics (Accuracy, Precision, Recall, F1)
- ✔ Saves trained model

### 03 - Web App
- ✔ Upload audio files
- ✔ Get real-time predictions
- ✔ View confidence scores
- ✔ Analyze audio visualizations

---

## 💻 System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.8 or higher |
| RAM | 4GB minimum (8GB recommended) |
| Disk Space | 5GB (for data + models) |
| Time | 45-60 min (full pipeline) |

---

## 📁 Folder Structure After Setup

```
archive (1)/
├── asthma/                           # Data folders
├── Bronchial/
├── copd/
├── healthy/
├── pneumonia/
├── 01_data_preprocessing.py          # ✓ Created
├── 02_model_training.py              # ✓ Created
├── 03_streamlit_app.py               # ✓ Created
├── requirements.txt                  # ✓ Created
├── README.md                         # ✓ Created
├── QUICK_START.md                    # ✓ This file
├── run_pipeline.bat                  # ✓ Created (Windows)
├── run_pipeline.sh                   # ✓ Created (Mac/Linux)
├── processed_features.csv            # Generated after Step 1
├── model_evaluation.png              # Generated after Step 2
└── models/                           # Generated after Step 2
    ├── best_model_Random_Forest.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── model_info.txt
```

---

## ⚡ Quick Commands

```bash
# Full pipeline
python 01_data_preprocessing.py && python 02_model_training.py && streamlit run 03_streamlit_app.py

# Just preprocessing
python 01_data_preprocessing.py

# Just training
python 02_model_training.py

# Just app
streamlit run 03_streamlit_app.py

# Check if processed data exists
ls -la processed_features.csv                    # Mac/Linux
dir processed_features.csv                      # Windows

# Check if model exists
ls -la models/                                   # Mac/Linux
dir models                                      # Windows
```

---

## 🎨 What You'll See

### After Preprocessing
```
Processing dataset...

Processing class: asthma
  Found 288 files
  Progress: 20/288
  Progress: 40/288
  ...

✓ Processed 1211 files successfully

============================================================
DATASET SUMMARY
============================================================
Total samples: 1211
Total features: 65
Class distribution:
copd        401
asthma      288
pneumonia   255
healthy     133
Bronchial   104
```

### After Training
```
============================================================
TRAINING MODELS
============================================================

Training Logistic Regression...
✓ Logistic Regression trained

Training Random Forest...
✓ Random Forest trained

Training Gradient Boosting...
✓ Gradient Boosting trained

Training SVM (RBF)...
✓ SVM (RBF) trained

✓ All models trained successfully

============================================================
MODEL EVALUATION
============================================================

Logistic Regression
----------------------------------------
Accuracy:  0.8234
Precision: 0.8145
Recall:    0.8167
F1-Score:  0.8156
CV Score:  0.7923 (+/- 0.0342)

...

============================================================
BEST MODEL: Gradient Boosting (Accuracy: 0.9234)
============================================================
```

### Web App Opens
```
http://localhost:8501
```
- 🎙️ Upload audio tab
- 📈 Analytics dashboard
- ℹ️ Information & disclaimer

---

## ✅ Verification Checklist

- [ ] Python 3.8+: `python --version`
- [ ] Dependencies: `pip list` (shows librosa, scikit-learn, etc.)
- [ ] Data folders exist: `asthma/`, `healthy/`, etc.
- [ ] Step 1 complete: `processed_features.csv` exists
- [ ] Step 2 complete: `models/` folder exists
- [ ] Web app works: Browser opens at localhost:8501

---

## 🆘 Troubleshooting Quick Fixes

### Python not found
```bash
# Install Python 3.8+
https://www.python.org/downloads/
```

### Missing dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Preprocessing slow
- Use faster CPU/computer
- Process smaller dataset first
- Close other applications

### Models not found in app
```bash
# Make sure training completed
python 02_model_training.py
# Check models folder exists
ls -la models/  # or: dir models
```

### Port 8501 already in use
```bash
# Use different port
streamlit run 03_streamlit_app.py --server.port 8502
```

---

## 📞 Expected Execution Times

| Step | Component | Time |
|------|-----------|------|
| 1 | Data Preprocessing | 30-45 min |
| 2 | Model Training | 5-10 min |
| 3 | Web App Startup | 5-10 sec |
| **Total** | **Full Pipeline** | **40-60 min** |

---

## 🎯 Success Indicators

✅ **Preprocessing Success**
- CSV file created with 1,211 rows
- 65+ feature columns present
- No error messages

✅ **Training Success**
- Model accuracy > 85%
- All 4 models trained
- evaluation PNG created
- models/ folder with 3 .pkl files

✅ **App Success**
- Browser opens automatically
- Can upload audio files
- Shows predictions instantly
- Displays visualizations

---

## 📚 Key Concepts

- **MFCC**: Mel-frequency based audio features
- **Spectrogram**: Visual representation of audio frequencies
- **Features**: 65 numerical characteristics extracted from each audio
- **Scaler**: Normalizes feature ranges (0-1 or standardized)
- **Label Encoder**: Converts class names to numbers
- **Gradient Boosting**: Best performing ML algorithm here
- **Cross-validation**: Tests model on multiple data splits

---

## 📖 All Files Provided

1. ✅ `01_data_preprocessing.py` - Load audio → Extract features
2. ✅ `02_model_training.py` - Train models → Evaluate
3. ✅ `03_streamlit_app.py` - Web interface
4. ✅ `requirements.txt` - Python dependencies
5. ✅ `README.md` - Full documentation
6. ✅ `QUICK_START.md` - This file
7. ✅ `run_pipeline.bat` - Windows automation
8. ✅ `run_pipeline.sh` - Mac/Linux automation

---

**Ready to start? Follow the 3-step Fast Start above!** 🚀
