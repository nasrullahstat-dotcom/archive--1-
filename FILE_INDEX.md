# 📑 Pipeline Files Index

## 🎯 START HERE

**For Windows users:**
```
Double-click: run_pipeline.bat
```

**For Mac/Linux users:**
```
Terminal: chmod +x run_pipeline.sh && ./run_pipeline.sh
```

**For manual execution:**
```
Read: QUICK_START.md
```

---

## 📂 Complete File Listing

### 🐍 Python Scripts (Ready to Run)

#### 1. `01_data_preprocessing.py` (565 lines)
**Purpose**: Convert raw audio → Features for ML  
**Input**: Audio files in folders (asthma/, healthy/, etc.)  
**Output**: `processed_features.csv` (1,211 rows × 67 columns)  
**Time**: 30-45 minutes  
**Run**: `python 01_data_preprocessing.py`

**What it does:**
- Loads all 1,211 WAV audio files
- Extracts 65 features from each (MFCC, spectral, temporal)
- Normalizes samples to 5 seconds
- Applies StandardScaler normalization
- Combines into single CSV file

**Features extracted:**
- MFCC (13 coefficients) - Mean & Std
- Spectral Centroid - Mean & Std
- Spectral Rolloff - Mean & Std
- Zero Crossing Rate - Mean & Std
- Chroma (12 bins) - Mean & Std
- RMS Energy - Mean & Std
- Onset Strength - Mean & Std

---

#### 2. `02_model_training.py` (406 lines)
**Purpose**: Train, evaluate, and save ML models  
**Input**: `processed_features.csv`  
**Output**: `models/` folder with trained models  
**Time**: 5-10 minutes  
**Run**: `python 02_model_training.py`

**What it does:**
- Trains 4 different ML models:
  1. Logistic Regression (baseline)
  2. Random Forest (ensemble)
  3. Gradient Boosting (best ⭐)
  4. SVM with RBF kernel (nonlinear)
- Evaluates each with:
  - Accuracy, Precision, Recall, F1-Score
  - 5-fold cross-validation
  - Confusion matrix
- Selects best model
- Saves for deployment

**Output files:**
- `models/best_model_Random_Forest.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/label_encoder.pkl` - Class encoder
- `models/model_info.txt` - Metadata
- `model_evaluation.png` - Evaluation charts

---

#### 3. `03_streamlit_app.py` (484 lines)
**Purpose**: Interactive web application for predictions  
**Input**: Trained models + new audio files  
**Output**: Web interface on localhost:8501  
**Time**: Instant (real-time)  
**Run**: `streamlit run 03_streamlit_app.py`

**What it does:**
- Web interface with Streamlit
- Upload audio files (drag & drop)
- Real-time predictions
- Shows confidence scores
- Visualizes audio (waveform, spectrogram)
- Charts probability distribution
- Displays model analytics

**Web interface tabs:**
1. 🎙️ **Prediction** - Upload & classify audio
2. 📈 **Analytics** - Model performance metrics
3. ℹ️ **Info** - System information & disclaimer

---

### 📚 Documentation Files (Read These)

#### 4. `README.md` (Complete Guide)
**Purpose**: Full project documentation  
**Contains**: 
- Architecture overview
- Installation guide
- Usage instructions
- Troubleshooting
- Medical disclaimers
- References

**Read this for**: Understanding the entire system

---

#### 5. `QUICK_START.md` (Fast Reference)
**Purpose**: Get running in 3 easy steps  
**Contains**:
- Quick start commands
- Expected outputs
- Verification checklist
- Troubleshooting shortcuts
- Time estimates

**Read this for**: Immediate execution

---

#### 6. `CONFIGURATION.md` (Advanced Customization)
**Purpose**: Customize parameters and optimize  
**Contains**:
- Parameter explanations
- Performance tuning recipes
- Hyperparameter optimization
- Ensemble methods
- Deployment strategies

**Read this for**: Customization & optimization

---

#### 7. `PROJECT_SUMMARY.md` (This Project)
**Purpose**: Overview of everything created  
**Contains**:
- Files delivered
- Dataset info
- Pipeline workflow
- Expected results
- Next steps

**Read this for**: Project overview

---

#### 8. `FILE_INDEX.md` (This File)
**Purpose**: Guide to all project files  
**Contains**: 
- File descriptions
- Quick links
- Execution order

---

### ⚙️ Configuration & Automation

#### 9. `requirements.txt`
**Purpose**: Python dependencies  
**Install**: `pip install -r requirements.txt`  
**Contains**:
- librosa (audio processing)
- scikit-learn (ML models)
- streamlit (web app)
- pandas (data handling)
- matplotlib/seaborn (visualization)

---

#### 10. `run_pipeline.bat` (Windows)
**Purpose**: One-click execution  
**Why use**: Automatic environment setup + menu-driven  
**How**: Double-click the file

---

#### 11. `run_pipeline.sh` (Mac/Linux)
**Purpose**: One-click execution  
**Why use**: Automatic environment setup + menu-driven  
**How**: `chmod +x run_pipeline.sh && ./run_pipeline.sh`

---

### 📊 Data Folders (Your Data)

#### 12-16. Data Folders (Existing)
```
asthma/          ← 288 audio files
Bronchial/       ← 104 audio files
copd/            ← 401 audio files
healthy/         ← 133 audio files
pneumonia/       ← 255 audio files
```

---

### 📈 Generated After Running (Don't Edit)

#### 17. `processed_features.csv`
**Created by**: `01_data_preprocessing.py`  
**Contains**: 1,211 rows × 67 columns (samples × features)  
**Use**: Input for training

---

#### 18. `model_evaluation.png`
**Created by**: `02_model_training.py`  
**Contains**: Evaluation charts (performance comparison)  
**View**: Shows model accuracy, confusion matrix, CV scores

---

#### 19. `models/` Folder
**Created by**: `02_model_training.py`  
**Contents**:
- `best_model_Random_Forest.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Class encoder
- `model_info.txt` - Model metadata

---

## 🚀 Execution Order

```
1. Read README.md (understand system)
   ↓
2. Read QUICK_START.md (see quick commands)
   ↓
3. Run: python 01_data_preprocessing.py
   ↓ (wait 30-45 min)
   ↓
4. Run: python 02_model_training.py
   ↓ (wait 5-10 min)
   ↓
5. Run: streamlit run 03_streamlit_app.py
   ↓ (opens browser at localhost:8501)
   ↓
6. Upload audio & test predictions!
```

---

## 📋 File Sizes (Approximate)

| File | Size | Type |
|------|------|------|
| 01_data_preprocessing.py | 20 KB | Script |
| 02_model_training.py | 16 KB | Script |
| 03_streamlit_app.py | 25 KB | Script |
| README.md | 35 KB | Doc |
| QUICK_START.md | 15 KB | Doc |
| CONFIGURATION.md | 30 KB | Doc |
| requirements.txt | 1 KB | Config |
| processed_features.csv | 50-100 MB | Data |

---

## 🎯 Quick Navigation

### By Purpose

**"How do I run everything?"**
→ QUICK_START.md or run_pipeline.bat

**"What does the pipeline do?"**
→ README.md or PROJECT_SUMMARY.md

**"How do I customize it?"**
→ CONFIGURATION.md

**"Help, I have an error!"**
→ README.md (Troubleshooting section)

**"What are the requirements?"**
→ requirements.txt or README.md (Installation)

---

### By Role

**Developer/Data Scientist:**
- Read: README.md + CONFIGURATION.md
- Edit: Python scripts for customization

**Business/Manager:**
- Read: PROJECT_SUMMARY.md
- Run: run_pipeline.bat/sh

**Student/Learner:**
- Read: README.md + CONFIGURATION.md
- Study: Python scripts
- Run: Each step manually to understand

---

## ✅ Pre-Execution Checklist

- [ ] Python 3.8+ installed
- [ ] All files present in same folder
- [ ] Data folders exist (asthma/, healthy/, etc.)
- [ ] At least 5GB disk space available
- [ ] Read README.md (first time only)
- [ ] Read QUICK_START.md (before first run)

---

## 🔗 File Dependencies

```
Data Folders (asthma/, etc.)
    ↓
    → 01_data_preprocessing.py
        ↓
        → processed_features.csv
            ↓
            → 02_model_training.py
                ↓
                → models/ folder
                    ↓
                    → 03_streamlit_app.py (uses models/)
                        ↓
                        → Web App on port 8501
```

---

## 📞 Common Questions

**Q: Where do I start?**
A: Run `run_pipeline.bat` (Windows) or follow QUICK_START.md

**Q: How long does it take?**
A: 45-60 minutes total (30-45 min preprocessing, 5-10 min training, instant for app)

**Q: Is this for clinical use?**
A: No - research purposes only. Always consult medical professionals.

**Q: Can I modify the parameters?**
A: Yes! See CONFIGURATION.md for all customization options.

**Q: What if preprocessing fails?**
A: Check README.md troubleshooting section or verify data files.

---

## 🎓 Learning Recommendations

1. **Beginner**: Run pipeline once end-to-end (follow QUICK_START.md)
2. **Intermediate**: Read Python scripts, understand each step
3. **Advanced**: Customize parameters (CONFIGURATION.md), optimize models
4. **Expert**: Modify algorithms, add new features, deploy to production

---

**Status**: ✅ Complete and ready to use  
**Version**: 1.0  
**Last Updated**: April 18, 2026

🚀 **You're all set! Start with QUICK_START.md or run_pipeline.bat**
