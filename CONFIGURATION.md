"""
Advanced Configuration Guide
Customize the ML pipeline for your specific needs
"""

# =============================================================================
# SECTION 1: DATA PREPROCESSING CONFIGURATION
# =============================================================================

# File: 01_data_preprocessing.py

# Customize audio parameters
SAMPLE_RATE = 22050          # Hz (higher = more detail but slower)
                              # Common: 22050, 44100, 48000

MAX_DURATION = 5.0           # seconds (match your audio length)
                              # Your data: 1.5-5 seconds

# Add/remove features
# Modify extract_features() method to include/exclude:

ENABLE_MFCC = True           # Mel-Frequency Cepstral Coefficients
MFCC_COEFFICIENTS = 13       # Number of coefficients (higher = more detail)

ENABLE_SPECTRAL = True       # Spectral features
ENABLE_CHROMA = True         # Chromatic features
ENABLE_TEMPORAL = True       # Time-domain features

# Batch processing (for large datasets)
BATCH_SIZE = 100             # Process N files at a time

# Save processed data
OUTPUT_FORMAT = 'csv'        # or 'pickle' for faster loading
NORMALIZED = True            # Normalize features to 0-1 range


# =============================================================================
# SECTION 2: MODEL TRAINING CONFIGURATION
# =============================================================================

# File: 02_model_training.py

# Data split configuration
TEST_SIZE = 0.2              # 20% for testing, 80% for training
VALIDATION_SIZE = 0.1        # Optional: hold-out validation set
RANDOM_STATE = 42            # For reproducibility

# Cross-validation
CV_FOLDS = 5                 # 5-fold cross-validation
                              # Higher = slower but more robust

# Model parameters

# 1. Logistic Regression
LR_MAX_ITER = 1000
LR_C = 1.0                   # Regularization strength (lower = more regulation)
LR_SOLVER = 'lbfgs'          # or 'liblinear', 'newton-cg'
LR_MULTI_CLASS = 'multinomial'

# 2. Random Forest
RF_N_ESTIMATORS = 200        # Number of trees (higher = better but slower)
RF_MAX_DEPTH = 20            # Tree depth (higher = more complex, risk overfitting)
RF_MIN_SAMPLES_SPLIT = 5     # Min samples to split node
RF_N_JOBS = -1               # -1 uses all CPU cores

# 3. Gradient Boosting (BEST MODEL)
GB_N_ESTIMATORS = 200        # Number of boosting stages
GB_LEARNING_RATE = 0.1       # Step size (lower = slower learning, more robust)
GB_MAX_DEPTH = 5             # Tree depth (keep small)
GB_SUBSAMPLE = 0.8           # Fraction of samples used for fitting

# 4. SVM
SVM_KERNEL = 'rbf'           # 'linear', 'rbf', 'poly', 'sigmoid'
SVM_C = 10                   # Regularization (higher = less regulation)
SVM_GAMMA = 'scale'          # Kernel coefficient

# Feature scaling
USE_STANDARD_SCALER = True   # Standardize features (important for SVM, LR)
USE_MIN_MAX_SCALER = False   # Normalize to 0-1 range


# =============================================================================
# SECTION 3: STREAMLIT APP CONFIGURATION
# =============================================================================

# File: 03_streamlit_app.py

# Model selection
MODEL_PATH = 'models/best_model_Random_Forest.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

# App parameters
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB
SUPPORTED_FORMATS = ['wav', 'mp3', 'ogg', 'flac']

# Audio analysis
WAVEFORM_FIGSIZE = (12, 3)
SPECTROGRAM_FIGSIZE = (12, 4)
PROB_CHART_FIGSIZE = (10, 5)

# Prediction confidence threshold
HIGH_CONFIDENCE = 0.80       # Green indicator
MEDIUM_CONFIDENCE = 0.60     # Orange indicator  
LOW_CONFIDENCE = 0.40        # Red indicator

# Class-specific colors
CLASS_COLORS = {
    'Healthy': '#51CF66',     # Green
    'Asthma': '#FF6B6B',      # Red
    'Bronchial': '#FFA07A',   # Orange
    'COPD': '#45B7D1',        # Blue
    'Pneumonia': '#9B59B6'    # Purple
}


# =============================================================================
# SECTION 4: PERFORMANCE TUNING
# =============================================================================

# For faster training (but lower accuracy)
FAST_MODE = False            # Reduce features, smaller models if True

# For best accuracy (but slower)
ACCURACY_MODE = True         # More models, more cross-validation if True

# Parallel processing
N_JOBS = -1                  # Use all cores (-1), or specific number

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# SECTION 5: ADVANCED FEATURE ENGINEERING
# =============================================================================

# Add custom features (modify AudioDataProcessor class)

def extract_custom_features(audio, sr=22050):
    """Add your own features here"""
    import librosa
    import numpy as np
    
    features = {}
    
    # Example: Spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features['spec_bandwidth_mean'] = np.mean(spec_bw)
    features['spec_bandwidth_std'] = np.std(spec_bw)
    
    # Example: Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['spec_contrast_mean'] = np.mean(spec_contrast, axis=1)
    features['spec_contrast_std'] = np.std(spec_contrast, axis=1)
    
    # Example: Mel spectrogram energy
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_energy_mean'] = np.mean(mel_db)
    features['mel_energy_std'] = np.std(mel_db)
    
    return features


# =============================================================================
# SECTION 6: OPTIMIZATION RECIPES
# =============================================================================

# RECIPE 1: Fast Setup (10 minutes)
"""
1. Set TEST_SIZE = 0.3 (less data)
2. Set RF_N_ESTIMATORS = 50
3. Set GB_N_ESTIMATORS = 50
4. Set CV_FOLDS = 3
"""

# RECIPE 2: High Accuracy (2 hours)
"""
1. Set TEST_SIZE = 0.15
2. Set RF_N_ESTIMATORS = 500
3. Set GB_N_ESTIMATORS = 500
4. Set CV_FOLDS = 10
5. Enable more features
"""

# RECIPE 3: Production Deployment (balanced)
"""
1. Keep defaults
2. Set RF_N_ESTIMATORS = 300
3. Set GB_N_ESTIMATORS = 250
4. Set CV_FOLDS = 5
5. Save model with full preprocessing info
"""

# RECIPE 4: Mobile Deployment (small model)
"""
1. Use Logistic Regression only
2. Reduce features to 30
3. Use quantization for model compression
4. Set MAX_DURATION = 3 seconds
"""


# =============================================================================
# SECTION 7: HYPERPARAMETER OPTIMIZATION (Advanced)
# =============================================================================

# Use GridSearchCV or RandomizedSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)


# =============================================================================
# SECTION 8: CLASS IMBALANCE HANDLING
# =============================================================================

# If certain diseases have fewer samples:

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Use when initiating models:
# LR: class_weight='balanced'
# RF: class_weight='balanced'
# GB: no direct support, use sample weights
# SVM: class_weight='balanced'


# =============================================================================
# SECTION 9: MODEL PERSISTENCE & DEPLOYMENT
# =============================================================================

import pickle
import joblib

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Or use joblib (better for large ML models)
joblib.dump(model, 'model.joblib')

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

model = joblib.load('model.joblib')

# Convert to ONNX for framework-agnostic deployment
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
# 
# initial_type = [('float_input', FloatTensorType([None, 65]))]
# onnx_model = convert_sklearn(model, initial_types=initial_type)
# with open('model.onnx', "wb") as f:
#     f.write(onnx_model.SerializeToString())


# =============================================================================
# SECTION 10: MONITORING & EVALUATION
# =============================================================================

# Additional metrics to track

from sklearn.metrics import roc_auc_score, log_loss, hamming_loss

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Log Loss
log_loss_score = log_loss(y_test, y_pred_proba)

# Hamming Loss
hamming = hamming_loss(y_test, y_pred)

# Per-class metrics
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None
)

print("Per-class metrics:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")


# =============================================================================
# SECTION 11: ENSEMBLE METHODS
# =============================================================================

# Combine multiple models for better predictions

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200)),
        ('gb', GradientBoostingClassifier(n_estimators=200)),
        ('svm', SVC(probability=True))
    ],
    voting='soft'  # or 'hard'
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)


# =============================================================================
# SECTION 12: LOGGING & DEBUGGING
# =============================================================================

import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log key events
logger.info("Starting data preprocessing...")
logger.debug(f"Loaded {n_samples} audio files")
logger.warning("Audio file has low SNR")
logger.error("Failed to load model")


# =============================================================================
# QUICK TEMPLATE FOR CUSTOMIZATION
# =============================================================================

"""
To customize the pipeline:

1. Copy desired configuration from sections above
2. Open the corresponding Python file
3. Find the configuration section at the top
4. Update parameter values
5. Save and run the script

Example:
    # Original
    RF_N_ESTIMATORS = 200
    
    # Modified for faster training
    RF_N_ESTIMATORS = 50

Common customizations:
    - Faster pipeline: reduce estimators, folds, features
    - Better accuracy: increase estimators, folds, features
    - Production use: balance both, add validation set
    - Mobile deployment: smaller features, simpler model
"""


# =============================================================================
# TESTING CONFIGURATIONS
# =============================================================================

# Config for testing (quick run)
TEST_CONFIG = {
    'sample_rate': 22050,
    'max_duration': 3.0,          # Shorter clips
    'test_size': 0.3,
    'rf_n_estimators': 50,        # Fewer trees
    'gb_n_estimators': 50,
    'cv_folds': 3,
    'expected_time': '10 minutes'
}

# Config for demo (medium run)
DEMO_CONFIG = {
    'sample_rate': 22050,
    'max_duration': 5.0,
    'test_size': 0.2,
    'rf_n_estimators': 150,
    'gb_n_estimators': 150,
    'cv_folds': 5,
    'expected_time': '25 minutes'
}

# Config for production (full run)
PRODUCTION_CONFIG = {
    'sample_rate': 22050,
    'max_duration': 5.0,
    'test_size': 0.15,
    'rf_n_estimators': 300,
    'gb_n_estimators': 250,
    'cv_folds': 10,
    'expected_time': '60 minutes'
}


print("""
📚 Configuration Guide
=====================

To get started with customization:
1. Read the section above that applies to you
2. Make changes in the corresponding Python file
3. Test with smaller dataset first
4. Run full pipeline when satisfied

For questions, refer to README.md and QUICK_START.md
""")
