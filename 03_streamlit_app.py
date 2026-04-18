"""
Streamlit Web Application for Respiratory Disease Detection
Real-time audio analysis and disease classification
"""

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import report generator
try:
    from report_generator import ReportGenerator
except:
    ReportGenerator = None


# Set page config
st.set_page_config(
    page_title="Respiratory Disease Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_processors():
    """Load trained model, scaler, and label encoder"""
    try:
        models_dir = Path('models')
        
        # Load model
        with open(models_dir / 'best_model_SVM_(RBF).pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoder
        with open(models_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


class AudioAnalyzer:
    """Analyze audio and extract features for prediction"""
    
    def __init__(self, sample_rate=22050, max_duration=5.0):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.n_samples = int(sample_rate * max_duration)
    
    def load_audio(self, file_path):
        """Load audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.max_duration)
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.n_samples]
            return audio, sr
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            return None, None
    
    def extract_features(self, audio):
        """Extract comprehensive audio features"""
        if audio is None:
            return None
        
        features = {}
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, n_chroma=12)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Tempogram
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        features['onset_mean'] = np.mean(onset_env)
        features['onset_std'] = np.std(onset_env)
        
        return features
    
    def flatten_features(self, features_dict):
        """Flatten features for model input"""
        flattened = []
        for key, value in sorted(features_dict.items()):
            if isinstance(value, np.ndarray):
                flattened.extend(value.flatten())
            else:
                flattened.append(value)
        return np.array(flattened).reshape(1, -1)


def main():
    """Main app"""
    st.title("🫁 Respiratory Disease Detection System")
    st.markdown("**AI-Powered Respiratory Sound Analysis**")
    
    # Initialize session state for prediction history
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None
    
    # Load models
    model, scaler, label_encoder = load_models_and_processors()
    
    if model is None:
        st.error("❌ Models not found. Please train the model first (run 02_model_training.py)")
        return
    
    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    # Initialize report generator
    if ReportGenerator is not None:
        report_gen = ReportGenerator(label_encoder, model, scaler)
    else:
        report_gen = None
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This application uses machine learning to analyze respiratory sounds
        and classify respiratory conditions:
        - **Asthma**
        - **Bronchial Disorder**
        - **COPD**
        - **Healthy**
        - **Pneumonia**
        
        Upload or record an audio sample for analysis.
        """)
        
        st.header("📊 Dataset Info")
        st.write("""
        - Total Samples: 1,211
        - Features Extracted: 60 per sample
        - Model: SVM (RBF) Classifier
        - Cross-validation: 5-fold
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Prediction", "📈 Analytics", "📋 Reports", "ℹ️ Info"])
    
    with tab1:
        st.header("Upload or Test Audio")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Option 1: Upload Audio File")
            uploaded_file = st.file_uploader("Choose a WAV file", type=['wav', 'mp3', 'ogg'])
            
            if uploaded_file is not None:
                # Display audio player
                st.audio(uploaded_file)
                
                # Save temp file
                with st.spinner("Processing audio..."):
                    temp_path = f"temp_{datetime.now().timestamp()}.wav"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and analyze
                    audio, sr = analyzer.load_audio(temp_path)
                    if audio is not None:
                        features = analyzer.extract_features(audio)
                        feature_vector = analyzer.flatten_features(features)
                        feature_scaled = scaler.transform(feature_vector)
                        
                        # Prediction
                        prediction = model.predict(feature_scaled)[0]
                        probabilities = model.predict_proba(feature_scaled)[0]
                        
                        disease = label_encoder.classes_[prediction]
                        confidence = probabilities[prediction]
                        
                        # Save to history
                        prediction_record = {
                            'timestamp': datetime.now().isoformat(),
                            'filename': uploaded_file.name,
                            'duration': len(audio) / sr,
                            'predicted_class': disease,
                            'confidence': confidence,
                        }
                        for i, cls in enumerate(label_encoder.classes_):
                            prediction_record[f'prob_{cls}'] = probabilities[i]
                        
                        st.session_state.predictions_history.append(prediction_record)
                        
                        # Generate detailed report if available
                        if report_gen:
                            st.session_state.current_report = report_gen.generate_individual_report(
                                uploaded_file.name,
                                len(audio) / sr,
                                disease,
                                probabilities,
                                None
                            )
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        st.markdown("### 🎯 Prediction Results")
                        col_pred1, col_pred2 = st.columns([2, 1])
                        
                        with col_pred1:
                            # Color based on disease
                            if disease == "Healthy":
                                st.markdown(f"### 🟢 **{disease}** (NORMAL)")
                            else:
                                st.markdown(f"### 🔴 **{disease}** (ABNORMAL)")
                        
                        with col_pred2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Probability distribution
                        st.markdown("### 📊 Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Disease': label_encoder.classes_,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        # Bar chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['#FF6B6B' if disease != 'Healthy' else '#51CF66' 
                                 for disease in prob_df['Disease']]
                        ax.barh(prob_df['Disease'], prob_df['Probability'], color=colors)
                        ax.set_xlabel('Probability')
                        ax.set_title('Disease Classification Probabilities')
                        ax.set_xlim([0, 1])
                        for i, v in enumerate(prob_df['Probability']):
                            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                        st.pyplot(fig)
                        
                        # Waveform visualization
                        st.markdown("### 🔊 Audio Waveform")
                        fig, ax = plt.subplots(figsize=(12, 3))
                        times = np.linspace(0, len(audio) / sr, len(audio))
                        ax.plot(times, audio, linewidth=0.5)
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Amplitude')
                        ax.set_title('Audio Signal')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Spectrogram
                        st.markdown("### 📡 Spectrogram")
                        fig, ax = plt.subplots(figsize=(12, 4))
                        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                        img = librosa.display.specshow(D, sr=sr, hop_length=512, 
                                                       x_axis='time', y_axis='log', ax=ax)
                        ax.set_title('Spectrogram')
                        fig.colorbar(img, ax=ax, format='%+2.0f dB')
                        st.pyplot(fig)
                    
                    # Cleanup
                    os.remove(temp_path)
        
        with col2:
            st.subheader("Option 2: Test with Sample")
            
            test_folder = Path(".")
            classes = ['asthma', 'Bronchial', 'copd', 'healthy', 'pneumonia']
            
            # Try to find sample files
            sample_files = {}
            for class_name in classes:
                class_path = test_folder / class_name
                if class_path.exists():
                    wav_files = list(class_path.glob('*.wav'))
                    if wav_files:
                        sample_files[class_name] = wav_files[0]
            
            if sample_files:
                selected_class = st.selectbox("Select test class", list(sample_files.keys()))
                
                if st.button("🔍 Analyze Sample"):
                    with st.spinner("Processing sample audio..."):
                        sample_file = sample_files[selected_class]
                        audio, sr = analyzer.load_audio(str(sample_file))
                        
                        if audio is not None:
                            features = analyzer.extract_features(audio)
                            feature_vector = analyzer.flatten_features(features)
                            feature_scaled = scaler.transform(feature_vector)
                            
                            prediction = model.predict(feature_scaled)[0]
                            probabilities = model.predict_proba(feature_scaled)[0]
                            
                            disease = label_encoder.classes_[prediction]
                            confidence = probabilities[prediction]
                            
                            st.success("✅ Analysis Complete!")
                            
                            col_p1, col_p2 = st.columns([2, 1])
                            with col_p1:
                                st.markdown(f"### **{disease}**")
                            with col_p2:
                                st.metric("Confidence", f"{confidence*100:.1f}%")
                            
                            prob_df = pd.DataFrame({
                                'Disease': label_encoder.classes_,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(9, 4))
                            ax.barh(prob_df['Disease'], prob_df['Probability'])
                            ax.set_xlabel('Probability')
                            st.pyplot(fig)
            else:
                st.info("📁 No test samples found in data folders")
    
    with tab2:
        st.header("Model Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Classes", len(label_encoder.classes_))
        with col2:
            st.metric("Total Features", 60)
        with col3:
            st.metric("Model Type", "SVM (RBF)")
        
        st.markdown("### 📊 Disease Distribution in Dataset")
        
        # Load original data for stats
        try:
            df = pd.read_csv('processed_features.csv')
            class_counts = df['class'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            class_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
            ax.set_title('Sample Distribution by Disease')
            ax.set_xlabel('Disease')
            ax.set_ylabel('Number of Samples')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        except:
            st.info("Data file not found")
        
        if os.path.exists('model_evaluation.png'):
            st.markdown("### 📈 Model Evaluation Metrics")
            from PIL import Image
            img = Image.open('model_evaluation.png')
            st.image(img, use_column_width=True)
    
    with tab3:
        st.header("System Information")
        
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Audio Upload**: Submit a respiratory sound recording (WAV format)
        2. **Feature Extraction**: Extract 60 features including:
           - MFCC (Mel-Frequency Cepstral Coefficients)
           - Spectral features (centroid, rolloff)
           - Zero crossing rate
           - Energy and chroma features
        3. **Model Prediction**: SVM (RBF) classifier analyzes features
        4. **Classification**: Outputs probability for each respiratory condition
        """)
        
        st.markdown("### 📋 Supported Diseases")
        diseases_info = {
            "🟢 Healthy": "Normal respiratory sounds",
            "🔴 Asthma": "Difficulty breathing, wheezing",
            "🔴 Bronchial": "Inflammation of airways",
            "🔴 COPD": "Chronic obstructive pulmonary disease",
            "🔴 Pneumonia": "Lung infection with fluid"
        }
        
        for disease, description in diseases_info.items():
            st.write(f"**{disease}**: {description}")
        
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        This system is an AI-based analytical tool for research purposes.
        **It is NOT a medical device and should NOT be used for diagnosis.**
        Always consult qualified healthcare professionals for medical diagnosis and treatment.
        """)
        
        st.markdown("### 📚 Citation")
        st.code("""
        Dataset: Asthma Detection Dataset: Version 2
        Paper: Tawfik, M., Al-Zidi, N. M., Fathail, I., & Nimbhore, S. (2022)
        """)
    
    with tab4:
        st.header("📋 Detailed Reports")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predictions Made", len(st.session_state.predictions_history))
        with col2:
            if st.session_state.predictions_history:
                confidences = [p.get('confidence', 0) for p in st.session_state.predictions_history]
                avg_confidence = sum(confidences) / len(confidences) * 100
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            else:
                st.metric("Average Confidence", "N/A")
        with col3:
            if st.session_state.predictions_history:
                disease_counts = {}
                for p in st.session_state.predictions_history:
                    disease = p.get('predicted_class', 'Unknown')
                    disease_counts[disease] = disease_counts.get(disease, 0) + 1
                top_disease = max(disease_counts, key=disease_counts.get)
                st.metric("Most Common", top_disease)
            else:
                st.metric("Most Common", "N/A")
        
        st.divider()
        
        # Current report section
        st.subheader("Current Prediction Report")
        if st.session_state.current_report:
            try:
                report_text = report_gen.format_report_text(st.session_state.current_report)
                st.text(report_text)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    json_str = json.dumps(st.session_state.current_report, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=json_str,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=report_text,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.warning(f"Error formatting report: {str(e)}")
        else:
            st.info("📁 No predictions made yet. Upload an audio file to generate a report.")
        
        st.divider()
        
        # Prediction history
        st.subheader("Prediction History")
        if st.session_state.predictions_history:
            # Convert history to DataFrame for display
            history_data = []
            for idx, pred in enumerate(st.session_state.predictions_history, 1):
                history_data.append({
                    'ID': idx,
                    'Timestamp': pred.get('timestamp', 'N/A'),
                    'File': pred.get('filename', 'N/A'),
                    'Predicted': pred.get('predicted_class', 'N/A'),
                    'Confidence': f"{pred.get('confidence', 0)*100:.1f}%",
                    'Duration (s)': f"{pred.get('duration', 0):.2f}"
                })
            
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df_history.to_csv(index=False)
                st.download_button(
                    label="📊 Export as CSV",
                    data=csv_data,
                    file_name=f"predictions_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if report_gen:
                    try:
                        # Generate batch report from history
                        batch_report = report_gen.generate_batch_report(
                            st.session_state.predictions_history
                        )
                        batch_text = report_gen.format_batch_report_text(batch_report)
                        st.download_button(
                            label="📋 Export Batch Report",
                            data=batch_text,
                            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error generating batch report: {str(e)}")
            
            with col3:
                if st.button("🗑️ Clear History"):
                    st.session_state.predictions_history = []
                    st.session_state.current_report = None
                    st.rerun()
        else:
            st.info("📁 No prediction history yet. Make predictions to see them here.")


if __name__ == "__main__":
    main()
