"""
Data Preprocessing and Feature Extraction Pipeline
Loads audio files, extracts features, and prepares data for ML training
"""

import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioDataProcessor:
    """Process audio files and extract features for ML"""
    
    def __init__(self, data_dir, sample_rate=22050, max_duration=5.0):
        """
        Initialize the audio processor
        
        Args:
            data_dir: Path to base data directory
            sample_rate: Sampling rate for audio (Hz)
            max_duration: Maximum duration to load (seconds)
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.n_samples = int(sample_rate * max_duration)
        self.features_data = []
        
    def load_audio(self, filepath):
        """Load and normalize audio file"""
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate, duration=self.max_duration)
            # Pad or trim to fixed length
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.n_samples]
            return audio
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract comprehensive audio features"""
        if audio is None:
            return None
        
        features = {}
        
        # MFCC (Mel-frequency cepstral coefficients) - most important
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
        """Flatten feature dictionary into a single array"""
        flattened = []
        for key, value in sorted(features_dict.items()):
            if isinstance(value, np.ndarray):
                flattened.extend(value.flatten())
            else:
                flattened.append(value)
        return np.array(flattened)
    
    def process_dataset(self):
        """Process all audio files and return feature dataframe"""
        print("Processing dataset...")
        data_list = []
        
        class_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for class_folder in class_folders:
            class_name = class_folder.name
            print(f"\nProcessing class: {class_name}")
            
            audio_files = list(class_folder.glob("*.wav"))
            print(f"  Found {len(audio_files)} files")
            
            for i, audio_file in enumerate(audio_files):
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{len(audio_files)}")
                
                # Load and process audio
                audio = self.load_audio(str(audio_file))
                if audio is None:
                    continue
                
                # Extract features
                features = self.extract_features(audio)
                if features is None:
                    continue
                
                # Create feature vector
                feature_vector = self.flatten_features(features)
                
                # Store with label
                data_list.append({
                    'filename': audio_file.name,
                    'class': class_name,
                    'features': feature_vector
                })
        
        print(f"\n✓ Processed {len(data_list)} files successfully")
        
        # Create dataframe
        df = pd.DataFrame(data_list)
        
        # Add features as separate columns
        feature_cols = pd.DataFrame(
            np.array([x for x in df['features']]),
            columns=[f'feature_{i}' for i in range(len(df['features'].iloc[0]))]
        )
        
        df = pd.concat([df[['filename', 'class']], feature_cols], axis=1)
        
        return df
    
    def save_processed_data(self, df, output_file='processed_data.csv'):
        """Save processed data to CSV"""
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"✓ Data saved to {output_path}")
        return output_path


def main():
    """Main pipeline execution"""
    # Set data directory
    data_dir = Path(".")  # Current directory containing class folders
    
    # Initialize processor
    processor = AudioDataProcessor(
        data_dir=data_dir,
        sample_rate=22050,
        max_duration=5.0
    )
    
    # Process all data
    df = processor.process_dataset()
    
    # Display info
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len([c for c in df.columns if c.startswith('feature_')])}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts())
    print("\nFirst few rows:")
    print(df.head())
    
    # Save processed data
    processor.save_processed_data(df, 'processed_features.csv')


if __name__ == "__main__":
    main()
