"""
Batch Processor for Respiratory Disease Detection
Process multiple audio files and generate comprehensive reports
"""

import os
import pickle
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from report_generator import ReportGenerator, create_csv_report


class BatchAudioProcessor:
    """Process multiple audio files in batch"""
    
    def __init__(self, model_path='models/best_model_SVM_(RBF).pkl',
                 scaler_path='models/scaler.pkl',
                 encoder_path='models/label_encoder.pkl',
                 sample_rate=22050, max_duration=5.0):
        """
        Initialize batch processor
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            encoder_path: Path to label encoder
            sample_rate: Audio sample rate
            max_duration: Max audio duration
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.n_samples = int(sample_rate * max_duration)
        
        # Load model components
        print("Loading model components...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.report_generator = ReportGenerator(self.label_encoder, self.model, self.scaler)
        self.predictions = []
        print("✓ Model loaded successfully")
    
    def load_audio(self, filepath):
        """Load and preprocess audio"""
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate, duration=self.max_duration)
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.n_samples]
            return audio
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract audio features"""
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
        
        # Onset
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
    
    def process_file(self, filepath):
        """Process single audio file"""
        filename = Path(filepath).name
        
        # Load audio
        audio = self.load_audio(filepath)
        if audio is None:
            return None
        
        # Extract features
        features = self.extract_features(audio)
        if features is None:
            return None
        
        # Get feature vector
        feature_vector = self.flatten_features(features)
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.model.predict(feature_scaled)[0]
        probabilities = self.model.predict_proba(feature_scaled)[0]
        
        predicted_class = self.label_encoder.classes_[prediction]
        audio_duration = len(audio) / self.sample_rate
        
        return {
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': probabilities[prediction],
            'probabilities': probabilities,
            'audio_duration': audio_duration
        }
    
    def process_directory(self, directory, pattern='*.wav', progress_interval=10):
        """
        Process all audio files in directory
        
        Args:
            directory: Path to directory containing audio files
            pattern: File pattern to match
            progress_interval: Show progress every N files
        """
        directory = Path(directory)
        audio_files = list(directory.glob(pattern))
        
        if not audio_files:
            print(f"✗ No audio files found in {directory}")
            return []
        
        print(f"\nProcessing {len(audio_files)} audio files from {directory.name}")
        print("=" * 60)
        
        successful = 0
        failed = 0
        
        for i, filepath in enumerate(audio_files, 1):
            if i % progress_interval == 0 or i == len(audio_files):
                print(f"Progress: {i}/{len(audio_files)}")
            
            result = self.process_file(filepath)
            if result:
                self.predictions.append(result)
                successful += 1
            else:
                failed += 1
        
        print(f"✓ Processed {successful} files successfully")
        if failed > 0:
            print(f"✗ Failed to process {failed} files")
        
        return self.predictions
    
    def process_multiple_directories(self, base_path, directories):
        """
        Process audio files from multiple disease directories
        
        Args:
            base_path: Base path containing all directories
            directories: List of directory names to process
        """
        base_path = Path(base_path)
        all_predictions = []
        
        for dir_name in directories:
            dir_path = base_path / dir_name
            if dir_path.exists():
                results = self.process_directory(dir_path)
                all_predictions.extend(results)
            else:
                print(f"✗ Directory not found: {dir_path}")
        
        self.predictions = all_predictions
        return all_predictions
    
    def generate_batch_report(self):
        """Generate summary report for batch"""
        if not self.predictions:
            print("✗ No predictions to report")
            return None
        
        batch_report = self.report_generator.generate_batch_report(self.predictions)
        return batch_report
    
    def export_csv_report(self, output_file='batch_predictions.csv'):
        """Export predictions to CSV"""
        if not self.predictions:
            print("✗ No predictions to export")
            return None
        
        csv_file = create_csv_report(self.predictions, output_file)
        return csv_file
    
    def export_text_report(self, output_file='batch_report.txt'):
        """Export batch report as text"""
        batch_report = self.generate_batch_report()
        if batch_report:
            report_text = self.report_generator.format_batch_report_text(batch_report)
            self.report_generator.save_report_text(report_text, output_file)
            print(f"✓ Report saved to {output_file}")
            return output_file
        return None


def main():
    """Main batch processing execution"""
    
    # Initialize processor
    processor = BatchAudioProcessor()
    
    # Define directories to process
    base_path = Path('.')
    disease_directories = ['asthma', 'Bronchial', 'copd', 'healthy', 'pneumonia']
    
    print("\n" + "="*60)
    print("BATCH RESPIRATORY DISEASE DETECTION")
    print("="*60)
    
    # Process all directories
    predictions = processor.process_multiple_directories(base_path, disease_directories)
    
    # Generate reports
    print("\n" + "="*60)
    print("GENERATING REPORTS")
    print("="*60)
    
    # Batch text report
    processor.export_text_report('batch_analysis.txt')
    
    # CSV export
    processor.export_csv_report('predictions_batch.csv')
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH SUMMARY")
    print("="*60)
    batch_report = processor.generate_batch_report()
    
    if batch_report:
        print(f"\nTotal samples analyzed: {batch_report['total_samples_analyzed']}")
        print(f"Average confidence: {batch_report['average_confidence']}")
        print(f"High confidence samples: {batch_report['high_confidence_samples']}")
        
        print("\nDisease Distribution:")
        for disease, stats in batch_report['disease_distribution'].items():
            print(f"  {disease}: {stats['count']} samples ({stats['percentage']})")
    
    print("\n✓ Batch processing completed!")
    print("Files generated:")
    print("  • batch_analysis.txt (detailed report)")
    print("  • predictions_batch.csv (data export)")


if __name__ == "__main__":
    main()
