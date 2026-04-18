"""
Report Generation Module
Generates detailed reports for respiratory disease predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json


class ReportGenerator:
    """Generate detailed prediction reports"""
    
    def __init__(self, label_encoder=None, model=None, scaler=None):
        """
        Initialize report generator
        
        Args:
            label_encoder: Sklearn LabelEncoder for classes
            model: Trained ML model
            scaler: Feature scaler
        """
        self.label_encoder = label_encoder
        self.model = model
        self.scaler = scaler
        self.timestamp = datetime.now()
        
    def generate_individual_report(self, filename, audio_duration, predicted_class, 
                                   probabilities, features_dict=None):
        """
        Generate report for a single prediction
        
        Args:
            filename: Name of audio file analyzed
            audio_duration: Duration of audio in seconds
            predicted_class: Predicted disease class
            probabilities: Probability array for all classes
            features_dict: Optional dictionary of extracted features
            
        Returns:
            dict: Complete report data
        """
        if self.label_encoder is None:
            classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']
        else:
            classes = self.label_encoder.classes_
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1]
        top_3_classes = [classes[i] for i in top_indices[:3]]
        top_3_probs = [probabilities[i] for i in top_indices[:3]]
        
        # Calculate confidence level
        confidence = probabilities[self.label_encoder.transform([predicted_class])[0]] * 100
        confidence_level = "Very High" if confidence > 90 else "High" if confidence > 80 else "Moderate" if confidence > 70 else "Low"
        
        # Risk assessment
        if predicted_class == 'healthy':
            risk_status = "Low Risk - Normal respiratory sounds detected"
            recommendation = "No immediate action needed. Routine monitoring recommended."
        else:
            risk_status = "Abnormal Pattern Detected - Consult healthcare professional"
            recommendation = "Recommend medical consultation for confirmatory diagnosis."
        
        report = {
            'timestamp': self.timestamp.isoformat(),
            'filename': filename,
            'audio_duration': f"{audio_duration:.1f}s",
            'prediction': {
                'primary_disease': predicted_class,
                'confidence': f"{confidence:.1f}%",
                'confidence_level': confidence_level,
            },
            'probability_distribution': {
                'rank_1': f"{classes[top_indices[0]]} - {probabilities[top_indices[0]]*100:.1f}%",
                'rank_2': f"{classes[top_indices[1]]} - {probabilities[top_indices[1]]*100:.1f}%",
                'rank_3': f"{classes[top_indices[2]]} - {probabilities[top_indices[2]]*100:.1f}%",
            },
            'risk_assessment': {
                'status': risk_status,
                'recommendation': recommendation,
            },
            'all_probabilities': {
                classes[i]: f"{probabilities[i]*100:.2f}%" for i in range(len(classes))
            }
        }
        
        return report
    
    def format_report_text(self, report):
        """Format report as readable text"""
        text = f"""
{'='*70}
RESPIRATORY DISEASE DETECTION - DETAILED ANALYSIS REPORT
{'='*70}

ANALYSIS DETAILS
{'-'*70}
Timestamp:           {report['timestamp']}
Audio File:          {report['filename']}
Audio Duration:      {report['audio_duration']}

PRIMARY PREDICTION
{'-'*70}
Predicted Disease:   {report['prediction']['primary_disease'].upper()}
Confidence Level:    {report['prediction']['confidence_level']} ({report['prediction']['confidence']})

PROBABILITY DISTRIBUTION
{'-'*70}
"""
        for disease, prob in report['all_probabilities'].items():
            bar_length = int(float(prob.strip('%')) / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            text += f"{disease:15} {prob:>7} {bar}\n"
        
        text += f"""
TOP 3 PREDICTIONS
{'-'*70}
1. {report['probability_distribution']['rank_1']}
2. {report['probability_distribution']['rank_2']}
3. {report['probability_distribution']['rank_3']}

RISK ASSESSMENT
{'-'*70}
Status:       {report['risk_assessment']['status']}
Recommendation: {report['risk_assessment']['recommendation']}

DISEASE INFORMATION
{'-'*70}
"""
        disease_info = {
            'healthy': 'Normal respiratory function. No abnormalities detected.',
            'asthma': 'Characterized by wheezing and difficulty breathing. Consider medication and trigger avoidance.',
            'bronchial': 'Inflammation of bronchial tubes. May cause persistent cough and mucus production.',
            'copd': 'Chronic obstructive condition. Limited airflow and progressive breathing difficulty.',
            'pneumonia': 'Lung infection with fluid accumulation. Requires prompt medical attention.',
        }
        
        predicted = report['prediction']['primary_disease'].lower()
        text += f"{disease_info.get(predicted, 'See healthcare provider for detailed information.')}\n"
        
        text += f"""
{'='*70}
DISCLAIMER
{'='*70}
This analysis is for research and supportive purposes only.
⚠️  NOT a substitute for professional medical diagnosis.
Consult qualified healthcare providers for medical decisions.
{'='*70}
"""
        return text
    
    def generate_batch_report(self, results_list):
        """
        Generate batch analysis report
        
        Args:
            results_list: List of prediction results
            
        Returns:
            dict: Batch report summary
        """
        df = pd.DataFrame(results_list)
        
        # Overall statistics
        total_samples = len(results_list)
        class_counts = df['predicted_class'].value_counts().to_dict()
        avg_confidence = df['confidence'].mean()
        high_confidence_count = len(df[df['confidence'] > 0.90])
        
        # Disease-specific stats
        disease_stats = {}
        for disease in df['predicted_class'].unique():
            disease_df = df[df['predicted_class'] == disease]
            disease_stats[disease] = {
                'count': len(disease_df),
                'percentage': f"{len(disease_df)/total_samples*100:.1f}%",
                'avg_confidence': f"{disease_df['confidence'].mean()*100:.1f}%",
                'min_confidence': f"{disease_df['confidence'].min()*100:.1f}%",
                'max_confidence': f"{disease_df['confidence'].max()*100:.1f}%",
            }
        
        batch_report = {
            'timestamp': self.timestamp.isoformat(),
            'total_samples_analyzed': total_samples,
            'average_confidence': f"{avg_confidence*100:.1f}%",
            'high_confidence_samples': f"{high_confidence_count}/{total_samples}",
            'disease_distribution': disease_stats,
            'processing_summary': {
                'total_abnormal': total_samples - class_counts.get('healthy', 0),
                'total_normal': class_counts.get('healthy', 0),
                'abnormal_percentage': f"{(1 - class_counts.get('healthy', 0)/total_samples)*100:.1f}%",
            }
        }
        
        return batch_report
    
    def format_batch_report_text(self, batch_report):
        """Format batch report as readable text"""
        text = f"""
{'='*70}
BATCH ANALYSIS REPORT - RESPIRATORY DISEASE DETECTION
{'='*70}

BATCH SUMMARY
{'-'*70}
Timestamp:                  {batch_report['timestamp']}
Total Samples Analyzed:     {batch_report['total_samples_analyzed']}
Average Confidence:         {batch_report['average_confidence']}
High Confidence Samples:    {batch_report['high_confidence_samples']}

DISEASE DISTRIBUTION
{'-'*70}
"""
        for disease, stats in batch_report['disease_distribution'].items():
            text += f"\n{disease.upper()}\n"
            text += f"  Count:              {stats['count']} ({stats['percentage']})\n"
            text += f"  Avg Confidence:     {stats['avg_confidence']}\n"
            text += f"  Confidence Range:   {stats['min_confidence']} - {stats['max_confidence']}\n"
        
        summary = batch_report['processing_summary']
        text += f"""
PROCESSING SUMMARY
{'-'*70}
Normal Samples:             {summary['total_normal']}
Abnormal Samples:           {summary['total_abnormal']}
Abnormal Percentage:        {summary['abnormal_percentage']}

KEY FINDINGS
{'-'*70}
1. Disease prevalence identified in batch
2. Confidence patterns across samples
3. Outliers for review (if any)

{'='*70}
RECOMMENDATIONS
{'-'*70}
• Review samples with confidence < 70% for manual verification
• Flag abnormal cases for clinical follow-up
• Consider re-testing low confidence samples
{'='*70}
"""
        return text
    
    def save_report_json(self, report, output_path='report.json'):
        """Save report as JSON"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        return output_path
    
    def save_report_text(self, report_text, output_path='report.txt'):
        """Save report as text file"""
        with open(output_path, 'w') as f:
            f.write(report_text)
        return output_path
    
    def create_full_prediction_record(self, filename, audio_duration, predicted_class,
                                     probabilities, audio_features=None):
        """
        Create a complete prediction record with all details
        
        Returns:
            dict: Full record for database/CSV storage
        """
        if self.label_encoder is None:
            classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']
        else:
            classes = self.label_encoder.classes_
        
        record = {
            'timestamp': self.timestamp.isoformat(),
            'filename': filename,
            'duration': audio_duration,
            'predicted_class': predicted_class,
            'confidence': probabilities[self.label_encoder.transform([predicted_class])[0]],
        }
        
        # Add all probabilities
        for i, cls in enumerate(classes):
            record[f'prob_{cls}'] = probabilities[i]
        
        return record


def create_csv_report(predictions_list, output_file='predictions_report.csv'):
    """
    Create a CSV report from multiple predictions
    
    Args:
        predictions_list: List of prediction dictionaries
        output_file: Output CSV file path
        
    Returns:
        Path to saved CSV
    """
    df = pd.DataFrame(predictions_list)
    df.to_csv(output_file, index=False)
    print(f"✓ CSV report saved to {output_file}")
    return output_file


# Export convenience functions
__all__ = [
    'ReportGenerator',
    'create_csv_report'
]
