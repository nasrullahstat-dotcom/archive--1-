"""
Model Training and Evaluation Pipeline
Trains multiple ML models and evaluates their performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize


class RespiratoryDiseaseClassifier:
    """Complete ML pipeline for respiratory disease classification"""
    
    def __init__(self, data_path='processed_features.csv', test_size=0.2, random_state=42):
        """
        Initialize classifier
        
        Args:
            data_path: Path to processed feature CSV
            test_size: Proportion of data for testing
            random_state: Random seed
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
        
        self.results = {}
    
    def load_data(self):
        """Load processed features"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and labels
        X = df.iloc[:, 2:]  # Skip filename and class
        y = df['class']
        
        print(f"Loaded {len(df)} samples with {X.shape[1]} features")
        print(f"Classes: {sorted(df['class'].unique())}")
        
        return X, y
    
    def prepare_data(self):
        """Split and normalize data"""
        print("\nPreparing data...")
        
        # Load data
        X, y = self.load_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_original = self.label_encoder.inverse_transform(y_train)
        self.y_test_original = self.label_encoder.inverse_transform(y_test)
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        print(f"✓ Features normalized using StandardScaler")
    
    def train_models(self):
        """Train multiple models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        if self.X_train is None:
            self.prepare_data()
        
        # Define models
        model_configs = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=self.random_state
            ),
            'SVM (RBF)': SVC(kernel='rbf', C=10, probability=True, random_state=self.random_state),
        }
        
        # Train each model
        for model_name, model in model_configs.items():
            print(f"\nTraining {model_name}...")
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
            print(f"✓ {model_name} trained")
        
        print("\n✓ All models trained successfully")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        best_accuracy = 0
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = (model_name, model)
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model[0]} (Accuracy: {best_accuracy:.4f})")
        print("="*60)
    
    def detailed_classification_report(self):
        """Print detailed classification report for best model"""
        model_name, model = self.best_model
        y_pred = model.predict(self.X_test)
        
        print(f"\nDETAILED REPORT - {model_name}")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(
            self.y_test_original, 
            self.label_encoder.inverse_transform(y_pred),
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        return cm, y_pred
    
    def plot_evaluation(self):
        """Create evaluation visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        f1_scores = [self.results[m]['f1'] for m in model_names]
        
        ax1 = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.35
        ax1.bar(x - width/2, accuracies, width, label='Accuracy')
        ax1.bar(x + width/2, f1_scores, width, label='F1-Score')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Confusion Matrix for best model
        ax2 = axes[0, 1]
        model_name, model = self.best_model
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2, 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        ax2.set_title(f'Confusion Matrix - {model_name}')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # 3. Cross-validation scores
        ax3 = axes[1, 0]
        cv_means = [self.results[m]['cv_mean'] for m in model_names]
        cv_stds = [self.results[m]['cv_std'] for m in model_names]
        ax3.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        ax3.set_ylabel('CV Score')
        ax3.set_title('Cross-Validation Performance')
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Metrics for best model
        ax4 = axes[1, 1]
        best_results = self.results[model_name]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            best_results['accuracy'],
            best_results['precision'],
            best_results['recall'],
            best_results['f1']
        ]
        ax4.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax4.set_ylabel('Score')
        ax4.set_title(f'Best Model Metrics - {model_name}')
        ax4.set_ylim([0, 1])
        for i, v in enumerate(values):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation plots saved to 'model_evaluation.png'")
        plt.close()
    
    def save_model(self, output_dir='models'):
        """Save best model and scaler"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        model_name, model = self.best_model
        
        # Save model
        model_path = output_dir / f'best_model_{model_name.replace(" ", "_")}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to {model_path}")
        
        # Save scaler
        scaler_path = output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {scaler_path}")
        
        # Save label encoder
        encoder_path = output_dir / 'label_encoder.pkl'
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Label encoder saved to {encoder_path}")
        
        # Save model info
        info_path = output_dir / 'model_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"Best Model: {model_name}\n")
            f.write(f"Accuracy: {self.results[model_name]['accuracy']:.4f}\n")
            f.write(f"F1-Score: {self.results[model_name]['f1']:.4f}\n")
            f.write(f"Classes: {', '.join(self.label_encoder.classes_)}\n")
        print(f"✓ Model info saved to {info_path}")


def main():
    """Main execution"""
    # Initialize classifier
    clf = RespiratoryDiseaseClassifier(
        data_path='processed_features.csv',
        test_size=0.2,
        random_state=42
    )
    
    # Train and evaluate
    clf.train_models()
    clf.evaluate_models()
    
    # Detailed results
    clf.detailed_classification_report()
    
    # Visualizations
    clf.plot_evaluation()
    
    # Save best model
    clf.save_model(output_dir='models')
    
    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
