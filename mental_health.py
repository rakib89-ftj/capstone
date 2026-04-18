#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mental Health Text Classification - Complete End-to-End Pipeline
Classes: Normal, Depression, Suicidal, Anxiety
Author: Capstone Project
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
class Config:
    # Paths
    DATA_DIR = "mental-health-text-classification-dataset/" # Corrected data directory
    TRAIN_FILE = r"C:\Users\rakib\Downloads\mental_heath_unbanlanced.csv" # Corrected typo in filename
    TEST_FILE = r"C:\Users\rakib\Downloads\mental_health_combined_test.csv"
    MODEL_DIR = "models/"

    # Training params
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    BATCH_SIZE = 16
    MAX_LEN = 128
    BERT_MODEL_NAME = "bert-base-uncased"
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    # Traditional ML params
    TFIDF_MAX_FEATURES = 20000
    TFIDF_NGRAM_RANGE = (1, 3)

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class labels
    LABELS = ['Normal', 'Depression', 'Suicidal', 'Anxiety']

config = Config()

# -------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def plot_confusion_matrix(y_true, y_pred, label_encoder, title="Confusion Matrix"):
    """Plot and save confusion matrix"""
    # Get the unique encoded labels present in y_true and y_pred
    unique_encoded_labels = np.unique(np.concatenate((y_true, y_pred)))
    # Compute confusion matrix using the integer encoded labels
    cm = confusion_matrix(y_true, y_pred, labels=unique_encoded_labels)

    # Get the corresponding string labels for display
    display_labels = label_encoder.inverse_transform(unique_encoded_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_labels, yticklabels=display_labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{config.MODEL_DIR}{title.replace(' ', '_')}.png")
    plt.show()
    return cm

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and print classification report"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=config.LABELS))

    return accuracy, precision, recall, f1, y_pred

# -------------------------------
# 3. DATA LOADING & EDA
# -------------------------------
def load_and_explore_data():
    """Load datasets and perform exploratory analysis"""
    print("\n" + "="*60)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*60)

    # Load training data
    train_path = config.TRAIN_FILE if os.path.isabs(config.TRAIN_FILE) else os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    train_df = pd.read_csv(train_path)
    print(f"\n✅ Training data loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

    # Check column names - handle variations
    text_col = 'text' if 'text' in train_df.columns else 'statement'
    label_col = 'status' if 'status' in train_df.columns else 'label'

    train_df = train_df[[text_col, label_col]].rename(
        columns={text_col: 'text', label_col: 'label'}
    )

    # Clean text
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    train_df = train_df[train_df['cleaned_text'].str.len() > 10]

    # Display class distribution
    print("\n📊 Class Distribution in Training Data:")
    class_counts = train_df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"   {label}: {count:,} ({count/len(train_df)*100:.1f}%)")

    # Visualize distribution
    plt.figure(figsize=(8, 5))
    class_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Mental Health Category')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{config.MODEL_DIR}class_distribution.png")
    plt.show()

    # Load test data
    test_path = config.TEST_FILE if os.path.isabs(config.TEST_FILE) else os.path.join(config.DATA_DIR, config.TEST_FILE)
    test_df = pd.read_csv(test_path)
    test_df = test_df[[text_col, label_col]].rename(
        columns={text_col: 'text', label_col: 'label'}
    )
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)

    print(f"\n✅ Test data loaded: {test_df.shape[0]} rows (balanced: 248 per class)")

    return train_df, test_df

# -------------------------------
# 4. TRADITIONAL ML MODELS
# -------------------------------
def train_traditional_models(X_train, y_train, X_test, y_test, label_encoder):
    """Train and evaluate traditional ML models with TF-IDF"""
    print("\n" + "="*60)
    print("STEP 2: TRADITIONAL MACHINE LEARNING MODELS")
    print("="*60)

    # TF-IDF Vectorization
    print("\n🔧 Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        stop_words='english'
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"   TF-IDF shape: {X_train_tfidf.shape}")

    # Apply SMOTE to handle class imbalance
    print("\n⚖️  Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=config.RANDOM_SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
    print(f"   Balanced shape: {X_train_balanced.shape}")

    results = {}
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=config.RANDOM_SEED
        ),
        'Linear SVM': LinearSVC(
            class_weight='balanced', random_state=config.RANDOM_SEED, max_iter=2000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=config.RANDOM_SEED, n_jobs=-1
        )
    }

    for name, model in models.items():
        print(f"\n📈 Training {name}...")
        model.fit(X_train_balanced, y_train_balanced)
        acc, prec, rec, f1, y_pred = evaluate_model(
            model, X_test_tfidf, y_test, model_name=name
        )
        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred,
            'tfidf': tfidf
        }
        plot_confusion_matrix(y_test, y_pred, label_encoder, f"Confusion_Matrix_{name.replace(' ', '_')}")

    return results

# -------------------------------
# 5. BERT DATASET CLASS
# -------------------------------
class MentalHealthDataset(Dataset):
    """PyTorch Dataset for BERT fine-tuning"""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# 6. BERT TRAINING PIPELINE
# -------------------------------
def train_bert_model(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, label_encoder):
    """Complete BERT fine-tuning pipeline"""
    print("\n" + "="*60)
    print("STEP 3: BERT FINE-TUNING PIPELINE")
    print("="*60)

    # Initialize tokenizer
    print(f"\n🔤 Loading BERT tokenizer: {config.BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    # Create datasets
    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
    val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)
    test_dataset = MentalHealthDataset(test_texts, test_labels, tokenizer, config.MAX_LEN)

    # Load pre-trained BERT model for classification
    print(f"🤖 Loading BERT model for {len(config.LABELS)} classes...")
    model = BertForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels=len(config.LABELS),
        problem_type="single_label_classification"
    )
    model.to(config.DEVICE)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.MODEL_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        warmup_steps=500,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=f"{config.MODEL_DIR}logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # Define metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train the model
    print("\n🏋️  Starting BERT training...")
    print(f"   Device: {config.DEVICE}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Max epochs: {config.EPOCHS}")
    print(f"   Max sequence length: {config.MAX_LEN}")

    trainer.train()

    # Evaluate on test set
    print("\n📊 Evaluating BERT on test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=1)

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='weighted'
    )

    print(f"\n{'='*50}")
    print("BERT Final Test Results")
    print(f"{'='*50}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=config.LABELS))

    plot_confusion_matrix(test_labels, test_preds, label_encoder, "Confusion_Matrix_BERT")

    # Save the model
    model.save_pretrained(f"{config.MODEL_DIR}bert_final")
    tokenizer.save_pretrained(f"{config.MODEL_DIR}bert_final")
    print(f"\n💾 BERT model saved to: {config.MODEL_DIR}bert_final")

    return model, tokenizer, test_accuracy

# -------------------------------
# 7. COMPARISON & DEPLOYMENT PREP
# -------------------------------
def compare_and_save_results(ml_results, bert_accuracy):
    """Compare all models and save results summary"""
    print("\n" + "="*60)
    print("STEP 4: MODEL COMPARISON")
    print("="*60)

    comparison = []
    for name, res in ml_results.items():
        comparison.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1']
        })
    comparison.append({
        'Model': 'BERT (Fine-tuned)',
        'Accuracy': bert_accuracy,
        'Precision': None,
        'Recall': None,
        'F1-Score': None
    })

    comparison_df = pd.DataFrame(comparison)
    print("\n📊 Model Performance Comparison:")
    print(comparison_df.to_string(index=False))

    # Find best model
    best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
    print(f"\n🏆 Best Performing Model: {best_model_name}")

    return comparison_df

# -------------------------------
# 8. MAIN EXECUTION PIPELINE
# -------------------------------
def main():
    """Run complete end-to-end pipeline"""
    print("\n" + "="*60)
    print("🧠 MENTAL HEALTH TEXT CLASSIFICATION - CAPSTONE PROJECT")
    print("="*60)

    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    set_seed(config.RANDOM_SEED)

    # Load data
    train_df, test_df = load_and_explore_data()

    # Prepare data for modeling
    X = train_df['cleaned_text'].values
    y = train_df['label'].values

    X_test = test_df['cleaned_text'].values
    y_test = test_df['label'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_test_encoded = label_encoder.transform(y_test)

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=config.VAL_SIZE,
        stratify=y_encoded, random_state=config.RANDOM_SEED
    )

    # Traditional ML Models
    ml_results = train_traditional_models(X_train, y_train, X_test, y_test_encoded, label_encoder)

    # BERT Model (use smaller subset for faster training - remove if you have GPU time)
    print("\n⚠️  Note: BERT training on full dataset may take time on CPU.")
    print("   For faster experimentation, consider using a sample or GPU.")

    # For demonstration, we'll use a stratified sample if needed
    sample_size = min(10000, len(X_train))
    if len(X_train) > sample_size:
        print(f"   Using {sample_size} samples for BERT training (stratified)")
        _, X_train_sample, _, y_train_sample = train_test_split(
            X_train, y_train, train_size=sample_size, stratify=y_train, random_state=config.RANDOM_SEED
        )
    else:
        X_train_sample, y_train_sample = X_train, y_train

    bert_model, bert_tokenizer, bert_accuracy = train_bert_model(
        X_train_sample, y_train_sample, X_val, y_val, X_test, y_test_encoded, label_encoder
    )

    # Compare results
    comparison_df = compare_and_save_results(ml_results, bert_accuracy)

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Results saved in: {config.MODEL_DIR}")
    print("   - Class distribution plot")
    print("   - Confusion matrices for all models")
    print("   - BERT model saved for deployment")

    return ml_results, bert_model, bert_tokenizer, label_encoder

if __name__ == "__main__":
    ml_results, bert_model, bert_tokenizer, label_encoder = main()