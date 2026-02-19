import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from preprocess import load_data, clean_text

def train_model():
    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'sentiment.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl')
    PLOT_PATH = os.path.join(BASE_DIR, 'models', 'confusion_matrix.png')
    
    # 1. Load Data
    print("Loading data...")
    # Using 100,000 rows for a good balance of speed and accuracy.
    # Set sample_size=None to train on full dataset (~1.6M rows, takes longer).
    df = load_data(DATA_PATH, sample_size=100000)
    
    if df is None:
        return

    # 2. Preprocess
    print("Pre-processing text (with lemmatization)...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Check for empty text after cleaning
    df = df[df['clean_text'].str.strip() != ""]
    
    X = df['clean_text']
    y = df['target']

    # 3. Split Data
    print("Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build Pipeline
    # TF-IDF with bigrams + Logistic Regression (better accuracy than Naive Bayes)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 5. Train Model
    print("Training Logistic Regression model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # 7. Save Confusion Matrix Plot
    print(f"Saving confusion matrix to {PLOT_PATH}...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print("Confusion matrix saved!")

    # 8. Save Model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
