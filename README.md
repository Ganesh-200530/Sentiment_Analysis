# 🔍 Sentiment Analysis Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)

A complete sentiment analysis system that classifies text into **Positive** or **Negative** categories using NLP and Machine Learning. Features text preprocessing with lemmatization, TF-IDF feature extraction with bigrams, a Logistic Regression classifier, and a **Streamlit web app** for interactive predictions.

---

## 📌 Features
- ✅ Text preprocessing (URL removal, stopwords, lemmatization)
- ✅ TF-IDF Vectorization (unigrams + bigrams, 10K features)
- ✅ Logistic Regression classifier (~77% accuracy)
- ✅ Confusion Matrix visualization
- ✅ CLI prediction tool
- ✅ Streamlit Web App (single + batch analysis)
- ✅ Model serialization with Joblib

---

## 🛠️ Tech Stack
| Technology | Purpose |
|---|---|
| Python | Core language |
| NLTK | Stopwords, Lemmatization |
| scikit-learn | TF-IDF, Logistic Regression, Metrics |
| Matplotlib | Confusion Matrix plot |
| Streamlit | Interactive Web App |
| Joblib | Model save/load |

---

## 📊 Model Performance
| Metric | Negative | Positive |
|---|---|---|
| Precision | 0.77 | 0.77 |
| Recall | 0.75 | 0.78 |
| F1-Score | 0.76 | 0.78 |
| **Overall Accuracy** | | **76.95%** |

- **Algorithm:** Logistic Regression
- **Features:** TF-IDF (10,000 features, unigrams + bigrams)
- **Dataset:** Sentiment140 (100K sample from 1.6M tweets)

---

## � Dataset
The dataset is too large for GitHub (~240MB). Download it from Kaggle and place it in the `data/` folder:

🔗 **Download:** [Sentiment140 - Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

After downloading, rename the file and place it as:
```
data/sentiment.csv
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ganesh-200530/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### 2. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140), extract, and place the CSV file as `data/sentiment.csv`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python src/train.py
```
> Uses 100,000 tweets by default. Set `sample_size=None` in `src/train.py` to train on the full 1.6M dataset.

### 4. Run Predictions (CLI)
```bash
python src/predict.py
```

### 5. Run Web App (Streamlit)
```bash
python -m streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 📁 Project Structure
```
Sentiment-Analysis-ML/
│
├── data/
│   └── sentiment.csv            # Dataset (Sentiment140)
│
├── src/
│   ├── preprocess.py            # Text cleaning, lemmatization, data loading
│   ├── train.py                 # Model training, evaluation, confusion matrix
│   ├── predict.py               # CLI prediction script
│
├── models/
│   ├── sentiment_model.pkl      # Trained model (generated after training)
│   └── confusion_matrix.png     # Performance visualization
│
├── app.py                       # Streamlit web application
├── main.py                      # Simple project runner
├── report.pdf                   # Full project report (PDF)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 📝 Full Report
A detailed project report is available at [`report.pdf`](report.pdf) covering:
- Dataset description & structure
- Preprocessing pipeline
- TF-IDF math explanation
- Logistic Regression theory
- Complete results & confusion matrix
- Future scope

---

## 🔮 Future Scope
- Train on full 1.6M dataset
- Try SVM / Random Forest classifiers
- Use Word2Vec / GloVe embeddings
- Implement LSTM / BERT deep learning models
- Deploy to Streamlit Cloud
- Add real-time Twitter API integration

---

## 📜 License
This project is for educational purposes.
