import sys
import os
import streamlit as st
import joblib

# Add src folder to path so we can import preprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import clean_text

@st.cache_resource
def load_model():
    """Load the trained model (cached so it only loads once)."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def predict(model, text):
    """Clean text and predict sentiment."""
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "Unknown", 0.0
    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0]
    confidence = max(probability) * 100
    label = "Positive 😊" if prediction == 1 else "Negative 😞"
    return label, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="Sentiment Analysis", page_icon="🔍", layout="centered")

st.title("🔍 Sentiment Analysis")
st.markdown("Analyze the sentiment of any text using a trained ML model.")

model = load_model()

if model is None:
    st.error("Model not found! Please run `python src/train.py` first to train the model.")
else:
    # Single text input
    st.subheader("Enter text to analyze")
    user_input = st.text_area("Type or paste your text here:", height=120, placeholder="e.g., I love this product! It works great.")

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            label, confidence = predict(model, user_input)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", label)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
        else:
            st.warning("Please enter some text first.")

    # Divider
    st.divider()

    # Batch analysis
    st.subheader("Batch Analysis")
    st.markdown("Enter multiple sentences (one per line) to analyze them all at once.")
    batch_input = st.text_area("Batch input:", height=150, placeholder="I am so happy today\nThis is terrible\nWhat a wonderful experience")

    if st.button("Analyze Batch"):
        if batch_input.strip():
            lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
            results = []
            for line in lines:
                label, confidence = predict(model, line)
                results.append({"Text": line, "Sentiment": label, "Confidence": f"{confidence:.1f}%"})
            st.table(results)
        else:
            st.warning("Please enter some text first.")

    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **Sentiment Analysis ML**
    
    - **Model:** Logistic Regression
    - **Features:** TF-IDF (unigrams + bigrams)
    - **Preprocessing:** Lemmatization, stopword removal
    - **Dataset:** Sentiment140 (Twitter)
    """)

    # Show confusion matrix if available
    cm_path = os.path.join(os.path.dirname(__file__), 'models', 'confusion_matrix.png')
    if os.path.exists(cm_path):
        st.sidebar.header("Model Performance")
        st.sidebar.image(cm_path, caption="Confusion Matrix")
