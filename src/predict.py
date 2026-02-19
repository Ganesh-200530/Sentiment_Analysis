import os
import joblib
from preprocess import clean_text

def load_model():
    """
    Loads the trained model from disk.
    """
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'sentiment_model.pkl')
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
        
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_sentiment(model, text):
    """
    Predicts sentiment for a given text using the loaded model.
    """
    # Preprocess text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return "Could not determine sentiment (empty after cleaning)"
        
    # Predict
    prediction = model.predict([cleaned_text])[0]
    
    # Map back to readable label
    sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral", 4: "Positive"}
    return sentiment_map.get(prediction, "Unknown")
    
if __name__ == "__main__":
    print("\n--- Sentiment Analysis CLI ---")
    model = load_model()
    
    if model:
        print("Enter text to analyze (type 'exit' to quit):")
        while True:
            text = input("\nInput: ")
            if text.lower() == 'exit':
                break
            
            result = predict_sentiment(model, text)
            print(f"Sentiment: {result}")
