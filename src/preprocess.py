import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not present)
for resource in ['corpora/stopwords', 'corpora/wordnet']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[1], quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans text by removing URLs, special characters, stopwords,
    and applies lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize, remove stopwords, and lemmatize
    text_tokens = text.split()
    filtered_text = [
        LEMMATIZER.lemmatize(word)
        for word in text_tokens
        if word not in STOP_WORDS and len(word) > 1
    ]
    
    return " ".join(filtered_text)

def load_data(file_path, sample_size=None):
    """
    Loads data from CSV file. Assumes Sentiment140 format (no header).
    Columns: 0=target, 1=id, 2=date, 3=flag, 4=user, 5=text
    """
    try:
        # Columns based on Sentiment140 dataset structure
        columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        
        # Read CSV with latin-1 encoding as it's common for this dataset
        df = pd.read_csv(file_path, encoding='latin-1', header=None, names=columns)
        

        # Keep only target and text
        df = df[['target', 'text']].copy()
        
        # Map target: 0=Negative, 2=Neutral, 4=Positive
        # We'll map to just Negative(0) and Positive(1) for binary classification
        # Or keep as is. Let's map for clarity. 4 -> 1 (Positive)
        df['target'] = df['target'].replace({4: 1})
        
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
            
        print(f"Data loaded: {df.shape}")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
