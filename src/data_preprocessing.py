# src/data_preprocessing.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords if not already present
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def clean_text(text):
    # Lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_data(data, text_column):
    data[text_column] = data[text_column].apply(clean_text)
    return data

def split_data(data, text_column, target_column):
    X = data[text_column]
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage
if __name__ == "__main__":
    data = load_data("data/raw/imdb_reviews.csv")
    data = preprocess_data(data, 'review')
    X_train, X_test, y_train, y_test = split_data(data, 'review', 'sentiment')
    # Save the processed data
    X_train.to_csv("data/processed/X_train.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
