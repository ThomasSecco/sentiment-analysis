# src/feature_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch

def tfidf_features(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

def word2vec_features(X_train, X_test):
    model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1)
    def get_avg_vector(sentence):
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        return sum(vectors) / len(vectors) if vectors else np.zeros(100)
    X_train_w2v = [get_avg_vector(sentence.split()) for sentence in X_train]
    X_test_w2v = [get_avg_vector(sentence.split()) for sentence in X_test]
    return X_train_w2v, X_test_w2v

def bert_features(X_train, X_test):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.pooler_output.detach().numpy()
    X_train_bert = [get_bert_embedding(text) for text in X_train]
    X_test_bert = [get_bert_embedding(text) for text in X_test]
    return X_train_bert, X_test_bert
