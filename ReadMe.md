# 🎬 Sentiment Analysis on IMDb Movie Reviews 🎥

Welcome to the **Sentiment Analysis on IMDb Movie Reviews** project! This project leverages various NLP techniques to determine whether movie reviews are positive or negative. Dive in to explore how we use traditional machine learning models, deep learning models, and transformer-based models like BERT to tackle this problem!

---

## 📌 Project Overview

Sentiment analysis is a key task in NLP that helps determine the emotional tone behind a series of words. Here, we apply sentiment analysis to IMDb movie reviews to classify them as either positive 😊 or negative 😞. We experiment with several models to understand which ones perform best for this dataset.


---

## 💻 Getting Started

### 🛠️ Prerequisites

To get started, clone this repository and install the dependencies listed in `requirements.txt`.

### 📊 Dataset

We’re using the [IMDb Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/) 📁, which contains 50,000 movie reviews labeled as **positive** or **negative**. 

1. **Download** the dataset and place it in `data/raw/imdb_reviews.csv`.
2. **Run** the preprocessing script to prepare the data for modeling.

---

## 🚀 Key Components

### 1. **Data Preprocessing** 🧹

The `data_preprocessing.py` script:
- Cleans text data (lowercasing, removing stop words, etc.).
- Splits the dataset into training and testing sets.
  
### 2. **Feature Extraction** 🧬

In `feature_extraction.py`, we explore different feature extraction techniques:
- **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **Word2Vec** embeddings
- **BERT** embeddings using the Transformers library

### 3. **Model Training** 🧠

The `model_training.py` script trains multiple models, including:
- **Logistic Regression** & **Support Vector Machine (SVM)**
- **Random Forest**
- **LSTM** (Long Short-Term Memory)
- **BERT-based classifier**

### 4. **Evaluation** 📈

Using `evaluation.py`, we evaluate each model using metrics such as:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

We also generate **confusion matrices** for each model, which are visualized in the Model Comparison notebook 📊.


## 📘 Notebooks

- **[EDA.ipynb](notebooks/EDA.ipynb)**: Exploratory Data Analysis to understand the dataset and review text statistics.
- **[Model_Comparison.ipynb](notebooks/Model_Comparison.ipynb)**: Performance comparison of each model with metrics and confusion matrix visualizations.

---

## 📈 Future Work

Here’s what could be added or improved:
- 🧩 **Hyperparameter Tuning** for optimized model performance.
- 🚀 **Data Augmentation** to increase the training data size.
- 🤖 **Additional Transformer Models** like RoBERTa or GPT-3 for comparison.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests if you have ideas to improve this project.

Happy coding! 💻😊
