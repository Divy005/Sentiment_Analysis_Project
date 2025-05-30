# 🧠 Sentiment Analysis with TF-IDF and Logistic Regression
## 📌 Project Overview

This project is a Natural Language Processing (NLP) pipeline designed to classify product or service reviews into three sentiment categories: **positive**, **neutral**, and **negative**. It leverages the power of **TF-IDF vectorization** to convert raw text into numerical features and applies **Logistic Regression** for multi-class classification.

The primary goal is to automatically interpret customer feedback and derive sentiment insights from review texts, making it easier for businesses to monitor user satisfaction, detect issues early, and understand customer trends at scale.

## 📚 Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Output](#output)

## ✨ Features

- 🧹 **Text Cleaning & Preprocessing**  
  Automatically removes HTML tags, punctuation, numbers, and common stopwords from raw review text, ensuring clean and consistent input for the model.

- 🏷️ **Sentiment Labeling Based on Ratings**  
  Converts numerical review ratings into sentiment categories — `positive`, `neutral`, or `negative` — using a defined thresholding strategy.

- 🧠 **TF-IDF Vectorization**  
  Transforms cleaned text into numerical feature vectors using Term Frequency–Inverse Document Frequency (TF-IDF), capturing the importance of words in the dataset.

- 📊 **Multiclass Logistic Regression Model**  
  Trains a multinomial Logistic Regression classifier to predict sentiment classes based on textual features.

- 💾 **Model and Vectorizer Persistence**  
  Saves both the trained model and TF-IDF vectorizer using `pickle`, enabling easy reuse or deployment without retraining.

- ✅ **High Accuracy Performance**  
  Demonstrated high classification accuracy (e.g., `0.93`) on test data, showing the effectiveness of the pipeline.


## 🛠 Installation

To get started, make sure you have **Python 3.7+** installed, then install the required Python libraries using `pip`:

```bash
pip install pandas scikit-learn nltk matplotlib seaborn

## 🚀 Usage

Follow these steps to run the sentiment analysis pipeline on your review dataset:

### 1. Prepare Your Dataset

Make sure your dataset (e.g., 'reviews.csv') contains at least the following two columns:

- 'reviewText': The actual text of the customer review  
- 'overall': The numerical rating (e.g., from 1.0 to 5.0)

Example:

| 'reviewText'                | 'overall' |
|----------------------------|-----------|
| 'Fantastic product!'        | 5.0       |
| 'Not what I expected.'      | 2.0       |

---

### 2. Run the Script

If your main Python script is named 'sentiment_analysis.py', run it using:

'''bash
python 'sentiment_analysis.py'


## 📁 Project Structure

├── app.py # Python script that handles text cleaning, sentiment labeling, TF-IDF vectorization, model training, evaluation, and saving
├── sentiment_model.pkl # Serialized Logistic Regression model trained to classify reviews as positive, neutral, or negative
├── tfidf_vectorizer.pkl # Serialized TF-IDF vectorizer fitted on the cleaned review texts for feature extraction
├── amazone_reviews.csv # Input dataset containing review texts and corresponding ratings

✅ Accuracy: 0.93

## 📬 Contact

If you have any questions, suggestions, or run into issues, feel free to:

- Open an [issue](https://github.com/your-username/your-repo/issues) here on GitHub  
- Submit a pull request to contribute improvements or fixes  
- Reach out via email: divydobariya11@gmail.com  

Your feedback and contributions are always welcome!
