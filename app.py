# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Download stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Configure Streamlit page settings
st.set_page_config(page_title="Sentiment Analyzer App", layout="wide")
st.title("🧠 Sentiment Analyzer App")

# File uploader widget for CSV input
uploaded_file = st.file_uploader("Upload your CSV file with reviews", type=["csv"])

# Function to clean and preprocess review text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()  # Lowercase text
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letter characters
    text = text.split()  # Tokenize
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return ' '.join(text)

# Function to map numeric star ratings to sentiment labels
def map_sentiment(score):
    if score <= 2.5:
        return 'negative'
    elif score <= 3.5:
        return 'neutral'
    else:
        return 'positive'

# Load pretrained BERT model and tokenizer
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

# Predict sentiment using BERT model
def predict_bert(texts, tokenizer, model):
    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item() + 1  # Class index shifted from 0–4 to 1–5
        # Map prediction to sentiment
        if pred <= 2:
            sentiments.append("negative")
        elif pred == 3:
            sentiments.append("neutral")
        else:
            sentiments.append("positive")
    return sentiments

# If a file has been uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Check for required 'reviewText' column
    if 'reviewText' not in df.columns:
        st.error("Your CSV must have a 'reviewText' column.")
    else:
        # Clean the review text
        df['cleaned'] = df['reviewText'].apply(clean_text)

        # Map numeric ratings to sentiment if 'overall' column is present
        if 'overall' in df.columns:
            df['true_sentiment'] = df['overall'].apply(map_sentiment)

        # Let the user choose between models
        model_choice = st.selectbox("Choose a model:", ["TF-IDF + LogisticRegression", "BERT (Transformers)"])

        # Use traditional ML model
        if model_choice == "TF-IDF + LogisticRegression":
            model = pickle.load(open("sentiment_model.pkl", 'rb'))
            vectorizer = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))
            X = vectorizer.transform(df['cleaned'])
            df['sentiment'] = model.predict(X)

        # Use BERT-based deep learning model
        elif model_choice == "BERT (Transformers)":
            with st.spinner("Loading BERT model and analyzing text. Please wait..."):
                tokenizer, bert_model = load_bert()
                df['sentiment'] = predict_bert(df['reviewText'].astype(str).tolist(), tokenizer, bert_model)

        # 📊 Display sentiment prediction results
        st.subheader("📊 Sentiment Analysis Results")
        st.write(df[['reviewText', 'sentiment']].head())

        # 🔍 Sentiment Distribution (Smaller Plot)
        st.subheader("🔍 Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller size
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2', ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # ☁️ WordCloud - Positive Reviews (Smaller)
        st.subheader("☁️ WordCloud - Positive Reviews")
        pos_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned'])
        wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(pos_text)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # ☁️ WordCloud - Negative Reviews (Smaller)
        st.subheader("☁️ WordCloud - Negative Reviews")
        neg_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned'])
        wordcloud = WordCloud(width=600, height=400, background_color='black', colormap='Reds').generate(neg_text)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # ⚠️ Show misleading 5-star reviews
        st.subheader("⚠️ Misleading 5-Star Reviews")
        if 'overall' in df.columns:
            misleading = df[(df['overall'] >= 5) & (df['sentiment'] != 'positive')]
            st.write(misleading[['reviewText', 'overall', 'sentiment']])

        # 🕵️ Filter Reviews by Sentiment
        st.subheader("🕵️ Filter Reviews by Sentiment")
        option = st.radio("Select sentiment to view reviews:", ['positive', 'neutral', 'negative'])
        filtered = df[df['sentiment'] == option]
        st.write(filtered[['reviewText', 'sentiment']])

        # 📅 Time-based analysis (Smaller plots)
        if 'reviewTime' in df.columns:
            st.subheader("📅 Star Rating and Sentiment Trend Over Time")
            df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')
            df = df.dropna(subset=['reviewTime'])

            time_freq = st.selectbox("Choose time granularity:", ['Month', 'Week', 'Day'])

            if time_freq == 'Month':
                df['time_period'] = df['reviewTime'].dt.to_period('M').dt.to_timestamp()
            elif time_freq == 'Week':
                df['time_period'] = df['reviewTime'].dt.to_period('W').dt.to_timestamp()
            else:
                df['time_period'] = df['reviewTime'].dt.date

            # 📈 Smaller Line Chart - Average Rating Over Time
            avg_rating = df.groupby('time_period')['overall'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.lineplot(data=avg_rating, x='time_period', y='overall', marker="o", ax=ax)
            ax.set_title("📈 Average Star Rating Over Time")
            ax.set_ylabel("Average Rating")
            ax.set_xlabel("Time")
            st.pyplot(fig)

            # 💬 Smaller Line Chart - Sentiment Trend Over Time
            sentiment_trend = df.groupby(['time_period', 'sentiment']).size().unstack(fill_value=0).reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            sentiment_trend.set_index('time_period').plot(kind='line', ax=ax, marker='o')
            ax.set_title("💬 Sentiment Count Over Time")
            ax.set_ylabel("Count")
            ax.set_xlabel("Time")
            st.pyplot(fig)

        # 📁 CSV Download Button
        st.subheader("📅 Download Annotated Data")
        st.download_button("Download CSV", df.to_csv(index=False), "annotated_reviews.csv", "text/csv")
