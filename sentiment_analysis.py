# sentiment_analysis.py

# Import required libraries
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import kagglehub
import os

# Download the dataset using kagglehub
path = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")

# Print path to verify
print("Dataset downloaded to:", path)

# Find the target CSV file
csv_filename = "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
csv_path = os.path.join(path, csv_filename)

# Load dataset
df = pd.read_csv(csv_path)

# Drop rows where the review is missing
df = df.dropna(subset=['reviews.text'])

# Extract the reviews column
reviews_data = df['reviews.text']

# Load spaCy model and add TextBlob sentiment pipeline
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("spacytextblob")

# Preprocessing function to clean the review text
def preprocess_text(text):
    doc = nlp(text.lower().strip())  # Normalize casing and whitespace
    tokens = [tok.text for tok in doc if tok.is_alpha and not tok.is_stop]
    return " ".join(tokens)

# Sentiment analysis function
def analyze_sentiment(review):
    cleaned = preprocess_text(review)
    doc = nlp(cleaned)
    polarity = doc._.blob.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

# Test on a few sample reviews
print("\nSample sentiment predictions:\n")
for i, review in enumerate(reviews_data.sample(5, random_state=42), 1):
    sentiment, polarity = analyze_sentiment(review)
    print(f"{i}. {sentiment} (Polarity: {polarity:.2f})\n   Review: {review[:100]}...\n")

# Optional: Compare similarity between two sample reviews
r1 = preprocess_text(reviews_data.iloc[0])
r2 = preprocess_text(reviews_data.iloc[1])
doc1 = nlp(r1)
doc2 = nlp(r2)
print(f"Similarity between two sample reviews: {doc1.similarity(doc2):.2f}")
