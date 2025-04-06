import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import nltk
import joblib

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('data/fake_and_real_news.csv')

# Optional: Reduce dataset size for testing (remove for final training)
df = df.sample(frac=0.1, random_state=42)  # Use only 10% of the data for testing

# Check the dataset shape (optional)
print(f'Dataset shape: {df.shape}')

# Clean text function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    
    # Convert stopwords to set once to speed up lookup
    stop_words_set = set(stopwords.words('english'))  
    words = [word for word in words if word not in stop_words_set]  # Remove stopwords
    
    return ' '.join(words)

# Apply cleaning function to the 'text' column
df['text'] = df['text'].astype(str).apply(clean_text)

# Split the dataset into training and testing sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data (use CountVectorizer for faster training)
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print("Classification Report on test data:")
print(classification_report(y_test, y_pred))

# Optional: Check the accuracy on training data to detect overfitting
print("Accuracy on training data:", model.score(X_train, y_train))

# Save the model and vectorizer
joblib.dump(model, 'models/fake_news_model.pkl')
joblib.dump(vectorizer, 'models/fake_news_vectorizer.pkl')
