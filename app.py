import joblib
import re
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/fake_news_vectorizer.pkl')

# Clean text function (same as in train.py)
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    
    stop_words_set = set(stopwords.words('english'))  # Using stopwords set for faster lookup
    words = [word for word in words if word not in stop_words_set]  # Remove stopwords
    
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the form
    text = request.form['text']
    
    # Clean and preprocess the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text using the loaded vectorizer
    text_vector = vectorizer.transform([cleaned_text])
    
    # Predict using the model
    prediction = model.predict(text_vector)
    
    # Map the prediction result (0 or 1) to a human-readable string
    prediction_text = 'Real News' if prediction[0] == 1 else 'Fake News'
    
    # Render the result in the HTML template
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
