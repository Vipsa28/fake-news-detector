# Fake News Detection using Machine Learning

## Overview

This project utilizes **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to classify news articles as either **Real News** or **Fake News**. The goal of the project is to build a model that can detect fake news based on text input and deploy it as a simple web application using **Flask**.

## Tech Stack

- **Python**: Programming language for data preprocessing and machine learning.
- **Scikit-learn**: For building the machine learning model.
- **NLTK (Natural Language Toolkit)**: For text preprocessing (stopword removal, text cleaning).
- **Flask**: To build a simple web application to deploy the model.
- **HTML/CSS**: For creating the user interface of the web app.

## Dataset

The dataset used in this project is the [Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle. This dataset contains labeled news articles (Real or Fake) used to train the model.

### Dataset Columns:

- `text`: The content of the news article.
- `label`: The label of the article (0 = Fake, 1 = Real).

## Getting Started

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd fake-news-detector
   ```

2. Set up the virtual environment:

   ```bash
   python -m venv venv
   ```

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:

   ```bash
   python train.py
   ```

5. Run the Flask web application:
   ```bash
   python app.py
   ```

The app will be available at `http://127.0.0.1:5000/`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
