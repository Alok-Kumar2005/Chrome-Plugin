from flask import Flask, request, jsonify, render_template
import mlflow
import pickle
import os
import pandas as pd
import re
import string
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask_cors import CORS
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import numpy as np
from datetime import datetime, timedelta
import random
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# -----------------------------
# Prometheus Metrics Setup
# -----------------------------
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)

# -----------------------------
# TEXT PREPROCESSING FUNCTIONS
# -----------------------------
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# -----------------------------
# YOUTUBE COMMENT FETCH FUNCTIONS
# -----------------------------
def extract_video_id(url):
    parsed = urlparse(url)
    if "youtube" in parsed.netloc:
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]
    elif "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None

def fetch_youtube_comments(video_url, api_key):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL: could not extract video ID")
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        ).execute()
        for item in response.get("items", []):
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

# -----------------------------
# MLflow Model and Vectorizer Setup
# -----------------------------
mlflow.set_tracking_uri("https://dagshub.com/vishwajitsarakpatil/Chrome-Plugin.mlflow")
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open(r'C:\Users\vishw\OneDrive\Desktop\project2\Chrome-Plugin-2\models\vectorizer.pkl', 'rb'))

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    REQUEST_COUNT.labels(method="POST", endpoint="/analyze").inc()
    start_time = time.time()
    
    data = request.get_json()
    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "No video_url provided"}), 400
    
    # Replace with your actual YouTube Data API key
    api_key = "AIzaSyDOhIz4rbNMM7VtgsxS0XJWPpVd0XLnFkM"
    
    try:
        # 1) Fetch comments
        comments = fetch_youtube_comments(video_url, api_key)
        if not comments:
            return jsonify({"error": "No comments found"}), 404
        
        # 2) Preprocess comments
        processed_comments = [normalize_text(c) for c in comments]
        
        # 3) Vectorize comments
        X = vectorizer.transform(processed_comments)
        
        # 4) Make predictions
        predictions = model.predict(X)
        predictions = predictions.tolist()
        
        # Calculate analytics
        total_comments = len(comments)
        unique_comments = len(set(comments))
        positive_count = predictions.count(1)
        negative_count = predictions.count(0)
        # For average rating, assume positive=5 and negative=1
        average_rating = (positive_count * 5 + negative_count * 1) / total_comments
        
        # Dummy time series forecast for next 7 days
        forecast_dates = []
        forecast_values = []
        for i in range(1, 8):
            date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            # Simulate forecast comment counts between min and max of current counts
            forecast_value = random.randint(min(positive_count, negative_count), max(positive_count, negative_count) + 10)
            forecast_dates.append(date)
            forecast_values.append(forecast_value)
        
        response_data = {
            "video_url": video_url,
            "total_comments": total_comments,
            "unique_comments": unique_comments,
            "positive_comments": positive_count,
            "negative_comments": negative_count,
            "average_rating": average_rating,
            "time_series_forecast": {
                "dates": forecast_dates,
                "values": forecast_values
            }
        }
        
        REQUEST_LATENCY.labels(endpoint="/analyze").observe(time.time() - start_time)
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
