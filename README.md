# YouTube Sentiment Analysis Chrome Extension
A full-stack machine learning project that analyzes **YouTube video comments in real time**, providing sentiment insights through a **Chrome extension interface**.


## 📌 Project Overview

This project combines **Machine Learning**, **MLOps**, and **Browser Extension Development** to deliver sentiment analysis for YouTube comments.

With a single click, users can analyze any YouTube video's comment section and instantly view:
- Sentiment distribution
- Average ratings
- Forecast trends

---

## 🚀 Features

- **Real-time Comment Analysis**  
  Fetch and analyze YouTube comments on demand

- **Sentiment Classification**  
  Binary classification (Positive / Negative) using Logistic Regression

- **Visual Analytics**  
  Interactive charts for sentiment distribution and forecasts

- **Chrome Extension**  
  Seamless integration with the YouTube browsing experience

- **MLOps Pipeline**  
  End-to-end CI/CD using DVC and MLflow

- **Performance Monitoring**  
  Prometheus metrics for prediction counts and latency tracking

---

## 🛠️ Tech Stack

### Machine Learning & NLP
- **ML Framework**: scikit-learn
- **NLP**: NLTK
- **Feature Engineering**: Bag of Words (CountVectorizer)
- **Model**: Logistic Regression (L1 Regularization)

### MLOps & Experimentation
- **Experiment Tracking**: MLflow
- **Model Registry**: MLflow Model Registry
- **Data Versioning**: DVC
- **Remote Storage**: DagsHub

### Backend & Deployment
- **Web Framework**: Flask
- **API Design**: RESTful APIs
- **Monitoring**: Prometheus Client
- **External API**: YouTube Data API v3

### Frontend
- **Browser Extension**: Chrome Manifest V3
- **Charts**: Chart.js
- **Styling**: Vanilla CSS

---



## 📂 Project Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Clone Project
```
https://github.com/Alok-Kumar2005/Chrome-Plugin.git
```

## Create Environment and Install Dependencies
```
uv init
uv venv
.venv\Scripts\activate
uv add -r requirements.txt
```

## Running the Flask Apps
```
cd flask_app
python app.py
```
