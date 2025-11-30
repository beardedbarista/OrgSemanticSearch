# FastAPI Hybrid Search Service

This project provides a **FastAPI-based hybrid search API** leveraging both **keyword search** and **semantic search** using `SentenceTransformer`.  
A Dockerfile is included so the service can be easily containerized.

---

## 🚀 Features
- FastAPI web service with:
  - Semantic vector search (SentenceTransformer)
  - Keyword-ranking hybrid search
  - Caching for fast startup
- Docker-ready deployment
- Uses `organizations-100.csv` sample dataset for demonstration

---

## 📦 Project Structure
```
/app
│── main.py
│── organizations-100.csv
requirements.txt
Dockerfile
README.md
```

---

## 🔧 Requirements

The `requirements.txt` should include:

```
fastapi
uvicorn
pandas
numpy
sentence-transformers
```

---

## 🐳 Docker Instructions

### 1. Build the Docker Image
```bash
docker build -t fastapi-hybrid-search .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 fastapi-hybrid-search
```

### 3. Access the API
Open the browser at:

```
http://localhost:8000/docs
```

---

## 🧱 Dockerfile (included)

```dockerfile
# Use slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /myenv/FastAPI_TEST/app

# Install system dependencies for Kaggle
RUN apt-get update && apt-get install -y     gcc     g++     && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY app ./app

# Expose FastAPI port

---

## ▶️ Running Locally (without Docker)

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Start the API:
```bash
uvicorn app.main:app --reload
```

---

## 📚 API Example

GET /org-semantic-search

Example:/org-semantic-search?query= companies in Turkey

Process:

Encode query with SentenceTransformer

Compare with precomputed embeddings

Return ranked results

Include cosine similarity score

##📝 Logging

All requests + performance metrics logged to:
app.log

## ✨ License
MIT License (or specify your preferred license)