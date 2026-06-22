# 📚 Bibliophile Pro: The Next-Gen Book Discovery Platform

Bibliophile Pro is a state-of-the-art book recommendation system that leverages advanced Machine Learning, a Hybrid Recommendation Engine, and modern UI aesthetics to provide a premium reading discovery experience.

## ✨ Core Features

*   **Hybrid Intelligence**: Combines **Content-Based Filtering** (TF-IDF) and **Collaborative Filtering** (KNN) to provide hyper-personalized suggestions.
*   **Persistent User Ecosystem**: Secure login/signup system with SQLite integration to save reading history and favorite books.
*   **Pro Analytics**: Interactive data visualization dashboard powered by Plotly for exploring library trends and reader behavior.
*   **AI-Powered Narratives**: Dynamic summaries generated for over 10,000 titles to help you decide your next read instantly.
*   **Ultra-Modern UI**: Glassmorphism-inspired dark mode interface with smooth transitions and high-performance layout.
*   **Smart Search & Discovery**: Real-time search with similarity scoring across metadata.

## 🛠️ Technology Stack

*   **Frontend**: Streamlit (Advanced UI/UX Customization)
*   **Intelligence**: Scikit-Learn (TF-IDF, Nearest Neighbors), SciPy (Sparse Matrices)
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly Express
*   **Persistence**: SQLite3 (Secure relational storage)
*   **Multimedia**: SpeechRecognition (Voice Search), Pillow (Image processing)

## 🚀 Deployment Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Setup Environment
Install all professional dependencies:
```bash
pip install -r requirements.txt
```

### 3. Initialize AI Pipeline
Generate the ML assets and populate the database (automatic on first run, or manual via):
```bash
python model_gen.py
```

### 4. Launch Application
```bash
streamlit run app.py
```

---
*Developed as a premier showcase of Machine Learning and Modern Web Design.*
