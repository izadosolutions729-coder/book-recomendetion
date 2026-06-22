"""
ML Pipeline for Book Recommendation System
Handles data ingestion, preprocessing, and model generation.
"""
import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def load_raw_data():
    """Download Goodbooks-10k datasets from standard repository."""
    urls = {
        "books": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
        "ratings": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
        "tags": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv",
        "book_tags": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv"
    }
    return [pd.read_csv(urls[k]) for k in ["books", "ratings", "tags", "book_tags"]]

def process_pipeline(books, ratings, tags, book_tags):
    """Execute preprocessing and model training."""
    
    # 1. Book Metadata Cleaning
    books = books.loc[:, ~books.columns.duplicated()]
    if 'title' in books.columns and 'original_title' in books.columns:
        books.drop(columns=['title'], inplace=True)
        books.rename(columns={'original_title': 'title'}, inplace=True)
    
    # 2. Genre Extraction
    merged_tags = pd.merge(book_tags, tags, on='tag_id')
    common_genres = ['fiction', 'mystery', 'romance', 'science-fiction', 'fantasy', 'biography', 'history', 'horror', 'thriller', 'young-adult']
    merged_tags['tag_name'] = merged_tags['tag_name'].str.lower()
    genre_tags = merged_tags[merged_tags['tag_name'].isin(common_genres)]
    book_genres = genre_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ', '.join(list(set(x)))).reset_index()
    
    # Remove existing tag_name if present to avoid merge duplicates
    if 'tag_name' in books.columns:
        books.drop(columns=['tag_name'], inplace=True)

    books = pd.merge(books, book_genres, on='goodreads_book_id', how='left')
    books['tag_name'] = books['tag_name'].fillna('General')
    books = books.drop_duplicates(subset=['book_id']).reset_index(drop=True)
    
    # 3. Content-Based Model (TF-IDF)
    titles, authors, genres = [books[c].fillna('').values for c in ['title', 'authors', 'tag_name']]
    books['content'] = [f"{t} {a} {g}" for t, a, g in zip(titles, authors, genres)]
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books['content'])
    
    # Efficient similarity: Only store Top 100 neighbors to save massive space
    raw_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
    
    # Create a reduced similarity matrix (indices and scores for top 100)
    # This reduces a 10000x10000 matrix (400MB) to a fraction of that.
    top_n = 100
    reduced_sim = []
    for row in raw_sim:
        # Get indices of top N similarities
        idx = np.argpartition(row, -top_n)[-top_n:]
        # Sort them by score
        idx = idx[np.argsort(row[idx])[::-1]]
        reduced_sim.append(list(zip(idx, row[idx])))
    
    # 5. Global Popularity (Bayesian Weighted Rating)
    C, m = books['average_rating'].mean(), books['ratings_count'].quantile(0.9)
    v, R = books['ratings_count'].values, books['average_rating'].values
    books['weighted_score'] = ((v/(v+m) * R) + (m/(m+v) * C))
    
    # We drop 'pivot' and 'knn_model' as they aren't used in app.py
    return books, reduced_sim, books.sort_values('weighted_score', ascending=False)

def initialize_persistence():
    """Setup SQLite environment."""
    conn = sqlite3.connect('users.db')
    queries = [
        'CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)',
        'CREATE TABLE IF NOT EXISTS favorites (username TEXT, book_id INTEGER)',
        'CREATE TABLE IF NOT EXISTS history (username TEXT, book_id INTEGER, timestamp TEXT)'
    ]
    for q in queries: conn.execute(q)
    conn.close()

def main():
    if not os.path.exists('data'): os.makedirs('data')
    
    # Load and process
    urls = {
        "books": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
        "ratings": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
        "tags": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv",
        "book_tags": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv"
    }
    raw_data = [pd.read_csv(urls[k]) for k in ["books", "ratings", "tags", "book_tags"]]
    
    books, reduced_sim, popular = process_pipeline(*raw_data)
    initialize_persistence()
    
    with open('data/processed_model.pkl', 'wb') as f:
        pickle.dump({
            'books': books, 
            'content_sim': reduced_sim, 
            'popular': popular
        }, f)
    print("Pipeline Execution Successful: Optimized ML Assets Saved.")

if __name__ == "__main__":
    main()
