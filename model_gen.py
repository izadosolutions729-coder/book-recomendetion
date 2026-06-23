"""
ML Pipeline for Book Recommendation System
Handles data ingestion, preprocessing, and model generation.
"""
import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_data():
    """Download Goodbooks-10k datasets from standard repository."""
    urls = {
        "books": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
        "ratings": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
        "tags": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv",
        "book_tags": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv"
    }
    data = {}
    for name, url in urls.items():
        logger.info(f"Downloading {name} dataset...")
        try:
            data[name] = pd.read_csv(url)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            raise
    return data["books"], data["ratings"], data["tags"], data["book_tags"]

def process_pipeline(books, ratings, tags, book_tags):
    """Execute preprocessing and model training."""
    logger.info("Starting preprocessing pipeline...")
    
    # 1. Book Metadata Cleaning
    books = books.loc[:, ~books.columns.duplicated()]
    
    # Handle titles
    if 'title' in books.columns and 'original_title' in books.columns:
        books.drop(columns=['title'], inplace=True)
        books.rename(columns={'original_title': 'title'}, inplace=True)
    
    books['title'] = books['title'].fillna("Unknown Title").astype(str)
    books['authors'] = books['authors'].fillna("Anonymous").astype(str)
    
    # 2. Genre Extraction
    logger.info("Extracting genres...")
    merged_tags = pd.merge(book_tags, tags, on='tag_id')
    common_genres = ['fiction', 'mystery', 'romance', 'science-fiction', 'fantasy', 'biography', 'history', 'horror', 'thriller', 'young-adult']
    merged_tags['tag_name'] = merged_tags['tag_name'].str.lower()
    genre_tags = merged_tags[merged_tags['tag_name'].isin(common_genres)]
    book_genres = genre_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ', '.join(list(set(x)))).reset_index()
    
    if 'tag_name' in books.columns:
        books.drop(columns=['tag_name'], inplace=True)

    books = pd.merge(books, book_genres, on='goodreads_book_id', how='left')
    books['tag_name'] = books['tag_name'].fillna('General')
    books = books.drop_duplicates(subset=['book_id']).reset_index(drop=True)
    
    # 3. Content-Based Model (TF-IDF)
    logger.info("Building recommendation engine...")
    titles, authors, genres = [books[c].fillna('').values for c in ['title', 'authors', 'tag_name']]
    books['content'] = [f"{t} {a} {g}" for t, a, g in zip(titles, authors, genres)]
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books['content'])
    
    # Efficient similarity: Only store Top 100 neighbors to save space
    raw_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
    
    top_n = 100
    reduced_sim = []
    for row in raw_sim:
        idx = np.argpartition(row, -top_n)[-top_n:]
        idx = idx[np.argsort(row[idx])[::-1]]
        reduced_sim.append(list(zip(idx, row[idx])))
    
    # 4. Global Popularity (Bayesian Weighted Rating)
    logger.info("Calculating weighted scores...")
    C = books['average_rating'].mean()
    m = books['ratings_count'].quantile(0.9)
    v = books['ratings_count'].values
    R = books['average_rating'].values
    books['weighted_score'] = ((v/(v+m) * R) + (m/(m+v) * C))
    
    popular = books.sort_values('weighted_score', ascending=False).reset_index(drop=True)
    
    return books, reduced_sim, popular

def initialize_persistence():
    """Setup SQLite environment."""
    logger.info("Initializing database...")
    conn = sqlite3.connect('users.db')
    queries = [
        'CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)',
        'CREATE TABLE IF NOT EXISTS favorites (username TEXT, book_id INTEGER)',
        'CREATE TABLE IF NOT EXISTS history (username TEXT, book_id INTEGER, timestamp TEXT)'
    ]
    for q in queries:
        conn.execute(q)
    conn.commit()
    conn.close()

def main():
    if not os.path.exists('data'):
        os.makedirs('data')
    
    try:
        books_raw, ratings_raw, tags_raw, book_tags_raw = load_raw_data()
        books, reduced_sim, popular = process_pipeline(books_raw, ratings_raw, tags_raw, book_tags_raw)
        initialize_persistence()
        
        output_path = 'data/processed_model.pkl'
        logger.info(f"Saving assets to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'books': books, 
                'content_sim': reduced_sim, 
                'popular': popular
            }, f)
        
        logger.info("Pipeline Execution Successful.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
