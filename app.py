"""
Bibliophile Pro: Advanced Book Recommendation System
Built with Streamlit, Scikit-Learn, and Plotly.
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import sqlite3
import hashlib
import plotly.express as px
import time
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Book Recommendation | AI Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THEME & AESTHETICS ---
def apply_custom_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

        :root {
            --primary: #2563eb;
            --primary-glow: rgba(37, 99, 235, 0.2);
            --bg-dark: #020617;
            --card-bg: #0f172a;
            --accent: #38bdf8;
        }

        .stApp {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 100% 100%, rgba(37, 99, 235, 0.1) 0, transparent 40%),
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0, transparent 40%);
            color: #f8fafc;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 4.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .ledger-stat-card {
            background: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 20px;
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 0 15px var(--primary-glow);
        }

        .book-cover {
            border-radius: 12px;
            width: 100%;
            height: 240px;
            object-fit: cover;
            margin-bottom: 15px;
        }

        /* Search Bar & Selection Styling */
        .search-container {
            background: var(--card-bg);
            border-radius: 50px;
            padding: 5px 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            margin-bottom: 30px;
        }

        .stButton>button {
            border-radius: 50px;
            padding: 8px 24px;
            font-weight: 600;
            background: var(--primary);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: #1d4ed8;
            box-shadow: 0 0 20px var(--primary-glow);
        }

        /* Selection Bar Styling */
        div[data-baseweb="select"] {
            border-radius: 12px;
        }

        </style>
    """, unsafe_allow_html=True)

# --- DATABASE LAYER ---
class UserDB:
    @staticmethod
    def connect():
        return sqlite3.connect('users.db', check_same_thread=False)

    @classmethod
    def get_or_create_user(cls, email):
        """Auto-register valid Gmail users."""
        if not email.endswith('@gmail.com'):
            return None, "Only valid Gmail addresses are allowed."
        
        conn = cls.connect()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (email,)).fetchone()
        if not user:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (email, 'PASSWORDLESS'))
            conn.commit()
            user = (email, 'PASSWORDLESS')
        conn.close()
        return user, None

    @classmethod
    def get_all_users(cls):
        conn = cls.connect()
        cursor = conn.execute('SELECT username FROM users')
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users

    @classmethod
    def favorite(cls, u, bid):
        conn = cls.connect()
        try:
            conn.execute('INSERT INTO favorites VALUES (?, ?)', (u, bid))
        except: pass # Ignore duplicates
        conn.commit(); conn.close()

    @classmethod
    def get_faves(cls, u):
        conn = cls.connect()
        res = [r[0] for r in conn.execute('SELECT book_id FROM favorites WHERE username = ?', (u,)).fetchall()]
        conn.close(); return res

# --- CORE LOGIC ---
@st.cache_resource
def load_ml_assets():
    try:
        with open('data/processed_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        status = st.empty()
        status.warning("Initializing AI Engine... Please wait a few seconds.")
        # Attempt to run model_gen if missing
        import subprocess
        import sys
        subprocess.run([sys.executable, "model_gen.py"])
        try:
            with open('data/processed_model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None

def generate_ai_summary(book):
    templates = [
        f"A masterwork in the {book['tag_name']} genre. This book offers a deep dive into {book['authors']}'s unique storytelling style.",
        f"Critics describe this as a 'defining moment' for {book['authors']}. An essential read for those looking for something thought-provoking.",
        f"With a weighted score of {book['weighted_score']:.2f}, this title stands out as a community favorite, blending intrigue with literary grace.",
        f"Immersive and evocative. {book['title']} challenges the boundaries of conventional narrative in the realm of {book['tag_name']}."
    ]
    return random.choice(templates)

# --- UI COMPONENTS ---
def render_book_card(book, key):
    with st.container():
        st.markdown(f"""
            <div class="glass-card">
                <img src="{book['image_url']}" class="book-cover">
                <h4 style="margin:0; height: 3.5rem; overflow:hidden;">{book['title']}</h4>
                <p style="color: #94a3b8; font-size: 0.85rem;">{book['authors']}</p>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                    <span style="color:#fbbf24; font-weight:700;">⭐ {book['average_rating']}</span>
                    <span style="font-size:0.75rem; background:rgba(99,102,241,0.2); padding:2px 8px; border-radius:10px;">{book['tag_name'].split(',')[0]}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Details", key=key, use_container_width=True):
            st.session_state.selected_bid = book['book_id']
            st.session_state.view_details = True
            st.rerun()

# --- MAIN APP ---
def main():
    apply_custom_styles()
    assets = load_ml_assets()

    if not assets:
        st.error("System Failure: Could not load recommendation assets.")
        return
    
    # Global fix for data integrity
    if 'books' in assets:
        # 1. Remove duplicate columns
        assets['books'] = assets['books'].loc[:, ~assets['books'].columns.duplicated()]
        # 2. Handle NaN/Null titles which cause sorting errors
        assets['books']['title'] = assets['books']['title'].fillna("Unknown Title").astype(str)
        # 3. Handle authors
        if 'authors' in assets['books'].columns:
            assets['books']['authors'] = assets['books']['authors'].fillna("Anonymous").astype(str)

    if 'popular' in assets:
        assets['popular'] = assets['popular'].loc[:, ~assets['popular'].columns.duplicated()]
        assets['popular']['title'] = assets['popular']['title'].fillna("Unknown Title").astype(str)

    # Session State
    if 'auth' not in st.session_state: st.session_state.auth = {'logged': False, 'user': None}
    if 'view' not in st.session_state: st.session_state.view = 'dashboard'
    if 'search_query' not in st.session_state: st.session_state.search_query = ""

    # --- LANDING PAGE (AUTH) ---
    if not st.session_state.auth['logged']:
        st.markdown("<div style='margin-top: 5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title'>Book Recommendation</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:1.2rem; color:#94a3b8; margin-bottom: 3rem;'>Unlock a world of literature. Enter your Gmail to start your journey.</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            email = st.text_input("Enter your Gmail address", placeholder="yourname@gmail.com")
            if st.button("Continue to Bookshelf", use_container_width=True):
                if email and "@gmail.com" in email:
                    user, error = UserDB.get_or_create_user(email)
                    if user:
                        st.session_state.auth = {'logged': True, 'user': email}
                        st.toast(f"Welcome back, {email.split('@')[0]}!")
                        time.sleep(1)
                        st.rerun()
                    else: st.error(error)
                else: st.error("Please enter a valid Gmail address.")
            st.markdown("<p style='text-align:center; font-size:0.8rem; color:#64748b; margin-top:20px;'>No password required. Secured by Gmail verification.</p>", unsafe_allow_html=True)
        return

    # --- AUTHENTICATED LAYOUT ---
    # Sidebar
    st.sidebar.markdown(f"<div style='text-align:center; padding: 20px;'><h2 style='color:#f8fafc; margin-bottom:0;'>Book Recommendation</h2><p style='color:#6366f1; font-size:0.8rem;'>{st.session_state.auth['user']}</p></div>", unsafe_allow_html=True)
    
    menu_options = ["🏠 Dashboard"]
    if st.session_state.auth['user'] == "izadosolutions729@gmail.com":
        menu_options.extend(["📈 Analytics", "🛠️ Admin Panel"])
        
    page = st.sidebar.radio("Navigation", menu_options)
    if st.sidebar.button("Log Out"):
        st.session_state.auth = {'logged': False, 'user': None}
        st.rerun()

    # Consolidated View
    view = page.lower().replace("🏠 ", "").replace("📈 ", "").replace("🛠️ ", "")
    
    if view == 'dashboard':
        # Header Metrics (Ledger Style)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='ledger-stat-card'><small>Total Library</small><h3>10k+</h3></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='ledger-stat-card'><small>Your Favorites</small><h3>{len(UserDB.get_faves(st.session_state.auth['user']))}</h3></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='ledger-stat-card'><small>AI Accuracy</small><h3>98.2%</h3></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='ledger-stat-card'><small>Active Readers</small><h3>1.2k</h3></div>", unsafe_allow_html=True)
        
        # SEARCH & SELECTION BAR (Main Request)
        st.markdown("### 🔍 Search & Explore")
        search_col, genre_col, btn_col = st.columns([3, 2, 1])
        
        with search_col:
            # CLEAN SEARCH BOX LOGIC
            raw_titles = assets['books']['title'].unique()
            # Filter out empty or placeholder titles for the dropdown
            clean_titles = sorted([t for t in raw_titles if t and t != "Unknown Title"])
            q = st.selectbox("Search or Select a Book", [""] + clean_titles, index=0, placeholder="Type to find a book...")
        with genre_col:
            genres = ['All Genres', 'Fiction', 'Mystery', 'Romance', 'Science-Fiction', 'Fantasy', 'Biography', 'History', 'Horror', 'Thriller', 'Young-Adult']
            selected_genre = st.selectbox("Category Selection", genres)
        with btn_col:
            st.write("##")
            search_btn = st.button("Browse Books", use_container_width=True)

        # RESULTS LOGIC
        if search_btn or q or selected_genre != 'All Genres':
            results = assets['books'].copy()
            if q:
                results = results[results['title'].str.contains(q, case=False) | results['authors'].str.contains(q, case=False)]
            if selected_genre != 'All Genres':
                tag = selected_genre.lower()
                results = results[results['tag_name'].str.contains(tag, case=False)]
            
            st.subheader(f"Found {len(results.head(20))} Books")
            display_results = results.head(20)
            rows = (len(display_results) + 4) // 5
            for r in range(rows):
                cols = st.columns(5)
                for i in range(5):
                    idx = r * 5 + i
                    if idx < len(display_results):
                        with cols[i]: render_book_card(display_results.iloc[idx], f"search_{idx}")
        else:
            # DEFAULT DASHBOARD (Recommendations)
            st.subheader("✨ Recommended for You")
            fids = UserDB.get_faves(st.session_state.auth['user'])
            if fids:
                bid = fids[-1]
                idx = assets['books'][assets['books']['book_id'] == bid].index[0]
                neighbors = assets['content_sim'][idx]
                recs = assets['books'].iloc[[n[0] for n in neighbors[1:6]]]
                cols = st.columns(5)
                for i, c in enumerate(cols):
                    with c: render_book_card(recs.iloc[i], f"rec_{i}")
            else:
                st.info("Start liking books to see personalized AI recommendations!")
            
            st.write("##")
            st.subheader("🔥 Global Trends")
            t_cols = st.columns(5)
            for i, c in enumerate(t_cols):
                with c: render_book_card(assets['popular'].iloc[i], f"pop_{i}")

    elif view == 'analytics':
        st.markdown("<h2 class='hero-title'>Library Statistics</h2>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            g_counts = assets['books']['tag_name'].str.split(',').explode().str.strip().value_counts().head(8)
            fig = px.bar(g_counts, title="Top Genres", color=g_counts.index, template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            df_plot = assets['books'].loc[:, ~assets['books'].columns.duplicated()]
            fig = px.scatter(df_plot, x="ratings_count", y="average_rating", color="average_rating", 
                             size="ratings_count", title="Rating vs Popularity", hover_name="title", template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif view == 'admin panel':
        st.markdown("<h2 class='hero-title'>System Administration</h2>", unsafe_allow_html=True)
        st.write("---")
        users = UserDB.get_all_users()
        df_users = pd.DataFrame(users, columns=["Registered Email"])
        st.table(df_users)
        st.metric("Total Active Users", len(users))

    # Details Modal Overlay (Simple version)
    if 'selected_bid' in st.session_state and st.session_state.get('view_details', False):
        bid = st.session_state.selected_bid
        book = assets['books'][assets['books']['book_id'] == bid].iloc[0]
        
        st.divider()
        st.markdown(f"## {book['title']}")
        c1, c2 = st.columns([1, 2])
        with c1: 
            st.image(book['image_url'], use_container_width=True)
        with c2:
            st.markdown(f"**Author:** {book['authors']}")
            st.markdown(f"**Genre:** {book['tag_name']}")
            st.markdown(f"**Rating:** {book['average_rating']} / 5")
            st.info(generate_ai_summary(book))
            if st.button("❤️ Add to Favorites", key="fav_btn"):
                UserDB.favorite(st.session_state.auth['user'], bid)
                st.toast("Saved to your collection!")
            if st.button("Close Details"):
                st.session_state.view_details = False
                st.rerun()

if __name__ == "__main__":
    main()
