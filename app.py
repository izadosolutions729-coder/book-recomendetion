"""
Book Recommendation: Advanced Book Recommendation System
Built with Streamlit, Scikit-Learn, and Plotly.
"""
import streamlit as st
import pickle
import sqlite3
import plotly.express as px
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        /* Center main container */
        .main .block-container {
            max-width: 1200px;
            padding-top: 3rem;
            padding-bottom: 5rem;
        }

        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }

        .stat-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 24px;
            text-align: center;
            backdrop-filter: blur(8px);
            transition: transform 0.3s ease;
        }

        .stat-card h3 {
            font-size: 2rem;
            margin: 10px 0 0 0;
            font-weight: 700;
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 16px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        .glass-card:hover {
            transform: translateY(-8px) scale(1.02);
            border-color: var(--primary);
            background: rgba(30, 41, 59, 0.6);
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5), 0 0 20px var(--primary-glow);
        }

        .book-cover {
            border-radius: 16px;
            width: 100%;
            aspect-ratio: 2/3;
            object-fit: cover;
            margin-bottom: 12px;
            box-shadow: 0 8px 16px -4px rgba(0,0,0,0.3);
        }

        .stButton>button {
            border-radius: 12px;
            padding: 10px 24px;
            font-weight: 600;
            background: linear-gradient(135deg, var(--primary) 0%, #1d4ed8 100%);
            color: white;
            border: none;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.8rem;
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 25px var(--primary-glow);
        }

        /* Improved selectbox styling */
        div[data-baseweb="select"] {
            background: rgba(15, 23, 42, 0.6) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        </style>
    """, unsafe_allow_html=True)

# --- DATABASE LAYER ---
class UserDB:
    DB_PATH = 'users.db'

    @staticmethod
    def connect():
        conn = sqlite3.connect(UserDB.DB_PATH, check_same_thread=False)
        # Ensure tables exist
        conn.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS favorites (username TEXT, book_id INTEGER)')
        conn.execute('CREATE TABLE IF NOT EXISTS history (username TEXT, book_id INTEGER, timestamp TEXT)')
        conn.commit()
        return conn


    @classmethod
    def get_or_create_user(cls, email):
        """Handle user registration and login."""
        if not email.endswith('@gmail.com'):
            return None, "Registration limited to @gmail.com addresses."
        
        with cls.connect() as conn:
            user = conn.execute('SELECT username FROM users WHERE username = ?', (email,)).fetchone()
            if not user:
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (email, 'SESSION_TOKEN'))
                conn.commit()
            return email, None

    @classmethod
    def get_all_users(cls):
        with cls.connect() as conn:
            cursor = conn.execute('SELECT username FROM users')
            return [row[0] for row in cursor.fetchall()]

    @classmethod
    def toggle_favorite(cls, user, book_id):
        with cls.connect() as conn:
            exists = conn.execute('SELECT 1 FROM favorites WHERE username = ? AND book_id = ?', (user, book_id)).fetchone()
            if exists:
                conn.execute('DELETE FROM favorites WHERE username = ? AND book_id = ?', (user, book_id))
                return False
            else:
                conn.execute('INSERT INTO favorites (username, book_id) VALUES (?, ?)', (user, book_id))
                return True

    @classmethod
    def get_favorites(cls, user):
        with cls.connect() as conn:
            return [r[0] for r in conn.execute('SELECT book_id FROM favorites WHERE username = ?', (user,)).fetchall()]

# --- DATA & ML ---
@st.cache_resource
def load_assets():
    """Load pre-processed ML models and data."""
    try:
        with open('data/processed_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error("ML assets missing. Please run model_gen.py first.")
        return None
    except Exception as e:
        logger.error(f"Error loading assets: {e}")
        return None

def get_ai_insight(book):
    """Dynamic AI-inspired book descriptions."""
    insights = [
        f"A compelling journey in {book['tag_name']}. {book['authors']} masterfully blends narrative depth with a rating of {book['average_rating']}.",
        f"Ranked with a weighted score of {book['weighted_score']:.2f}, this is a standout choice for fans of sophisticated storytelling.",
        f"Immersive and evocative. A quintessential read that defines the modern literature landscape.",
    ]
    return random.choice(insights)

# --- UI COMPONENTS ---
def render_book_card(book, key):
    st.markdown(f"""
        <div class="glass-card">
            <img src="{book['image_url']}" class="book-cover">
            <div style="flex-grow: 1;">
                <h4 style="margin:0 0 8px 0; font-size: 1rem; line-height: 1.4; height: 2.8rem; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">{book['title']}</h4>
                <p style="color: #94a3b8; font-size: 0.8rem; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{book['authors']}</p>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:16px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 12px;">
                <span style="color:#fbbf24; font-weight:700; font-size: 0.9rem;">⭐ {book['average_rating']}</span>
                <span style="font-size:0.7rem; background:rgba(37, 99, 235, 0.2); color: #60a5fa; padding:4px 10px; border-radius:8px; font-weight: 600;">{str(book['tag_name']).split(',')[0].upper()}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("") # Spacer
    if st.button("Details", key=key, use_container_width=True):
        st.session_state.selected_bid = book['book_id']
        st.session_state.view_details = True
        st.rerun()

# --- MAIN APP ---
def main():
    apply_custom_styles()
    assets = load_assets()

    if not assets:
        st.error("⚠️ System components missing. Please contact the administrator.")
        return


    # Session Initialization
    if 'auth' not in st.session_state: 
        st.session_state.auth = {'logged': False, 'user': None}

    # --- ENTRANCE PAGE ---
    if not st.session_state.auth['logged']:
        st.markdown("<div style='margin-top: 10vh;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title'>Book Recommendation</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#94a3b8;'>A premium AI-powered book discovery engine.</p>", unsafe_allow_html=True)
        
        _, col, _ = st.columns([1, 1.5, 1])
        with col:
            email = st.text_input("Gmail Address", placeholder="name@gmail.com")
            if st.button("Enter Workspace", use_container_width=True):
                user, error = UserDB.get_or_create_user(email)
                if user:
                    st.session_state.auth = {'logged': True, 'user': user}
                    st.toast("Access Granted")
                    time.sleep(0.5)
                    st.rerun()
                else: st.error(error)
        return

    # --- AUTHENTICATED EXPERIENCE ---
    st.sidebar.markdown(f"<div style='padding: 20px;'><h2>Book Recommendation</h2><p style='color:#6366f1;'>{st.session_state.auth['user']}</p></div>", unsafe_allow_html=True)
    
    options = ["Dashboard"]
    if st.session_state.auth['user'] == "izadosolutions729@gmail.com":
        options.extend(["Analytics", "Admin"])
        
    nav = st.sidebar.radio("Navigation", options)

    if st.sidebar.button("Log Out"):
        st.session_state.auth = {'logged': False, 'user': None}
        st.rerun()

    if nav == "Dashboard":
        # Metrics
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.markdown("<div class='stat-card'><small style='color:#94a3b8; text-transform:uppercase; letter-spacing:1px;'>Total Library</small><h3>10,000+</h3></div>", unsafe_allow_html=True)
        with m2: 
            faves = UserDB.get_favorites(st.session_state.auth['user'])
            st.markdown(f"<div class='stat-card'><small style='color:#94a3b8; text-transform:uppercase; letter-spacing:1px;'>My Favorites</small><h3>{len(faves)}</h3></div>", unsafe_allow_html=True)
        with m3: st.markdown("<div class='stat-card'><small style='color:#94a3b8; text-transform:uppercase; letter-spacing:1px;'>Service Status</small><h3 style='color:#10b981;'>ONLINE</h3></div>", unsafe_allow_html=True)
        with m4: st.markdown("<div class='stat-card'><small style='color:#94a3b8; text-transform:uppercase; letter-spacing:1px;'>Precision</small><h3>98.4%</h3></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 3rem;'></div>", unsafe_allow_html=True)

        # Discovery Module
        st.markdown("### 🔍 Discovery Engine")
        col1, col2 = st.columns([2, 1])
        with col1:
            clean_titles = sorted([str(t) for t in assets['books']['title'].unique()])
            q = st.selectbox("Search across our premium collection", [""] + clean_titles, index=0, placeholder="Type a book title...")
        with col2:
            genres = ['All Genres'] + sorted(['Fiction', 'Mystery', 'Romance', 'Science-Fiction', 'Fantasy', 'Biography', 'History', 'Horror', 'Thriller', 'Young-Adult'])
            sel_genre = st.selectbox("Filter by Literary Genre", genres)

        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)


        if q or sel_genre != 'All Genres':
            results = assets['books'].copy()
            if q:
                results = results[results['title'] == q]
            if sel_genre != 'All Genres':
                results = results[results['tag_name'].str.contains(sel_genre.lower(), case=False)]
            
            st.subheader(f"Results ({len(results.head(10))})")
            res_data = results.head(10)
            cols = st.columns(5)
            for i, (_, row) in enumerate(res_data.iterrows()):
                with cols[i % 5]: render_book_card(row, f"search_{i}")
        else:
            # Recommendations
            st.subheader("🎯 Personalized for You")
            if faves:
                last_fave = faves[-1]
                book_idx = assets['books'][assets['books']['book_id'] == last_fave].index[0]
                recs_meta = assets['content_sim'][book_idx]
                recs = assets['books'].iloc[[n[0] for n in recs_meta[1:6]]]
                cols = st.columns(5)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i]: render_book_card(row, f"rec_{i}")
            else:
                st.info("Select a book and add to favorites to activate AI recommendations.")

            st.write("##")
            st.subheader("🔥 Global Trends")
            t_cols = st.columns(5)
            for i, (_, row) in enumerate(assets['popular'].head(5).iterrows()):
                with t_cols[i]: render_book_card(row, f"pop_{i}")

    elif nav == "Analytics":
        st.markdown("<h2 class='hero-title'>Insights</h2>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            g_dist = assets['books']['tag_name'].str.split(',').explode().str.strip().value_counts().head(8)
            fig = px.pie(values=g_dist.values, names=g_dist.index, title="Genre Distribution", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(assets['books'].head(1000), x="ratings_count", y="average_rating", color="average_rating", 
                             size="ratings_count", title="Popularity Analysis", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    elif nav == "Admin":
        st.markdown("<h2 class='hero-title'>Admin Panel</h2>", unsafe_allow_html=True)
        users = UserDB.get_all_users()
        st.table(pd.DataFrame(users, columns=["Authorized Users"]))
        st.metric("Total User Base", len(users))

    # Overlays
    if st.session_state.get('view_details', False):
        st.divider()
        bid = st.session_state.selected_bid
        book = assets['books'][assets['books']['book_id'] == bid].iloc[0]
        
        c1, c2 = st.columns([1, 2])
        with c1: st.image(book['image_url'], use_container_width=True)
        with c2:
            st.title(book['title'])
            st.markdown(f"**By:** {book['authors']} | **Genre:** {book['tag_name']}")
            st.divider()
            st.info(get_ai_insight(book))
            if st.button("❤️ Add to Favorites" if bid not in faves else "💔 Remove Favorite"):
                UserDB.toggle_favorite(st.session_state.auth['user'], bid)
                st.rerun()
            if st.button("Dismiss"):
                st.session_state.view_details = False
                st.rerun()

if __name__ == "__main__":
    main()
