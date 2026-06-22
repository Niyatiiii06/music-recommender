import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import re

# =========================
# LOAD MODEL
# =========================
with open("spotify_model.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
song_to_index = data["song_to_index"]
text_matrix = data["text_matrix"]
features = data["features"]

def clean_name(name):
    name = str(name).lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return name.strip()

df["_song_key"] = (
    df["track_name"].apply(clean_name)
    + "|"
    + df["artists"].astype(str).str.lower().str.strip()
)

# =========================
# FUNCTIONS (UNCHANGED)
# =========================
def get_audio_scores(song_id, n=100):
    if song_id not in song_to_index:
        return {}
    song_index = song_to_index[song_id]
    selected_song = df.loc[song_index]
    song_genre = selected_song["track_genre"]
    filtered_df = df[df["track_genre"] == song_genre].copy()
    filtered_df = filtered_df.reset_index()
    X = filtered_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=min(n + 1, len(filtered_df)), metric="cosine")
    knn.fit(X_scaled)
    filtered_index = filtered_df[filtered_df["song_id"] == song_id].index[0]
    distances, indices = knn.kneighbors(X_scaled[filtered_index].reshape(1, -1))
    audio_scores = {}
    for idx, dist in zip(indices[0], distances[0]):
        original_index = filtered_df.iloc[idx]["index"]
        audio_scores[original_index] = 1 - dist
    return audio_scores

def get_text_scores(song_id):
    if song_id not in song_to_index:
        return None
    song_index = song_to_index[song_id]
    query_vector = text_matrix[song_index]
    return cosine_similarity(query_vector, text_matrix).flatten()

def recommend_hybrid(song_id, n=10):
    audio_scores = get_audio_scores(song_id, n=100)
    text_scores  = get_text_scores(song_id)
    final_scores = {}
    for idx, audio_score in audio_scores.items():
        text_score = text_scores[idx]
        final_scores[idx] = {
            "final": 0.70 * audio_score + 0.30 * text_score,
            "audio": audio_score,
            "text":  text_score
        }
    sorted_songs = sorted(final_scores.items(), key=lambda x: x[1]["final"], reverse=True)
    results = []
    selected_index = song_to_index[song_id]
    seen = {df.loc[selected_index]["_song_key"]}
    for idx, scores in sorted_songs:
        row = df.loc[idx]
        key = row["_song_key"]
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "track_name":  row["track_name"],
            "artist":      row["artists"],
            "genre":       row["track_genre"],
            "score":       round(scores["final"] * 100, 1),
            "audio_score": round(scores["audio"]  * 100, 1),
            "text_score":  round(scores["text"]   * 100, 1)
        })
        if len(results) == n:
            break
    return results

# =========================
# UI
# =========================
st.set_page_config(page_title="Hybrid Music Recommender", page_icon="🎵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0d1a; }
.block-container { padding: 2.5rem 3rem; max-width: 860px; }

.rcard {
    background: #13132a;
    border: 1px solid #1e1e3a;
    border-radius: 18px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.2s;
}
.rcard:hover { border-color: #7c3aed; background: #16163a; }
.rcard-left { display: flex; align-items: center; gap: 14px; }
.rrank { font-size: 1.2rem; font-weight: 800; color: #4b5563; min-width: 36px; }
.rtitle { font-size: 1rem; font-weight: 700; color: #f3f4f6; margin-bottom: 2px; }
.rartist { font-size: 0.8rem; color: #6b7280; }
.rgenre {
    display: inline-block; background: #1e1e3a; color: #8b5cf6;
    border-radius: 99px; padding: 2px 10px; font-size: 0.7rem;
    font-weight: 600; margin-top: 5px;
}
.rcard-right { text-align: right; flex-shrink: 0; }
.rscore { font-size: 1.5rem; font-weight: 800; }
.rsubscore { font-size: 0.68rem; color: #4b5563; margin-top: 1px; }
.sbar-wrap { width: 72px; height: 3px; background: #1e1e3a; border-radius: 99px; margin-top: 5px; overflow: hidden; margin-left: auto; }
.sbar-fill { height: 100%; border-radius: 99px; }
.slabel {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #4b5563; margin-bottom: 10px;
}
div[data-baseweb="select"] > div {
    background: #13132a !important;
    border: 1px solid #2d2d44 !important;
    border-radius: 12px !important;
    color: #f3f4f6 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 0.95rem !important; padding: 0.6rem 2rem !important;
    width: 100% !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.footer { text-align: center; color: #2d2d44; font-size: 0.72rem; margin-top: 2rem; }
h1 { color: #f3f4f6 !important; }
p, .stMarkdown p { color: #6b7280 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🎵 Hybrid Music Recommender")
st.write("Audio similarity × Semantic NLP · KNN + TF-IDF · 81,344 songs")

song_list = sorted(song_to_index.keys())
selected_song = st.selectbox("Search a song", song_list, label_visibility="collapsed",
                             placeholder="Type to search a song...")

row_in = df.loc[song_to_index[selected_song]]
st.markdown(f'''
<div style="background:#13132a;border:1px solid #2d2d44;border-radius:20px;padding:1.4rem 1.6rem;margin:1rem 0 1.5rem">
  <div style="font-size:1.3rem;font-weight:800;color:#f3f4f6;margin-bottom:2px">{row_in["track_name"]}</div>
  <div style="font-size:0.85rem;color:#6b7280;margin-bottom:12px">{row_in["artists"]}</div>
  <span style="display:inline-block;padding:4px 12px;border-radius:99px;font-size:0.75rem;font-weight:600;margin-right:6px;margin-top:4px;background:#1e1e3a;color:#60a5fa">⚡ Energy {row_in["energy"]:.2f}</span>
  <span style="display:inline-block;padding:4px 12px;border-radius:99px;font-size:0.75rem;font-weight:600;margin-right:6px;margin-top:4px;background:#1e1e3a;color:#34d399">💃 Dance {row_in["danceability"]:.2f}</span>
  <span style="display:inline-block;padding:4px 12px;border-radius:99px;font-size:0.75rem;font-weight:600;margin-right:6px;margin-top:4px;background:#1e1e3a;color:#f472b6">😊 Valence {row_in["valence"]:.2f}</span>
  <span style="display:inline-block;padding:4px 12px;border-radius:99px;font-size:0.75rem;font-weight:600;margin-right:6px;margin-top:4px;background:#1e1e3a;color:#fbbf24">⭐ Popularity {int(row_in["popularity"])}</span>
</div>
''', unsafe_allow_html=True)

if st.button("Recommend 🎯"):
    with st.spinner("Scanning 81,344 songs..."):
        results = recommend_hybrid(selected_song)

    st.markdown(f'<div class="slabel">Top {len(results)} recommendations for <span style="color:#a78bfa">{selected_song}</span></div>', unsafe_allow_html=True)

    for i, song in enumerate(results, 1):
        score = song["score"]
        if score >= 80:
            color = "#34d399"
        elif score >= 65:
            color = "#60a5fa"
        else:
            color = "#f59e0b"

        st.markdown(f"""
        <div class="rcard">
          <div class="rcard-left">
            <div class="rrank">#{i}</div>
            <div>
              <div class="rtitle">{song['track_name']}</div>
              <div class="rartist">{song['artist']}</div>
              <span class="rgenre">{song['genre']}</span>
            </div>
          </div>
          <div class="rcard-right">
            <div class="rscore" style="color:{color}">{score}%</div>
            <div class="rsubscore">Audio {song['audio_score']} · NLP {song['text_score']}</div>
            <div class="sbar-wrap"><div class="sbar-fill" style="width:{score}%;background:{color}"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Audio · NLP · Genre-filtered · Built with Streamlit</div>', unsafe_allow_html=True)
