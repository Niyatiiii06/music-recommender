🎵 Hybrid Music Recommender

A machine learning-based music recommendation system that suggests similar songs using a hybrid recommendation approach combining audio similarity and semantic NLP similarity.

---

📌 Project Overview

Music recommendation systems often rely only on user listening history or metadata. This project improves recommendations by combining:

- Audio-based similarity using song characteristics
- NLP-based similarity using textual metadata

This hybrid approach helps generate more relevant song recommendations.

---

🚀 Features

- Search songs from a dataset of 81,000+ tracks
- Get top 10 similar song recommendations
- Hybrid recommendation engine:
  - Audio Similarity
  - Semantic NLP Similarity
- Duplicate recommendation removal
- Interactive Streamlit UI
- Displays:
  - Song name
  - Artist
  - Genre
  - Final match score
  - Audio vs NLP contribution

---

🧠 Machine Learning Workflow

1. Data Preprocessing

- Handled missing values
- Cleaned metadata
- Removed duplicates
- Selected important audio features

2. Audio Recommendation Engine

Used audio features such as:

- Danceability
- Energy
- Loudness
- Tempo
- Valence
- Acousticness
- Instrumentalness
- Speechiness

Algorithms used:

- StandardScaler
- K-Nearest Neighbors (KNN)
- Cosine Similarity

3. NLP Recommendation Engine

Text features used:

- Track Name
- Artist
- Genre

NLP techniques:

- Text preprocessing
- TF-IDF Vectorization
- Cosine Similarity

4. Hybrid Scoring

Final recommendation score:

70% Audio Similarity + 30% NLP Similarity

This ensures recommendations are based on both musical characteristics and semantic similarity.

---

🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

📂 Project Structure

music-recommender/
│
├── app.py
├── spotify_model.pkl
├── notebook.ipynb
└── README.md

---

📊 Dataset

Dataset contains 81,000+ songs with:

- Song metadata
- Artists
- Genres
- Audio features

---

🎯 Future Improvements

- Mood-based recommendation
- Artist diversity penalty
- Deep learning embeddings
- Spotify API integration
- Collaborative filtering

---

👩‍💻 Author

Niyati Singh
Aspiring Data Scientist | Machine Learning & NLP Enthusiast