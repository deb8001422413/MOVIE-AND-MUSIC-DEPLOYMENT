import streamlit as st
import pickle
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Movie & Music Recommender",
    layout="centered"
)


# ==========================
# CSS
# ==========================

st.markdown("""
<style>

.fade-in {
    animation: fadeIn 0.8s ease-in-out;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.rainbow-card {
    position: relative;
    padding: 14px;
    margin: 12px 0;
    border-radius: 14px;
    background: red;
    color: white;
    overflow: hidden;
    isolation: isolate;
}

.rainbow-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 14px;
    padding: 12px;
    background: conic-gradient(
        #00ff7f,
        #00c8ff,
        #ff00ff,
        #ff9800,
        #00ff7f
    );
    animation: spin 5s linear infinite;

    -webkit-mask:
        linear-gradient(#029 0 0) content-box,
        linear-gradient(#029 0 0);

    -webkit-mask-composite: xor;
    mask-composite: exclude;
}

@keyframes spin {
    0% {
        transform: rotate(0deg);
    }

    100% {
        transform: rotate(360deg);
    }
}

.rainbow-card:hover {
    background: #0b2e1a;
    box-shadow: 0 0 20px rgba(0,255,127,0.6);
    transform: scale(1.03);
    transition: all 0.3s ease;
}

div.stButton > button {
    position: relative;
    padding: 0.6em 1.5em;
    font-size: 16px;
    font-weight: 600;
    color: white;
    background: rgba(0,255,127,0.6);
    border-radius: 14px;
    border: none;
    cursor: pointer;
    overflow: hidden;
    z-index: 0;
}

div.stButton > button::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 16px;
    background: conic-gradient(
        #00ff7f,
        #00c8ff,
        #ff00ff,
        #ff9800,
        #00ff7f
    );
    animation: spin 4s linear infinite;
    z-index: -1;
}

div.stButton > button::after {
    content: "";
    position: absolute;
    inset: 2px;
    border-radius: 12px;
    background: #181;
    z-index: -1;
    box-shadow: 0 0 18px orange;
}

div.stButton > button:hover::after {
    background: linear-gradient(
        135deg,
        #00ff7f,
        #00c8ff
    );
}

div.stButton > button:hover {
    box-shadow: 0 0 18px rgba(0,255,127,0.7);
    transform: scale(1.05);
    transition: all 0.3s ease;
}

div[data-baseweb="select"],
div[data-baseweb="input"] {
    position: relative;
    border-radius: 45px;
    padding: 12px;
    background: linear-gradient(
        135deg,
        #00ff7f,
        #00c8ff,
        #ff00ff,
        #ff9800,
        #00ff7f
    );
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 22px !important;
    background: lightyellow !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] * {
    color: red !important;
}

div[data-baseweb="select"]:hover,
div[data-baseweb="input"]:hover {
    box-shadow: 0 0 25px rgba(0,255,127,0.6);
}

.stApp {
    background-color: black;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# LOAD DATA
# ==========================

@st.cache_resource
def load_data():

    # Movie dictionary
    with open("movies_dict.pkl", "rb") as f:
        movies_dict = pickle.load(f)

    # Movie vectors
    movie_vectors = load_npz("movie_vectors.npz")

    # Music dictionary
    with open("music_dict.pkl", "rb") as f:
        music_dict = pickle.load(f)

    # Music vectors
    music_vectors = load_npz("music_vectors.npz")

    return (
        movies_dict,
        movie_vectors,
        music_dict,
        music_vectors
    )


movies_dict, movie_vectors, music_dict, music_vectors = load_data()


# ==========================
# DATAFRAMES
# ==========================

movies = pd.DataFrame(movies_dict)
music = pd.DataFrame(music_dict)


# ==========================
# MOVIE RECOMMENDATION
# ==========================

def recommend_movie(movie_title):

    index = movies[
        movies["title"] == movie_title
    ].index[0]

    # Calculate similarity only for selected movie
    distances = cosine_similarity(
        movie_vectors[index],
        movie_vectors
    ).flatten()

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    st.markdown(
        f"### Movies like **{movie_title}**"
    )

    for i, score in movies_list:

        title = movies.iloc[i]["title"]

        st.markdown(
            f"""
            <div class="rainbow-card fade-in">
                🎞️ <b>{title}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================
# MUSIC RECOMMENDATION
# ==========================

def recommend_music(song_name):

    index = music[
        music["track_name"] == song_name
    ].index[0]

    artist = music.iloc[index]["track_artist"]

    # Calculate similarity only for selected song
    distances = cosine_similarity(
        music_vectors[index],
        music_vectors
    ).flatten()

    sorted_songs = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    st.markdown(
        f"### 🎵 Songs like **{song_name}**"
    )

    count = 0

    for i, score in sorted_songs:

        track = music.iloc[i]["track_name"]
        art = music.iloc[i]["track_artist"]

        # Skip the selected song
        if track == song_name and art == artist:
            continue

        st.markdown(
            f"""
            <div class="rainbow-card fade-in">
                <b>{track}</b><br>
                {art}
            </div>
            """,
            unsafe_allow_html=True
        )

        count += 1

        if count == 5:
            break


# ==========================
# TITLE
# ==========================

st.title(
    "🎬 Movie & 🎵 Music Recommendation System"
)


# ==========================
# TABS
# ==========================

tabs = st.tabs(
    ["🎬 Movies", "🎵 Music"]
)


# ==========================
# MOVIE TAB
# ==========================

with tabs[0]:

    movie = st.selectbox(
        "Select a movie",
        movies["title"].values
    )

    if st.button(
        "Recommend Movies"
    ):

        recommend_movie(movie)


# ==========================
# MUSIC TAB
# ==========================

with tabs[1]:

    song = st.selectbox(
        "Select a song",
        music["track_name"].values
    )

    if st.button(
        "Recommend Songs"
    ):

        recommend_music(song)
