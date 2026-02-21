import pandas as pd

# =============================
# LOAD DATA (READ-ONLY)
# =============================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.join(BASE_DIR, "data", "indian_movies.csv")
songs_path = os.path.join(BASE_DIR, "data", "indian_songs.csv")

movies_df = pd.read_csv(movies_path)
songs_df = pd.read_csv(songs_path)


# =============================
# NORMALIZE SONG GENRES
# =============================
def normalize_song_genre(genre):
    g = str(genre).lower()

    if "hip" in g or "rap" in g:
        return "energetic"
    elif "romance" in g or "melody" in g:
        return "romantic"
    elif "pop" in g or "dance" in g:
        return "pop"
    elif "sad" in g:
        return "sad"
    elif "classical" in g or "instrumental" in g:
        return "calm"
    else:
        return "filmi"

songs_df["norm_genre"] = songs_df["genre"].apply(normalize_song_genre)

# =============================
# EMOTION → GENRE MAPPING
# =============================
movie_emotion_genres = {
    "sadness": ["Romance", "Drama"],
    "joy": ["Comedy", "Family"],
    "love": ["Romance"],
    "anger": ["Action"],
    "fear": ["Thriller"],
    "surprise": ["Adventure"]
}

song_emotion_genres = {
    "sadness": ["romantic", "sad", "calm"],
    "joy": ["pop", "energetic"],
    "love": ["romantic"],
    "anger": ["energetic"],
    "fear": ["calm"],
    "surprise": ["pop"]
}

# =============================
# SAFE HYBRID WEIGHT FUNCTION
# =============================
def compute_hybrid_weight(emotion, current_scores, user_profile):
    current_weight = current_scores.get(emotion, 0)
    history_weight = user_profile.get(emotion, 0)

    return (0.7 * current_weight) + (0.3 * history_weight)

# =============================
# VALIDATION FUNCTION
# =============================
def validate_emotion_scores(emotion_scores):
    if not isinstance(emotion_scores, dict):
        raise ValueError(
            "emotion_scores must be a dictionary. "
            "Make sure you call predict_top_k() in app.py "
            "before passing data to engine."
        )
    return emotion_scores

# =============================
# MOVIE RECOMMENDATION
# =============================
def weighted_movie_recommendation(
    emotion_scores,
    user_profile=None,
    top_n=3
):

    emotion_scores = validate_emotion_scores(emotion_scores)

    if user_profile is None:
        user_profile = {}

    temp_df = movies_df.copy()
    scores = []

    for _, row in temp_df.iterrows():
        total_score = 0
        genres = str(row["genre"])

        for emotion in emotion_scores:
            hybrid_weight = compute_hybrid_weight(
                emotion,
                emotion_scores,
                user_profile
            )

            if emotion in movie_emotion_genres:
                for g in movie_emotion_genres[emotion]:
                    if g.lower() in genres.lower():
                        total_score += hybrid_weight

        scores.append(total_score)

    temp_df["emotion_score"] = scores

    ranked = temp_df.sort_values(
        by="emotion_score",
        ascending=False
    )

    # Safe fallback
    if ranked.empty:
        return temp_df.head(top_n)

    return ranked.head(top_n)

# =============================
# SONG RECOMMENDATION
# =============================
def weighted_song_recommendation(
    emotion_scores,
    user_profile=None,
    top_n=3
):

    emotion_scores = validate_emotion_scores(emotion_scores)

    if user_profile is None:
        user_profile = {}

    temp_df = songs_df.copy()
    scores = []

    for _, row in temp_df.iterrows():
        total_score = 0
        genre = str(row["norm_genre"]).lower()

        for emotion in emotion_scores:
            hybrid_weight = compute_hybrid_weight(
                emotion,
                emotion_scores,
                user_profile
            )

            if emotion in song_emotion_genres:
                if genre in song_emotion_genres[emotion]:
                    total_score += hybrid_weight

        scores.append(total_score)

    temp_df["emotion_score"] = scores

    ranked = temp_df.sort_values(
        by="emotion_score",
        ascending=False
    )

    # Safe fallback
    if ranked.empty:
        return temp_df.head(top_n)

    return ranked.head(top_n)
