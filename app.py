import streamlit as st
import altair as alt
import pandas as pd
from bert_inference import predict_top_k
from engine import weighted_movie_recommendation, weighted_song_recommendation

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Emotion AI Enterprise",
    page_icon="📊",
    layout="wide"
)

# =============================
# SESSION STATE INIT
# =============================
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "emotion_weights": {}
    }

# =============================
# HEADER
# =============================
st.title("📊 Emotion-Aware Media Intelligence")
st.subheader("Personalized AI Recommendation System")

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙️ AI Dashboard Controls")

user_name = st.sidebar.text_input("User Name")

if st.sidebar.button("Clear Emotion History"):
    st.session_state.emotion_history = []
    st.session_state.user_profile["emotion_weights"] = {}

st.sidebar.metric("Sessions Tracked", len(st.session_state.emotion_history))

# =============================
# INPUT
# =============================
user_text = st.text_area("🧠 Describe how you're feeling")

if st.button("🚀 Analyze Emotion") and user_text.strip():

    # 🔥 Step 1: Predict Emotion
    emotion_scores = predict_top_k(user_text)

    # 🔥 Step 2: Get Hybrid Recommendations
    movie_recs = weighted_movie_recommendation(
        emotion_scores,
        st.session_state.user_profile["emotion_weights"]
    )

    song_recs = weighted_song_recommendation(
        emotion_scores,
        st.session_state.user_profile["emotion_weights"]
    )

    # 🔥 Step 3: Save Session History
    st.session_state.emotion_history.append(emotion_scores)

    # 🔥 Step 4: Update User Profile
    for emotion, weight in emotion_scores.items():
        previous = st.session_state.user_profile["emotion_weights"].get(emotion, 0)
        st.session_state.user_profile["emotion_weights"][emotion] = previous + weight

    # =============================
    # KPI METRICS
    # =============================
    top_emotion = list(emotion_scores.keys())[0]
    top_score = list(emotion_scores.values())[0]

    k1, k2, k3 = st.columns(3)

    k1.metric("Top Emotion", top_emotion.upper())
    k2.metric("Confidence", f"{top_score:.2f}")
    k3.metric("Emotions Detected", len(emotion_scores))

    st.markdown("---")

    # =============================
    # EMOTION DISTRIBUTION
    # =============================
    colA, colB = st.columns(2)

    with colA:
        st.subheader("📈 Emotion Distribution")

        prob_df = pd.DataFrame(
            emotion_scores.items(),
            columns=["Emotion", "Probability"]
        )

        chart = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X("Emotion", sort="-y"),
            y="Probability",
            color="Emotion"
        )

        st.altair_chart(chart, use_container_width=True)

    # =============================
    # EMOTION TREND
    # =============================
    with colB:
        st.subheader("📊 Emotion Trend")

        if len(st.session_state.emotion_history) > 1:
            trend_df = pd.DataFrame(st.session_state.emotion_history).fillna(0)
            trend_df["Session"] = range(1, len(trend_df) + 1)

            trend_melt = trend_df.melt(
                id_vars=["Session"],
                var_name="Emotion",
                value_name="Score"
            )

            trend_chart = alt.Chart(trend_melt).mark_line(point=True).encode(
                x="Session",
                y="Score",
                color="Emotion"
            )

            st.altair_chart(trend_chart, use_container_width=True)
        else:
            st.info("Analyze multiple inputs to see trend.")

    st.markdown("---")

    # =============================
    # USER PROFILE
    # =============================
    st.subheader("👤 User Emotional Fingerprint")

    profile_df = pd.DataFrame(
        st.session_state.user_profile["emotion_weights"].items(),
        columns=["Emotion", "Accumulated Weight"]
    )

    if not profile_df.empty:
        profile_chart = alt.Chart(profile_df).mark_bar().encode(
            x="Emotion",
            y="Accumulated Weight",
            color="Emotion"
        )
        st.altair_chart(profile_chart, use_container_width=True)

    st.markdown("---")

    # =============================
    # RECOMMENDATIONS
    # =============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎬 Movie Intelligence Engine")
        for _, row in movie_recs.iterrows():
            st.write(f"• {row['title']} ({row['genre']})")

    with col2:
        st.subheader("🎵 Music Intelligence Engine")
        for _, row in song_recs.iterrows():
            st.write(f"• {row['title']} — {row['artist']}")

    # =============================
    # EXPLAINABLE AI
    # =============================
    st.markdown("### 🧠 Why These Recommendations?")
    for emotion, weight in emotion_scores.items():
        st.write(f"• {emotion.upper()} influenced ranking with weight {weight:.2f}")
