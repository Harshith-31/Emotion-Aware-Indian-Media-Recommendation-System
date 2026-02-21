## 🌐 Live Demo
https://emotion-aware-indian-media-recommendation-system-hfnx3iiqimwit.streamlit.app
🎭 Emotion-Aware Indian Media Recommendation System
📌 Project Overview
The Emotion-Aware Indian Media Recommendation System is an enterprise-grade AI application that detects user emotions using a Transformer-based NLP model and generates personalized Indian movie and music recommendations through a hybrid scoring engine.
Unlike traditional preference-based recommender systems, this system adapts to the user's emotional state and historical behavior, enabling dynamic and context-aware recommendations.
🚀 Key Features
🧠 Transformer-based Top-K Emotion Detection (DistilBERT)
📊 Emotion Probability Distribution Visualization
🎬 Weighted Movie Recommendation Engine
🎵 Weighted Song Recommendation Engine
👤 Personalized Emotional Fingerprint Tracking
📈 Multi-Session Emotion Trend Analytics
🔍 Explainable AI Scoring Breakdown
🏢 Enterprise Dashboard UI (Streamlit)
⚙️ Modular Production-Ready Architecture
🧠 System Architecture
User Input (Text)
        ↓
BERT Emotion Detection (Top-K)
        ↓
Hybrid Scoring Engine
        ↓
Movie & Song Ranking
        ↓
Enterprise Dashboard Visualization
Project Structure
mini_project_1/
│
├── app.py                # Streamlit Dashboard UI
├── engine.py             # Hybrid Recommendation Engine
├── bert_inference.py     # Transformer Emotion Detection
├── requirements.txt      # Dependencies
├── README.md
│
└── data/
    ├── indian_movies.csv
    └── indian_songs.csv
⚙️ Hybrid Recommendation Model
The system uses a hybrid scoring formula:
Final Score =
    0.7 × Current Emotion Score
  + 0.3 × User Historical Emotion Profile
This allows:
Adaptive personalization
Multi-emotion influence
Ranked recommendation outputs
Behavioral tracking across sessions
📊 Dashboard Capabilities
Emotion KPI metrics
Emotion distribution bar charts
Emotion trend across sessions
User emotional fingerprint visualization
Ranked movie and song suggestions
Explainable emotion influence panel
🛠 Tech Stack
Python
Streamlit
HuggingFace Transformers
PyTorch
Pandas
Altair
Scikit-learn (baseline model)
📦 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/emotion-aware-media-recommendation.git
cd emotion-aware-media-recommendation
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
streamlit run app.py
The app will open in your browser.
📈 Project Evolution
The system evolved through multiple development phases:
Baseline Emotion Classifier (TF-IDF + SVM)
Transformer-Based Emotion Detection
Weighted Recommendation Engine
Personalized Hybrid Scoring (Level-3)
Enterprise Dashboard Development
Adaptive Intelligence Preparation (Level-4 Blueprint)
🎯 Current Capabilities
✔ Context-aware emotion detection
✔ Top-K emotion probability ranking
✔ Hybrid personalized recommendation engine
✔ Emotion trend analytics
✔ Explainable AI scoring
✔ Modular architecture ready for deployment
🔮 Future Enhancements
Temporal Emotion Decay (Adaptive AI)
Popularity & Rating-based Multi-Factor Scoring
Diversity Boost Logic (Exploration vs Exploitation)
Multi-user Authentication
Deployment on Streamlit Cloud
API Integration (TMDB / Spotify)
📊 Potential Applications
Emotion-driven content platforms
AI-powered entertainment assistants
Mood-based streaming personalization
Behavioral analytics dashboards
👨‍💻 Author
Harshith Reddy
Data Science & AI Enthusiast
