import streamlit as st
import pickle
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources (one-time cached)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load trained model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Page settings
st.set_page_config(
    page_title="News Topic Classification",
    page_icon="📰",
    layout="centered"
)

# Sidebar
st.sidebar.title("📊 Model Info")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Vectorizer:** TF-IDF (bi-grams)")
st.sidebar.write("**Training Data:** HuffPost")
st.sidebar.write("**Accuracy:** ~76.7%")
st.sidebar.markdown("---")
st.sidebar.write("**Categories:**")
st.sidebar.write("- Politics")
st.sidebar.write("- Business")
st.sidebar.write("- Sports")
st.sidebar.write("- Technology")
st.sidebar.write("- Entertainment")
st.sidebar.write("- Science")
st.sidebar.write("- Health")
st.sidebar.write("- Lifestyle")

# Main UI
st.title("📰 News Topic Classification ")
st.write("Enter any news text to predict its topic with confidence scores.")

user_input = st.text_area(
    "✍️ Enter News Text",
    height=150,
    placeholder="Example: The government announced new reforms to boost the economy..."
)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        probs = model.predict_proba(vectorized)[0]
        classes = model.classes_

        # 🔹 TECHNOLOGY KEYWORD BOOST (HYBRID AI)
        tech_keywords = [
            "software", "application", "app", "mobile",
            "technology", "developer", "developers",
            "data", "security", "performance", "platform"
        ]

        if any(word in cleaned for word in tech_keywords):
            for i, cls in enumerate(classes):
                if cls == "Technology":
                    probs[i] += 0.20

        # Build results table
        df = pd.DataFrame({
            "Topic": classes,
            "Confidence (%)": (probs * 100).round(2)
        }).sort_values(by="Confidence (%)", ascending=False)

        st.subheader("📈 Prediction Confidence")
        st.bar_chart(df.set_index("Topic"))

        best_topic = df.iloc[0]["Topic"]
        best_conf = df.iloc[0]["Confidence (%)"]

        st.markdown("---")

        # Final decision
        if best_conf < 25:
            st.error("❓ Final Decision: Unknown / Other")
            st.info("Low confidence due to ambiguous or unfamiliar wording.")
        else:
            emojis = {
                "Politics": "🏛️",
                "Business": "💼",
                "Sports": "🏏",
                "Technology": "💻",
                "Entertainment": "🎬",
                "Science": "🔬",
                "Health": "🩺",
                "Lifestyle": "🏖️"
            }
            st.success(
                f"{emojis.get(best_topic,'📰')} **Final Topic:** {best_topic} ({best_conf}%)"
            )

        st.markdown("### 🧠 How to Read This")
        st.write(
            "Higher bars mean stronger association with learned patterns. "
            "If confidence is low, the system avoids overconfident predictions."
        )
