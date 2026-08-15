import streamlit as st
import joblib
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai


load_dotenv()


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #777;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    border: 1px solid #ddd;
}

.source-box {
    padding: 12px;
    border-radius: 8px;
    background-color: #f7f7f7;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.markdown(
    '<div class="title"> AI Fake News Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning + Web Search + Generative AI By Suryans</div>',
    unsafe_allow_html=True
)

# =========================
# LOAD ML MODEL
# =========================

@st.cache_resource
def load_model():

    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


logistic_model, tfidf_vectorizer = load_model()

# =========================
# API CLIENTS
# =========================

# Put your API keys here

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# =========================
# NEWS INPUT
# =========================

st.subheader("Enter News or Claim")

news = st.text_area(
    "Paste the news/claim you want to verify:",
    height=150,
    placeholder="XYZ"
)

# =========================
# CHECK BUTTON
# =========================

if st.button("🔍 Check News", use_container_width=True):

    if news.strip() == "":
        st.warning("Please enter a news claim first.")

    else:

        # =========================
        # ML PREDICTION
        # =========================

        with st.spinner("Running machine learning model..."):

            X_news = tfidf_vectorizer.transform([news])

            ml_prediction = logistic_model.predict(X_news)[0]

            if ml_prediction == 0:
                ml_result = "FAKE"
            else:
                ml_result = "REAL"

        # =========================
        # WEB SEARCH
        # =========================

        with st.spinner("Searching latest information on the web..."):

            response = tavily.search(
                query=news,
                search_depth="advanced",
                max_results=3
            )

        evidence = ""

        for result in response["results"]:

            evidence += f"""
Source: {result['title']}
URL: {result['url']}
Content: {result['content'][:1000]}
"""

        # =========================
        # GEMINI
        # =========================

        with st.spinner("AI is analyzing the evidence..."):

            prompt = f"""
You are an AI fact-checking assistant.

User Claim:
{news}

Machine Learning Prediction:
{ml_result}

Web Evidence:
{evidence}

Analyze the claim using the provided web evidence.

Return the answer in exactly this structure:

FINAL VERDICT: TRUE, FALSE, or UNCERTAIN

ML PREDICTION: {ml_result}

EXPLANATION:
Give a clear and simple explanation of why the claim is true, false,
or uncertain.

SOURCES:
List the important sources used.
"""

            interaction = client.interactions.create(
                model="gemini-3.1-flash-lite",
                input=prompt
            )

            final_answer = interaction.output_text

        # =========================
        # RESULTS
        # =========================

        st.divider()

        st.subheader("🔎 Fact Check Result")

        st.markdown(
            f"""
            <div class="result-box">
            <b>Machine Learning Prediction:</b> {ml_result}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("🤖 AI Analysis")

        st.write(final_answer)

        # =========================
        # WEB SOURCES
        # =========================

        st.subheader("🌐 Web Sources")

        for result in response["results"]:

            st.markdown(
                f"""
                <div class="source-box">
                <b>{result['title']}</b><br>
                <a href="{result['url']}" target="_blank">
                Open Source
                </a>
                </div>
                """,
                unsafe_allow_html=True
            )