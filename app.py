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

.verdict-card {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 22px;
    border-radius: 14px;
    margin: 15px 0 20px 0;
    border: 1px solid rgba(255,255,255,0.15);
}

.true-card {
    background: rgba(46, 204, 113, 0.12);
    border-left: 6px solid #2ecc71;
}

.false-card {
    background: rgba(231, 76, 60, 0.12);
    border-left: 6px solid #e74c3c;
}

.uncertain-card {
    background: rgba(241, 196, 15, 0.12);
    border-left: 6px solid #f1c40f;
}

.verdict-icon {
    font-size: 38px;
}

.verdict-label {
    font-size: 13px;
    color: #999;
    font-weight: 600;
    letter-spacing: 1px;
}

.verdict-text {
    font-size: 30px;
    font-weight: 800;
    margin-top: 2px;
}

.info-card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 25px;
}

.card-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
}

.ml-result {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    background: rgba(100,100,255,0.15);
    font-weight: 700;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

.explanation-card {
    padding: 22px;
    border-radius: 14px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.10);
    line-height: 1.8;
    font-size: 16px;
    margin-bottom: 25px;
}

.source-card {
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 10px;
}

.source-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(100,100,255,0.15);
    font-weight: 700;
}

.source-title {
    font-weight: 600;
    margin-bottom: 5px;
}

.source-content a {
    text-decoration: none;
    font-size: 14px;
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
Keep the explanation concise and easy to read.
Use 3-5 short sentences maximum.
Do not repeat the verdict.
Do not repeat the ML prediction.
Do not include unnecessary details.
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

# -------------------------
# VERDICT
# -------------------------

if "FINAL VERDICT: TRUE" in final_answer.upper():
    verdict = "TRUE"
    verdict_icon = "✅"
elif "FINAL VERDICT: FALSE" in final_answer.upper():
    verdict = "FALSE"
    verdict_icon = "❌"
else:
    verdict = "UNCERTAIN"
    verdict_icon = "⚠️"

# -------------------------
# RESULT CARD
# -------------------------

if verdict == "TRUE":
    verdict_class = "true-card"
elif verdict == "FALSE":
    verdict_class = "false-card"
else:
    verdict_class = "uncertain-card"

st.markdown(
    f"""
    <div class="verdict-card {verdict_class}">
        <div class="verdict-icon">{verdict_icon}</div>
        <div>
            <div class="verdict-label">FINAL VERDICT</div>
            <div class="verdict-text">{verdict}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# ML PREDICTION
# -------------------------

st.markdown(
    f"""
    <div class="info-card">
        <div class="card-title">🤖 Machine Learning Prediction</div>
        <div class="ml-result">{ml_result}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# AI ANALYSIS
# -------------------------

st.markdown(
    """
    <div class="section-title">🧠 AI Analysis</div>
    """,
    unsafe_allow_html=True
)

# Remove verdict and ML prediction from Gemini output
clean_answer = final_answer

clean_answer = clean_answer.replace(
    "FINAL VERDICT: TRUE", ""
).replace(
    "FINAL VERDICT: FALSE", ""
).replace(
    "FINAL VERDICT: UNCERTAIN", ""
)

clean_answer = clean_answer.replace(
    f"ML PREDICTION: {ml_result}", ""
)

# Remove SOURCES section because sources are shown separately
if "SOURCES:" in clean_answer:
    clean_answer = clean_answer.split("SOURCES:")[0]

clean_answer = clean_answer.replace("EXPLANATION:", "").strip()

st.markdown(
    f"""
    <div class="explanation-card">
        {clean_answer}
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# WEB SOURCES
# -------------------------

st.markdown(
    """
    <div class="section-title">🌐 Web Sources</div>
    """,
    unsafe_allow_html=True
)

for i, result in enumerate(response["results"], 1):

    st.markdown(
        f"""
        <div class="source-card">

            <div class="source-number">{i}</div>

            <div class="source-content">

                <div class="source-title">
                    {result['title']}
                </div>

                <a href="{result['url']}" target="_blank">
                    🔗 Open Source
                </a>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )