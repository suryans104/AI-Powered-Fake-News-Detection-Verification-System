# AI-Powered Fake News Detection & Verification System

## 📌 About the Project

This project is an AI-powered Fake News Detection and Verification System.

The Machine Learning model is trained on a historical news dataset containing
real and fake news articles. Logistic Regression is used as the classification
algorithm, with TF-IDF used to convert news text into numerical features.

However, a Machine Learning model trained on historical data may not be able
to reliably verify newly published or breaking news. To improve this limitation,
a real-time web search and AI-based verification layer has been added.

The system combines three components:

1. 🤖 Machine Learning
   - Logistic Regression
   - TF-IDF Vectorization
   - Historical news dataset

2. 🌐 Real-Time Web Search
   - Searches the web for relevant and recent information
   - Collects supporting evidence from multiple sources

3. 🧠 AI-Based Verification
   - Gemini analyzes the user's claim together with the retrieved web evidence
   - Generates a final verdict: TRUE, FALSE, or UNCERTAIN
   - Provides a short explanation and source information

## 🔄 How It Works

User enters a news claim
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
ML Prediction
        ↓
Real-Time Web Search
        ↓
Evidence Collection
        ↓
Gemini AI Verification
        ↓
Final Verdict + Explanation + Sources

## 🎯 Why Web Search and Gemini Were Added?

The original ML model is trained on historical news data. Therefore, it has
limited knowledge of events that happened after the training data was collected.

To address this limitation, real-time web search was added to retrieve
current evidence, while Gemini is used to analyze that evidence and provide
a more understandable verification result.

This makes the project a combination of:

**Traditional Machine Learning + Real-Time Web Search + Generative AI**

## ⚠️ Disclaimer

The ML prediction is based on patterns learned from the historical training
dataset. The final verification also depends on the quality and availability
of web search results. Therefore, the system should be treated as an
assistive fact-checking tool rather than an absolute source of truth.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- TF-IDF
- Joblib
- Streamlit
- Tavily Web Search
- Google Gemini
