import streamlit as st
import cohere
from dotenv import load_dotenv
import os
import speech_recognition as sr  # For voice input
import re  # For parsing medicines
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(cohere_api_key)

# Load and train ML model (runs once on app start)
@st.cache_resource  # Cache for efficiency
def train_ml_model():
    # Use absolute path to ensure it finds the CSV
    csv_path = r"C:\Users\Shreya Reddy\PHARMABOT\symptom_data.csv"  # Matches your provided path
    data = pd.read_csv(csv_path)  # Assumes columns: 'text' (symptoms), 'label' (disease)
    X = data['text']  # Updated to match actual dataset column
    y = data['label']
    
    # Split and train simple classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    model.fit(X_train, y_train)
    return model

ml_model = train_ml_model()

# Function for refined prompt (updated to avoid repetition and ensure completeness)
def generate_prompt(symptoms, ml_prediction):
    return f"""
You are PharmaBot, a friendly AI health assistant for rural India. Start with: 'This is not professional medical advice. Consult a doctor.' Do not repeat this disclaimer.

User symptoms: {symptoms}
ML-predicted condition: {ml_prediction} (for reference only)

Respond in this structure:
1. **Possible Causes**: 1-2 simple reasons.
2. **Suggested OTC Medicines**: 1-2 Indian brands.
3. **Home Remedies**: 1-2 easy tips.
4. **When to See a Doctor**: Warnings.

Rules: Simple English, empathetic, under 200 words. For serious symptoms, advise doctor immediately. End with a brief summary.
"""

# Streamlit app config
st.set_page_config(page_title="💊 PharmaBot", page_icon="💊", layout="wide")
st.title("💊 PharmaBot - AI Health Assistant")
st.markdown("Describe your symptoms in simple words or use voice input. I'll provide helpful advice!")

# Sidebar for notes and voice input
st.sidebar.title("Important")
st.sidebar.markdown("**Note:** This is for educational use only. Always see a doctor.")
if st.sidebar.button("🎤 Record Voice"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.info("Listening... Speak your symptoms (say 'stop' to end).")
        try:
            audio = recognizer.listen(source, timeout=10)  # Listen for up to 10 seconds
            symptoms = recognizer.recognize_google(audio)  # Convert to text using Google API
            st.sidebar.success(f"You said: {symptoms}")
        except sr.UnknownValueError:
            st.sidebar.error("Sorry, couldn't understand the audio. Try again.")
            symptoms = None
        except sr.RequestError:
            st.sidebar.error("Voice service unavailable. Please type instead.")
            symptoms = None
else:
    symptoms = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input as chat (text or from voice)
text_symptoms = st.chat_input("Your symptoms:")
if text_symptoms:
    symptoms = text_symptoms  # Prioritize text input if provided

if symptoms:
    # Validate input
    if len(symptoms.strip()) < 5:
        st.warning("Please provide more details about your symptoms.")
    else:
        st.session_state.messages.append({"role": "user", "content": symptoms})
        with st.chat_message("user"):
            st.markdown(symptoms)
        
        # ML Prediction
        with st.spinner("Predicting with ML..."):
            ml_prediction = ml_model.predict([symptoms])[0]
            st.info(f"ML Prediction: Possible {ml_prediction} (experimental— not a diagnosis)")
        
        # Generate advice with ML input (updated params for complete responses)
        with st.spinner("Analyzing..."):
            response = co.generate(model='command', prompt=generate_prompt(symptoms, ml_prediction), max_tokens=500, temperature=0.7)
            advice = response.generations[0].text.strip()
            
            # Fallback
            if not advice or len(advice) < 50:
                advice = "Sorry, couldn't analyze. Try again or see a doctor."
        
        st.session_state.messages.append({"role": "assistant", "content": advice})
        with st.chat_message("assistant"):
            st.markdown(advice)
            
            # Flexible extraction for OTC section
            otc_section = re.search(r"(?:Suggested OTC Medicines|OTC Medicines):?\s*(.*?)(?=(?:Home Remedies|When to See a Doctor)|\Z)", advice, re.DOTALL | re.IGNORECASE)
            if otc_section:
                otc = otc_section.group(1).strip()
                medicines = re.findall(r'([A-Za-z0-9\s]+(?:\s*\([A-Za-z0-9\s]+\))?)', otc)  # Catches names like "Ibuprofen (Advil)"
                medicines = [med.strip() for med in medicines if len(med.strip()) > 5 and 'such as' not in med]  # Filter noise
                if medicines:
                    with st.expander("🛒 Buy Suggested Medicines (External Links)"):
                        st.markdown("**Note:** These links open trusted sites—verify availability and consult a pharmacist before purchase.")
                        for med in medicines:
                            search_url = f"https://www.pharmeasy.in/search/all?name={med.replace(' ', '%20')}"
                            st.markdown(f"- {med}: [Buy on PharmEasy]({search_url})", unsafe_allow_html=True)
        
        st.warning("Remember: AI-generated advice. Consult a professional.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit, Cohere, and scikit-learn. Project by [Your Name].")
