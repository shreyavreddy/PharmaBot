import streamlit as st
import cohere
from dotenv import load_dotenv
import os
import speech_recognition as sr
import re
import pandas as pd

# ---------- PAGE CUSTOM STYLING ----------
st.markdown("""
<style>
.main-title {font-size:2.7em;font-weight:800;color:#ED145B;margin-bottom:7px}
.sub-title {font-size:1.15em;color:#444;margin-bottom:22px}
.ml-pred {background:#F2F8FD;padding:10px 16px;border-radius:8px;margin-bottom:10px;font-size:1.1em;color:#0884fc;display:inline-block}
.advice-card {background:#F7F9FA;border-radius:10px;padding:18px 23px 14px 23px;box-shadow:0 4px 12px #0001}
</style>
""", unsafe_allow_html=True)

# ---------- ENV / API ----------
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY").strip())

# ---------- DATA ----------
@st.cache_resource
def load_symptom_data():
    csv_path = r"C:\Users\Shreya Reddy\PHARMABOT\symptom_data.csv"  # adjust if needed
    return pd.read_csv(csv_path)       # must contain columns: text , label

symptom_df = load_symptom_data()

# ---------- WORD CLEAN / STOPWORDS ----------
STOP = {
    "and","or","the","is","a","an","to","of","in","on","for","with","that",
    "this","these","those","have","has","had","be","been","being"
}
tok_pat = re.compile(r"[a-z]+")

def tokenize(s: str):
    return {w for w in tok_pat.findall(s.lower()) if len(w) >= 4 and w not in STOP}

# ---------- PREDICT BY WORD OVERLAP ----------
def predict_disease(user_input: str, df: pd.DataFrame):
    uwords = tokenize(user_input)
    max_overlap, best_label = 0, None
    for _, row in df.iterrows():
        sw = tokenize(str(row["text"]))
        ov = len(uwords & sw)
        if ov > max_overlap:
            max_overlap = ov
            best_label = row["label"]
    return best_label if max_overlap > 0 else None   # strict: at least ONE real overlap

# ---------- VALIDATE WITH COHERE ----------
def is_valid_symptom_input(co, user_input: str):
    prompt = f"""
You are a medical expert. Determine if the following text is a valid description of medical symptoms.
- Respond ONLY with 'YES' if it describes actual symptoms (e.g., 'fever headache cold').
- Respond ONLY with 'NO' if it's nonsense, questions, or not symptoms (e.g., 'how to dance', 'what is cohere').
- Do not explain.

Text: {user_input}
"""
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=5,
        temperature=0.0
    ).generations[0].text.strip()
    return response == "YES"

# ---------- STREAMLIT LAYOUT ----------
st.sidebar.image("https://img.icons8.com/color/96/pill.png", width=70)
st.sidebar.title("Important")
st.sidebar.warning("This is for educational use only. Always see a doctor.")
st.sidebar.info("Tip: Use voice input for hands-free experience.")
st.set_page_config(page_title="PharmaBot", page_icon="💊", layout="wide")
st.markdown('<div class="main-title">💊 PharmaBot – AI Health Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Describe your symptoms or use voice!</div>', unsafe_allow_html=True)

# ---------- VOICE INPUT ----------
if st.sidebar.button("🎤 Record Voice"):
    r = sr.Recognizer()
    with sr.Microphone() as src:
        st.sidebar.info("Listening …")
        try:
            audio = r.listen(src, timeout=10)
            voice_symptoms = r.recognize_google(audio)
            st.sidebar.success(f"You said: {voice_symptoms}")
        except Exception:
            st.sidebar.error("Voice recognition failed; please type.")
            voice_symptoms = None
else:
    voice_symptoms = None

# ---------- CHAT HISTORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- USER INPUT ----------
typed_symptoms = st.chat_input("Your symptoms:")
symptoms = typed_symptoms or voice_symptoms

# ---------- MAIN LOGIC ----------
if symptoms:
    s_clean = symptoms.strip()
    st.session_state.messages.append({"role": "user", "content": "🗣️ " + s_clean})
    with st.chat_message("user"):
        st.markdown("🗣️ " + s_clean)

    if len(s_clean) < 5:
        st.warning("Please provide more descriptive symptoms (≥ 5 characters).")
    else:
        with st.spinner("Validating input …"):
            is_valid = is_valid_symptom_input(co, s_clean)

        if not is_valid:
            st.error("⚠️  Input does not describe valid medical symptoms. "
                     "Please list real symptoms separated by spaces or commas "
                     "(e.g., 'fever headache cold').")
        else:
            with st.spinner("Matching symptoms …"):
                disease = predict_disease(s_clean, symptom_df)

            if disease is None:
                st.error("⚠️  Could not match your input to any known symptoms. "
                         "Please list real symptoms separated by spaces or commas "
                         "(e.g., 'fever headache cold').")
            else:
                # ---------- DISPLAY PREDICTION ----------
                st.markdown(f'<div class="ml-pred">🔎 **Predicted Disease:** {disease} '
                            f'</div>', unsafe_allow_html=True)

                # ---------- LLM PROMPT ----------
                def prompt(sym, dis):
                    return f"""
You are PharmaBot, a friendly AI health assistant for rural India.
Start with: 'This is not professional medical advice. Consult a doctor.' Do not repeat this disclaimer.

User symptoms: {sym}
Predicted disease (treat as ground truth): {dis}

Respond in this structure:
🩺 **Why It Happened (Possible Causes)**: 1-2 points.
💊 **Suggested OTC Medicines**: 1-2 Indian brands, comma-separated, no explanations.
🛡️ **Precautions and Home Remedies**: 1-2 short tips.
⚠️ **When to See a Doctor**: Warnings.

Rules: ≤ 200 words, simple empathetic English, allopathic OTC only.
"""

                with st.spinner("Generating advice …"):
                    advice = co.generate(
                        model="command",
                        prompt=prompt(s_clean, disease),
                        max_tokens=500,
                        temperature=0.7
                    ).generations[0].text.strip()

                st.session_state.messages.append({"role": "assistant", "content": advice})
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="advice-card">{advice}</div>', unsafe_allow_html=True)

                    # ---------- OTC LINKS ----------
                    otc_match = re.search(
                        r"(?:Suggested OTC Medicines|OTC Medicines):?\s*(.*?)(?=(?:Precautions|When to See a Doctor)|$)",
                        advice,
                        flags=re.I | re.S
                    )
                    if otc_match:
                        meds = [m.strip() for m in otc_match.group(1).split(",") if len(m.strip()) > 2]
                        meds = list(dict.fromkeys(meds))
                        if meds:
                            with st.expander("🛒 Buy Suggested Medicines"):
                                st.markdown("**Note:** Verify with a pharmacist before buying.")
                                for m in meds:
                                    url1 = f"https://www.pharmeasy.in/search/all?name={m.replace(' ','%20')}"
                                    url2 = f"https://www.apollopharmacy.in/search-medicines/{m.replace(' ','%20')}"
                                    st.markdown(f"- **{m}**: [PharmEasy]({url1}) | [Apollo]({url2})",
                                                unsafe_allow_html=True)

                st.warning("Remember: This is never a substitute for professional medical advice.")

st.markdown("---")
st.caption("Powered by Streamlit · Cohere · Project by SHREYA")
