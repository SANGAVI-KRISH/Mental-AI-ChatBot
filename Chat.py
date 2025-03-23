import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import uuid
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from textblob import TextBlob
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# 🎨 Set Page Configuration
st.set_page_config(page_title="AI Mental Health Companion", page_icon="💙", layout="wide")
st.title("💙 AI Mental Health Companion")

# 📌 Sidebar with Daily Tip
st.sidebar.header("🌱 Daily Wellness Tip")
daily_tips = [
    "Take a deep breath. Inhale for 4 seconds, hold for 4, exhale for 6.",
    "Write down 3 things you're grateful for today. 💙",
    "Listen to calming music for 5 minutes. 🎶",
    "Take a short walk to refresh your mind. 🚶‍♂️",
    "Drink a glass of water and stay hydrated. 💧",
    "Stretch for a few minutes to relax your body. 🧘‍♂️"
]

# Allow users to cycle through daily tips
if "daily_tip_index" not in st.session_state:
    st.session_state.daily_tip_index = hash(str(datetime.date.today())) % len(daily_tips)

if st.sidebar.button("🔄 New Tip"):
    st.session_state.daily_tip_index = (st.session_state.daily_tip_index + 1) % len(daily_tips)

st.sidebar.write(f"💡 {daily_tips[st.session_state.daily_tip_index]}")

# 🌿 Store Chat History & Mood Log
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

# 🎤 Speech-to-Text Input Option
recognizer = sr.Recognizer()

def voice_input():
    """Captures and returns speech-to-text input."""
    with sr.Microphone() as source:
        st.sidebar.write("🎤 Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "I couldn't understand. Could you try again?"
        except sr.RequestError:
            return "Speech recognition service is unavailable."

# 🌈 Sentiment Analysis Function
def analyze_sentiment(text):
    """Analyzes sentiment and categorizes it as stressed, neutral, or relaxed."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # -1 (negative) to +1 (positive)
    
    if polarity < -0.3:
        return "stressed"
    elif polarity > 0.3:
        return "relaxed"
    else:
        return "neutral"

# 🤖 Define Chatbot Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a mental health assistant. Provide relaxation tips based on the user's stress level."),
    ("user", "User input: {query}")
])
llm = OllamaLLM(model="llama3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# 🎭 User Input Section
st.write("How are you feeling today?")
input_method = st.radio("Choose input method:", ["Text", "Voice 🎙️"])

if input_method == "Voice 🎙️":
    input_txt = voice_input()
    st.write(f"🗣️ You said: {input_txt}")
else:
    input_txt = st.text_input("Type your feelings here...")

# 🔄 Process Input
if input_txt and input_txt.lower() not in ["", "sorry, i couldn't hear you clearly.", "i couldn't understand. could you try again?", "speech recognition service is unavailable."]:
    sentiment = analyze_sentiment(input_txt)
    result = chain.invoke({"query": input_txt})

    # Store chat and mood log
    st.session_state.chat_history.append(("You", input_txt))
    st.session_state.chat_history.append(("AI", result))
    st.session_state.mood_log.append((input_txt, sentiment))

    # Mood display emojis
    mood_emojis = {"stressed": "😟", "relaxed": "😊", "neutral": "😐"}
    sentiment_display = f"{sentiment} {mood_emojis.get(sentiment, '')}"

    # 💬 Chat Display
    st.subheader("🗨️ Chat History")
    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.write(f"**{role}:** {message}")

    # 🔊 Text-to-Speech Output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts = gTTS(result)
        tts.save(temp_audio.name)
        st.audio(temp_audio.name)
    os.remove(temp_audio.name)  # Cleanup audio after playing

# 📈 Mood Tracking Visualization
st.sidebar.subheader("📊 Mood Tracker")
if len(st.session_state.mood_log) > 1:
    df = pd.DataFrame(st.session_state.mood_log, columns=["Mood", "Sentiment"])
    
    # Ensure consistent sentiment categories for plotting
    sentiment_order = ["stressed", "neutral", "relaxed"]
    df["Sentiment"] = pd.Categorical(df["Sentiment"], categories=sentiment_order, ordered=True)

    fig, ax = plt.subplots()
    df["Sentiment"].value_counts().reindex(sentiment_order).plot(kind="bar", ax=ax, color=["red", "yellow", "green"])
    st.sidebar.pyplot(fig)

# 🌡️ Stress Level Gauge
st.sidebar.subheader("📊 Stress Level Indicator")
if len(st.session_state.mood_log) > 1:
    stress_levels = {"stressed": -1, "neutral": 0, "relaxed": 1}
    avg_stress = sum(stress_levels[s] for _, s in st.session_state.mood_log) / len(st.session_state.mood_log)
    
    fig, ax = plt.subplots(figsize=(2, 0.5))
    ax.barh([0], [avg_stress], color="red" if avg_stress < -0.2 else "yellow" if avg_stress < 0.2 else "green")
    ax.set_xlim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["😟 Stressed", "😐 Neutral", "😊 Relaxed"])
    st.sidebar.pyplot(fig)

# 🌀 **Human Breathing Exercise GIF**
st.sidebar.subheader("🧘‍♂️ Try a Relaxing Breathing Exercise")

# 📝 Footer
st.sidebar.markdown("---")
st.sidebar.markdown("💙 Built for your mental well-being. Stay calm & relaxed! 🌿")