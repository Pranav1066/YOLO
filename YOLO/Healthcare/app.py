import streamlit as st
import pandas as pd
import datetime
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage  # Corrected import
import random
from PIL import Image

# Mock for voice and image processing
def mock_speech_to_text(audio_data):
    return "I have a fever and headache."

def mock_image_diagnosis(image):
    return "Possible Skin Rash: Eczema"

# Initialize ChatGroq for Groq-accelerated inference
llm = ChatGroq(
    temperature=0,
    groq_api_key="",
    model="llama-3.1-70b-versatile"
)

if "user_data" not in st.session_state:
    st.session_state["user_data"] = {
        "name": "Guest",
        "age": 25,
        "gender": "Prefer not to say",
        "medical_history": [],
        "chat_history": [],
    }

# Sidebar for User Profile
with st.sidebar:
    st.header("User Profile")
    name = st.text_input("Name", st.session_state["user_data"]["name"])
    age = st.slider("Age", 0, 100, st.session_state["user_data"]["age"])
    gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
    update = st.button("Update Profile")
    if update:
        st.session_state["user_data"].update({"name": name, "age": age, "gender": gender})
        st.success("Profile Updated!")

    st.header("Health History")
    chat_df = pd.DataFrame(st.session_state["user_data"]["chat_history"], columns=["Date", "Symptoms", "Conditions"])
    st.dataframe(chat_df)

# Main Interface
st.title("AI Health Ecosystem")
st.subheader("Describe Your Symptoms")

# Input Modes
mode = st.selectbox("Input Mode", ["Text", "Voice", "Image"])
symptoms = ""
submit_symptoms = False  # Initialize submit_symptoms as False

if mode == "Text":
    with st.form("symptom_form"):
        symptoms = st.text_area("Enter your symptoms...")
        submit_symptoms = st.form_submit_button("Submit")
elif mode == "Voice":
    st.write("Record your voice:")
    uploaded_audio = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if uploaded_audio:
        symptoms = mock_speech_to_text(uploaded_audio)
        st.write(f"Detected Symptoms: {symptoms}")
        submit_symptoms = st.button("Submit")
elif mode == "Image":
    uploaded_image = st.file_uploader("Upload Image of Symptom", type=["jpg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        symptoms = mock_image_diagnosis(image)
        st.write(f"Detected Condition: {symptoms}")
        submit_symptoms = st.button("Submit")

# Diagnosis and AI Suggestions
if submit_symptoms and symptoms:
    user_prompt = f"""
    You are an expert medical AI. Based on the patient's symptoms: '{symptoms}', 
    their age: {age}, gender: {gender}, and medical history: {st.session_state['user_data']['medical_history']},
    provide possible diagnoses and recommendations.
    """

    # Properly structured message input for ChatGroq
    messages = [
        SystemMessage(content="You are a highly skilled medical assistant."),
        HumanMessage(content=user_prompt)
    ]

    ai_response = llm(messages)

    # Mock Symptom Mapping and Remedies
    conditions = random.choices(["Flu", "COVID-19", "Asthma"], k=2)
    remedies = ["Rest", "Drink fluids", "Consult a specialist"]

    st.session_state["user_data"]["chat_history"].append((datetime.datetime.now(), symptoms, conditions))

    st.write(f"### Possible Conditions:")
    for condition in set(conditions):
        st.write(f"- {condition}")

    st.write(f"### Suggested Remedies:")
    for remedy in set(remedies):
        st.write(f"- {remedy}")

    st.write("### AI Doctor's Full Analysis:")
    st.write(ai_response.content)

# Community Health Insights
st.subheader("Community Health Trends")
location = st.text_input("Enter your location", "Bangalore")
if location:
    st.write(f"Disease trends in {location}:")
    st.bar_chart({"Flu": 50, "COVID-19": 30, "Dengue": 20})

st.subheader("Book an Appointment")
if st.button("Find Specialists Near You"):
    st.write("Suggested doctors:")
    doctors = [f"Dr. {name} - {random.randint(1, 5)} km away" for name in ["Patel", "Singh", "Gupta"]]
    for doctor in doctors:
        st.write(f"- {doctor}")