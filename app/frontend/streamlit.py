import streamlit as st
import requests

API_URL = "http://localhost:8000/classify"

# --- Settings explanations ---
SETTINGS_EXPLANATIONS = {
    "temperature": "Controls randomness in the model's output. Lower = more deterministic, Higher = more creative.",
    "reasoning_effort": "Specifies how much cognitive effort the model should use for reasoning tasks.",
    "verbosity": "Controls the length/detail of the model's responses.",
    "web": "Indicates whether the model should optionally use web information or stay offline."
}

# --- Streamlit UI ---
st.title("Real-Time Prompt Classifier")

prompt = st.text_area("Type your prompt:", height=150)

if prompt:
    with st.spinner("Classifying..."):
        response = requests.post(API_URL, json={"prompt": prompt})
        data = response.json()

    # Display predicted categories
    st.subheader("Predicted Categories")
    for cat in data.get("categories", []):
        name = cat["name"]
        confidence = cat["confidence"]
        st.write(f"**{name}** — confidence: {confidence:.2f}")

    # Display recommended settings with explanations
    st.subheader("Recommended Settings")
    for key, value in data.get("settings", {}).items():
        explanation = SETTINGS_EXPLANATIONS.get(key, "No explanation available.")
        st.write(f"**{key}**: {value}")
        st.caption(f"Explanation: {explanation}")


