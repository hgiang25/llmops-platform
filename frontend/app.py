import streamlit as st
import requests

st.set_page_config(page_title="LLMOps Platform", layout="wide")

st.title("LLMOps Platform - Inference UI")
st.markdown("Test the **Difficulty-Aware Routing** system.")

prompt = st.text_area("Enter your prompt:")

if st.button("Send Request"):
    if prompt:
        try:
            # Call API Gateway
            response = requests.post("http://localhost:8000/chat", json={"prompt": prompt})
            if response.status_code == 200:
                data = response.json()
                st.success("Request Processed!")
                st.write("**Difficulty Score:**", data.get("difficulty_score"))
                st.write("**Response:**", data.get("response"))
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}. Is the API Gateway running?")
