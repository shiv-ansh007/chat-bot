import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# Streamlit UI Setup
st.set_page_config(page_title="Titanic Chatbot", layout="centered")

st.title("🚢 Titanic Dataset Chatbot")
st.write("Ask questions about the Titanic dataset and get insights!")

# User Input
user_input = st.text_input("Enter your question:", placeholder="e.g., What percentage of passengers were male?")

# Backend API URL
API_URL = "http://localhost:8000/ask"

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = requests.get(API_URL, params={"query": user_input})
            if response.status_code == 200:
                data = response.json()
                
                # Display Text Response
                if data["response"]:
                    st.success(f"**Chatbot:** {data['response']}")
                
                # Display Image if Available
                if data["image"]:
                    img_data = base64.b64decode(data["image"])
                    img = Image.open(BytesIO(img_data))
                    st.image(img, caption="Visualization", use_column_width=True)
            else:
                st.error("Something went wrong. Please try again.")
    else:
        st.warning("Please enter a question!")
