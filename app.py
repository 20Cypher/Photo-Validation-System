import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000/validate/"

st.title("Photo Validation System")
st.write("Upload an image to check if it meets the official requirements.")

uploaded_file = st.file_uploader("Upload a Photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Validate Image"):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with open("temp_image.jpg", "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            results = response.json()
            st.subheader("Validation Results")
            for key, value in results.items():
                st.write(f"{key}: {value}")
        else:
            st.error("Failed to process the image. Please try again.")
