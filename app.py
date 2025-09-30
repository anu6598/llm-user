import streamlit as st
import pandas as pd
import requests
import os

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Hugging Face", layout="wide")
st.title("📊 CSV Q&A with Hugging Face")

def query_huggingface_free(prompt):
    """
    Use Hugging Face Inference API with FREE models that actually work
    No token required for some public models
    """
    try:
        # Using a public model that doesn't require authentication
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        
        response = requests.post(API_URL, json={
            "inputs": prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }, timeout=30)
        
        return response
    except Exception as e:
        return None

# CSV uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())
    
    st.subheader("Ask questions about your dataset")
    user_question = st.text_input("Type your question here:")
    
    if user_question:
        # Prepare prompt
        prompt = f"""
        As a data analyst, analyze this dataset with columns: {list(df.columns)}
        Data shape: {len(df)} rows, {len(df.columns)} columns
        
        Question: {user_question}
        
        Provide analytical insights based on the dataset structure.
        """
        
        with st.spinner("Analyzing with AI... (Free model - may take 20 seconds)"):
            response = query_huggingface_free(prompt)
            
            if response is None:
                st.error("Network error occurred")
            elif response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        answer = result[0]['generated_text']
                        st.success("✅ Analysis:")
                        st.write(answer)
                    else:
                        st.write("**Raw response:**", result)
                else:
                    st.error("Unexpected response format")
            elif response.status_code == 503:
                st.warning("""
                ⏳ Model is loading... This is normal for free Hugging Face models.
                **Please wait 20 seconds and try again** - the model needs to wake up.
                """)
            else:
                st.error(f"API Error {response.status_code}. Try the local analysis option below.")

else:
    st.info("👆 Please upload a CSV file to get started")
