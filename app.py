import streamlit as st
import pandas as pd
import requests
import json

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Hugging Face", layout="wide")
st.title("📊 CSV Q&A with Hugging Face")

def query_huggingface_chat(prompt, hf_token):
    """
    Use Hugging Face's Chat Completion API - this is the CORRECT way
    Uses models that are actually available on the inference API
    """
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # For text generation models, we use the simple inputs format
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        return response
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

# Main app
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("📊 Dataset Info")
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))

    # Question input
    st.subheader("💬 Ask Questions About Your Data")
    user_question = st.text_area("Enter your question:", height=100)

    if user_question:
        # Get HF Token
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        
        if not HF_TOKEN:
            st.error("Please add HF_TOKEN to Streamlit secrets")
            st.stop()
            
        # Prepare prompt
        prompt = f"""
        Analyze this dataset and answer the question.

        DATASET INFORMATION:
        - Columns: {list(df.columns)}
        - Total rows: {len(df)}
        - Total columns: {len(df.columns)}
        - Sample data: {df.head(2).to_string()}

        QUESTION: {user_question}

        Provide helpful analysis based on the dataset structure.
        """

        with st.spinner("🔄 Analyzing your data..."):
            response = query_huggingface_chat(prompt, HF_TOKEN)
            
            if response is None:
                st.error("Failed to connect to API")
            elif response.status_code == 200:
                result = response.json()
                st.success("✅ Analysis Results:")
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        answer = result[0]['generated_text']
                    else:
                        answer = str(result[0])
                    st.write(answer)
                else:
                    st.write("Raw response:", result)
                    
            elif response.status_code == 404:
                st.error("""
                ❌ Model not found. This usually means:
                
                1. The model isn't deployed on the free inference API
                2. The model name is incorrect
                3. The model is temporarily unavailable
                
                **Let's try a different approach...**
                """)
                
                # Alternative: Try without token for public models
                st.info("🔄 Trying public model access...")
                try_public_model(prompt)
                
            elif response.status_code == 503:
                st.warning("""
                ⏳ Model is loading. This is normal for free Hugging Face models.
                Please wait 20-30 seconds and try again.
                """)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

else:
    st.info("👆 Please upload a CSV file to get started")

def try_public_model(prompt):
    """Try accessing a model without authentication"""
    # Some models are available without tokens
    public_models = [
        "https://api-inference.huggingface.co/models/gpt2",
        "https://api-inference.huggingface.co/models/distilgpt2",
    ]
    
    for model_url in public_models:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {"max_length": 200}
            }
            
            response = requests.post(model_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Got response from public model!")
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get('generated_text', str(result[0]))
                    st.write(answer)
                return True
                
        except Exception as e:
            continue
    
    st.error("No public models available. Try the OpenRouter option below.")
    return False
