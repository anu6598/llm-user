import streamlit as st
import pandas as pd
import requests
import os

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Hugging Face", layout="wide")
st.title("📊 CSV Q&A with Hugging Face")

def get_working_models():
    """List of working free models on Hugging Face"""
    return [
        {"name": "Microsoft DialoGPT-small", "model": "microsoft/DialoGPT-small"},
        {"name": "GPT-2 Small", "model": "gpt2"},
        {"name": "DistilGPT-2", "model": "distilgpt2"},
        {"name": "Facebook Blenderbot-400M", "model": "facebook/blenderbot-400M-distill"},
    ]

def query_huggingface(prompt, hf_token, model_choice):
    """Query Hugging Face API with better error handling"""
    
    model_map = {
        "Microsoft DialoGPT-small": "microsoft/DialoGPT-small",
        "GPT-2 Small": "gpt2", 
        "DistilGPT-2": "distilgpt2",
        "Facebook Blenderbot-400M": "facebook/blenderbot-400M-distill"
    }
    
    model = model_map[model_choice]
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json={
            "inputs": prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.7,
                "do_sample": True
            }
        })
        return response
    except Exception as e:
        return None

# CSV uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())
    
    # Model selection
    st.subheader("Model Settings")
    models = get_working_models()
    model_names = [m["name"] for m in models]
    selected_model = st.selectbox("Choose a model:", model_names, index=0)
    
    st.subheader("Ask questions about your dataset")
    user_question = st.text_input("Type your question here:")
    
    if user_question:
        # Get Hugging Face token
        HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
        
        if not HF_TOKEN:
            st.error("Please set your HF_TOKEN in Streamlit secrets")
            st.stop()
            
        # Prepare prompt
        prompt = f"""
        You are a data analyst assistant. Analyze this dataset and answer the question.
        
        Dataset columns: {', '.join(df.columns)}
        Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns
        
        Question: {user_question}
        
        Provide a helpful analysis based on the available data.
        """
        
        with st.spinner(f"Generating answer using {selected_model}..."):
            response = query_huggingface(prompt, HF_TOKEN, selected_model)
            
            if response is None:
                st.error("Network error occurred. Please try again.")
            elif response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        answer = result[0]['generated_text']
                    else:
                        answer = str(result[0])
                    st.success("✅ Answer:")
                    st.write(answer)
                else:
                    st.error("Unexpected response format")
            elif response.status_code == 404:
                st.error(f"❌ Model not found. This model might be temporarily unavailable.")
                st.info("Try selecting a different model from the dropdown above.")
            elif response.status_code == 503:
                st.warning("⏳ Model is loading. This is normal for free models. Please wait 20-30 seconds and try again.")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

else:
    st.info("👆 Please upload a CSV file to get started")
