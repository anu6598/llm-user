import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="CSV Q&A", layout="wide")
st.title("📊 CSV Q&A with AI")

# Simple one-model approach
def query_simple(prompt, hf_token):
    """Use only one reliable model"""
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 300}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        return response
    except:
        return None

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    question = st.text_input("Ask about your data:")
    
    if question:
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        
        if not HF_TOKEN:
            st.error("Add HF_TOKEN to Streamlit secrets")
            st.stop()
            
        prompt = f"""
        Dataset columns: {list(df.columns)}
        Data shape: {df.shape}
        Question: {question}
        
        Analyze and provide insights:
        """
        
        with st.spinner("Analyzing..."):
            response = query_simple(prompt, HF_TOKEN)
            
            if response and response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    answer = result[0].get('generated_text', 'No response')
                    st.success("Answer:")
                    st.write(answer)
            else:
                st.error("Error. Make sure your Hugging Face token is valid.")
