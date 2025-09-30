import streamlit as st
import pandas as pd
import requests
import os
import time

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Hugging Face", layout="wide")
st.title("📊 CSV Q&A with Hugging Face")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def query_huggingface(prompt, hf_token, max_retries=3):
    """Query Hugging Face API with retry logic"""
    
    # Using a more reliable free model
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json={
                "inputs": prompt,
                "parameters": {
                    "max_length": 400,
                    "temperature": 0.1,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            })
            
            if response.status_code == 200:
                result = response.json()
                return result
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    st.info(f"Model is loading... waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"error": "Model is still loading. Please try again in a minute."}
            else:
                return {"error": f"API Error {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"error": f"Request failed: {str(e)}"}
    
    return {"error": "Max retries exceeded"}

# CSV uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    # Show basic dataset info
    with st.expander("Dataset Overview"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.write("**Columns:**", ", ".join(df.columns.tolist()))

    st.subheader("Ask questions about your dataset")
    user_question = st.text_area("Type your question here:", height=100, 
                                placeholder="E.g., What are the main trends in this data? What insights can you derive from the numeric columns?")

    if user_question:
        # Prepare prompt
        prompt = f"""
        You are a data analyst assistant. Analyze the following dataset and provide insights.

        Dataset columns: {', '.join(df.columns)}
        Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns

        Question: {user_question}

        Please provide a concise, data-driven answer. If the question requires specific calculations, explain what you would calculate and why.
        """

        with st.spinner("Analyzing your data... (This may take 10-30 seconds for free models)"):
            # Get Hugging Face token from secrets
            HF_TOKEN = st.secrets.get("HF_TOKEN")
            
            if not HF_TOKEN:
                st.error("""
                🔐 Hugging Face token not found. Please add it to your Streamlit secrets:
                
                1. Go to https://huggingface.co/settings/tokens
                2. Create a free token (no payment needed)
                3. Add it to your Streamlit Cloud secrets as `HF_TOKEN`
                """)
                st.stop()
            
            result = query_huggingface(prompt, HF_TOKEN)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                
                if "loading" in result['error'].lower():
                    st.info("""
                    💡 **Tip for free Hugging Face models:** 
                    - Models spin down after inactivity
                    - First request may take 20-30 seconds to wake up
                    - Subsequent requests will be faster
                    - Try again in 30 seconds!
                    """)
            else:
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        answer = result[0]['generated_text']
                    else:
                        # Handle different response formats
                        answer = str(result[0])
                    
                    st.success("✅ Analysis Results:")
                    st.write(answer)
                    
                    # Add some helpful follow-up
                    st.info("💡 **Follow-up ideas:** Try asking about specific columns, trends, or correlations in your data.")
                else:
                    st.error("Unexpected response format from the model")

else:
    st.info("👆 Please upload a CSV file to get started")
    st.markdown("""
    ### How to use:
    1. Upload a CSV file
    2. Ask questions about your data
    3. Get AI-powered insights
    
    ### Example questions:
    - "What are the main trends in this data?"
    - "Which columns have the strongest correlation?"
    - "What insights can you derive from the numeric columns?"
    - "Are there any data quality issues I should know about?"
    """)
