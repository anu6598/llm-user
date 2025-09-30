import streamlit as st
import pandas as pd
import requests
import json

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Free AI", layout="wide")
st.title("📊 CSV Q&A with Free AI")

def query_openrouter(prompt):
    """Use OpenRouter free tier - more reliable"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer free",  # Free tier
                "HTTP-Referer": "https://your-app.com",  # Required but can be any URL
                "X-Title": "CSV Analyzer"  # Required but can be any title
            },
            json={
                "model": "google/gemma-7b-it:free",  # Free model
                "messages": [
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            },
            timeout=30
        )
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
        You are a data analyst assistant.
        The dataset has the following columns: {', '.join(df.columns)}.
        Dataset preview: {df.head().to_string()}
        
        Answer this question based on the dataset: {user_question}
        
        Provide specific insights and analysis. If you need calculations, explain what should be calculated.
        """
        
        with st.spinner("Analyzing your data with AI..."):
            response = query_openrouter(prompt)
            
            if response is None:
                st.error("Network error. Please check your connection and try again.")
            elif response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                st.success("✅ Analysis Results:")
                st.write(answer)
            elif response.status_code == 429:
                st.warning("⚠️ Free tier limit reached. Please try again in a few minutes.")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

else:
    st.info("👆 Please upload a CSV file to get started")
