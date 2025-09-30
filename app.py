import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Streamlit secret or environment variable
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Model selection
MODEL_NAME = "gpt-4o-mini"  # Change to a model your account has access to

st.set_page_config(page_title="CSV Q&A with LLM", layout="wide")
st.title("📊 CSV Q&A with LLM")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    st.subheader("Ask questions about your dataset")
    user_question = st.text_input("Type your question here:")

    if user_question:
        prompt = f"""
        You are a data assistant. Use the following CSV data to answer the user's question accurately.
        CSV DATA:
        {df.to_csv(index=False)}
        USER QUESTION: {user_question}
        """
        with st.spinner("Generating answer..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                answer = response.choices[0].message.content
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error while processing your question: {e}")
else:
    st.info("Please upload a CSV file to get started.")
