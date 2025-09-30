import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env (optional)
load_dotenv()

# Set API key from Streamlit secrets or environment
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY in environment variables or Streamlit secrets.")
else:
    openai.api_key = OPENAI_API_KEY

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with OpenAI", layout="wide")
st.title("📊 CSV Q&A with OpenAI")

# CSV uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    st.subheader("Ask questions about your dataset")
    user_question = st.text_input("Type your question here:")

    if user_question and OPENAI_API_KEY:
        # Prepare prompt
        prompt = f"""
        You are a data analyst assistant.
        The dataset has the following columns: {', '.join(df.columns)}.
        Answer the following question based on the dataset. Use precise numbers and examples where possible.

        Question: {user_question}
        """

        with st.spinner("Generating answer..."):
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # you can change to "gpt-4" if you have access
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                answer = response.choices[0].message.content
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error while processing your question: {e}")
else:
    st.info("Please upload a CSV file to get started.")
