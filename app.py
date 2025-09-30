# app.py
import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# ------------------ Config ------------------
st.set_page_config(page_title="CSV Q&A with LLM", layout="wide")
st.title("📊 CSV Q&A with OpenAI")

# Load OpenAI API key from Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Please set your OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------ CSV Upload ------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    # ------------------ User Question ------------------
    st.subheader("Ask questions about your dataset")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("Generating answer..."):
            try:
                # Construct a prompt including column names for context
                columns = ", ".join(df.columns)
                prompt = f"""
You are a data analyst assistant. Answer questions about the following CSV dataset.
Columns: {columns}
Data (first 5 rows for context):
{df.head().to_csv(index=False)}

Question: {user_question}
Provide a clear answer based on the data.
"""
                # Call OpenAI API (v2)
                response = client.chat.completions.create(
                    model="gpt-4",
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
