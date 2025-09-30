import streamlit as st
import pandas as pd
import os
import openai

# Set API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📊 CSV Q&A with OpenAI")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    question = st.text_input("Ask a question about the dataset:")

    if question:
        # Prepare prompt
        df_preview = df.head(100).to_csv(index=False)  # limit data for token efficiency
        prompt = f"""
        You are a data analyst. Answer the user's question based on the following dataset:

        {df_preview}

        Question: {question}
        Answer:
        """

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        st.write(response['choices'][0]['message']['content'])
