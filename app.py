import streamlit as st
import pandas as pd
import openai
import os

# ------------------- CONFIG -------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # make sure you set this in your env

st.set_page_config(page_title="CSV Q&A", layout="wide")
st.title("📊 Ask Questions About Your CSV")

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # ------------------- USER QUESTION -------------------
    user_q = st.text_input("Ask a question about the data:")

    if st.button("Get Answer") and user_q:
        # Build context: describe dataset columns
        context = f"The dataset has {len(df)} rows and {len(df.columns)} columns.\n"
        context += "Columns are: " + ", ".join(df.columns) + ".\n"

        # Add a small sample of the data
        sample_data = df.head(5).to_dict(orient="records")
        context += f"Here are the first 5 rows: {sample_data}\n"

        # Send to OpenAI
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Answer based only on the provided dataset sample."},
                    {"role": "user", "content": context + "\n\nQuestion: " + user_q}
                ],
                max_tokens=500
            )
            answer = response.choices[0].message.content
            st.success(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")
