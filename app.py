import streamlit as st
import pandas as pd
import os
import openai

# -------------------- OpenAI Setup --------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.error("Please add your OPENAI_API_KEY in Streamlit Secrets!")
    st.stop()

openai.api_key = OPENAI_API_KEY

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="CSV Q&A with LLM", layout="wide")
st.title("📊 CSV Q&A with LLM (Token Efficient)")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    # Precompute summaries for LLM
    st.subheader("Generating summary for LLM...")
    summary = df.describe(include='all').to_string()
    user_device_summary = df.groupby("user_id")["device_license_split"].unique().to_dict()
    device_lessons_summary = df.groupby("device_license_split")["total_lessons"].sum().to_dict()

    summary_text = f"""
CSV Summary:
{summary}

User -> Devices mapping:
{user_device_summary}

Device -> Total Lessons:
{device_lessons_summary}
"""
    st.text_area("Summary sent to LLM", summary_text, height=200)

    # User question input
    user_question = st.text_input("Ask your question about the CSV dataset:")

    if user_question:
        st.subheader("LLM Answer")
        with st.spinner("Processing your question..."):
            try:
                # GPT prompt
                prompt = f"""
You are a data analyst. Use the following CSV summary to answer the question.
Do not assume anything outside the data.

CSV SUMMARY:
{summary_text}

QUESTION:
{user_question}

Answer concisely and clearly.
"""
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # use smaller model to avoid TPM issues
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                answer = response.choices[0].message.content
                st.success(answer)

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file to get started.")
