import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.experimental.pandas import PandasDataFrameAgent

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
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        agent = PandasDataFrameAgent(llm, df, verbose=True)

        with st.spinner("Generating answer..."):
            try:
                response = agent.run(user_question)
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error while processing your question: {e}")
else:
    st.info("Please upload a CSV file to get started.")
