import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# LangChain imports for latest version
from langchain.chat_models import ChatOpenAI
from langchain_experimental.pandas import create_pandas_dataframe_agent


# Load environment variables from .env if present
load_dotenv()

# Get OpenAI API key from Streamlit secrets or environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
        try:
            # Initialize LLM agent
            llm = ChatOpenAI(temperature=0, model_name="gpt-4")
            agent = PandasDataFrameAgent.from_llm(llm, df, verbose=True)
            
            with st.spinner("Generating answer..."):
                response = agent.run(user_question)
                st.success("✅ Answer:")
                st.write(response)

        except Exception as e:
            st.error(f"Error while processing your question: {e}")

else:
    st.info("Please upload a CSV file to get started.")
