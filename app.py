import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# or if using streamlit secrets directly
import streamlit as st
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Load environment variables from .env if present
load_dotenv()

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY in environment variables or a .env file.")
else:
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

    if user_question and OPENAI_API_KEY:
        # Initialize LLM agent
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        agent = create_pandas_dataframe_agent(llm, df, verbose=True)
        
        with st.spinner("Generating answer..."):
            try:
                response = agent.run(user_question)
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error while processing your question: {e}")
else:
    st.info("Please upload a CSV file to get started.")
