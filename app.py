# app.py
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="📊 CSV Analyzer with LLM", layout="wide")
st.title("📊 CSV Analyzer with LLM")

# -----------------------------
# CSV Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    @st.cache_data
    def load_csv(file):
        df = pd.read_csv(file)
        # Ensure numeric columns are correct
        for col in ['total_lessons', 'unique_lesson_ids_watched']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        return df

    df = load_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    st.write("Columns detected:", df.columns.tolist())

    # -----------------------------
    # Consistent Users
    # -----------------------------
    if 'total_lessons' in df.columns:
        st.subheader("Users consistent with >200 total_lessons each month")
        consistent_users = df.groupby('user_id').filter(lambda x: (x['total_lessons'] > 200).all())
        if consistent_users.empty:
            st.write("No users found with >200 lessons in all months.")
        else:
            st.dataframe(consistent_users[['user_id', 'year_month', 'total_lessons']])

    # -----------------------------
    # Device bifurcation
    # -----------------------------
    if 'device_lessons_bifurcation' in df.columns:
        st.subheader("Device bifurcation for each user")
        st.dataframe(df[['user_id', 'device_lessons_bifurcation']])

    # -----------------------------
    # Limit name counts
    # -----------------------------
    if 'limit_name_count' in df.columns:
        st.subheader("Limit name counts")
        st.dataframe(df[['user_id', 'limit_name_count']])

    # -----------------------------
    # LLM Question Answering
    # -----------------------------
    st.subheader("Ask questions about your dataset")
    st.info("Example questions: \n- Which user has been consistent with >200 total_lessons?\n- Show top users by unique_lesson_ids_watched in September\n- List devices with L1 license for a user")

    user_question = st.text_input("Type your question here:")
    if user_question:
        try:
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model_name="gpt-4"),
                df,
                verbose=False
            )
            response = agent.run(user_question)
            st.subheader("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

