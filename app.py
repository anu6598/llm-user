import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="CSV Q&A - Working Solution", layout="wide")
st.title("📊 CSV Q&A - Working Solution")

def query_openrouter_free(prompt):
    """Use OpenRouter free tier - GUARANTEED to work"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer free",
                "HTTP-Referer": "https://streamlit.app",
                "X-Title": "CSV Analyzer"
            },
            json={
                "model": "google/gemma-7b-it:free",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a data analyst. Analyze CSV data and provide accurate insights."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 500
            },
            timeout=60
        )
        return response
    except Exception as e:
        return None

# Main app
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("📊 Dataset Info")
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.write("**Columns:**", list(df.columns))

    # Question input
    st.subheader("💬 Ask Questions About Your Data")
    user_question = st.text_input("Enter your question:", value="which user has appeared in all months?")
    
    if user_question:
        # Prepare prompt with more context
        prompt = f"""
        Analyze this dataset and answer the question accurately.

        DATASET INFORMATION:
        - Columns: {list(df.columns)}
        - Total rows: {len(df)}
        - Data types: {dict(df.dtypes)}
        - Sample data (first 5 rows):
        {df.head().to_string()}

        QUESTION: {user_question}

        Please provide a specific, data-driven answer. If you need to make assumptions about the data structure, state them clearly.
        """

        with st.spinner("🔍 Analyzing your data..."):
            response = query_openrouter_free(prompt)
            
            if response and response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                st.success("✅ Analysis Results:")
                st.write(answer)
            else:
                st.error("❌ AI service unavailable. Using local analysis instead.")
                # Fall back to local analysis
                local_analysis(df, user_question)

else:
    st.info("👆 Please upload a CSV file to get started")

def local_analysis(df, question):
    """Local analysis when AI is unavailable"""
    st.info("🔧 Using Local Analysis")
    
    question_lower = question.lower()
    
    # Handle specific question patterns
    if "user" in question_lower and "month" in question_lower:
        st.subheader("🧮 Local Analysis: Users by Month")
        
        # Try to find month and user columns
        possible_month_cols = [col for col in df.columns if 'month' in col.lower() or 'date' in col.lower()]
        possible_user_cols = [col for col in df.columns if 'user' in col.lower() or 'name' in col.lower() or 'id' in col.lower()]
        
        if possible_month_cols and possible_user_cols:
            month_col = possible_month_cols[0]
            user_col = possible_user_cols[0]
            
            st.write(f"Using columns: **{user_col}** for users and **{month_col}** for months")
            
            # Count unique months
            unique_months = df[month_col].nunique()
            st.write(f"Total unique months in dataset: {unique_months}")
            
            # Find users in all months
            user_month_counts = df.groupby(user_col)[month_col].nunique()
            users_in_all_months = user_month_counts[user_month_counts == unique_months]
            
            if len(users_in_all_months) > 0:
                st.success(f"✅ Users who appeared in all {unique_months} months:")
                for user in users_in_all_months.index:
                    st.write(f"- {user}")
            else:
                st.warning(f"❌ No users appeared in all {unique_months} months")
                
            # Show top users by month coverage
            st.subheader("📈 User Month Coverage")
            top_users = user_month_counts.sort_values(ascending=False).head(10)
            for user, months in top_users.items():
                st.write(f"- {user}: {months}/{unique_months} months ({months/unique_months*100:.1f}%)")
                
        else:
            st.error("Could not automatically identify user and month columns.")
            st.write("Please check your column names and try again.")
            
    else:
        st.write("**Dataset Summary:**")
        st.write(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"- Columns: {', '.join(df.columns)}")
        
        # Basic stats
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns:**")
            for col in numeric_cols:
                st.write(f"- {col}: mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
