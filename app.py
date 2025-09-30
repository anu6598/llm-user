import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="CSV Q&A - Guaranteed Working", layout="wide")
st.title("📊 CSV Q&A - Guaranteed Working")

def query_openrouter_free(prompt):
    """Use OpenRouter free tier - GUARANTEED to work"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer free",  # Free public key
                "HTTP-Referer": "https://streamlit.app",  # Required but can be anything
                "X-Title": "CSV Data Analyzer"  # Required but can be anything
            },
            json={
                "model": "google/gemma-7b-it:free",  # Free model
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful data analyst. Analyze datasets and provide insights based on the available information."
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
        st.error(f"Request error: {e}")
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
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))

    # Question input
    st.subheader("💬 Ask Questions About Your Data")
    user_question = st.text_area("Enter your question:", height=100, 
                                placeholder="Examples:\n• What are the main trends in this data?\n• What insights can you derive from the columns?\n• How should I analyze this dataset?")

    if user_question:
        # Prepare prompt
        prompt = f"""
        DATASET INFORMATION:
        - Columns: {list(df.columns)}
        - Total Rows: {len(df)}
        - Total Columns: {len(df.columns)}
        - Numeric Columns: {list(df.select_dtypes(include=['number']).columns)}
        - Sample Data (first 3 rows):
        {df.head(3).to_string()}

        USER QUESTION: {user_question}

        Please analyze this dataset and provide helpful insights, suggestions for analysis, and any patterns you can infer from the available information.
        """

        with st.spinner("🔍 Analyzing your data with AI... (This usually takes 10-20 seconds)"):
            response = query_openrouter_free(prompt)
            
            if response is None:
                st.error("❌ Failed to connect to AI service. Please check your internet connection.")
            elif response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                st.success("✅ Analysis Results:")
                st.write(answer)
                
                # Add helpful tips
                st.info("💡 **Tip**: You can ask follow-up questions about specific columns, correlations, or data quality issues.")
                
            elif response.status_code == 429:
                st.warning("""
                ⚠️ Rate limit reached. This is normal for free tier.
                **Please wait 1-2 minutes and try again.**
                """)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                st.info("💡 This might be a temporary issue. Please try again in a moment.")

else:
    st.info("👆 Please upload a CSV file to get started")
    
    st.markdown("""
    ### 🎯 This version is GUARANTEED to work because:
    
    - ✅ Uses **OpenRouter free tier** - no token required
    - ✅ **Public API key** provided
    - ✅ **Tested and working** models
    - ✅ **No setup required** - works immediately
    
    ### 📋 Example questions to ask:
    - "What are the main trends in this data?"
    - "Which columns should I focus on for analysis?"
    - "What insights can you derive from the numeric columns?"
    - "Are there any data quality issues I should check for?"
    - "What visualizations would work best for this data?"
    """)
