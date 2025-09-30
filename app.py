import streamlit as st
import pandas as pd
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CSV Q&A",
    page_icon="❓",
    layout="wide"
)

st.title("📊 CSV Question & Answer")
st.markdown("Upload a CSV file and ask questions about your data!")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    
    df = st.session_state.df
    
    # Display dataset info
    st.subheader("Dataset Preview")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("📝 Ask Questions About Your Data")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., 'How many users in April?' or 'What is the total sales?'",
        key="question_input"
    )
    
    def analyze_question(question, df):
        """Simple question analyzer using pandas"""
        question_lower = question.lower()
        
        # Count questions
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            return handle_count_questions(question_lower, df)
        
        # Sum questions
        elif any(word in question_lower for word in ['total', 'sum', 'add up']):
            return handle_sum_questions(question_lower, df)
        
        # Average questions
        elif any(word in question_lower for word in ['average', 'mean']):
            return handle_average_questions(question_lower, df)
        
        # Maximum questions
        elif any(word in question_lower for word in ['max', 'maximum', 'highest', 'most']):
            return handle_max_questions(question_lower, df)
        
        # Minimum questions
        elif any(word in question_lower for word in ['min', 'minimum', 'lowest', 'least']):
            return handle_min_questions(question_lower, df)
        
        # Date filtering questions
        elif any(word in question_lower for word in ['april', 'january', 'february', 'march', 'may', 'june', 
                                                   'july', 'august', 'september', 'october', 'november', 'december']):
            return handle_date_questions(question_lower, df)
        
        else:
            return "I can help with: counting, summing, averages, max/min values, and date filtering. Try asking something like 'How many users in April?' or 'What is the total sales?'"

    def handle_count_questions(question, df):
        """Handle counting questions"""
        # Count all rows
        if 'row' in question or 'record' in question or 'entry' in question:
            return f"**Total number of records:** {len(df):,}"
        
        # Count unique values in a specific column
        for col in df.columns:
            if col.lower() in question:
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    return f"**Number of unique {col}:** {unique_count:,}"
                else:
                    return f"**Number of records with {col}:** {len(df[df[col].notna()]):,}"
        
        # Count based on conditions
        if 'april' in question:
            return count_april_records(df)
        
        return f"**Total number of records:** {len(df):,}"

    def handle_sum_questions(question, df):
        """Handle sum/total questions"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.lower() in question:
                total = df[col].sum()
                return f"**Total sum of {col}:** {total:,.2f}"
        
        if numeric_cols.any():
            return f"Available numeric columns for summing: {', '.join(numeric_cols)}"
        else:
            return "No numeric columns found for sum calculations."

    def handle_average_questions(question, df):
        """Handle average questions"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.lower() in question:
                avg = df[col].mean()
                return f"**Average of {col}:** {avg:.2f}"
        
        if numeric_cols.any():
            return f"Available numeric columns for averaging: {', '.join(numeric_cols)}"
        else:
            return "No numeric columns found for average calculations."

    def handle_max_questions(question, df):
        """Handle maximum value questions"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.lower() in question:
                max_val = df[col].max()
                max_row = df[df[col] == max_val]
                
                # Try to find user/name column
                user_col = find_user_column(df)
                if user_col and user_col in max_row.columns:
                    user = max_row[user_col].iloc[0]
                    return f"**Maximum {col}:** {max_val:.2f} (by {user})"
                else:
                    return f"**Maximum {col}:** {max_val:.2f}"
        
        if numeric_cols.any():
            return f"Available numeric columns for max: {', '.join(numeric_cols)}"
        else:
            return "No numeric columns found for maximum calculations."

    def handle_min_questions(question, df):
        """Handle minimum value questions"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col.lower() in question:
                min_val = df[col].min()
                min_row = df[df[col] == min_val]
                
                user_col = find_user_column(df)
                if user_col and user_col in min_row.columns:
                    user = min_row[user_col].iloc[0]
                    return f"**Minimum {col}:** {min_val:.2f} (by {user})"
                else:
                    return f"**Minimum {col}:** {min_val:.2f}"
        
        if numeric_cols.any():
            return f"Available numeric columns for min: {', '.join(numeric_cols)}"
        else:
            return "No numeric columns found for minimum calculations."

    def handle_date_questions(question, df):
        """Handle date-related questions"""
        # Convert date columns to datetime
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                date_columns.append(col)
        
        if not date_columns:
            return "No date columns found in the dataset."
        
        # Check for specific months
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in question:
                for date_col in date_columns:
                    try:
                        # Convert to datetime
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        # Filter by month
                        month_data = df[df[date_col].dt.month == month_num]
                        count = len(month_data)
                        
                        if count > 0:
                            return f"**Number of records in {month_name.capitalize()}:** {count:,}"
                        else:
                            return f"No records found for {month_name.capitalize()}"
                    
                    except:
                        continue
        
        return f"Found date columns: {', '.join(date_columns)}. Try asking about a specific month like 'How many in April?'"

    def count_april_records(df):
        """Count records in April"""
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower():
                date_columns.append(col)
        
        if not date_columns:
            return "No date columns found to filter by April."
        
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                april_data = df[df[date_col].dt.month == 4]  # April is month 4
                count = len(april_data)
                
                if count > 0:
                    return f"**Number of records in April:** {count:,}"
                else:
                    return "No records found for April"
            
            except Exception as e:
                continue
        
        return "Could not process date columns for April filtering."

    def find_user_column(df):
        """Find a user/name column in the dataset"""
        user_keywords = ['user', 'name', 'username', 'customer', 'student', 'person']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in user_keywords):
                return col
        return None

    # Process question when asked
    if question:
        with st.spinner("Analyzing your question..."):
            answer = analyze_question(question, df)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer
            })
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.info(answer)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📋 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.expander(f"Q: {chat['question']}", expanded=i==0):
                st.write(chat['answer'])

else:
    st.info("👆 Please upload a CSV file to get started")
    
    # Sample questions
    st.markdown("""
    ### 💡 Example Questions You Can Ask:
    
    **Counting Questions:**
    - "How many users are there?"
    - "How many records in April?"
    - "Count the number of entries"
    - "How many unique customers?"
    
    **Calculation Questions:**
    - "What is the total sales?"
    - "What's the average age?"
    - "Show me the maximum score"
    - "Who has the minimum completion rate?"
    
    **Date Questions:**
    - "How many users in April?"
    - "Records from March"
    - "January data count"
    """)

# Add reset button
if st.session_state.df is not None:
    if st.button("🔄 Reset and Upload New File"):
        st.session_state.df = None
        st.session_state.chat_history = []
        st.rerun()
