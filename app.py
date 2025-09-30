import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Streamlit page config
st.set_page_config(page_title="CSV Smart Analyzer", layout="wide")
st.title("📊 CSV Smart Analyzer - No APIs Required")

def generate_smart_analysis(df, question):
    """Generate intelligent analysis without external APIs"""
    
    analysis = ""
    question_lower = question.lower()
    
    # Basic dataset info
    analysis += "## 📋 Dataset Overview\n\n"
    analysis += f"- **Dataset Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n"
    analysis += f"- **Total Values**: {df.size} data points\n"
    analysis += f"- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
    
    # Column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    analysis += "## 🏷️ Column Analysis\n\n"
    analysis += f"- **Numeric Columns** ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}\n"
    analysis += f"- **Text Columns** ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}\n\n"
    
    # Answer specific question types
    if any(word in question_lower for word in ['stat', 'summary', 'overview', 'describe']):
        analysis += "## 📊 Statistical Summary\n\n"
        if numeric_cols:
            for col in numeric_cols:
                analysis += f"### {col}\n"
                analysis += f"- Mean: {df[col].mean():.2f}\n"
                analysis += f"- Median: {df[col].median():.2f}\n"
                analysis += f"- Min: {df[col].min():.2f} | Max: {df[col].max():.2f}\n"
                analysis += f"- Standard Deviation: {df[col].std():.2f}\n"
                analysis += f"- Missing Values: {df[col].isnull().sum()}\n\n"
    
    if any(word in question_lower for word in ['correlation', 'relationship', 'correlate']):
        analysis += "## 🔗 Correlation Analysis\n\n"
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Find strong correlations
            strong_corrs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        strong_corrs.append(f"{numeric_cols[i]} ↔ {numeric_cols[j]}: {corr:.3f}")
            
            if strong_corrs:
                analysis += "**Strong Correlations (> 0.7):**\n"
                for corr in strong_corrs:
                    analysis += f"- {corr}\n"
            else:
                analysis += "No strong correlations found (> 0.7)\n"
        else:
            analysis += "Need at least 2 numeric columns for correlation analysis\n"
    
    if any(word in question_lower for word in ['trend', 'pattern', 'insight']):
        analysis += "## 🔍 Data Insights\n\n"
        if numeric_cols:
            analysis += "**Numeric Column Insights:**\n"
            for col in numeric_cols:
                q75, q25 = np.percentile(df[col].dropna(), [75, 25])
                iqr = q75 - q25
                analysis += f"- **{col}**: Range {df[col].min():.2f}-{df[col].max():.2f}, IQR: {iqr:.2f}\n"
        
        if categorical_cols:
            analysis += "\n**Categorical Column Insights:**\n"
            for col in categorical_cols:
                top_value = df[col].value_counts().index[0] if not df[col].empty else "N/A"
                analysis += f"- **{col}**: Most frequent value: '{top_value}'\n"
    
    if any(word in question_lower for word in ['missing', 'null', 'quality']):
        analysis += "## 🧹 Data Quality Report\n\n"
        missing_total = df.isnull().sum().sum()
        missing_percent = (missing_total / df.size) * 100
        analysis += f"- **Total Missing Values**: {missing_total} ({missing_percent:.1f}%)\n"
        analysis += f"- **Duplicate Rows**: {df.duplicated().sum()}\n"
        
        if missing_total > 0:
            analysis += "\n**Columns with Missing Values:**\n"
            for col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    analysis += f"- {col}: {missing} missing values\n"
    
    # General advice
    analysis += "\n## 💡 Recommended Analysis\n"
    if numeric_cols:
        analysis += "- Create histograms for numeric columns to understand distributions\n"
    if len(numeric_cols) >= 2:
        analysis += "- Generate scatter plots to visualize relationships between numeric variables\n"
    if categorical_cols:
        analysis += "- Use bar charts to show frequency distributions of categorical variables\n"
    
    return analysis

# CSV uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Quick stats
        st.subheader("Quick Stats")
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    with col2:
        st.subheader("Smart Analysis")
        
        # Pre-defined questions
        question_option = st.selectbox(
            "Choose a question or ask your own:",
            [
                "Select a question...",
                "Give me a comprehensive data overview",
                "What are the main statistics for numeric columns?",
                "Are there strong correlations between variables?",
                "What data quality issues should I know about?",
                "What patterns or trends can you identify?",
                "Ask custom question..."
            ]
        )
        
        custom_question = ""
        if question_option == "Ask custom question...":
            custom_question = st.text_input("Your custom question:")
        elif question_option != "Select a question...":
            custom_question = question_option
        
        if custom_question:
            with st.spinner("Analyzing your data..."):
                analysis = generate_smart_analysis(df, custom_question)
                st.markdown(analysis)
                
                # Auto-generate some visualizations
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.subheader("📈 Auto-Generated Visualizations")
                    
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Correlation heatmap
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                            st.pyplot(fig)
                        
                        with col2:
                            # Distribution example
                            fig, ax = plt.subplots(figsize=(8, 6))
                            df[numeric_cols[0]].hist(bins=20, ax=ax)
                            ax.set_title(f'Distribution of {numeric_cols[0]}')
                            st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to get started")
    
    st.markdown("""
    ### 🎯 This analyzer provides:
    - **Smart data insights** without external APIs
    - **Statistical analysis** of your data
    - **Correlation detection** between variables
    - **Data quality assessment**
    - **Automated visualizations**
    - **100% free** - no API keys required
    - **No rate limits** - use as much as you want
    """)
