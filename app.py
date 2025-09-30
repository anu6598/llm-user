import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re

# Page configuration
st.set_page_config(
    page_title="CSV Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
    }
    .analysis-result {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class CSVDataAnalyzer:
    def __init__(self):
        self.df = None
        self.uploaded_file = None
    
    def load_data(self, uploaded_file):
        """Load CSV data into pandas DataFrame"""
        try:
            if uploaded_file is not None:
                # Try to read the CSV file
                self.df = pd.read_csv(uploaded_file)
                self.uploaded_file = uploaded_file
                return True
            return False
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def get_dataset_info(self):
        """Get basic information about the dataset"""
        if self.df is None:
            return None
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'sample_data': self.df.head(10)
        }
        return info
    
    def analyze_question(self, question):
        """Analyze the user's question and provide relevant insights"""
        if self.df is None:
            return "Please upload a CSV file first.", None
        
        question_lower = question.lower()
        
        # Handle specific analytical questions
        if any(word in question_lower for word in ['most', 'highest', 'top', 'maximum', 'max']):
            return self.analyze_max_min_questions(question, 'max')
        
        elif any(word in question_lower for word in ['least', 'lowest', 'bottom', 'minimum', 'min']):
            return self.analyze_max_min_questions(question, 'min')
        
        elif any(word in question_lower for word in ['average', 'mean']):
            return self.analyze_average_questions(question)
        
        elif any(word in question_lower for word in ['sum', 'total']):
            return self.analyze_sum_questions(question)
        
        elif any(word in question_lower for word in ['count', 'how many']):
            return self.analyze_count_questions(question)
        
        # Basic statistics
        elif any(word in question_lower for word in ['statistic', 'summary', 'describe', 'overview']):
            return self.get_basic_statistics()
        
        # Data information
        elif any(word in question_lower for word in ['info', 'information', 'columns', 'shape']):
            return self.get_data_info()
        
        # Missing values
        elif any(word in question_lower for word in ['missing', 'null', 'na', 'empty']):
            return self.get_missing_values()
        
        # Correlation analysis
        elif any(word in question_lower for word in ['correlation', 'relationship', 'correlate']):
            return self.get_correlation_analysis()
        
        # Specific column analysis
        elif any(word in question_lower for word in ['column', 'feature', 'variable']):
            return self.analyze_specific_columns(question)
        
        # Data distribution
        elif any(word in question_lower for word in ['distribution', 'histogram', 'frequency']):
            return self.analyze_distribution(question)
        
        # Default response for other questions
        else:
            return self.general_analysis(question)
    
    def analyze_max_min_questions(self, question, analysis_type):
        """Analyze questions about maximum/minimum values"""
        # Find numeric columns mentioned in the question
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        mentioned_columns = []
        
        for col in self.df.columns:
            if col.lower() in question.lower():
                mentioned_columns.append(col)
        
        # If no specific columns mentioned, use all numeric columns
        if not mentioned_columns:
            mentioned_columns = numeric_cols.tolist()
        
        results = []
        visualizations = []
        
        for col in mentioned_columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                if analysis_type == 'max':
                    max_value = self.df[col].max()
                    max_rows = self.df[self.df[col] == max_value]
                    
                    if len(max_rows) == 1:
                        result_text = f"**Maximum value in '{col}':** {max_value}"
                        # Try to find user/name column to show who has this value
                        user_col = self.find_user_column()
                        if user_col:
                            user_name = max_rows[user_col].iloc[0]
                            result_text += f"\n**User with maximum {col}:** {user_name}"
                    else:
                        result_text = f"**Maximum value in '{col}':** {max_value} (found in {len(max_rows)} records)"
                    
                    results.append(result_text)
                    
                    # Create visualization for top values
                    if len(self.df) > 1:
                        top_10 = self.df.nlargest(10, col)[[user_col if user_col else col, col]]
                        fig = px.bar(top_10, x=user_col if user_col else col, y=col, 
                                   title=f"Top 10 Values in {col}")
                        visualizations.append(fig)
                
                else:  # min analysis
                    min_value = self.df[col].min()
                    min_rows = self.df[self.df[col] == min_value]
                    
                    if len(min_rows) == 1:
                        result_text = f"**Minimum value in '{col}':** {min_value}"
                        user_col = self.find_user_column()
                        if user_col:
                            user_name = min_rows[user_col].iloc[0]
                            result_text += f"\n**User with minimum {col}:** {user_name}"
                    else:
                        result_text = f"**Minimum value in '{col}':** {min_value} (found in {len(min_rows)} records)"
                    
                    results.append(result_text)
        
        if not results:
            return f"No numeric columns found for {analysis_type} analysis.", None
        
        response = "\n\n".join(results)
        return response, visualizations
    
    def analyze_average_questions(self, question):
        """Analyze questions about averages"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        mentioned_columns = []
        
        for col in self.df.columns:
            if col.lower() in question.lower():
                mentioned_columns.append(col)
        
        if not mentioned_columns:
            mentioned_columns = numeric_cols.tolist()
        
        results = []
        
        for col in mentioned_columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                avg_value = self.df[col].mean()
                results.append(f"**Average of '{col}':** {avg_value:.2f}")
        
        if not results:
            return "No numeric columns found for average calculation.", None
        
        return "\n\n".join(results), None
    
    def analyze_sum_questions(self, question):
        """Analyze questions about sums/totals"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        mentioned_columns = []
        
        for col in self.df.columns:
            if col.lower() in question.lower():
                mentioned_columns.append(col)
        
        if not mentioned_columns:
            mentioned_columns = numeric_cols.tolist()
        
        results = []
        
        for col in mentioned_columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                total_value = self.df[col].sum()
                results.append(f"**Total sum of '{col}':** {total_value:,.2f}")
        
        if not results:
            return "No numeric columns found for sum calculation.", None
        
        return "\n\n".join(results), None
    
    def analyze_count_questions(self, question):
        """Analyze counting questions"""
        results = []
        
        # Count unique values in categorical columns
        if 'unique' in question.lower():
            for col in self.df.select_dtypes(include=['object']).columns:
                unique_count = self.df[col].nunique()
                results.append(f"**Unique values in '{col}':** {unique_count}")
        
        # Count specific conditions
        elif 'null' in question.lower() or 'missing' in question.lower():
            for col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    results.append(f"**Missing values in '{col}':** {null_count}")
        
        else:
            # General row count
            results.append(f"**Total number of records:** {len(self.df):,}")
            
            # Count by category if specific column mentioned
            for col in self.df.columns:
                if col.lower() in question.lower():
                    if self.df[col].dtype == 'object':
                        value_counts = self.df[col].value_counts().head(10)
                        results.append(f"**Value counts for '{col}':**\n{value_counts.to_string()}")
        
        if not results:
            return "Please specify what you'd like to count (e.g., 'unique users', 'missing values', 'total records').", None
        
        return "\n\n".join(results), None
    
    def find_user_column(self):
        """Try to identify a user/name column in the dataset"""
        user_like_columns = ['user', 'name', 'username', 'student', 'customer', 'person']
        for col in self.df.columns:
            if any(user_word in col.lower() for user_word in user_like_columns):
                return col
        return None
    
    def get_basic_statistics(self):
        """Provide basic statistics for numeric columns"""
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return "No numeric columns found for statistical analysis.", None
        
        stats = numeric_df.describe()
        return f"**Basic Statistics for Numeric Columns:**\n\n{stats.to_string()}", None
    
    def get_data_info(self):
        """Provide dataset information"""
        info = self.get_dataset_info()
        
        response = f"""
**Dataset Information:**
- **Shape**: {info['shape'][0]} rows × {info['shape'][1]} columns
- **Columns**: {', '.join(info['columns'])}
- **Numeric Columns**: {', '.join(info['numeric_columns']) if info['numeric_columns'] else 'None'}
- **Categorical Columns**: {', '.join(info['categorical_columns']) if info['categorical_columns'] else 'None'}
"""
        return response, None
    
    def get_missing_values(self):
        """Analyze missing values in the dataset"""
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        response = "**Missing Values Analysis:**\n\n"
        for col in missing_data.index:
            if missing_data[col] > 0:
                response += f"- **{col}**: {missing_data[col]} missing values ({missing_percentage[col]:.2f}%)\n"
        
        if missing_data.sum() == 0:
            response += "No missing values found in the dataset!"
        
        return response, None
    
    def get_correlation_analysis(self):
        """Perform correlation analysis"""
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return "No numeric columns found for correlation analysis.", None
        
        if len(numeric_df.columns) < 2:
            return "Need at least 2 numeric columns for correlation analysis.", None
        
        correlation_matrix = numeric_df.corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        response = "**Correlation Analysis:**\n\n"
        response += f"Correlation matrix computed for {len(numeric_df.columns)} numeric columns.\n\n"
        
        if high_corr_pairs:
            response += "**Strong Correlations (|r| > 0.5):**\n"
            for col1, col2, corr in high_corr_pairs:
                response += f"- {col1} & {col2}: {corr:.3f}\n"
        else:
            response += "No strong correlations found (|r| > 0.5)."
        
        return response, [fig]
    
    def analyze_specific_columns(self, question):
        """Analyze specific columns mentioned in the question"""
        columns_found = []
        for col in self.df.columns:
            if col.lower() in question.lower():
                columns_found.append(col)
        
        if not columns_found:
            return "Please specify which columns you'd like to analyze. Available columns: " + ", ".join(self.df.columns), None
        
        response = f"**Analysis for columns: {', '.join(columns_found)}**\n\n"
        visualizations = []
        
        for col in columns_found:
            if col in self.df.columns:
                dtype = self.df[col].dtype
                unique_count = self.df[col].nunique()
                missing_count = self.df[col].isnull().sum()
                
                response += f"**{col}** (dtype: {dtype}):\n"
                response += f"- Unique values: {unique_count}\n"
                response += f"- Missing values: {missing_count}\n"
                
                if self.df[col].dtype in ['int64', 'float64']:
                    response += f"- Mean: {self.df[col].mean():.2f}\n"
                    response += f"- Median: {self.df[col].median():.2f}\n"
                    response += f"- Std: {self.df[col].std():.2f}\n"
                    response += f"- Min: {self.df[col].min():.2f}\n"
                    response += f"- Max: {self.df[col].max():.2f}\n"
                    
                    # Create distribution plot for numeric columns
                    fig = px.histogram(self.df, x=col, title=f"Distribution of {col}")
                    visualizations.append(fig)
                
                response += "\n"
        
        return response, visualizations
    
    def analyze_distribution(self, question):
        """Analyze distribution of numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if numeric_cols.empty:
            return "No numeric columns found for distribution analysis.", None
        
        visualizations = []
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            fig = px.histogram(
                self.df, 
                x=col,
                title=f"Distribution of {col}",
                marginal="box"
            )
            visualizations.append(fig)
        
        response = f"Displayed distribution plots for numeric columns. Analyzed distributions for: {', '.join(numeric_cols[:3])}"
        return response, visualizations
    
    def general_analysis(self, question):
        """Provide general analysis for other types of questions"""
        response = f"I've analyzed your question: '{question}'\n\n"
        
        # Add some general insights
        info = self.get_dataset_info()
        response += f"**Dataset Overview:**\n"
        response += f"- The dataset has {info['shape'][0]} rows and {info['shape'][1]} columns\n"
        response += f"- Numeric columns: {len(info['numeric_columns'])}\n"
        response += f"- Categorical columns: {len(info['categorical_columns'])}\n\n"
        
        response += "**You can ask me about:**\n"
        response += "- 'Which user has the most total_lessons?'\n"
        response += "- 'Show me the highest values in each column'\n"
        response += "- 'What is the average age?'\n"
        response += "- 'Total sum of sales'\n"
        response += "- 'How many unique users?'\n"
        response += "- Basic statistics and correlations\n"
        response += "- Data distributions and missing values\n"
        
        return response, None

def main():
    st.markdown('<h1 class="main-header">📊 CSV Data Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file and ask questions about your data!")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CSVDataAnalyzer()
        st.session_state.chat_history = []
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            if analyzer.uploaded_file != uploaded_file:
                if analyzer.load_data(uploaded_file):
                    st.success("✅ File uploaded successfully!")
                    
                    # Show basic info
                    info = analyzer.get_dataset_info()
                    st.subheader("Dataset Info")
                    st.write(f"**Shape:** {info['shape'][0]} rows × {info['shape'][1]} columns")
                    
                    st.subheader("Columns")
                    for col in info['columns']:
                        st.write(f"- {col} ({info['data_types'][col]})")
        
        st.markdown("---")
        st.header("💡 Question Examples")
        st.write("""
        **Analytical Questions:**
        - "Which user has the most total_lessons?"
        - "Show the highest sales amount"
        - "What is the average score?"
        - "Total revenue by category"
        - "How many unique customers?"
        
        **General Questions:**
        - "Basic statistics"
        - "Correlation analysis"
        - "Missing values"
        - "Data distributions"
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if uploaded_file is not None and analyzer.df is not None:
            st.subheader("📈 Data Preview")
            
            # Show data preview with tabs
            tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Data Types"])
            
            with tab1:
                st.dataframe(analyzer.df.head(10), use_container_width=True)
            
            with tab2:
                st.dataframe(analyzer.df.tail(10), use_container_width=True)
            
            with tab3:
                dtype_info = pd.DataFrame({
                    'Column': analyzer.df.columns,
                    'Data Type': analyzer.df.dtypes.values,
                    'Non-Null Count': analyzer.df.count().values,
                    'Null Count': analyzer.df.isnull().sum().values
                })
                st.dataframe(dtype_info, use_container_width=True)
    
    with col2:
        if uploaded_file is not None and analyzer.df is not None:
            st.subheader("🔍 Quick Insights")
            
            info = analyzer.get_dataset_info()
            
            # Display key metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric("Total Rows", info['shape'][0])
                st.metric("Numeric Columns", len(info['numeric_columns']))
            
            with metric_col2:
                st.metric("Total Columns", info['shape'][1])
                st.metric("Categorical Columns", len(info['categorical_columns']))
            
            # Missing values summary
            total_missing = analyzer.df.isnull().sum().sum()
            if total_missing > 0:
                st.warning(f"⚠️ {total_missing} missing values found")
            else:
                st.success("✅ No missing values")
    
    # Question and answer section
    if uploaded_file is not None and analyzer.df is not None:
        st.markdown("---")
        st.subheader("❓ Ask Questions About Your Data")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., 'Which user has the most total_lessons?' or 'What is the average score?'",
            help="Ask any analytical question about your dataset"
        )
        
        # Pre-defined question buttons for common analytical questions
        st.write("**Quick Questions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👑 Top User"):
                question = "which user has the most"
        
        with col2:
            if st.button("📈 Max Value"):
                question = "show maximum values"
        
        with col3:
            if st.button("📊 Average"):
                question = "what is the average"
        
        with col4:
            if st.button("🔢 Count"):
                question = "how many unique"
        
        # Process question
        if question:
            with st.spinner("Analyzing your data..."):
                answer, visualizations = analyzer.analyze_question(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer,
                    'visualizations': visualizations
                })
                
                # Display answer
                st.markdown("### 💡 Analysis Result")
                st.markdown(f'<div class="analysis-result">{answer}</div>', unsafe_allow_html=True)
                
                # Display visualizations if any
                if visualizations:
                    st.markdown("### 📊 Visualizations")
                    for viz in visualizations:
                        st.plotly_chart(viz, use_container_width=True)
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📝 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question']}"):
                    st.markdown(chat['answer'])
                    if chat['visualizations']:
                        for viz in chat['visualizations']:
                            st.plotly_chart(viz, use_container_width=True)
    
    elif uploaded_file is None:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to CSV Data Analyzer! 🎉</h2>
            <p>Upload a CSV file to start exploring your data.</p>
            <p>You can ask analytical questions like:</p>
            <ul style='display: inline-block; text-align: left;'>
                <li>"Which user has the most total_lessons?"</li>
                <li>"What is the highest sales amount?"</li>
                <li>"Show me the average score by category"</li>
                <li>"How many unique customers are there?"</li>
                <li>"Who has the minimum completion rate?"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
