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
            return "Please upload a CSV file first."
        
        question_lower = question.lower()
        
        # Basic statistics
        if any(word in question_lower for word in ['statistic', 'summary', 'describe', 'overview']):
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
        
        # Top values
        elif any(word in question_lower for word in ['top', 'highest', 'lowest', 'maximum', 'minimum']):
            return self.analyze_extremes(question)
        
        # Default response for other questions
        else:
            return self.general_analysis(question)
    
    def get_basic_statistics(self):
        """Provide basic statistics for numeric columns"""
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return "No numeric columns found for statistical analysis."
        
        stats = numeric_df.describe()
        return f"**Basic Statistics for Numeric Columns:**\n\n{stats.to_string()}"
    
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
        return response
    
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
        
        return response
    
    def get_correlation_analysis(self):
        """Perform correlation analysis"""
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return "No numeric columns found for correlation analysis."
        
        if len(numeric_df.columns) < 2:
            return "Need at least 2 numeric columns for correlation analysis."
        
        correlation_matrix = numeric_df.corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        return response
    
    def analyze_specific_columns(self, question):
        """Analyze specific columns mentioned in the question"""
        # Extract column names from the question
        columns_found = []
        for col in self.df.columns:
            if col.lower() in question.lower():
                columns_found.append(col)
        
        if not columns_found:
            return "Please specify which columns you'd like to analyze. Available columns: " + ", ".join(self.df.columns)
        
        response = f"**Analysis for columns: {', '.join(columns_found)}**\n\n"
        
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
                
                response += "\n"
        
        return response
    
    def analyze_distribution(self, question):
        """Analyze distribution of numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if numeric_cols.empty:
            return "No numeric columns found for distribution analysis."
        
        # Create distribution plots
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns to avoid too many plots
            fig = px.histogram(
                self.df, 
                x=col,
                title=f"Distribution of {col}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        return f"Displayed distribution plots for numeric columns. Analyzed distributions for: {', '.join(numeric_cols[:3])}"
    
    def analyze_extremes(self, question):
        """Analyze top/bottom values"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if numeric_cols.empty:
            return "No numeric columns found for extreme value analysis."
        
        response = "**Extreme Values Analysis:**\n\n"
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            top_5 = self.df.nlargest(5, col)[[col]]
            bottom_5 = self.df.nsmallest(5, col)[[col]]
            
            response += f"**{col}**:\n"
            response += f"- Top 5 values:\n{top_5.to_string()}\n"
            response += f"- Bottom 5 values:\n{bottom_5.to_string()}\n\n"
        
        return response
    
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
        response += "- Basic statistics and summaries\n"
        response += "- Data distributions and histograms\n"
        response += "- Correlation between variables\n"
        response += "- Missing values analysis\n"
        response += "- Specific column information\n"
        response += "- Top/bottom values\n"
        response += "- Data quality issues\n"
        
        return response

def main():
    st.markdown('<h1 class="main-header">📊 CSV Data Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file and ask questions about your data!")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CSVDataAnalyzer()
    
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
        Try asking:
        - "Show me basic statistics"
        - "What are the correlations?"
        - "Are there missing values?"
        - "Show distribution of numeric columns"
        - "What are the top 5 values in each column?"
        - "Tell me about column X"
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
            placeholder="e.g., 'Show me basic statistics' or 'What are the correlations between variables?'",
            help="Ask any question about your dataset"
        )
        
        # Pre-defined question buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Basic Stats"):
                question = "show basic statistics"
        
        with col2:
            if st.button("🔗 Correlations"):
                question = "show correlations"
        
        with col3:
            if st.button("❓ Missing Values"):
                question = "are there missing values"
        
        with col4:
            if st.button("📈 Distributions"):
                question = "show distributions"
        
        # Process question
        if question:
            with st.spinner("Analyzing your data..."):
                answer = analyzer.analyze_question(question)
                
                # Display answer
                st.markdown("### 💡 Analysis Result")
                st.markdown(answer)
    
    elif uploaded_file is None:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to CSV Data Analyzer! 🎉</h2>
            <p>Upload a CSV file to start exploring your data.</p>
            <p>You can ask questions about:</p>
            <ul style='display: inline-block; text-align: left;'>
                <li>Basic statistics and summaries</li>
                <li>Data distributions and patterns</li>
                <li>Correlations between variables</li>
                <li>Missing values and data quality</li>
                <li>Specific column analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
