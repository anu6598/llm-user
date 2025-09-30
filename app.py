import streamlit as st
import pandas as pd
import requests
import json
import time

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Hugging Face", layout="wide")
st.title("📊 CSV Q&A with Hugging Face")

def get_available_models():
    """Return a list of reliable free Hugging Face models"""
    return [
        {
            "name": "Microsoft DialoGPT-medium", 
            "model": "microsoft/DialoGPT-medium",
            "description": "Good for conversational Q&A"
        },
        {
            "name": "GPT-2", 
            "model": "gpt2",
            "description": "Reliable text generation"
        },
        {
            "name": "Facebook Blenderbot-400M", 
            "model": "facebook/blenderbot-400M-distill",
            "description": "Good for dialogue"
        },
        {
            "name": "Google Flan-T5 Small", 
            "model": "google/flan-t5-small",
            "description": "Good for instruction following"
        }
    ]

def query_huggingface(prompt, model_name, hf_token, max_retries=3):
    """Query Hugging Face API with retry logic"""
    
    model_map = {model["name"]: model["model"] for model in get_available_models()}
    model_id = model_map[model_name]
    
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    for attempt in range(max_retries):
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 250,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10, 20, 30 seconds
                    st.info(f"🔄 Model is loading... Waiting {wait_time} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return type('obj', (object,), {
                        'status_code': 503,
                        'text': 'Model loading timeout after multiple retries'
                    })
            else:
                return response
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.info(f"⏰ Request timeout. Retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            else:
                return type('obj', (object,), {
                    'status_code': 408,
                    'text': 'Request timeout after multiple retries'
                })
        except Exception as e:
            return type('obj', (object,), {
                'status_code': 500,
                'text': f'Request failed: {str(e)}'
            })
    
    return type('obj', (object,), {
        'status_code': 500,
        'text': 'Max retries exceeded'
    })

def setup_huggingface_token():
    """Helper function to setup and validate Hugging Face token"""
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    
    if not HF_TOKEN:
        st.error("""
        ## 🔐 Hugging Face Token Required
        
        **To use this app, you need a FREE Hugging Face token:**
        
        1. **Create Account/Login** at [huggingface.co](https://huggingface.co)
        2. **Get Token** from [Settings → Access Tokens](https://huggingface.co/settings/tokens)
        3. **Choose Token Type**: Select **"Read"** 
        4. **Add to Streamlit Secrets**:
           - Go to your app settings in Streamlit Cloud
           - Add: `HF_TOKEN = "your_token_here"`
        
        ⚠️ **Token starts with** `hf_` - make sure to copy the entire token!
        """)
        return None
    
    # Validate token format
    if not HF_TOKEN.startswith('hf_'):
        st.error("""
        ❌ Invalid token format!
        - Hugging Face tokens should start with `hf_`
        - Please check you copied the entire token
        - Get a new one from [Hugging Face Settings](https://huggingface.co/settings/tokens)
        """)
        return None
    
    return HF_TOKEN

# Main app
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.subheader("📊 Dataset Info")
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))
        
        st.write("**Columns:**")
        for col in df.columns:
            st.write(f"- {col} ({df[col].dtype})")

    # Model selection
    st.subheader("🤖 Model Selection")
    models = get_available_models()
    model_names = [model["name"] for model in models]
    
    selected_model = st.selectbox(
        "Choose a model:",
        model_names,
        help="Different models have different strengths. DialoGPT-medium is recommended for Q&A."
    )
    
    # Show model description
    selected_model_info = next(model for model in models if model["name"] == selected_model)
    st.caption(f"ℹ️ {selected_model_info['description']}")

    # Question input
    st.subheader("💬 Ask Questions About Your Data")
    user_question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Examples:\n• What are the main trends in this data?\n• Which columns have the strongest correlation?\n• What insights can you derive from column X?\n• Are there any data quality issues?"
    )

    if user_question:
        # Setup token
        HF_TOKEN = setup_huggingface_token()
        if not HF_TOKEN:
            st.stop()

        # Prepare enhanced prompt
        prompt = f"""
        You are an expert data analyst. Analyze the following dataset and provide helpful insights.

        DATASET INFORMATION:
        - Columns: {list(df.columns)}
        - Total rows: {len(df)}
        - Total columns: {len(df.columns)}
        - Numeric columns: {list(numeric_cols)}
        - Data types: {dict(df.dtypes)}
        - Sample data (first 3 rows):
        {df.head(3).to_string()}

        USER QUESTION: {user_question}

        Please provide:
        1. Specific insights based on the available data structure
        2. Suggestions for relevant analyses
        3. Any data patterns or relationships you can infer
        4. Practical recommendations for further exploration

        Be analytical and helpful. If specific calculations are needed, explain what should be calculated and why.
        """

        with st.spinner(f"🔄 Analyzing with {selected_model}... (This may take 20-40 seconds for free models)"):
            response = query_huggingface(prompt, selected_model, HF_TOKEN)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        answer = result[0]['generated_text']
                    else:
                        answer = str(result[0])
                    
                    st.success("✅ Analysis Results:")
                    st.write(answer)
                    
                    # Add follow-up suggestions
                    st.info("""
                    💡 **Follow-up questions you might ask:**
                    - "What are the summary statistics for the numeric columns?"
                    - "Are there any correlations between variables?"
                    - "What data quality issues should I check for?"
                    - "Can you suggest visualizations for this data?"
                    """)
                    
                else:
                    st.error("Unexpected response format from the model")
                    st.write("Raw response:", result)
                    
            elif response.status_code == 503:
                st.warning("""
                ⏳ **Model is Loading...**
                
                This is normal for free Hugging Face models. They spin down when not in use.
                
                **What to do:**
                - Wait 30 seconds and try again
                - The model should be loaded and faster on subsequent requests
                - You can try a different model from the dropdown
                """)
                
            elif response.status_code == 401:
                st.error("""
                ❌ **Authentication Failed**
                - Please check your Hugging Face token is correct
                - Make sure it has **Read** permissions
                - Verify it's properly set in Streamlit secrets
                """)
                
            elif response.status_code == 429:
                st.warning("""
                ⚠️ **Rate Limit Reached**
                - You've made too many requests too quickly
                - Please wait a few minutes before trying again
                - Free tier has limited requests per minute
                """)
                
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                st.info("💡 Try selecting a different model from the dropdown above.")

else:
    st.info("👆 Please upload a CSV file to get started")
    
    with st.expander("ℹ️ How to get your FREE Hugging Face token"):
        st.markdown("""
        1. **Go to [huggingface.co](https://huggingface.co)**
        2. **Create account** (or login if you have one)
        3. **Click your profile picture** → **Settings**
        4. **Go to Access Tokens** in left sidebar
        5. **Click "New token"**
        6. **Choose "Read" role** and give it a name
        7. **Copy the token** (starts with `hf_`)
        8. **Add to Streamlit Cloud secrets** as `HF_TOKEN`
        
        ✅ **That's it! No credit card required. Completely free.**
        """)

    st.markdown("""
    ### 🎯 Example Questions to Ask:
    - **Trend Analysis**: "What are the main trends in this data?"
    - **Correlation**: "Which columns might be correlated?"
    - **Data Quality**: "Are there any data quality issues I should know about?"
    - **Insights**: "What business insights can I derive from this dataset?"
    - **Visualization**: "What charts would best represent this data?"
    """)
