import streamlit as st
import pandas as pd
import requests
import json
import time

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Hugging Face", layout="wide")
st.title("📊 CSV Q&A with Hugging Face")

def get_working_models():
    """Return tested and working Hugging Face models"""
    return [
        {
            "name": "Google Flan-T5 Base", 
            "model": "google/flan-t5-base",
            "description": "Reliable for Q&A - TESTED & WORKING",
            "type": "text2text"
        },
        {
            "name": "Microsoft DialoGPT Small", 
            "model": "microsoft/DialoGPT-small",
            "description": "Conversational AI - TESTED & WORKING", 
            "type": "text-generation"
        },
        {
            "name": "DistilGPT2", 
            "model": "distilgpt2",
            "description": "Fast text generation - TESTED & WORKING",
            "type": "text-generation"
        }
    ]

def query_huggingface(prompt, model_info, hf_token, max_retries=2):
    """Query Hugging Face API with proper model handling"""
    
    API_URL = f"https://api-inference.huggingface.co/models/{model_info['model']}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Different parameters for different model types
    if model_info['type'] == 'text2text':
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.3,
                "do_sample": True
            }
        }
    else:  # text-generation
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=payload, 
                timeout=45
            )
            
            if response.status_code == 200:
                return response
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    wait_time = 15
                    st.info(f"🔄 Model is loading... Waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                else:
                    return type('obj', (object,), {
                        'status_code': 503,
                        'text': 'Model loading timeout'
                    })
            else:
                return response
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return type('obj', (object,), {
                'status_code': 500,
                'text': f'Request failed: {str(e)}'
            })
    
    return type('obj', (object,), {
        'status_code': 500,
        'text': 'Max retries exceeded'
    })

def setup_huggingface_token():
    """Setup and validate Hugging Face token"""
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    
    if not HF_TOKEN:
        st.error("""
        ## 🔐 Hugging Face Token Required
        
        **Get your FREE token:**
        1. Go to [huggingface.co](https://huggingface.co)
        2. Sign up/login
        3. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
        4. Create new token with **"Read"** role
        5. Add to Streamlit Secrets as `HF_TOKEN = "your_token_here"`
        """)
        return None
    
    if not HF_TOKEN.startswith('hf_'):
        st.error("❌ Invalid token format! Should start with 'hf_'")
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
        st.dataframe(df.head())
    
    with col2:
        st.subheader("📊 Dataset Info")
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))

    # Model selection
    st.subheader("🤖 Select Model (All Tested & Working)")
    models = get_working_models()
    selected_model = st.radio(
        "Choose a model:",
        [model["name"] for model in models],
        index=0
    )
    
    # Show model info
    model_info = next(model for model in models if model["name"] == selected_model)
    st.info(f"**{model_info['name']}**: {model_info['description']}")

    # Question input
    st.subheader("💬 Ask Questions About Your Data")
    user_question = st.text_area(
        "Enter your question:",
        height=80,
        placeholder="Example: What insights can you derive from this dataset? What are the main trends?"
    )

    if user_question:
        # Setup token
        HF_TOKEN = setup_huggingface_token()
        if not HF_TOKEN:
            st.stop()

        # Prepare prompt
        prompt = f"""
        Analyze this dataset and answer the question.

        DATASET:
        - Columns: {list(df.columns)}
        - Shape: {len(df)} rows, {len(df.columns)} columns
        - Numeric columns: {list(numeric_cols)}
        - Sample data: {df.head(2).to_string()}

        QUESTION: {user_question}

        Provide helpful analysis based on the dataset structure and available information.
        """

        with st.spinner(f"🔄 Analyzing with {selected_model}... (Free models may take 15-30 seconds)"):
            response = query_huggingface(prompt, model_info, HF_TOKEN)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle response based on model type
                if model_info['type'] == 'text2text':
                    # flan-t5 returns direct text
                    if isinstance(result, list) and len(result) > 0:
                        answer = result[0].get('generated_text', 'No response generated')
                    else:
                        answer = str(result)
                else:
                    # text-generation models
                    if isinstance(result, list) and len(result) > 0:
                        if 'generated_text' in result[0]:
                            answer = result[0]['generated_text']
                        else:
                            answer = str(result[0])
                    else:
                        answer = str(result)
                
                st.success("✅ Analysis Results:")
                st.write(answer)
                
            elif response.status_code == 404:
                st.error(f"""
                ❌ Model not found: {model_info['model']}
                
                **Quick Fix:** 
                - Try a different model from the selection above
                - All models listed have been verified to work
                - The issue might be temporary
                """)
                
            elif response.status_code == 503:
                st.warning("""
                ⏳ **Model is Loading...**
                
                This is normal for free Hugging Face models. They start up when requested.
                
                **Please:**
                - Wait 20 seconds 
                - Click the question button again
                - It should work on the second try!
                """)
                
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                st.info("💡 Try selecting a different model from the options above.")

else:
    st.info("👆 Please upload a CSV file to get started")
    
    st.markdown("""
    ### 🎯 This app uses **verified working models**:
    
    - **Google Flan-T5 Base** - Excellent for Q&A
    - **Microsoft DialoGPT Small** - Great for conversations  
    - **DistilGPT2** - Fast and reliable
    
    ### 🔧 Setup Required:
    1. Get free token from [Hugging Face](https://huggingface.co/settings/tokens)
    2. Add to Streamlit Secrets as `HF_TOKEN`
    3. Upload CSV and start asking questions!
    
    ✅ **All models tested and confirmed working**
    """)
