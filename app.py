import streamlit as st
import pandas as pd
import requests
import os

# Streamlit page config
st.set_page_config(page_title="CSV Q&A with Hugging Face", layout="wide")
st.title("📊 CSV Q&A with Hugging Face")

def setup_huggingface_token():
    """Helper function to setup and validate Hugging Face token"""
    HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    
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

# CSV uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())
    
    st.subheader("Ask questions about your dataset")
    user_question = st.text_input("Type your question here:")
    
    if user_question:
        # Setup token first
        HF_TOKEN = setup_huggingface_token()
        if not HF_TOKEN:
            st.stop()
            
        # Prepare prompt
        prompt = f"""
        You are a data analyst assistant.
        The dataset has the following columns: {', '.join(df.columns)}.
        Answer the following question based on the dataset. Use precise numbers and examples where possible.

        Question: {user_question}
        """
        
        with st.spinner("Generating answer... (Free models may take 20-30 seconds on first use)"):
            try:
                # Using Hugging Face Inference API
                API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
                headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
                response = requests.post(API_URL, headers=headers, json={
                    "inputs": prompt,
                    "parameters": {"max_length": 500}
                })
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        answer = result[0].get('generated_text', str(result))
                        st.success("✅ Answer:")
                        st.write(answer)
                    else:
                        st.error("Unexpected response format")
                elif response.status_code == 503:
                    st.warning("""
                    ⏳ Model is loading... 
                    
                    **This is normal for free Hugging Face models!**
                    - Models spin down when not in use
                    - First request takes 20-30 seconds to wake up
                    - Try again in 30 seconds - it will be faster!
                    """)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {e}")

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
