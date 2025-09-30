import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="CSV Q&A Debug", layout="wide")
st.title("🔧 CSV Q&A Debug Mode")

# Debug function to test the token and API
def debug_huggingface(hf_token):
    """Test the Hugging Face token and API connection"""
    test_models = [
        "google/flan-t5-base",
        "microsoft/DialoGPT-small", 
        "distilgpt2",
        "facebook/bart-large-cnn"
    ]
    
    results = []
    
    for model in test_models:
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        test_prompt = "Hello, are you working? Answer with yes or no."
        
        try:
            start_time = time.time()
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": test_prompt, "parameters": {"max_length": 50}},
                timeout=30
            )
            end_time = time.time()
            
            results.append({
                "model": model,
                "status_code": response.status_code,
                "response_time": f"{end_time - start_time:.2f}s",
                "response_text": response.text[:200] if response.text else "No response"
            })
            
        except Exception as e:
            results.append({
                "model": model,
                "status_code": "Error",
                "response_time": "N/A",
                "response_text": str(e)
            })
    
    return results

# Simple query function
def query_huggingface_simple(prompt, hf_token):
    """Simple query function with detailed error reporting"""
    # Try multiple models in sequence
    models_to_try = [
        "google/flan-t5-base",
        "microsoft/DialoGPT-small",
        "distilgpt2"
    ]
    
    for model in models_to_try:
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.3
            }
        }
        
        try:
            st.info(f"🔄 Trying model: {model}")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return {
                        "success": True,
                        "model": model,
                        "answer": result[0].get('generated_text', str(result[0])),
                        "raw_response": result
                    }
            
            # If we get here, the model didn't work
            st.warning(f"Model {model} returned status {response.status_code}")
            
        except Exception as e:
            st.warning(f"Model {model} failed: {str(e)}")
            continue
    
    return {"success": False, "error": "All models failed"}

# Main app with debug options
st.sidebar.header("🔧 Debug Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())
    
    # Show dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
    with col2:
        st.metric("Numeric Columns", len(df.select_dtypes(include=['number']).columns))
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Get HF Token
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    
    if not HF_TOKEN:
        st.error("""
        ❌ No HF_TOKEN found in secrets!
        
        Please add your Hugging Face token to Streamlit secrets:
        1. Go to app settings → Secrets
        2. Add: `HF_TOKEN = "hf_your_actual_token_here"`
        """)
        st.stop()
    
    # Debug section
    if debug_mode:
        st.subheader("🔍 Debug Information")
        st.code(f"Token starts with: {HF_TOKEN[:10]}...")
        st.code(f"Token length: {len(HF_TOKEN)} characters")
        
        if st.button("Run Connection Test"):
            with st.spinner("Testing connection to Hugging Face..."):
                results = debug_huggingface(HF_TOKEN)
                
            st.subheader("Connection Test Results")
            for result in results:
                with st.expander(f"Model: {result['model']}"):
                    st.write(f"Status: {result['status_code']}")
                    st.write(f"Response Time: {result['response_time']}")
                    st.write(f"Response: {result['response_text']}")

    # Question input
    st.subheader("💬 Ask Questions About Your Data")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        # Prepare prompt
        prompt = f"""
        You are a data analyst. Analyze this dataset and answer the question.
        
        DATASET INFO:
        - Columns: {list(df.columns)}
        - Total rows: {len(df)}
        - Data types: {dict(df.dtypes)}
        
        QUESTION: {user_question}
        
        Provide helpful analysis and insights based on the dataset structure.
        """
        
        with st.spinner("🔄 Analyzing your data..."):
            result = query_huggingface_simple(prompt, HF_TOKEN)
            
            if result["success"]:
                st.success(f"✅ Answer (from {result['model']}):")
                st.write(result["answer"])
                
                if debug_mode:
                    with st.expander("Raw API Response"):
                        st.json(result["raw_response"])
            else:
                st.error("""
                ❌ All models failed. Possible issues:
                
                1. **Token Permissions**: Make sure your token has "Read" access
                2. **Token Validity**: Token might be expired or revoked
                3. **Rate Limits**: You might have hit API limits
                4. **Model Availability**: Models might be temporarily down
                
                **Next Steps:**
                - Check your token at https://huggingface.co/settings/tokens
                - Try creating a new token
                - Wait a few minutes and try again
                """)

else:
    st.info("👆 Please upload a CSV file to get started")

# Alternative approach section
with st.expander("🔄 Alternative Solutions"):
    st.markdown("""
    ### If Hugging Face continues to fail, try these:
    
    **Option 1: Use OpenRouter (Free)**
    ```python
    # No token needed for basic free tier
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer free",
            "HTTP-Referer": "https://your-app.com",
            "X-Title": "CSV Analyzer"
        },
        json={
            "model": "google/gemma-7b-it:free",
            "messages": [{"role": "user", "content": "Your prompt"}]
        }
    )
    ```
    
    **Option 2: Local Analysis (No API)**
    - Use pandas profiling
    - Generate automatic insights
    - Create visualizations
    
    **Option 3: Different Hugging Face Approach**
    - Try the inference API without a token for public models
    - Use a different model endpoint
    """)
