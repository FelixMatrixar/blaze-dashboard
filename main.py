# Streamlit dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Config
st.set_page_config(layout="wide")
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
# Summary
DATA_SUMMARY = "Dataset has 3 columns. Target might be ' \"Weight(Pounds)\"'. Row count approx 50+."

# Load data
df = pd.read_csv(DATA_URL)
# Clean column names
df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True).str.lower()

# Authentication
if 'IBM_API_KEY' in st.secrets and 'IBM_PROJECT_ID' in st.secrets:
    api_key = st.secrets['IBM_API_KEY']
    project_id = st.secrets['IBM_PROJECT_ID']
else:
    api_key = st.sidebar.text_input('IBM API Key', type='password')
    project_id = st.sidebar.text_input('IBM Project ID')

model_id = st.sidebar.selectbox('Model ID', ['meta-llama/llama-2-70b-chat', 'ibm/granite-13b-chat'])

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "Context‑Aware Analyst"])

with tab1:
    st.header("Data Overview")
    st.dataframe(df)
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Chat with AI Analyst")
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": DATA_SUMMARY}]
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"]) 
    prompt = st.chat_input("Ask a question about the data...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Model call
        model = ModelInference(model_id=model_id, credentials={"apikey": api_key}, project_id=project_id)
        response = model.chat(messages=st.session_state.messages)
        answer = response.get('generated_text', '')
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
