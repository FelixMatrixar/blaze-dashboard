import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Config
st.set_page_config(layout="wide")
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

# Load data
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Aggressive sanitization: Keep only letters, numbers, and underscores
    df.columns = [
        "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
        for col in df.columns
    ]
    return df

df = load_data(DATA_URL)

# Data Summary
DATA_SUMMARY = "Columns: Index, Height_Inches, Weight_Pounds. Target: Weight_Pounds. Rows: approx 200."

# Sidebar Authentication
api_key = st.secrets.get("IBM_API_KEY") or st.sidebar.text_input("IBM API Key", type="password")
project_id = st.secrets.get("IBM_PROJECT_ID") or st.sidebar.text_input("IBM Project ID")
ibm_url = "https://eu-gb.ml.cloud.ibm.com"

if not api_key or not project_id:
    st.warning("Please provide IBM credentials in secrets or sidebar.")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head())
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.subheader("Scatter Matrix")
        fig = px.scatter_matrix(df, dimensions=num_cols, title="Scatter Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Correlation Heatmap")
        corr = df[num_cols].corr()
        fig2 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Ask the Data Analyst")
    if "messages" not in st.session_state:
        context_str = DATA_SUMMARY
        st.session_state.messages = [{
            "role": "system",
            "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}" 
        }]
    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])
    # Input
    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
                model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
                response = model.chat(messages=st.session_state.messages)
                answer = response.get("generated_text", "")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
