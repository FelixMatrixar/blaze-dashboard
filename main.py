import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Page config
st.set_page_config(layout="wide")

# Dataset URL (hardcoded)
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

# Load data
@st.cache_data

def load_data(url):
    df = pd.read_csv(url)
    # Safe column cleaning
    df.columns = df.columns.astype(str).str.replace(r'[^\w]', '_', regex=True).str.lower()
    return df

df = load_data(DATA_URL)

# Summary for system prompt
DATA_SUMMARY = "Columns: index, height_inches, weight_pounds. Target: weight_pounds. Rows: approx 50+."

# Authentication
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not project_id:
    project_id = st.sidebar.text_input("IBM Project ID")
ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.header("Exploratory Dashboard")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        x_axis = st.selectbox("X‑axis", options=num_cols, index=0)
        y_axis = st.selectbox("Y‑axis", options=num_cols, index=1 if len(num_cols) > 1 else 0)
        chart_type = st.radio("Chart type", ["Scatter", "Line", "Bar"])
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        else:
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric columns found.")

with tab2:
    st.header("🧠 Context‑Aware Analyst")
    if "messages" not in st.session_state:
        # System prompt with dataset summary
        context_str = f"{DATA_SUMMARY}"
        st.session_state.messages = [{"role": "system", "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"}]
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])
    # User input
    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Prepare model
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get('result', {}).get('generated_text', 'No response')
        else:
            answer = "Please provide IBM API credentials in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
