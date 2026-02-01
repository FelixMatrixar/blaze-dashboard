import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Page config
st.set_page_config(layout="wide")

# Load data
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Clean column names
    df.columns = df.columns.astype(str).str.replace(r"[^\w]", "_", regex=True).str.lower()
    return df

df = load_data()

# Summary for AI
DATA_SUMMARY = "Columns: Index, Height_Inches, Weight_Pounds. Target: Weight_Pounds. Rows: approx 200."

# Authentication
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")
if not api_key:
    st.sidebar.text_input("IBM API Key", type="password", key="api_key_input")
    api_key = st.session_state.get("api_key_input")
if not project_id:
    st.sidebar.text_input("IBM Project ID", key="project_id_input")
    project_id = st.session_state.get("project_id_input")

ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context‑Aware Analyst"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df)
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) >= 2:
        x_col = st.selectbox("X axis", options=num_cols, index=0)
        y_col = st.selectbox("Y axis", options=num_cols, index=1)
        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Statistics")
    st.write(df.describe())

with tab2:
    st.header("Ask the Data Analyst")
    if "messages" not in st.session_state:
        context_str = DATA_SUMMARY
        st.session_state.messages = [{"role": "system", "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"}]
    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    # User input
    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Call IBM Granite model
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            try:
                response = model.chat(messages=st.session_state.messages)
                answer = response.get("generated_text", "")
            except Exception as e:
                answer = f"Error calling model: {e}"
        else:
            answer = "Please provide IBM API credentials in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
