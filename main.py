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
    # Clean column names
    df.columns = [c.strip().replace('"', '') for c in df.columns]
    return df

df = load_data(DATA_URL)

# Authentication
if "IBM_API_KEY" in st.secrets and "IBM_PROJECT_ID" in st.secrets:
    api_key = st.secrets["IBM_API_KEY"]
    project_id = st.secrets["IBM_PROJECT_ID"]
else:
    st.sidebar.header("IBM Credentials")
    api_key = st.sidebar.text_input("API Key", type="password")
    project_id = st.sidebar.text_input("Project ID")

ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Context Summary
DATA_SUMMARY = "Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: approx 200."

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context‑Aware Analyst"])

with tab1:
    st.title("Height vs Weight Dashboard")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Data Overview")
    st.dataframe(df.head())
    st.subheader("Scatter Plot")
    fig = px.scatter(df, x="Height(Inches)", y="Weight(Pounds)", hover_data=numeric_cols)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Statistics")
    st.write(df.describe())

with tab2:
    st.title("Context‑Aware Analyst")
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "system",
            "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {DATA_SUMMARY}"
        }]
    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])  # type: ignore
        else:
            st.chat_message("user").write(msg["content"])  # type: ignore
    # Prompt input
    if prompt := st.chat_input("Ask a question about the data:"):
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
            answer = "IBM credentials not provided."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
