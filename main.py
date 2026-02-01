# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Page config
st.set_page_config(layout="wide")

# Load dataset
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Clean column names safely
    df.columns = df.columns.astype(str).str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('"', '').str.replace("'", "")
    return df

df = load_data(DATA_URL)

# Smart authentication
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not project_id:
    project_id = st.sidebar.text_input("IBM Project ID")
ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) >= 2:
        x_axis = st.selectbox("X axis", options=num_cols, index=0)
        y_axis = st.selectbox("Y axis", options=num_cols, index=1)
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough numeric columns for a scatter plot.")

with tab2:
    st.header("Ask about the data")
    # Initialize session state for chat
    if "messages" not in st.session_state:
        # Hardcoded data summary from inspection
        context_str = "Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: approx 50+."
        st.session_state.messages = [{
            "role": "system",
            "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"
        }]
    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.markdown(f"**Assistant:** {msg['content']}")
        else:
            st.markdown(f"**User:** {msg['content']}")
    # Input
    prompt = st.chat_input("Ask a question about the dataset")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Prepare credentials
        creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
        model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
        # Call model
        response = model.chat(messages=st.session_state.messages)
        answer = response.get('generated_text', 'No response')
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f"**Assistant:** {answer}")
