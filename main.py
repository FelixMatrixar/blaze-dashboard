# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai import ModelInference

st.set_page_config(layout="wide")
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(DATA_URL)
# Clean column names
df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True).str.lower()

# DATA_SUMMARY
DATA_SUMMARY = "Columns: index, height_inches, weight_pounds. Rows: 200."

# Authentication
if "IBM_API_KEY" in st.secrets:
    api_key = st.secrets["IBM_API_KEY"]
else:
    api_key = st.sidebar.text_input("Watsonx API Key", type="password")

if "IBM_PROJECT_ID" in st.secrets:
    project_id = st.secrets["IBM_PROJECT_ID"]
else:
    project_id = st.sidebar.text_input("Watsonx Project ID", type="password")

model_id = st.sidebar.selectbox("Model", ["ibm/granite-13b-chat-v2", "meta-llama/llama-3-70b-instruct"])  # type: ignore

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.header("Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X axis", numeric_cols, index=0)
            y_col = st.selectbox("Y axis", numeric_cols, index=1)
            fig_scatter = px.scatter(df, x=x_col, y=y_col, trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.write("No numeric columns found.")

with tab2:
    st.header("Chat with Data Analyst")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": f"You are a Data Analyst. Answer based on this dataset structure: {DATA_SUMMARY}"}]
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    if api_key and project_id:
        prompt = st.chat_input("Ask a question about the data...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            # Call Watsonx AI
            model = ModelInference(
                model_id=model_id,
                credentials={"url": "https://us-south.ml.cloud.ibm.com", "apikey": api_key},
                project_id=project_id,
            )
            response = model.chat(messages=st.session_state.messages)
            answer = response.get("result", {}).get("generated_text", "")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
    else:
        st.warning("Please provide IBM API Key and Project ID in the sidebar or secrets.")
