import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ibm_watsonx_ai.foundation_models import ModelInference

# Configuration
st.set_page_config(layout="wide")

# Dataset URL (hardcoded)
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

# Load data
df = pd.read_csv(DATA_URL)

# Aggressive sanitization: Keep only letters, numbers, and underscores
df.columns = [
    "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
    for col in df.columns
]

# Authentication
api_key = st.secrets.get("IBM_API_KEY") or st.sidebar.text_input("IBM API Key", type="password")
project_id = st.secrets.get("IBM_PROJECT_ID") or st.sidebar.text_input("IBM Project ID")
ibm_url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22"

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df)
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) >= 2:
        st.subheader("Scatter Matrix")
        fig = px.scatter_matrix(df, dimensions=num_cols)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Correlation Heatmap")
        corr = df[num_cols].corr()
        heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis'))
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Not enough numeric columns for visualizations.")

with tab2:
    st.header("Ask the Data Analyst")
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "system",
            "content": (
                "You are a helpful Data Analyst. "
                "You are answering questions about a dataset with the following structure: "
                "Columns: index, height_inches, weight_pounds. Target: weight_pounds. Rows: 50+."
            )
        }]
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask a question about the data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Call IBM watsonx.ai
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get("generated_text", "")
        else:
            answer = "IBM credentials not provided."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
