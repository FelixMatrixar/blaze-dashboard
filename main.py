import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Config
st.set_page_config(layout="wide")
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

df = pd.read_csv(DATA_URL)
# Safe cleaning
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.strip("_")
    .str.lower()
)

# Sidebar for IBM credentials
api_key = st.secrets.get("IBM_API_KEY") or st.sidebar.text_input("IBM API Key", type="password")
project_id = st.secrets.get("IBM_PROJECT_ID") or st.sidebar.text_input("IBM Project ID")
ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.title("Height vs Weight Dashboard")
    num_cols = df.select_dtypes(include=np.number).columns
    st.write("Numeric columns:", list(num_cols))
    if set(["height(inches)", "weight(pounds)"]).issubset(set(df.columns)):
        fig = px.scatter(df, x="height(inches)", y="weight(pounds)", hover_data=[df.columns[0]])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Columns for height and weight not found.")
    st.dataframe(df)

with tab2:
    st.title("🧠 Context-Aware Analyst")
    # Initialize chat history
    if "messages" not in st.session_state:
        context_str = "Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: approx 50+."
        st.session_state.messages = [{"role": "system", "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"}]
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])  # type: ignore
    # User input
    if prompt := st.chat_input("Ask a question about the data:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Call IBM Granite model
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get("generated_text", "")
        else:
            answer = "Please provide IBM API credentials in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
