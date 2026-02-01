import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Height & Weight Dashboard")

# --- Load Data ---
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Clean column names
    df.columns = df.columns.astype(str).str.replace(r"[^\w]", "_", regex=True).str.lower()
    return df

df = load_data(DATA_URL)

# --- Summary Context ---
DATA_SUMMARY = "Columns: Index, Height(Inches), Weight(Pounds). Target (potential): Weight(Pounds). Rows: ~50+."

# --- Authentication ---
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not project_id:
    project_id = st.sidebar.text_input("IBM Project ID")
ibm_url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22"

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.header("Height vs Weight Overview")
    num_cols = df.select_dtypes(include=np.number).columns
    st.subheader("Data Preview")
    st.dataframe(df.head())
    # Scatter plot
    fig = px.scatter(df, x='height_inches', y='weight_pounds', trendline='ols',
                     labels={'height_inches':'Height (inches)', 'weight_pounds':'Weight (pounds)'},
                     title='Height vs Weight')
    st.plotly_chart(fig, use_container_width=True)
    # Distribution
    col1, col2 = st.columns(2)
    with col1:
        hist_h = px.histogram(df, x='height_inches', nbins=20, title='Height Distribution')
        st.plotly_chart(hist_h, use_container_width=True)
    with col2:
        hist_w = px.histogram(df, x='weight_pounds', nbins=20, title='Weight Distribution')
        st.plotly_chart(hist_w, use_container_width=True)

with tab2:
    st.header("🧠 Context‑Aware Analyst")
    # Initialise session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "system",
            "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {DATA_SUMMARY}"
        }]
    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])
    # Prompt input
    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Call IBM Granite model
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get('generated_text', 'No response')
        else:
            answer = "⚠️ IBM credentials not provided."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

st.sidebar.success("Ready to explore!")
