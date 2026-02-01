import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# --- Page Config ---
st.set_page_config(page_title="Context‑Aware Data Dashboard", layout="wide")

# --- Load Data ---
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[^"]*\"", "", regex=True)  # remove stray quotes
        .str.replace(r"[\s\"]+", "_", regex=True)
        .str.lower()
    )
    return df

df = load_data()

# --- Authentication ---
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not project_id:
    project_id = st.sidebar.text_input("IBM Project ID")
ibm_url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22"

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context‑Aware Analyst"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head())
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        fig = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

with tab2:
    st.header("Ask the Data Analyst")
    # Initialize chat history
    if "messages" not in st.session_state:
        # Context from Phase 1
        context_str = """Columns: Index, Height(Inches), Weight(Pounds). Target (likely): Weight(Pounds). Rows: ~50+."""
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
    # User input
    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Call IBM Granite model
        if api_key and project_id:
            try:
                creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
                model = ModelInference(
                    model_id="ibm/granite-13b-chat-v2",
                    credentials=creds,
                    project_id=project_id,
                )
                response = model.chat(messages=st.session_state.messages)
                answer = response.get("result", {}).get("generated_text", "")
            except Exception as e:
                answer = f"Error calling model: {e}"
        else:
            answer = "Please provide IBM API credentials in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
