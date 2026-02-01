import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Config
st.set_page_config(layout="wide")
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

# Load data
df = pd.read_csv(DATA_URL)
# Clean column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"\W+", "_", regex=True)
    .str.strip("_")
    .str.lower()
)

# Summary for AI
context_str = """Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: 200."""

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
    st.title("Height vs Weight Dashboard")
    num_cols = df.select_dtypes(include=np.number).columns
    st.subheader("Data Overview")
    st.dataframe(df.head())
    if len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[1], y=num_cols[2], hover_data=num_cols)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Statistics")
    st.write(df.describe())

with tab2:
    st.title("🧠 Context-Aware Analyst")
    if "messages" not in st.session_state:
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
    # Prompt input
    prompt = st.chat_input("Ask a question about the dataset...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Call IBM Granite model
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get("generated_text", "")
        else:
            answer = "API credentials not provided."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f"**Assistant:** {answer}")
