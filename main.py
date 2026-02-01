import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(layout="wide")

# Load dataset (hardcoded URL)
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Clean column names
    df.columns = [c.strip().replace('"', '') for c in df.columns]
    return df

df = load_data(DATA_URL)

# Summary for AI context
context_str = """Dataset Summary: Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: ~50+."""

# Authentication
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not project_id:
    project_id = st.sidebar.text_input("IBM Project ID")
ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "🧠 Context‑Aware Analyst"])

with tab1:
    st.header("Data Overview")
    st.dataframe(df)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        fig = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Height vs Weight")
    fig2 = px.scatter(df, x="Height(Inches)", y="Weight(Pounds)", trendline="ols")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Ask the Data Analyst")
    if "messages" not in st.session_state:
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
    # Prompt input
    if prompt := st.chat_input("Ask a question about the data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Call IBM watsonx AI
        try:
            from ibm_watsonx_ai.foundation_models import ModelInference
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get('generated_text', '')
        except Exception as e:
            answer = f"Error calling IBM API: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
