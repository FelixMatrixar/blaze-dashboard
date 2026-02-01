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
    # Aggressive sanitization: Keep only letters, numbers, and underscores
    df.columns = ["".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower() for col in df.columns]
    return df

df = load_data(DATA_URL)

# Sidebar for IBM credentials
st.sidebar.title("IBM Watsonx.ai Credentials")
api_key = st.secrets.get("IBM_API_KEY") or st.sidebar.text_input("API Key", type="password")
project_id = st.secrets.get("IBM_PROJECT_ID") or st.sidebar.text_input("Project ID")
ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "🧠 Context‑Aware Analyst"])

with tab1:
    st.header("Exploratory Dashboard")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Data Overview")
    st.dataframe(df.head())
    if len(num_cols) >= 2:
        x_axis = st.selectbox("X‑axis", options=num_cols, index=0)
        y_axis = st.selectbox("Y‑axis", options=num_cols, index=1)
        fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Statistics")
    st.write(df.describe())

with tab2:
    st.header("🧠 Context‑Aware Analyst")
    # Initialise chat history
    if "messages" not in st.session_state:
        context_str = "Columns: index, height_inches, weight_pounds. Target: weight_pounds. Rows: ~50."
        st.session_state.messages = [{
            "role": "system",
            "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"
        }]
    # Display chat
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])
    prompt = st.chat_input("Ask a question about the data...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
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
