import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Page config
st.set_page_config(layout="wide")

# Load data
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Safe column cleaning
    df.columns = [c.strip().replace('"', '').replace("'", "").replace(' ', '_').lower() for c in df.columns]
    return df

df = load_data(DATA_URL)

# Extract summary
DATA_SUMMARY = "Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: ~50."

# Sidebar for authentication
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
    if num_cols:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1] if len(num_cols)>1 else num_cols[0],
                         title=f"Scatter of {num_cols[0]} vs {num_cols[1] if len(num_cols)>1 else num_cols[0]}")
        st.plotly_chart(fig, use_container_width=True)
    st.write("## Statistics")
    st.write(df.describe())

with tab2:
    st.header("Ask the Analyst")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {DATA_SUMMARY}"}]
    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])  # type: ignore
        else:
            st.chat_message("user").write(msg["content"])  # type: ignore
    # User input
    if prompt := st.chat_input("Ask a question about the data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Call IBM Granite model
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get('generated_text', '')
        else:
            answer = "IBM credentials not provided."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
