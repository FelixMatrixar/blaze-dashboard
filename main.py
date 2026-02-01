import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Configuration
st.set_page_config(layout="wide")

DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

# Load data
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Clean column names
    df.columns = [c.strip().replace('"', '').replace("'", "").replace(' ', '_').lower() for c in df.columns]
    return df

df = load_data(DATA_URL)

# Sidebar for authentication (fallback)
if 'IBM_API_KEY' in st.secrets and 'IBM_PROJECT_ID' in st.secrets:
    api_key = st.secrets["IBM_API_KEY"]
    project_id = st.secrets["IBM_PROJECT_ID"]
else:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
    project_id = st.sidebar.text_input("IBM Project ID")

ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context-Aware Analyst"])

with tab1:
    st.header("Data Overview")
    st.dataframe(df)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Histograms")
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Scatter Plot: Height vs Weight")
    if set(["height_inches", "weight_pounds"]).issubset(df.columns):
        fig = px.scatter(df, x="height_inches", y="weight_pounds", trendline="ols", title="Height vs Weight")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Columns for Height and Weight not found.")

with tab2:
    st.header("🧠 Context‑Aware Analyst")
    # Initialize chat history
    if "messages" not in st.session_state:
        context_str = """Columns: Index, Height_Inches, Weight_Pounds. Target: Weight_Pounds. Rows: ~50. """
        st.session_state.messages = [{"role": "system", "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"}]
    
    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])
    
    # Prompt input
    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Call IBM Granite model
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            response = model.chat(messages=st.session_state.messages)
            answer = response.get("result", {}).get("generated_text", "")
        else:
            answer = "Please provide IBM API credentials in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
