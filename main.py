import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# Page config
st.set_page_config(layout="wide")

# Load dataset
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Clean column names
    df.columns = [c.strip().replace('"', '').replace("'", "") for c in df.columns]
    return df

df = load_data(DATA_URL)

# Summary for AI
context_str = """Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: 200."""

# Sidebar for IBM credentials fallback
if 'IBM_API_KEY' in st.secrets and 'IBM_PROJECT_ID' in st.secrets:
    api_key = st.secrets['IBM_API_KEY']
    project_id = st.secrets['IBM_PROJECT_ID']
else:
    api_key = st.sidebar.text_input('IBM API Key', type='password')
    project_id = st.sidebar.text_input('IBM Project ID')

ibm_url = "https://eu-gb.ml.cloud.ibm.com"

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"
    }]

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context‑Aware Analyst"])

with tab1:
    st.title("Height & Weight Dashboard")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Data Overview")
    st.dataframe(df.head())
    st.subheader("Statistics")
    st.write(df.describe())
    st.subheader("Scatter Plot: Height vs Weight")
    fig = px.scatter(df, x=df.columns[1], y=df.columns[2], trendline="ols", labels={df.columns[1]: 'Height (inches)', df.columns[2]: 'Weight (pounds)'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.title("🧠 Context‑Aware Analyst")
    user_input = st.text_input("Ask a question about the data:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Prepare credentials
        if api_key and project_id:
            creds = {"url": ibm_url, "apikey": api_key, "version": "5.0"}
            model = ModelInference(model_id="ibm/granite-13b-chat-v2", credentials=creds, project_id=project_id)
            try:
                response = model.chat(messages=st.session_state.messages)
                answer = response.get('generated_text', '')
            except Exception as e:
                answer = f"Error calling IBM model: {e}"
        else:
            answer = "Please provide IBM API Key and Project ID in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Display chat history
        for msg in st.session_state.messages[1:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")
