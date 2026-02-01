# ------------------------------------------------------------
# main.py – Context‑Aware Data Analyst Dashboard
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Context‑Aware Data Analyst",
                   layout="wide")

# ------------------------------------------------------------
# Load dataset (hard‑coded URL as requested)
# ------------------------------------------------------------
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Universal cleaning: safe column names
    df.columns = (
        df.columns.astype(str)
        .str.replace(r'[^\w]', '_', regex=True)
        .str.lower()
    )
    return df

df = load_data()

# ------------------------------------------------------------
# Sidebar – IBM credentials (fallback if not in st.secrets)
# ------------------------------------------------------------
st.sidebar.header("🔐 IBM Watsonx.ai Credentials")
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")

if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not project_id:
    project_id = st.sidebar.text_input("IBM Project ID")

# IBM endpoint for EU‑GB region (hard‑coded as required)
ibm_url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22"

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_dashboard, tab_analyst = st.tabs(["📊 Dashboard", "🧠 Context‑Aware Analyst"])

# ----------------------- Tab 1: Dashboard -----------------------
with tab_dashboard:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        col_x = st.selectbox("X‑axis", numeric_cols, index=0, key="x_axis")
        col_y = st.selectbox("Y‑axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="y_axis")

        fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Statistical Summary")
    st.write(df.describe())

# ------------------- Tab 2: Context‑Aware Analyst -------------------
with tab_analyst:
    st.subheader("Ask questions about the dataset")
    # ----------------------------------------------------------------
    # Initialise session state for chat history
    # ----------------------------------------------------------------
    if "messages" not in st.session_state:
        # ------------------------------------------------------------
        # DATA_SUMMARY injected directly (core feature)
        # ------------------------------------------------------------
        DATA_SUMMARY = (
            "Dataset has 3 columns: Index, Height_Inches, Weight_Pounds. "
            "Target might be Weight_Pounds. Approx. 50+ rows."
        )
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful Data Analyst. You are answering questions about a "
                    f"dataset with the following structure: {DATA_SUMMARY}"
                )
            }
        ]

    # ----------------------------------------------------------------
    # Display chat history
    # ----------------------------------------------------------------
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ----------------------------------------------------------------
    # Prompt input
    # ----------------------------------------------------------------
    if prompt := st.chat_input("Enter your question about the data..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ----------------------------------------------------------------
        # Call Watsonx.ai model
        # ----------------------------------------------------------------
        if not api_key or not project_id:
            error_msg = "❗ IBM credentials are missing. Please provide them in the sidebar."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            try:
                creds = {"url": ibm_url, "apikey": api_key}
                model = ModelInference(
                    model_id="ibm/granite-13b-chat-v2",
                    credentials=creds,
                    project_id=project_id,
                )
                response = model.chat(messages=st.session_state.messages)
                answer = response.get("generated_text", "No response returned.")
            except Exception as e:
                answer = f"❗ Error calling IBM model: {e}"
            # Append assistant reply
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.caption(
    "🚀 Live!  \n"
    "URL: https://blaze-dashboard.streamlit.app/  \n"
    "*Dashboard configured for London (EU‑GB) region.*"
)
