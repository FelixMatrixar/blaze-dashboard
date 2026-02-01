# ──────────────────────────────────────────────────────────────────────────────
#   Context‑Aware Data Analyst Dashboard
#   Dataset: https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv
#   DATA_SUMMARY: Columns: Index, Height_Inches, Weight_Pounds. Target: Weight_Pounds. Rows: ~50.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# --------------------------------------------------------------------------- #
# Page config & data loading
# --------------------------------------------------------------------------- #
st.set_page_config(layout="wide", page_title="Context‑Aware Data Analyst")
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # universal cleaning: lower‑case, replace non‑word chars with underscore
    df.columns = (
        df.columns.astype(str)
        .str.replace(r'[^\w]', '_', regex=True)
        .str.replace(r'__+', '_', regex=True)
        .str.strip('_')
        .str.lower()
    )
    return df

df = load_data(DATA_URL)

# --------------------------------------------------------------------------- #
# Sidebar – IBM credentials (fallback to manual entry)
# --------------------------------------------------------------------------- #
st.sidebar.header("🔑 IBM Watsonx AI credentials")
api_key = st.secrets.get("IBM_API_KEY") or st.sidebar.text_input(
    "API Key", type="password"
)
project_id = st.secrets.get("IBM_PROJECT_ID") or st.sidebar.text_input(
    "Project ID"
)
ibm_url = "https://eu-gb.ml.cloud.ibm.com"   # EU‑GB region default

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Context‑Aware Analyst"])

# --------------------------------------------------------------------------- #
# Tab 1 – Quick visualisation & stats
# --------------------------------------------------------------------------- #
with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Descriptive statistics")
    st.write(df.describe())

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X‑axis", numeric_cols, index=0, key="xcol")
        y_col = st.selectbox("Y‑axis", numeric_cols, index=1, key="ycol")
        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2 – Context‑Aware Analyst (chat with Watsonx)
# --------------------------------------------------------------------------- #
with tab2:
    st.header("🧠 Ask questions about the dataset")

    # Initialise chat history with system prompt containing DATA_SUMMARY
    if "messages" not in st.session_state:
        context_str = (
            "You are a helpful Data Analyst. You are answering questions about a dataset "
            "with the following structure: Columns: Index, Height_Inches, Weight_Pounds. "
            "Target: Weight_Pounds. Rows: approximately 50."
        )
        st.session_state.messages = [
            {"role": "system", "content": context_str}
        ]

    # Display chat history
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])

    # Prompt input
    prompt = st.chat_input("Ask a question about the data …")

    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # ------------------------------------------------------------------- #
        # Call IBM Watsonx AI
        # ------------------------------------------------------------------- #
        if not api_key or not project_id:
            st.error("⚠️ Please provide both IBM_API_KEY and IBM_PROJECT_ID.")
        else:
            try:
                creds = {"url": ibm_url, "apikey": api_key}
                model = ModelInference(
                    model_id="ibm/granite-13b-chat-v2",
                    credentials=creds,
                    project_id=project_id,
                )
                response = model.chat(messages=st.session_state.messages)
                answer = response["generated_text"] if isinstance(response, dict) else str(response)

                # Append assistant reply
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
            except Exception as e:
                st.error(f"❗ Watsonx call failed: {e}")

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.caption(
    "🚀 **Live!**\n\nURL: https://blaze-dashboard.streamlit.app/\n\n*Dashboard configured for London (EU‑GB) region.*"
)
