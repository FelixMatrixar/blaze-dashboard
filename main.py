# ------------------------------------------------------------
# main.py – Context‑Aware Data Analyst Dashboard
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# ------------------------------------------------------------
# 0️⃣ Page configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Height‑Weight Analyzer",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------
# 1️⃣ Load dataset (hard‑coded URL)
# ------------------------------------------------------------
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    # The source CSV has a quirky header; we clean it on load
    df = pd.read_csv(url, skipinitialspace=True)
    # Strip stray quotes & spaces from column names
    df.columns = [c.replace('"', '').strip() for c in df.columns]
    return df

df = load_data(DATA_URL)

# ------------------------------------------------------------
# 2️⃣ Sidebar – IBM credentials (fallback to manual entry)
# ------------------------------------------------------------
st.sidebar.header("🔐 IBM Watsonx.ai Authentication")
ibm_api_key = st.secrets.get("IBM_API_KEY") or st.sidebar.text_input(
    "IBM API Key", type="password"
)
ibm_project_id = st.secrets.get("IBM_PROJECT_ID") or st.sidebar.text_input(
    "IBM Project ID"
)
ibm_url = st.secrets.get("IBM_URL") or st.sidebar.text_input(
    "IBM Service URL", value="https://us-south.ml.cloud.ibm.com"
)

# ------------------------------------------------------------
# 3️⃣ Tabs
# ------------------------------------------------------------
tab_dashboard, tab_analyst = st.tabs(["📊 Dashboard", "🧠 Context‑Aware Analyst"])

# ------------------------------------------------------------
# 📊 Tab 1 – Exploratory Dashboard
# ------------------------------------------------------------
with tab_dashboard:
    st.title("Height‑Weight Explorer")
    st.subheader("Dataset preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Statistical Summary")
    st.write(df.describe())

    # Height vs Weight scatter
    if {"Height(Inches)", "Weight(Pounds)"} <= set(df.columns):
        fig = px.scatter(df,
                         x="Height(Inches)",
                         y="Weight(Pounds)",
                         hover_data=numeric_cols,
                         title="Height vs. Weight")
        st.plotly_chart(fig, use_container_width=True)

    # Histogram controls
    st.subheader("Distributions")
    col = st.selectbox("Select numeric column", numeric_cols, index=0)
    bins = st.slider("Number of bins", 5, 100, 30)
    hist_fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
    st.plotly_chart(hist_fig, use_container_width=True)

# ------------------------------------------------------------
# 🧠 Tab 2 – Context‑Aware Analyst (Chat)
# ------------------------------------------------------------
with tab_analyst:
    st.title("🧠 Context‑Aware Analyst")
    st.caption(
        "Ask any question about the dataset. The assistant is pre‑loaded with a concise description of the data."
    )

    # ---- 2️⃣ Initialise session state for chat history ----
    if "messages" not in st.session_state:
        # ----- DATA_SUMMARY injected below -----
        context_str = """Dataset has 3 columns: Index, Height(Inches), Weight(Pounds). Target might be 'Weight(Pounds)'. Approximately 200 rows."""
        st.session_state.messages = [
            {
                "role": "system",
                "content": f"You are a helpful Data Analyst. You are answering questions about a dataset with the following structure: {context_str}"
            }
        ]

    # ---- 3️⃣ Display chat history ----
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        elif msg["role"] == "user":
            st.chat_message("user").write(msg["content"])

    # ---- 4️⃣ Prompt input ----
    if prompt := st.chat_input("Ask a question about the data…"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # ---- 5️⃣ Call IBM Granite model ----
        if not ibm_api_key or not ibm_project_id:
            st.error(
                "🚨 Missing IBM credentials. Please provide IBM_API_KEY and IBM_PROJECT_ID in the sidebar."
            )
        else:
            try:
                creds = {"url": ibm_url, "apikey": ibm_api_key, "version": "5.0"}
                model = ModelInference(
                    model_id="ibm/granite-13b-chat-v2",
                    credentials=creds,
                    project_id=ibm_project_id,
                )
                response = model.chat(messages=st.session_state.messages)

                assistant_reply = response["generated_text"]
                # Append assistant reply
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_reply}
                )
                st.chat_message("assistant").write(assistant_reply)

            except Exception as e:
                st.error(f"❌ Error invoking model: {e}")

# ------------------------------------------------------------
# End of file
# ------------------------------------------------------------
