import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ibm_watsonx_ai.foundation_models import ModelInference

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="Height‑Weight Dashboard")

# -------------------------------------------------
# Load data
# -------------------------------------------------
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data

def load_data(url):
    df = pd.read_csv(url)
    # Clean column names
    df.columns = (
        df.columns.str.replace(r"[^"]+", "", regex=True)  # remove stray quotes
        .str.replace(r"[\s\(\)]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.lower()
    )
    return df

df = load_data(DATA_URL)

# -------------------------------------------------
# Universal cleaning (extra safety)
# -------------------------------------------------
df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True).str.lower()

# -------------------------------------------------
# Sidebar – IBM Watsonx credentials & model selection
# -------------------------------------------------
st.sidebar.header("🔐 IBM Watsonx Settings")
api_key = st.sidebar.text_input("API Key", type="password")
project_id = st.sidebar.text_input("Project ID")
model_id = st.sidebar.text_input("Model ID", value="ibm/granite-13b-v2")

# Critical IBM endpoint – hard‑coded as required
ibm_url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22"

# -------------------------------------------------
# Session state for chat history
# -------------------------------------------------
if "messages" not in st.session_state:
    # System prompt injects the dataset summary
    st.session_state.messages = [
        {
            "role": "system",
            "content": "The dataset contains 3 columns: Index, Height(Inches), and Weight(Pounds). It has roughly 50 rows (sampled). The likely target variable is Weight(Pounds)."
        }
    ]

# -------------------------------------------------
# Tabs layout
# -------------------------------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Context‑Aware Analyst"]) 

with tab1:
    st.title("Height vs. Weight Exploration")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Data preview")
    st.dataframe(df.head())

    # Scatter plot
    if "height_inches" in df.columns and "weight_pounds" in df.columns:
        fig = px.scatter(
            df,
            x="height_inches",
            y="weight_pounds",
            hover_data=numeric_cols,
            title="Height vs. Weight",
            labels={"height_inches": "Height (inches)", "weight_pounds": "Weight (pounds)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Expected columns not found after cleaning.")

    # Simple statistics
    st.subheader("Summary statistics")
    st.write(df.describe())

with tab2:
    st.title("💬 Context‑Aware Analyst")
    st.caption("Ask any question about the dataset. The model will use the embedded summary above.")

    # Chat input
    user_prompt = st.chat_input("Enter your question about the data…")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        # Build ModelInference instance
        try:
            creds = {"url": ibm_url, "apikey": api_key}
            model = ModelInference(
                model_id=model_id,
                credentials=creds,
                project_id=project_id,
            )
            response = model.chat(messages=st.session_state.messages)
            answer = response.get("generated_text", "[No response]")
        except Exception as e:
            answer = f"Error calling IBM Watsonx: {e}"
        # Append assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Display chat history
        for msg in st.session_state.messages[1:]:  # skip system message
            with st.chat_message(msg["role"]):
                st.write(msg["content"]) 
