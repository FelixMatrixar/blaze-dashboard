import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Secrets ----
api_key = st.secrets.get("IBM_API_KEY")
ibm_url = st.secrets.get("IBM_URL")
ibm_project_id = st.secrets.get("IBM_PROJECT_ID")

# Fallbacks (if secrets missing)
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not ibm_url:
    ibm_url = st.sidebar.text_input("IBM URL")
if not ibm_project_id:
    ibm_project_id = st.sidebar.text_input("IBM Project ID")

# ---- Page Config ----
st.set_page_config(layout="wide")

# ---- Load Data ----
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data

def load_data(url):
    df = pd.read_csv(url)
    # Aggressive sanitization: Keep only letters, numbers, and underscores
    df.columns = [
        "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
        for col in df.columns
    ]
    return df

df = load_data(DATA_URL)

# ---- DATA_SUMMARY ----
DATA_SUMMARY = "Columns: index, height_inches, weight_pounds. Target: weight_pounds. Rows: 50+."

# ---- Watsonx Model ----
from ibm_watsonx_ai.foundation_models import ModelInference
model = ModelInference(
    model_id="ibm/granite-13b-chat-v2",
    credentials={
        "apikey": api_key,
        "url": ibm_url,
        "version": "5.0"
    },
    project_id=ibm_project_id,
)

# ---- App Layout ----
tabs = st.tabs(["📊 Dashboard", "🧠 Analyst"])

with tabs[0]:
    st.header("Data Overview")
    st.dataframe(df)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("Scatter Matrix")
        sns.pairplot(df[numeric_cols])
        st.pyplot(plt.gcf())
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
    else:
        st.info("Not enough numeric columns for visualizations.")

with tabs[1]:
    st.header("AI Analyst")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.chat_input("Ask a question about the dataset...")
    if user_input:
        # Prepare prompt with data summary
        system_prompt = f"You are an analyst. {DATA_SUMMARY} Use this information to answer user queries."
        response = model.generate(
            prompt=system_prompt + "\nUser: " + user_input,
            max_new_tokens=300,
            temperature=0.7,
        )
        answer = response['results'][0]['generated_text'] if isinstance(response, dict) else str(response)
        st.session_state.chat_history.append((user_input, answer))
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
