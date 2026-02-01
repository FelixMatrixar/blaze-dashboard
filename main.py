# main/main.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibm_watsonx_ai.foundation_models import ModelInference

# Page config
st.set_page_config(layout="wide")

# Secrets
api_key = st.secrets.get("IBM_API_KEY")
ibm_url = st.secrets.get("IBM_URL")
ibm_project_id = st.secrets.get("IBM_PROJECT_ID")

# Fallbacks if secrets missing
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not ibm_url:
    ibm_url = st.sidebar.text_input("IBM URL", value="https://eu-gb.ml.cloud.ibm.com")
if not ibm_project_id:
    ibm_project_id = st.sidebar.text_input("IBM Project ID")

# Load data
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

# Watsonx model
model = ModelInference(
    model_id="ibm/granite-13b-chat-v2",
    credentials={
        "apikey": api_key,
        "url": ibm_url
    },
    project_id=ibm_project_id
)

# Data summary for analyst
DATA_SUMMARY = "Columns: index, height_inches, weight_pounds. Target: weight_pounds. Rows: 200."

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Analyst"])

with tab1:
    st.header("Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    st.subheader("Scatter Matrix")
    if len(numeric_cols) >= 2:
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)
    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig2, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig2)

with tab2:
    st.header("AI Analyst")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.chat_input("Ask a question about the data")
    if user_input:
        # Build prompt
        system_prompt = f"You are a data analyst. Use the following data summary to answer questions: {DATA_SUMMARY}"
        response = model.generate(
            prompt=system_prompt + "\nUser: " + user_input,
            max_new_tokens=200,
            temperature=0.7,
        )
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Assistant", response.get('results')[0].get('generated_text')))
    for role, message in st.session_state.chat_history:
        if role == "User":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)
