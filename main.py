# main/main.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibm_watsonx_ai.foundation_models import ModelInference

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
st.set_page_config(layout="wide")

# ---------------------------------------------------
# Secrets handling
# ---------------------------------------------------
api_key = st.secrets.get("IBM_API_KEY")
ibm_url = st.secrets.get("IBM_URL")
ibm_project_id = st.secrets.get("IBM_PROJECT_ID")

# Fallback values (only used if secrets are missing)
if not api_key:
    api_key = "YOUR_API_KEY"
if not ibm_url:
    ibm_url = "https://us-south.ml.cloud.ibm.com"
if not ibm_project_id:
    ibm_project_id = "YOUR_PROJECT_ID"

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(DATA_URL)

# ---------------------------------------------------
# Column sanitization (mandatory)
# ---------------------------------------------------
df.columns = [
    "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
    for col in df.columns
]

# ---------------------------------------------------
# IBM watsonx.ai model initialization
# ---------------------------------------------------
model = ModelInference(
    model_id="ibm/granite-13b-chat-v2",
    credentials={
        "apikey": api_key,
        "url": ibm_url,
        "version": "5.0"
    },
    project_id=ibm_project_id,
)

# ---------------------------------------------------
# Data summary for system prompt
# ---------------------------------------------------
DATA_SUMMARY = "Columns: index, height_inches, weight_pounds. Target: weight_pounds. Rows: 50+."

# ---------------------------------------------------
# Streamlit app layout
# ---------------------------------------------------
st.title("Human Height & Weight Dashboard")

tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Analyst"])

with tab1:
    st.header("Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.subheader("Scatter Matrix")
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)
        
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig2, ax2 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.write("No numeric columns detected.")

with tab2:
    st.header("AI Analyst")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.chat_input("Ask a question about the data...")
    if user_input:
        # Prepare prompt with data summary
        system_prompt = f"You are an analyst. Use the following data summary to answer questions. {DATA_SUMMARY}"
        # Call Watsonx AI model
        response = model.generate(
            prompt=system_prompt + "\nUser: " + user_input,
            max_new_tokens=300,
        )
        answer = response.get('generated_text', 'No response')
        st.session_state.chat_history.append((user_input, answer))
        for q, a in st.session_state.chat_history:
            st.write(f"**User:** {q}")
            st.write(f"**Assistant:** {a}")
