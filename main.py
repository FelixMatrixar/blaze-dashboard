import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibm_watsonx_ai.foundation_models import ModelInference

# Page config
st.set_page_config(layout="wide")

# Load data
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

df = load_data(DATA_URL)

# Column sanitization (mandatory)
df.columns = [
    "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
    for col in df.columns
]

# DATA_SUMMARY injected for the analyst chat
DATA_SUMMARY = "Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: 200."

# Tabs
tab_dashboard, tab_analyst = st.tabs(["📊 Dashboard", "🧠 Analyst"])

with tab_dashboard:
    st.header("Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        # Scatter matrix
        st.subheader("Scatter Matrix")
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig2, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig2)
    else:
        st.write("No numeric columns detected.")

with tab_analyst:
    st.header("AI Analyst")
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    # User input
    if prompt := st.chat_input("Ask a question about the data..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Prepare system prompt with data summary
        system_prompt = f"You are an AI data analyst. Use the following data summary to answer questions. {DATA_SUMMARY}"
        # Initialize Watsonx model (placeholder model name)
        model = ModelInference(
            model_id="google/flan-t5-xl",
            params={"decoding_method": "greedy", "max_new_tokens": 200},
            project_id=st.secrets.get("IBM_PROJECT_ID"),
            api_key=st.secrets.get("IBM_API_KEY"),
            url="https://eu-gb.ml.cloud.ibm.com"
        )
        # Perform inference
        response = model.generate(prompt, system_prompt=system_prompt)
        answer = response.get("generated_text", "[No response]")
        # Append assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Display assistant message
        with st.chat_message("assistant"):
            st.write(answer)
