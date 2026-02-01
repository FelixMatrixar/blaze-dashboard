import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(layout="wide")

# Load data
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(DATA_URL)

# Aggressive sanitization: Keep only letters, numbers, and underscores
df.columns = [
    "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
    for col in df.columns
]

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Analyst"])

with tab1:
    st.header("Scatter Matrix and Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        # Scatter matrix
        st.subheader("Scatter Matrix")
        sns.pairplot(df[numeric_cols])
        st.pyplot(plt.gcf())
        # Heatmap
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for visualizations.")

with tab2:
    st.header("Watsonx.ai Analyst")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # System prompt with data summary
        st.session_state.messages.append({"role": "system", "content": "You are a data analyst. Dataset summary: Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: ~200."})
    # User input
    user_input = st.chat_input("Ask a question about the data...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Placeholder for Watsonx.ai call – replace with actual SDK usage
        # Example:
        # from ibm_watsonx_ai import Credentials, ModelInference
        # credentials = Credentials(apikey="YOUR_API_KEY", url="https://eu-gb.ml.cloud.ibm.com")
        # model = ModelInference(credentials=credentials, project_id="YOUR_PROJECT_ID")
        # response = model.generate_text(prompt=st.session_state.messages)
        # assistant_reply = response.result['generated_text']
        assistant_reply = "(Watsonx.ai response would appear here)"
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
