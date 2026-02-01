import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ibm_watsonx_ai import Credentials, AI

# Page config
st.set_page_config(layout="wide")

# Data URL
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

# Load data
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

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Analyst"])

with tab1:
    st.header("Data Overview")
    st.dataframe(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("Scatter Matrix")
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig2, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig2)
    else:
        st.info("Not enough numeric columns for visualisations.")

with tab2:
    st.header("Watsonx Analyst")
    # Initialize Watsonx credentials (replace with actual keys securely)
    # These placeholders should be set as Streamlit secrets.
    credentials = Credentials(
        url="https://eu-gb.ml.cloud.ibm.com",
        apikey=st.secrets["watsonx_apikey"],
        instance_id=st.secrets["watsonx_instance_id"]
    )
    ai = AI(credentials)
    # System prompt with dataset summary
    system_prompt = "You are an analyst. Data summary: Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: 50+."
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    # User input
    if user_input := st.chat_input("Ask a question about the data"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = ai.chat.completions.create(
                model_id="ibm/granite-13b-v2",
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
                max_tokens=500
            )
            answer = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
