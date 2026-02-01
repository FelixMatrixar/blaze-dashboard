import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibm_watsonx_ai.foundation_models import ModelInference

DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

st.set_page_config(layout="wide")

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

# Sidebar for navigation
tab = st.sidebar.selectbox("Select Tab", ["📊 Dashboard", "🧠 Analyst"])

if tab == "📊 Dashboard":
    st.title("📊 Data Dashboard")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    st.subheader("Data Overview")
    st.dataframe(df)

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
        st.write("Not enough numeric columns for visualizations.")

elif tab == "🧠 Analyst":
    st.title("🧠 Watsonx Analyst")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # System prompt with data summary
    system_prompt = "You are an analyst assistant. Use the following data summary to answer questions. Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: ~50+."
    user_input = st.text_input("Ask a question about the data:")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Initialize model inference (placeholder – credentials to be set in Streamlit secrets)
        model = ModelInference(
            model_id="meta-llama/llama-2-70b-chat",
            params={"decoding_method": "greedy", "max_new_tokens": 200},
            credentials={"url": "https://eu-gb.ml.cloud.ibm.com", "apikey": st.secrets["watsonx"]["apikey"]}
        )
        response = model.generate(
            prompt=system_prompt + "\nUser: " + user_input + "\nAssistant:",
            max_new_tokens=200,
        )
        assistant_msg = response.get("generated_text", "")
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {msg["content"]}")
