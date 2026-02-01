import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(layout="wide")

# Load data
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(DATA_URL)

# Aggressive sanitization: Keep only letters, numbers, and underscores
df.columns = [
    "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
    for col in df.columns
]

# Watsonx AI placeholder (credentials should be set in Streamlit secrets)
def call_watsonx(prompt: str) -> str:
    import json
    from ibm_watsonx_ai import Credentials, ModelInference
    credentials = Credentials(
        url="https://eu-gb.ml.cloud.ibm.com",
        apikey=st.secrets["watsonx"]["apikey"],
        instance_id=st.secrets["watsonx"]["instance_id"],
        project_id=st.secrets["watsonx"]["project_id"]
    )
    model_id = "meta-llama/llama-2-70b-chat"
    model_params = {"decoding_method": "greedy", "max_new_tokens": 200}
    model = ModelInference(
        model_id=model_id,
        params=model_params,
        credentials=credentials,
        project_id=st.secrets["watsonx"]["project_id"],
    )
    response = model.generate(prompt)
    return response.get("results", [{}])[0].get("generated_text", "")

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Analyst"])

with tab1:
    st.header("Scatter Matrix & Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        # Scatter matrix
        st.subheader("Scatter Matrix")
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)
        # Heatmap
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig2, ax2 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.write("Not enough numeric columns for visualizations.")

with tab2:
    st.header("Data Analyst Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        system_prompt = f"You are an analyst. Use the following dataset summary to answer questions. {"Columns: Index, Height(Inches), Weight(Pounds). Target: Weight(Pounds). Rows: ~50."}"
        st.session_state.messages.append({"role": "system", "content": system_prompt})
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])  # type: ignore
    if user_input := st.chat_input("Ask a question about the data"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = call_watsonx(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
