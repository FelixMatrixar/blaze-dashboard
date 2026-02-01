import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibm_watsonx_ai.foundation_models import ModelInference

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
st.set_page_config(layout="wide")

# ------------------------------------------------------------
# Secrets handling
# ------------------------------------------------------------
api_key = st.secrets.get("IBM_API_KEY")
ibm_url = st.secrets.get("IBM_URL")
ibm_project_id = st.secrets.get("IBM_PROJECT_ID")

# Fallbacks (if needed)
if not api_key:
    api_key = st.sidebar.text_input("IBM API Key", type="password")
if not ibm_url:
    ibm_url = st.sidebar.text_input("IBM URL")
if not ibm_project_id:
    ibm_project_id = st.sidebar.text_input("IBM Project ID")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data

def load_data(url):
    return pd.read_csv(url)

df = load_data(DATA_URL)

# ------------------------------------------------------------
# Column sanitization (mandatory)
# ------------------------------------------------------------
df.columns = [
    "".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower()
    for col in df.columns
]

# ------------------------------------------------------------
# Watsonx model setup
# ------------------------------------------------------------
model = ModelInference(
    model_id="ibm/granite-13b-chat-v2",
    credentials={{
        "apikey": api_key,
        "url": ibm_url,
        "version": "5.0"
    }},
    project_id=ibm_project_id,
)

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🧠 Analyst"])

# ------------------------------------------------------------
# DATA_SUMMARY injection
# ------------------------------------------------------------
DATA_SUMMARY = "Columns: index, height_inches, weight_pounds. Target: weight_pounds. Rows: 50+."

if page == "📊 Dashboard":
    st.title("📊 Data Dashboard")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

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
        st.info("Not enough numeric columns for visualizations.")

elif page == "🧠 Analyst":
    st.title("🧠 Watsonx Analyst")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Ask a question about the data:")
    if st.button("Send") and user_input:
        # Build prompt
        system_prompt = f"You are a data analyst. Use the following dataset summary to answer questions. {DATA_SUMMARY}"
        # Perform inference (placeholder - actual call may vary)
        try:
            response = model.generate(
                prompt=system_prompt + "\n\n" + user_input,
                max_new_tokens=200,
                temperature=0.7,
            )
            answer = response.get('results', [{}])[0].get('generated_text', 'No response')
        except Exception as e:
            answer = f"Error during model inference: {e}"
        st.session_state.chat_history.append((user_input, answer))
    if st.session_state.chat_history:
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
