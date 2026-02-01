# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(layout="wide")

DATASET_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

df = pd.read_csv(DATASET_URL)
# Clean column names
df.columns = [c.strip().replace('"','').replace(' ','_') for c in df.columns]

# Secret Management
if "IBM_API_KEY" in st.secrets:
    ibm_api_key = st.secrets["IBM_API_KEY"]
else:
    ibm_api_key = st.sidebar.text_input("Enter IBM API Key", type="password")

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 AI Analyst"])

with tab1:
    st.header("Human Height & Weight Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Height (in)", f"{df.Height_Inches.mean():.2f}")
        st.metric("Average Weight (lb)", f"{df.Weight_Pounds.mean():.2f}")
    with col2:
        st.metric("Min Height", f"{df.Height_Inches.min():.2f}")
        st.metric("Max Weight", f"{df.Weight_Pounds.max():.2f}")
    st.subheader("Scatter: Height vs Weight")
    fig_scatter = px.scatter(df, x="Height_Inches", y="Weight_Pounds", trendline="ols")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.subheader("Correlation Heatmap")
    corr = df[["Height_Inches","Weight_Pounds"]].corr()
    fig_heat = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.subheader("Data Table")
    st.dataframe(df)

with tab2:
    st.header("🤖 AI Analyst")
    prompt = st.chat_input("Ask a question about the data…")
    if prompt:
        if not ibm_api_key:
            st.warning("⚠️ API Key missing. Please enter it in the sidebar.")
        else:
            url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22/v1/assistants/chat"
            headers = {"Authorization": f"Bearer {ibm_api_key}", "Content-Type": "application/json"}
            payload = {"input": {"text": prompt}}
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                reply = data.get('output', {}).get('generic', [{}])[0].get('text', str(data))
                st.write(reply)
            except Exception as e:
                st.error(f"Error contacting Watsonx: {e}")
