import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(layout="wide")

DATASET_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(DATASET_URL)
# Clean column names
df.columns = [c.strip().replace('"','').replace(' ', '_') for c in df.columns]

# Secret management
if "IBM_API_KEY" in st.secrets:
    ibm_api_key = st.secrets["IBM_API_KEY"]
else:
    ibm_api_key = st.sidebar.text_input("Enter IBM API Key", type="password")

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 AI Analyst"])

with tab1:
    st.title("Height vs Weight Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Height (in)", f"{df.Height_Inches.mean():.2f}")
        st.metric("Mean Weight (lb)", f"{df.Weight_Pounds.mean():.2f}")
    with col2:
        st.metric("Min Height", f"{df.Height_Inches.min():.2f}")
        st.metric("Max Height", f"{df.Height_Inches.max():.2f}")
    fig_scatter = px.scatter(df, x="Height_Inches", y="Weight_Pounds", trendline="ols", title="Height vs Weight")
    st.plotly_chart(fig_scatter, use_container_width=True)
    fig_heat = px.density_heatmap(df, x="Height_Inches", y="Weight_Pounds", nbinsx=30, nbinsy=30, title="Density Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(df)

with tab2:
    st.header("🤖 AI Analyst")
    prompt = st.chat_input("Ask a question about the data...")
    if prompt:
        if ibm_api_key:
            url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22/v1/assistants/chat"
            headers = {"Authorization": f"Bearer {ibm_api_key}", "Content-Type": "application/json"}
            payload = {"input": {"text": prompt}}
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                answer = data.get('output', {}).get('generic', [{}])[0].get('text', str(data))
            except Exception as e:
                answer = f"Error calling Watsonx: {e}"
        else:
            answer = "⚠️ API Key missing. Please provide it in the sidebar or Streamlit secrets."
        st.write(f"**You:** {prompt}")
        st.write(f"**AI:** {answer}")
