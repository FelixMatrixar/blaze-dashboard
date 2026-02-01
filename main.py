import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(layout="wide")

DATASET_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(DATASET_URL)

# Clean column names
df.columns = [c.strip().replace('"','') for c in df.columns]

# Secret Management
if "IBM_API_KEY" in st.secrets:
    ibm_api_key = st.secrets["IBM_API_KEY"]
else:
    ibm_api_key = st.sidebar.text_input("Enter IBM API Key", type="password")

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 AI Analyst"])

with tab1:
    st.title("Height & Weight Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Height (in)", f"{df['Height(Inches)'].mean():.2f}")
    with col2:
        st.metric("Average Weight (lb)", f"{df['Weight(Pounds)'].mean():.2f}")
    
    fig_scatter = px.scatter(df, x="Height(Inches)", y="Weight(Pounds)", trendline="ols", title="Height vs Weight")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    fig_heat = px.density_heatmap(df, x="Height(Inches)", y="Weight(Pounds)", nbinsx=30, nbinsy=30, title="Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.subheader("Data Preview")
    st.dataframe(df)

with tab2:
    st.title("🤖 AI Analyst")
    prompt = st.chat_input("Ask a question about the data...")
    if prompt:
        if not ibm_api_key:
            st.warning("⚠️ API Key missing.")
        else:
            url = "https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/b3247552-26de-498f-a5d5-545480fbda22/v1/assistants/chat"
            headers = {"Authorization": f"Bearer {ibm_api_key}", "Content-Type": "application/json"}
            payload = {"input": {"text": prompt}}
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                answer = data.get('output', {}).get('generic', [{}])[0].get('text', str(data))
                st.write(answer)
            except Exception as e:
                st.error(f"Error contacting Watsonx: {e}")
