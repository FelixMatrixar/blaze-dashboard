import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# Load data
DATASET_URL = 'https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv'
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Clean column names
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True).str.lower()
    return df

df = load_data(DATASET_URL)

st.title('Universal Crash‑Proof Dashboard')

# Identify column types
num_cols = df.select_dtypes(include=np.number).columns.tolist()
obj_cols = df.select_dtypes(include='object').columns.tolist()

# KPI Row
if len(num_cols) > 0:
    primary_col = num_cols[0]
    col1, col2 = st.columns(2)
    col1.metric('Mean', f"{df[primary_col].mean():.2f}")
    col2.metric('Max', f"{df[primary_col].max():.2f}")

# Visualizations
if len(num_cols) >= 2:
    fig = px.scatter(df, x=num_cols[0], y=num_cols[1], trendline='ols',
                     title=f"{num_cols[0]} vs {num_cols[1]}")
    st.plotly_chart(fig)
elif len(num_cols) == 1:
    fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
    st.plotly_chart(fig)

# Show data table
st.subheader('Data Sample')
st.dataframe(df)
