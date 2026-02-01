import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

data_url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv'
df = pd.read_csv(data_url)
# Clean column names
df.columns = [col.strip().replace('"','').replace("'",'') for col in df.columns]
# Header
st.title('Height & Weight Dashboard')
st.caption('Powered by BlazeWatson')
# KPI Row
col1, col2, col3 = st.columns(3)
col1.metric('Total Records', len(df))
col2.metric('Avg Height (in)', f"{df['Height(Inches)'].mean():.2f}")
col3.metric('Avg Weight (lb)', f"{df['Weight(Pounds)'].mean():.2f}")
# Correlation Heatmap
if len(df.select_dtypes(include='number').columns) > 2:
    corr = df.select_dtypes(include='number').corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto')
    st.subheader('Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)
# Scatter plot Height vs Weight
fig2 = px.scatter(df, x='Height(Inches)', y='Weight(Pounds)', trendline='ols', title='Height vs Weight')
st.plotly_chart(fig2, use_container_width=True)
# Raw data
st.subheader('Raw Data')
st.dataframe(df)
