import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv')

st.title('Height vs Weight Analysis')
st.subheader('Powered by BlazeWatson')

# KPI metrics
avg_height = df[' Height(Inches)"'].mean()
avg_weight = df[' "Weight(Pounds)"'].mean()
max_height = df[' Height(Inches)"'].max()

col1, col2, col3 = st.columns(3)
col1.metric('Average Height (in)', f"{avg_height:.2f}")
col2.metric('Average Weight (lb)', f"{avg_weight:.2f}")
col3.metric('Max Height (in)', f"{max_height:.2f}")

# Scatter plot with trendline
fig = px.scatter(df, x=' Height(Inches)"', y=' "Weight(Pounds)"', trendline='ols', labels={' Height(Inches)"':'Height (in)', ' "Weight(Pounds)"':'Weight (lb)'})
st.plotly_chart(fig)

# Show data
st.dataframe(df)
