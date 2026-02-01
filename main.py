import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv')

# Clean column names
df.columns = [col.strip().replace('"', '').replace(' ', '_') for col in df.columns]

st.title('Height & Weight Dashboard')
st.caption('Powered by BlazeWatson')

# KPI metrics
total_count = len(df)
avg_height = df['Height(Inches)'].mean()
avg_weight = df['Weight(Pounds)'].mean()
col1, col2, col3 = st.columns(3)
col1.metric('Total Records', total_count)
col2.metric('Avg Height (in)', f"{avg_height:.2f}")
col3.metric('Avg Weight (lb)', f"{avg_weight:.2f}")

# Correlation heatmap
corr = df[['Height(Inches)', 'Weight(Pounds)']].corr()
fig = px.imshow(corr, text_auto=True, aspect='auto')
st.plotly_chart(fig)

# Histograms
fig_h = px.histogram(df, x='Height(Inches)', nbins=20, title='Height Distribution')
fig_w = px.histogram(df, x='Weight(Pounds)', nbins=20, title='Weight Distribution')
st.plotly_chart(fig_h)
st.plotly_chart(fig_w)

# Scatter plot
fig_scatter = px.scatter(df, x='Height(Inches)', y='Weight(Pounds)', trendline='ols', title='Height vs Weight')
st.plotly_chart(fig_scatter)

# Show raw data
st.subheader('Raw Data')
st.dataframe(df)