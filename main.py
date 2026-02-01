import streamlit as st
import pandas as pd
import plotly.express as px

data_url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv'
df = pd.read_csv(data_url)
# Clean column names
df.columns = [c.strip().replace('"','').replace(' ', '_') for c in df.columns]

st.title('Height & Weight Dashboard')
st.caption('Powered by BlazeWatson')

# KPI metrics
total = len(df)
avg_height = df['Height(Inches)'].mean()
max_weight = df['Weight(Pounds)'].max()
col1, col2, col3 = st.columns(3)
col1.metric('Total Records', total)
col2.metric('Avg Height (in)', f"{avg_height:.2f}")
col3.metric('Max Weight (lb)', f"{max_weight:.2f}")

# Correlation heatmap
corr = df[['Height(Inches)','Weight(Pounds)']].corr()
fig_heat = px.imshow(corr, text_auto=True, aspect='auto')
st.plotly_chart(fig_heat, use_container_width=True)

# Scatter plot
fig_scatter = px.scatter(df, x='Height(Inches)', y='Weight(Pounds)', trendline='ols')
st.plotly_chart(fig_scatter, use_container_width=True)

# Show raw data
st.subheader('Raw Data')
st.dataframe(df)
