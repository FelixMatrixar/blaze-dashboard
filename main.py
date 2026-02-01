import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Safe column cleaning
    df.columns = ["_".join("".join(c if c.isalnum() else " " for c in col).lower().split()) for col in df.columns]
    return df

df = load_data()

# AutoML Report
AUTOML_REPORT = {
    "winner": "Random Forest",
    "primary_score": 1.0,
    "detailed_metrics": "Accuracy: 1.0",
    "tuning_status": "Baseline",
    "best_params": "Default",
    "verdict": "Excellent performance"
}

st.set_page_config(layout="wide", page_title="Iris Dashboard")
st.title("Iris Dataset Exploration & Model Report")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Preview", "Distributions", "Feature Relationships",
    "Correlations", "PCA", "Clustering", "AutoML Report"
])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(df)
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

with tab2:
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

with tab3:
    st.subheader("Pairwise Relationships")
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)

with tab4:
    st.subheader("Correlation Matrix")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab5:
    st.subheader("PCA Projection")
    pca = PCA(n_components=2)
    numeric_df = df.select_dtypes(include=np.number).dropna()
    components = pca.fit_transform(numeric_df)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="species", ax=ax)
    st.pyplot(fig)

with tab6:
    st.subheader("KMeans Clustering (k=3)")
    kmeans = KMeans(n_clusters=3, random_state=42)
    numeric_df = df.select_dtypes(include=np.number).dropna()
    kmeans.fit(numeric_df)
    labels = kmeans.labels_
    cluster_df = numeric_df.copy()
    cluster_df["cluster"] = labels
    fig, ax = plt.subplots()
    sns.scatterplot(data=cluster_df, x=numeric_df.columns[0], y=numeric_df.columns[1], hue="cluster", palette="deep", ax=ax)
    st.pyplot(fig)

with tab7:
    st.subheader("AutoML Report")
    st.metric(label="Primary Score (Accuracy)", value=AUTOML_REPORT["primary_score"])
    st.write(AUTOML_REPORT["detailed_metrics"])
    st.write(f"Model: {AUTOML_REPORT['winner']}")
    st.write(f"Tuning Status: {AUTOML_REPORT['tuning_status']}")
    st.write(f"Best Params: {AUTOML_REPORT['best_params']}")
    st.success(AUTOML_REPORT['verdict'])
    st.markdown("---")
    st.subheader("Leaderboard (Baseline)")
    leaderboard = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        "Accuracy": [1.0, 1.0, 1.0]
    })
    st.table(leaderboard)
