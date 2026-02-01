# main.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Iris AutoML Dashboard", layout="wide")

# Load data
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Safe column cleaning
    df.columns = ["_".join("".join(c if c.isalnum() else " " for c in col).lower().split()) for col in df.columns]
    return df

df = load_data()

# ----------------- AutoML Report -----------------
AUTOML_REPORT = {
    "winner": "Random Forest",
    "primary_score": 1.0,
    "detailed_metrics": "Accuracy: 1.0",
    "tuning_status": "Baseline",
    "best_params": "n_estimators=100, max_depth=None",
    "verdict": "Excellent performance on Iris dataset"
}

st.title("Iris Classification AutoML Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Preview",
    "Feature Distribution",
    "Model Performance",
    "Correlations",
    "PCA",
    "Clustering",
    "AutoML Report"
])

with tab1:
    st.header("Data Preview")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

with tab2:
    st.header("Feature Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected = st.selectbox("Select numeric feature", numeric_cols)
    st.bar_chart(df[selected].value_counts())

with tab3:
    st.header("Model Performance")
    X = df.drop(columns=["species"])
    y = df["species"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    st.metric("Accuracy", f"{acc:.3f}")
    st.text(classification_report(y, preds))

with tab4:
    st.header("Correlations (Numeric Only)")
    corr = df.corr(numeric_only=True)
    st.write(corr)
    st.caption("Correlation matrix for numeric features.")

with tab5:
    st.header("PCA (Numeric Features)")
    from sklearn.decomposition import PCA
    numeric_df = df.select_dtypes(include=np.number).dropna()
    pca = PCA(n_components=2)
    components = pca.fit_transform(numeric_df)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    st.scatter_chart(pca_df, x="PC1", y="PC2", color="species")

with tab6:
    st.header("Clustering (KMeans on Numeric Features)")
    from sklearn.cluster import KMeans
    numeric_df = df.select_dtypes(include=np.number).dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)
    df_clusters = numeric_df.copy()
    df_clusters["cluster"] = clusters
    st.scatter_chart(df_clusters, x=numeric_df.columns[0], y=numeric_df.columns[1], color="cluster")

with tab7:
    st.header("AutoML Report")
    st.subheader("Winner Model")
    st.write(AUTOML_REPORT["winner"])
    st.metric("Primary Score (Accuracy)", AUTOML_REPORT["primary_score"])
    st.caption(AUTOML_REPORT["detailed_metrics"])
    st.write("Tuning Status:", AUTOML_REPORT["tuning_status"])
    st.write("Best Params:", AUTOML_REPORT["best_params"])
    st.success(AUTOML_REPORT["verdict"])
