# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv(DATA_URL)
# Clean column names
df.columns = ["_".join("".join(c if c.isalnum() else " " for c in col).lower().split()) for col in df.columns]

# AutoML Report
AUTOML_REPORT = {
    "winner": "Ridge",
    "score": 0.3073,
    "tuning_status": "Hyper‑Tuned",
    "best_params": {},
    "verdict": "Ridge regression after tuning"
}

st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.title("📊 AutoML Dashboard for Height‑Weight Data")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Overview",
    "Statistics",
    "Correlation",
    "PCA Plot",
    "Model Performance",
    "Feature Importance",
    "AutoML Report"
])

with tab1:
    st.header("Raw Data")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

with tab2:
    st.header("Descriptive Statistics")
    st.write(df.describe())

with tab3:
    st.header("Correlation Matrix")
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("PCA (2‑Component) Projection")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(df.select_dtypes(include=np.number))
    fig2, ax2 = plt.subplots()
    sc = ax2.scatter(components[:,0], components[:,1], c=df["weight_pounds"], cmap="viridis")
    plt.colorbar(sc, label="Weight (pounds)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    st.pyplot(fig2)

with tab5:
    st.header("Model Performance")
    st.metric(label="Model", value=AUTOML_REPORT["winner"])
    st.metric(label="MSE (tuned)", value=AUTOML_REPORT["score"])

with tab6:
    st.header("Feature Importance (Linear Coefficients)")
    from sklearn.linear_model import Ridge
    X = df.drop(columns=["weight_pounds"]).values
    y = df["weight_pounds"].values
    model = Ridge()
    model.fit(X, y)
    coeff = model.coef_
    feat_names = df.drop(columns=["weight_pounds"]).columns
    fig3, ax3 = plt.subplots()
    ax3.barh(feat_names, coeff)
    ax3.set_xlabel("Coefficient Value")
    st.pyplot(fig3)

with tab7:
    st.header("AutoML Report")
    st.json(AUTOML_REPORT)

# End of file