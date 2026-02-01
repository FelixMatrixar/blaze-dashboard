import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
dataset_url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Clean column names (snake_case)
    df.columns = ["_".join("".join(c if c.isalnum() else " " for c in col).lower().split()) for col in df.columns]
    return df

df = load_data(dataset_url)

# AutoML Report (baseline Ridge)
AUTOML_REPORT = {
    "winner": "Ridge",
    "primary_score": 0.1627,
    "detailed_metrics": "R2: 0.1627 | MSE: 133.5091",
    "tuning_status": "Baseline",
    "best_params": "N/A",
    "verdict": "Baseline model for Weight prediction"
}

st.title("AutoML Dashboard – Weight Prediction")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Overview",
    "Statistics",
    "Correlation",
    "PCA",
    "Feature Distributions",
    "Target Distribution",
    "AutoML Report"
])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

with tab2:
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

with tab3:
    st.subheader("Correlation Matrix")
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab4:
    st.subheader("PCA (2 components)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(df.select_dtypes(include=np.number))
    fig2, ax2 = plt.subplots()
    ax2.scatter(components[:,0], components[:,1])
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    st.pyplot(fig2)

with tab5:
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

with tab6:
    st.subheader("Target Distribution (Weight)")
    fig3, ax3 = plt.subplots()
    sns.histplot(df["weight_pounds"], kde=True, ax=ax3)
    st.pyplot(fig3)

with tab7:
    st.subheader("AutoML Report")
    st.metric(label="Primary Score (R2)", value=AUTOML_REPORT["primary_score"])
    st.write(AUTOML_REPORT["detailed_metrics"])
    st.write("**Best Model:**", AUTOML_REPORT["winner"])
    st.write("**Tuning Status:**", AUTOML_REPORT["tuning_status"])
    st.write("**Best Params:**", AUTOML_REPORT["best_params"])
    st.write("**Verdict:**", AUTOML_REPORT["verdict"])
