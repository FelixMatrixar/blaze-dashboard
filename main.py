import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Clean column names
    df.columns = ["_".join("".join(c if c.isalnum() else " " for c in col).lower().split()) for col in df.columns]
    return df

df = load_data()

# AutoML report
AUTOML_REPORT = {
    "winner": "Gradient Boosting",
    "best_score": 0.0333,
    "leaderboard": [
        {"Model": "Random Forest", "MAE": 1.4516, "Params": "{'model__max_depth': 10, 'model__n_estimators': 100}", "Score": -0.137},
        {"Model": "Gradient Boosting", "MAE": 1.3733, "Params": "{'model__learning_rate': 0.01, 'model__n_estimators': 100}", "Score": 0.0333},
        {"Model": "Ridge Regression", "MAE": 1.4397, "Params": "{}", "Score": -0.0117}
    ],
    "top_features": [],
    "verdict": "Gradient Boosting provides the best regression performance with a MAE of 1.37 on predicting Height (inches)."
}

class NarrativeIntelligence:
    @staticmethod
    def profile_data(df):
        return df.describe().T
    @staticmethod
    def analyze_distribution(df, column):
        fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
        return fig

tabs = st.tabs(["Stats", "Histograms", "Correlations", "PCA", "Clustering", "Feature Insights", "🏆 AutoML Report"])

with tabs[0]:
    st.header("Dataset Statistics")
    st.dataframe(NarrativeIntelligence.profile_data(df))

with tabs[1]:
    st.header("Histograms")
    for col in df.select_dtypes(include=np.number).columns:
        st.subheader(col)
        st.plotly_chart(NarrativeIntelligence.analyze_distribution(df, col))

with tabs[2]:
    st.header("Correlation Matrix")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig)

with tabs[3]:
    st.header("PCA (2 Components)")
    from sklearn.decomposition import PCA
    numeric_df = df.select_dtypes(include=np.number).dropna()
    pca = PCA(n_components=2)
    components = pca.fit_transform(numeric_df)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Scatter Plot")
    st.plotly_chart(fig)

with tabs[4]:
    st.header("KMeans Clustering (k=3)")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)
    scatter_df = numeric_df.copy()
    scatter_df["cluster"] = clusters
    fig = px.scatter(scatter_df, x=scatter_df.columns[0], y=scatter_df.columns[1], color="cluster", title="Clustering Scatter")
    st.plotly_chart(fig)

with tabs[5]:
    st.header("Feature Insights")
    st.info("No additional feature engineering performed.")

with tabs[6]:
    st.header("AutoML Report")
    st.subheader(f"🏆 Best Model: {AUTOML_REPORT['winner']}")
    st.metric(label="Best Score (MAE)", value=AUTOML_REPORT['best_score'])
    st.write(AUTOML_REPORT['verdict'])
    st.info("This analysis was run by BlazeWatson Agent on the target: Height(Inches).")
    st.subheader("Leaderboard")
    st.table(pd.DataFrame(AUTOML_REPORT['leaderboard']))
