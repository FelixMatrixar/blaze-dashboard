import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
st.set_page_config(page_title="Height‑Weight AutoML Dashboard", layout="wide")

# -------------------------------------------------
# DATA LOADING & CLEANING
# -------------------------------------------------
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Safe snake_case cleaning (no regex)
    cleaned = []
    for col in df.columns:
        # replace non‑alphanumeric with space, lower, split, join with underscore
        safe = "_".join("".join(c if c.isalnum() else " " for c in col).lower().split())
        cleaned.append(safe)
    df.columns = cleaned
    return df

df = load_data(DATA_URL)

# -------------------------------------------------
# INJECTED AUTOML RESULTS
# -------------------------------------------------
AUTOML_REPORT = {
    "winner": "Ridge Regression",
    "best_score": 0.1627,
    "leaderboard": [
        {"Model": "Random Forest", "MAE": 10.4121, "Params": "{'model__max_depth': 10, 'model__n_estimators': 100}", "Score": -0.141},
        {"Model": "Gradient Boosting", "MAE": 9.1252, "Params": "{'model__learning_rate': 0.01, 'model__n_estimators': 100}", "Score": 0.0781},
        {"Model": "Ridge Regression", "MAE": 8.7351, "Params": "{}", "Score": 0.1627}
    ],
    "top_features": [
        {"feature": "height_inches", "importance": 1.0}
    ],
    "verdict": "Ridge Regression emerged as the top performer with an MAE of 8.74, achieving the highest score (0.1627) among the tested models. While ensemble methods like Gradient Boosting and Random Forest offered competitive error rates, the linear nature of the relationship between height and weight allowed Ridge Regression to capture the pattern efficiently with minimal over‑fitting. The modest improvement over the tree‑based models suggests that the dataset is relatively clean and low‑dimensional, making regularized linear regression the optimal choice for this height‑to‑weight prediction task."
}

# -------------------------------------------------
# NARRATIVE INTELLIGENCE CLASS
# -------------------------------------------------
class NarrativeIntelligence:
    @staticmethod
    def profile_data(df: pd.DataFrame) -> pd.DataFrame:
        return df.describe().transpose()

    @staticmethod
    def missing_summary(df: pd.DataFrame) -> pd.Series:
        return df.isnull().sum()

    @staticmethod
    def analyze_distribution(df: pd.DataFrame, column: str):
        fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
        return fig

# -------------------------------------------------
# LAYOUT WITH TABS
# -------------------------------------------------
st.title("📊 Height‑Weight AutoML Dashboard")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📄 Data Overview",
    "📈 Histograms",
    "📊 Correlations",
    "🔎 PCA",
    "🧩 Clustering",
    "⚙️ Settings",
    "🏆 AutoML Report"
])

# ---------- Tab 1: Data Overview ----------
with tab1:
    st.subheader("Raw Data")
    st.dataframe(df.head())
    st.subheader("Statistical Profile")
    profile = NarrativeIntelligence.profile_data(df)
    st.dataframe(profile)
    st.subheader("Missing Values")
    st.write(NarrativeIntelligence.missing_summary(df))

# ---------- Tab 2: Histograms ----------
with tab2:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected = st.selectbox("Select column for histogram", numeric_cols, index=0)
    fig = NarrativeIntelligence.analyze_distribution(df, selected)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 3: Correlations ----------
with tab3:
    st.subheader("Correlation Matrix")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 4: PCA ----------
with tab4:
    st.subheader("PCA Projection (2 Components)")
    numeric_df = df.select_dtypes(include=np.number).dropna(axis=1)
    if numeric_df.shape[1] >= 2:
        pca = PCA(n_components=2)
        components = pca.fit_transform(numeric_df)
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Scatter")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for PCA.")

# ---------- Tab 5: Clustering ----------
with tab5:
    st.subheader("K‑Means Clustering")
    numeric_df = df.select_dtypes(include=np.number).dropna(axis=1)
    k = st.slider("Number of clusters", 2, 6, 3)
    if numeric_df.shape[0] > 0:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(numeric_df)
        df_clust = numeric_df.copy()
        df_clust["cluster"] = clusters
        fig = px.scatter_matrix(df_clust, dimensions=numeric_df.columns.tolist(), color="cluster", title="Cluster Scatter Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset is empty.")

# ---------- Tab 6: Settings ----------
with tab6:
    st.subheader("Dataset Settings")
    st.write("You can adjust the data source URL below and reload the data.")
    new_url = st.text_input("Dataset URL", DATA_URL)
    if st.button("Reload Data"):
        st.experimental_rerun()

# ---------- Tab 7: AutoML Report ----------
with tab7:
    st.subheader("🏆 Champion Model")
    st.metric(label="Winner", value=AUTOML_REPORT["winner"], delta=f"Score: {AUTOML_REPORT['best_score']:.4f}")

    st.subheader("📊 Leaderboard")
    leaderboard_df = pd.DataFrame(AUTOML_REPORT["leaderboard"])
    st.dataframe(leaderboard_df)

    st.subheader("🔝 Top Feature Importance")
    top_feat = pd.DataFrame(AUTOML_REPORT["top_features"])
    fig_feat = px.bar(top_feat, x="feature", y="importance", title="Feature Importance")
    st.plotly_chart(fig_feat, use_container_width=True)

    st.subheader("📝 Verdict")
    st.write(AUTOML_REPORT["verdict"])

# End of app
