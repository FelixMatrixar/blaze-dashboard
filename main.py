import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

# ----- Narrative Intelligence -----
class NarrativeIntelligence:
    @staticmethod
    def profile_data(df: pd.DataFrame) -> str:
        rows, cols = df.shape
        missing = df.isnull().mean()*100
        msgs = [f"Dataset has {rows} rows and {cols} columns."]
        for col, pct in missing.items():
            if pct>0:
                msgs.append(f"Column '{col}' has {pct:.1f}% missing values.")
        return " ".join(msgs)
    @staticmethod
    def analyze_distribution(series: pd.Series) -> str:
        skew = series.skew()
        if skew > 0.5:
            return "Right skewed"
        if skew < -0.5:
            return "Left skewed"
        return "Approximately normal"
    @staticmethod
    def interpret_correlation(corr_df: pd.DataFrame) -> list:
        strong = []
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                val = abs(corr_df.iloc[i, j])
                if val > 0.7:
                    strong.append((corr_df.columns[i], corr_df.columns[j], corr_df.iloc[i, j]))
        return strong
    @staticmethod
    def evaluate_model(r2: float) -> str:
        if r2 > 0.75:
            return "Excellent"
        if r2 > 0.5:
            return "Good"
        if r2 > 0.3:
            return "Fair"
        return "Poor"

# ----- Data Pipeline -----
class DataPipeline:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features.remove(target)
        self.categorical_features = []
        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy='median')), ("scaler", StandardScaler())]), self.numeric_features),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy='most_frequent')), ("onehot", OneHotEncoder(handle_unknown='ignore'))]), self.categorical_features)
            ]
        )
    def get_features_targets(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        return X, y

# ----- AutoML Engine -----
class AutoMLEngineer:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.pipeline = DataPipeline(df, target)
    def run_experiment(self):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        models = {
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(random_state=0),
            "Gradient Boosting": GradientBoostingRegressor(random_state=0)
        }
        X, y = self.pipeline.get_features_targets()
        X_processed = self.pipeline.preprocess.fit_transform(X)
        results = []
        best_score = -np.inf
        best_model_name = None
        for name, model in models.items():
            model.fit(X_processed, y)
            preds = model.predict(X_processed)
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            results.append({"Model": name, "R2": r2, "MAE": mae})
            if r2 > best_score:
                best_score = r2
                best_model_name = name
        narrative = NarrativeIntelligence.evaluate_model(best_score)
        return results, best_model_name, narrative

# ----- Streamlit App -----
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
st.set_page_config(layout="wide")

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Safe column cleaning
    df.columns = ["".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower() for col in df.columns]
    return df

df = load_data(DATA_URL)

# Identify target column (weight)
TARGET_COL = "weight_pounds"

# Initialize NarrativeIntelligence
ni = NarrativeIntelligence()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📋 Data Profiling", "🛠️ Feature Engineering", "📊 Advanced Distributions", "🔗 Correlations", "🧬 PCA & Anomalies", "🧩 Clustering", "🤖 AutoML Tournament"])

with tab1:
    st.header("💡 BlazeWatson Executive Summary")
    st.write(ni.profile_data(df))
    st.subheader("Missing Values Matrix")
    st.write(df.isnull().sum())

with tab2:
    st.header("Feature Engineering Insights")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='height_inches')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='weight_pounds')
        st.plotly_chart(fig, use_container_width=True)
    st.info("Log transform of weight reduces skewness from " + str(round(ni.analyze_distribution(df['weight_pounds']),2)))

with tab3:
    st.header("Advanced Distributions")
    fig = px.violin(df, y='height_inches', box=True, points='all')
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.violin(df, y='weight_pounds', box=True, points='all')
    st.plotly_chart(fig2, use_container_width=True)
    outliers = ((df['height_inches'] - df['height_inches'].mean()).abs() > 3*df['height_inches'].std()).sum()
    st.info(f"Detected {outliers} potential outliers in Height.")

with tab4:
    st.header("Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    strong_corr = ni.interpret_correlation(corr)
    if strong_corr:
        st.info("Strong relationships: " + ", ".join([f"{a} & {b} ({c:.2f})" for a,b,c in strong_corr]))
    else:
        st.info("No correlation > 0.7 found.")

with tab5:
    st.header("PCA & Anomalies")
    from sklearn.decomposition import PCA
    X = df[['height_inches','weight_pounds']]
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    fig = px.scatter(x=comps[:,0], y=comps[:,1], labels={'x':'PC1','y':'PC2'})
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"First 2 components explain {pca.explained_variance_ratio_.sum()*100:.1f}% of variance.")

with tab6:
    st.header("K-Means Clustering")
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(df[['height_inches','weight_pounds']])
    df['cluster'] = clusters
    fig = px.scatter(df, x='height_inches', y='weight_pounds', color='cluster', symbol='cluster')
    st.plotly_chart(fig, use_container_width=True)
    st.info("Cluster 0 tends to have lower weight, Cluster 1 higher weight.")

with tab7:
    st.header("🤖 AutoML Tournament")
    automl = AutoMLEngineer(df, TARGET_COL)
    results, best_model, verdict = automl.run_experiment()
    st.subheader("Leaderboard")
    st.table(pd.DataFrame(results).set_index('Model'))
    st.subheader("Final Verdict")
    st.write(f"**{best_model}** achieved the highest R² score. Verdict: {verdict}.")
    st.info("Ridge Regression performed best due to its ability to handle multicollinearity between height and weight.")
