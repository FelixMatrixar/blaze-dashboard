import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Clean column names
    df.columns = [c.strip().replace('"', '').replace(' ', '_').lower() for c in df.columns]
    return df

df = load_data()

# AutoML report
AUTOML_REPORT = {
    "winner": "Ridge",
    "primary_score": 0.1627,
    "detailed_metrics": "MSE: 133.5091, R2: 0.1627",
    "leaderboard": [
        {"model": "Ridge", "R2": 0.1627, "MSE": 133.5091},
        {"model": "Random Forest", "R2": -0.1101, "MSE": 177.0031},
        {"model": "Gradient Boosting", "R2": -0.1555, "MSE": 184.2448}
    ],
    "tuning_status": "not performed",
    "best_params": "N/A",
    "verdict": "Ridge regression achieved the highest R² (0.16) among tested models, indicating a modest linear relationship between height and weight."
}

# Narrative Intelligence
class NarrativeIntelligence:
    @staticmethod
    def interpret_histogram(col, data):
        series = data[col]
        skew = series.skew()
        if skew > 0.5:
            return f"The {col} distribution is right‑skewed (skew={skew:.2f})."
        elif skew < -0.5:
            return f"The {col} distribution is left‑skewed (skew={skew:.2f})."
        else:
            return f"The {col} distribution is fairly symmetric (skew={skew:.2f})."
    @staticmethod
    def interpret_correlation(df):
        corr = df.corr()
        height_weight = corr.loc['height_inches', 'weight_pounds']
        if abs(height_weight) > 0.7:
            strength = "strong"
        elif abs(height_weight) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        direction = "positive" if height_weight > 0 else "negative"
        return f"There is a {strength} {direction} correlation ({height_weight:.2f}) between height_inches and weight_pounds."
    @staticmethod
    def interpret_pca(explained_variance):
        total = sum(explained_variance[:2]) * 100
        return f"The first two PCA components explain {total:.1f}% of the variance."

# App layout – 7 tabs
st.set_page_config(page_title="HW 200 AutoML Dashboard", layout="wide")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Overview",
    "Height Distribution",
    "Weight Distribution",
    "Height vs Weight",
    "Correlation Matrix",
    "PCA Insight",
    "🏆 AutoML Report"
])

with tab1:
    st.header("Dataset Preview")
    st.dataframe(df.head())
    st.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

with tab2:
    st.header("Height (inches) Distribution")
    fig = px.histogram(df, x="height_inches", nbins=30, title="Height Histogram")
    st.plotly_chart(fig)
    st.info(NarrativeIntelligence.interpret_histogram("height_inches", df))

with tab3:
    st.header("Weight (pounds) Distribution")
    fig = px.histogram(df, x="weight_pounds", nbins=30, title="Weight Histogram")
    st.plotly_chart(fig)
    st.info(NarrativeIntelligence.interpret_histogram("weight_pounds", df))

with tab4:
    st.header("Height vs Weight Scatter")
    fig = px.scatter(df, x="height_inches", y="weight_pounds", trendline="ols", title="Height vs Weight")
    st.plotly_chart(fig)
    st.info(NarrativeIntelligence.interpret_correlation(df))

with tab5:
    st.header("Correlation Matrix")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig)
    st.info(NarrativeIntelligence.interpret_correlation(df))

with tab6:
    st.header("PCA Insight")
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(df[["height_inches", "weight_pounds"]])
    expl = pca.explained_variance_ratio_
    fig = px.bar(x=[f"PC{i+1}" for i in range(len(expl))], y=expl, labels={"x":"Component", "y":"Explained Variance"}, title="PCA Explained Variance")
    st.plotly_chart(fig)
    st.info(NarrativeIntelligence.interpret_pca(expl))

with tab7:
    st.header("🏆 AutoML Report")
    st.subheader("Winner Model")
    st.metric(label="Model", value=AUTOML_REPORT["winner"], delta=f"R²={AUTOML_REPORT['primary_score']:.3f}")
    st.write(AUTOML_REPORT["verdict"])
    st.subheader("Full Leaderboard")
    st.dataframe(pd.DataFrame(AUTOML_REPORT["leaderboard"]))
    st.caption("Tuning status: " + AUTOML_REPORT["tuning_status"])
