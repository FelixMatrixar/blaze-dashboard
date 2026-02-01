import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Clean column names
    df.columns = [c.strip().replace('"', '').replace(' ', '_') for c in df.columns]
    return df

df = load_data()

# AutoML Report
AUTOML_REPORT = {
    "winner": "Ridge",
    "primary_score": 0.1627,
    "detailed_metrics": "MSE: 133.51, R2: 0.1627",
    "leaderboard": [
        {"Model": "Ridge", "R2": 0.1627, "MSE": 133.51},
        {"Model": "Random Forest", "R2": -0.1101, "MSE": 177.00},
        {"Model": "Gradient Boosting", "R2": -0.1555, "MSE": 184.24}
    ],
    "tuning_status": "Not performed",
    "best_params": "N/A",
    "verdict": "Ridge regression provides the best R² score among tested models, though overall predictive power is modest."
}

class NarrativeIntelligence:
    @staticmethod
    def interpret_histogram(col, df):
        desc = df[col].describe()
        skew = df[col].skew()
        direction = "right" if skew > 0 else "left" if skew < 0 else "approximately symmetric"
        return f"The distribution of {col} is {direction} skewed (skew={skew:.2f})."
    @staticmethod
    def interpret_correlation(df):
        corr = df.corr().iloc[0,1]
        if abs(corr) > 0.7:
            strength = "strong"
        elif abs(corr) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        return f"Correlation between Height_Inches and Weight_Pounds is {strength} (r={corr:.2f})."
    @staticmethod
    def interpret_pca(explained_variance):
        total = sum(explained_variance[:2]) * 100
        return f"First two PCA components explain {total:.1f}% of variance."

st.set_page_config(layout="wide", page_title="AutoML Dashboard")

st.title("📊 AutoML Dashboard for Height‑Weight Data")

# Tabs
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
    st.subheader("Raw Data")
    st.dataframe(df)

with tab2:
    fig = px.histogram(df, x="Height_Inches", nbins=20, title="Height Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.info(NarrativeIntelligence.interpret_histogram("Height_Inches", df))

with tab3:
    fig = px.histogram(df, x="Weight_Pounds", nbins=20, title="Weight Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.info(NarrativeIntelligence.interpret_histogram("Weight_Pounds", df))

with tab4:
    fig = px.scatter(df, x="Height_Inches", y="Weight_Pounds", trendline="ols", title="Height vs Weight")
    st.plotly_chart(fig, use_container_width=True)
    st.info(NarrativeIntelligence.interpret_correlation(df))

with tab5:
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    st.info(NarrativeIntelligence.interpret_correlation(df))

with tab6:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(df[["Height_Inches", "Weight_Pounds"]])
    explained = pca.explained_variance_ratio_
    fig = px.scatter(x=components[:,0], y=components[:,1], title="PCA Scatter (2 components)")
    st.plotly_chart(fig, use_container_width=True)
    st.info(NarrativeIntelligence.interpret_pca(explained))

with tab7:
    st.subheader("🏆 AutoML Report")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Winner Model", value=AUTOML_REPORT["winner"])
    with col2:
        st.metric(label="Primary R² Score", value=AUTOML_REPORT["primary_score"])
    st.markdown(f"**Verdict:** {AUTOML_REPORT['verdict']}")
    st.markdown("### Full Leaderboard")
    st.dataframe(pd.DataFrame(AUTOML_REPORT["leaderboard"]))

# End of dashboard
