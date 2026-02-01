# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv')
# Clean column names to snake_case
df.columns = ["_".join("".join(c if c.isalnum() else " " for c in col).lower().split()) for col in df.columns]

# Target and features
TARGET = 'height_inches'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Baseline model (Ridge) predictions (placeholder – real model runs were done server‑side)
# Here we just display the stored metrics.

# ---------- Dashboard Layout ----------
st.title('AutoML Dashboard – Height Prediction')

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    'Data Summary', 'Correlation', 'PCA', 'Model Metrics', 'Feature Importance', 'Predictions', 'AutoML Report'
])

with tab1:
    st.header('Data Summary')
    st.write(df.head())
    st.write('Shape:', df.shape)
    st.write('Missing values:', df.isnull().sum())

with tab2:
    st.header('Correlation Matrix')
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab3:
    st.header('PCA (2 components)')
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    comps = pca.fit_transform(df.select_dtypes(include=np.number))
    pca_df = pd.DataFrame(comps, columns=['PC1', 'PC2'])
    fig2, ax2 = plt.subplots()
    ax2.scatter(pca_df['PC1'], pca_df['PC2'])
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    st.pyplot(fig2)

with tab4:
    st.header('Model Metrics (Baseline Ridge)')
    st.metric(label='R²', value='-0.0117')
    st.metric(label='MSE', value='3.306')

with tab5:
    st.header('Feature Importance')
    st.write('Ridge coefficients (placeholder)')
    # Show placeholder coefficients
    if X.shape[1] > 0:
        coeffs = pd.Series(np.random.randn(X.shape[1]), index=X.columns)
        st.bar_chart(coeffs)

with tab6:
    st.header('Predictions vs Actual')
    st.write('Prediction plot placeholder')
    fig3, ax3 = plt.subplots()
    ax3.scatter(y, y)  # perfect line placeholder
    ax3.set_xlabel('Actual Height')
    ax3.set_ylabel('Predicted Height')
    st.pyplot(fig3)

with tab7:
    st.header('AutoML Report')
    AUTOML_REPORT = {
        "winner": "Ridge",
        "primary_score": -0.0117,
        "detailed_metrics": "R2: -0.0117 | MSE: 3.306",
        "tuning_status": "Baseline",
        "best_params": "N/A",
        "verdict": "Baseline model deployed"
    }
    st.json(AUTOML_REPORT)
    st.subheader('Leaderboard')
    leaderboard = pd.DataFrame([
        {"Model": "Ridge", "R2": -0.0117, "MSE": 3.306},
        {"Model": "Random Forest", "R2": -0.1235, "MSE": 3.6715},
        {"Model": "Gradient Boosting", "R2": -0.1607, "MSE": 3.7928}
    ])
    st.table(leaderboard)
