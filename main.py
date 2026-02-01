import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

# Config
st.set_page_config(layout="wide")

# Data URL
DATA_URL = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Safe column cleaning
    df.columns = ["".join(c for c in col if c.isalnum() or c == " ").strip().replace(" ", "_").lower() for col in df.columns]
    return df

df = load_data()

# AutoML Engine
class AutoMLEngineer:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_classifier = False
        self.le = None

    def preprocess(self, df, target):
        X = df.drop(columns=[target])
        y = df[target]
        # Handle missing values
        for col in X.columns:
            if X[col].dtype.kind in "bifc":
                X[col].fillna(X[col].mean(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)
        # Encode categoricals if any
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        # Target preprocessing
        if y.dtype.kind in "bifc":
            y = y.fillna(y.mean())
        else:
            self.le = LabelEncoder()
            y = self.le.fit_transform(y.astype(str))
        self.is_classifier = (y.nunique() < 20 and y.dtype.kind not in "bifc")
        self.feature_names = X.columns.tolist()
        return X, y

    def train(self, X, y):
        if self.is_classifier:
            self.model = RandomForestClassifier(random_state=42)
        else:
            self.model = RandomForestRegressor(random_state=42)
        self.model.fit(X, y)
        return self.model

    def get_feature_importance(self):
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importances_
        return pd.DataFrame({"feature": self.feature_names, "importance": importance}).sort_values(by="importance", ascending=False)

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["📋 Data Profiling", "📊 Advanced Distributions", "🔗 Correlations & Interactions", "🧬 PCA & Anomalies", "🤖 AutoML Engineer"]
selection = st.sidebar.radio("Go to", pages)

if selection == "📋 Data Profiling":
    st.header("Data Profiling")
    st.subheader("Head of Data")
    st.dataframe(df.head())
    st.subheader("Describe")
    st.dataframe(df.describe())
    st.subheader("Missing Values Matrix")
    missing = df.isnull()
    st.dataframe(missing)
    st.subheader("Data Types")
    st.write(df.dtypes)

elif selection == "📊 Advanced Distributions":
    st.header("Advanced Distributions")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.violin(df, y=col, box=True, points="all", title=f"Violin of {col}")
            st.plotly_chart(fig, use_container_width=True)

elif selection == "🔗 Correlations & Interactions":
    st.header("Correlations & Interactions")
    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="Viridis"))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Interactive Scatter")
    cols = df.columns.tolist()
    x_axis = st.selectbox("X Axis", cols, index=0)
    y_axis = st.selectbox("Y Axis", cols, index=1)
    color = st.selectbox("Color", cols, index=2)
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f"Scatter: {x_axis} vs {y_axis}")
    st.plotly_chart(fig, use_container_width=True)

elif selection == "🧬 PCA & Anomalies":
    st.header("PCA & Anomalies")
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Scatter (first 2 components)")
    st.plotly_chart(fig, use_container_width=True)
    # Anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(scaled)
    preds = iso.predict(scaled)
    pca_df["anomaly"] = preds
    fig2 = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["anomaly"].map({1: "Inlier", -1: "Outlier"}), title="IsolationForest Anomalies on PCA")
    st.plotly_chart(fig2, use_container_width=True)

elif selection == "🤖 AutoML Engineer":
    st.header("AutoML Engineer")
    target = st.selectbox("Select Target Column", df.columns.tolist())
    if st.button("Train Model"):
        engine = AutoMLEngineer()
        X, y = engine.preprocess(df, target)
        engine.train(X, y)
        # Metrics
        preds = engine.model.predict(X)
        if engine.is_classifier:
            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds, average="weighted")
            st.write(f"**Accuracy:** {acc:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
        else:
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            st.write(f"**R2:** {r2:.4f}")
            st.write(f"**MAE:** {mae:.4f}")
        # Feature importance
        fi = engine.get_feature_importance()
        fig = px.bar(fi, x="importance", y="feature", orientation="h", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        # Actual vs Predicted
        if not engine.is_classifier:
            fig2 = px.scatter(x=y, y=preds, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2 = px.scatter(x=y, y=preds, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted (Classification)")
            st.plotly_chart(fig2, use_container_width=True)
