"""
Premium-Sportswear App • Analytics Workbench
===========================================

• Cleans extreme outliers (1.5×IQR) on key numeric columns
• Uses RobustScaler for all numeric preprocessing
• Compares 4 algorithms for classification and regression tasks

Tabs
1. Overview               – quick stats & plots
2. Segments (k-means)     – scatter + scatter-matrix
3. Willingness Classifier – 4 models, metrics table, best CM
4. Spend Regression       – 4 models, metrics table, best scatter
5. Association Rules      – Apriori market-basket
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- scikit-learn ---
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             r2_score, mean_absolute_error)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# --- frequent pattern mining ---
from mlxtend.frequent_patterns import apriori, association_rules


# ------------------------------------------------------------------ #
# 0.   Helpers
# ------------------------------------------------------------------ #

def fmt(x, nd=2):       # nice number formatting for tables
    return f"{x:.{nd}f}"

def drop_outliers(df: pd.DataFrame, cols, k=1.5) -> pd.DataFrame:
    """
    Remove rows where ANY listed numeric column falls outside
    [Q1 − k·IQR, Q3 + k·IQR].
    """
    clean = df.copy()
    for c in cols:
        q1, q3 = clean[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        clean = clean[(clean[c] >= low) & (clean[c] <= high)]
    return clean


# ------------------------------------------------------------------ #
# 1.   Load data + clean
# ------------------------------------------------------------------ #

st.set_page_config(page_title="Sportswear-App Analytics", layout="wide")

upl = st.sidebar.file_uploader("Upload survey CSV "
                               "(or leave empty to use bundled sample)",
                               type=["csv"])

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

raw_df = load_csv(upl) if upl else load_csv("sportswear_survey_synthetic.csv")

num_cols_for_clean = ["Q1_Age", "Q5_MonthlyIncome",
                      "Q10_HoursPerWeek", "Q13_SpendLast12Months"]

df = drop_outliers(raw_df, num_cols_for_clean)   # <- outliers removed
st.sidebar.info(f"Rows after cleaning: **{len(df)}** "
                f"(dropped {len(raw_df) - len(df)})")

st.title("🏅 Premium-Sportswear – Analytics Workbench")


# ------------------------------------------------------------------ #
# 2.   Tabs
# ------------------------------------------------------------------ #

tab_ov, tab_seg, tab_cls, tab_reg, tab_rule = st.tabs(
    ["Overview", "Segments", "Willingness Classifier",
     "Spend Regression", "Association Rules"]
)


# ------------------------------------------------------------------ #
# 2a. Overview tab
# ------------------------------------------------------------------ #
with tab_ov:
    st.subheader("Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("High-Intent users", df["HighIntent"].sum())
    c3.metric("Countries", df["Q3_Country"].nunique())

    # Age distribution
    st.plotly_chart(px.histogram(df, x="Q1_Age", nbins=30,
                                 title="Age distribution"),
                    use_container_width=True)

    # Gender pie
    st.plotly_chart(
        px.pie(df, names="Q2_Gender", hole=.4,
               title="Gender split")
          .update_traces(textposition="inside",
                         textinfo="percent+label"),
        use_container_width=True
    )

    # Brands bar
    brands_ex = (
        df["Q17_BrandsBought"].str.split(",", expand=True)
          .stack().str.strip()
          .replace("", np.nan).dropna()
          .rename("Brand")
    )
    st.plotly_chart(px.histogram(brands_ex, x="Brand",
                                 title="Brands bought (count)"),
                    use_container_width=True)

    # Income vs Spend scatter
    st.plotly_chart(
        px.scatter(df, x="Q5_MonthlyIncome", y="Q13_SpendLast12Months",
                   color=df["HighIntent"].map({0: "Low/Med", 1: "High"}),
                   labels={"color": "HighIntent"},
                   title="Monthly income vs Annual sportswear spend"),
        use_container_width=True
    )

    # Correlation heat-map
    corr_cols = ["Q1_Age", "Q5_MonthlyIncome",
                 "Q10_HoursPerWeek", "Q13_SpendLast12Months",
                 "Q8_SportsCount"]
    st.plotly_chart(px.imshow(df[corr_cols].corr(), text_auto=True,
                              title="Correlation matrix (numeric features)"),
                    use_container_width=True)

    # Sports practised histogram
    sports_ex = (
        df["Q9_SportsPractised"].str.split(",", expand=True)
          .stack().str.strip()
          .replace("", np.nan).dropna()
          .rename("Sport")
    )
    st.plotly_chart(px.histogram(sports_ex, x="Sport",
                                 title="Sports practised"),
                    use_container_width=True)


# ------------------------------------------------------------------ #
# 2b. Segments tab
# ------------------------------------------------------------------ #
with tab_seg:
    st.subheader("Consumer Segments – k-means")

    vars_km = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
               "Q13_SpendLast12Months", "Q8_SportsCount"]

    k = st.slider("k (clusters)", 2, 8, 4)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    df["Cluster"] = km.fit_predict(RobustScaler().fit_transform(df[vars_km]))

    # Scatter age × income
    st.plotly_chart(
        px.scatter(df, x="Q1_Age", y="Q5_MonthlyIncome",
                   color="Cluster",
                   hover_data=["Q13_SpendLast12Months"]),
        use_container_width=True
    )

    # Scatter-matrix
    st.plotly_chart(
        px.scatter_matrix(df, dimensions=vars_km, color="Cluster",
                          height=700)
          .update_traces(diagonal_visible=False),
        use_container_width=True
    )

    st.markdown("**Cluster means**")
    st.dataframe(df.groupby("Cluster")[vars_km].mean().round(1))


# ------------------------------------------------------------------ #
# 2c. Willingness Classifier tab  (4 algorithms)
# ------------------------------------------------------------------ #
with tab_cls:
    st.subheader("Predict High-Intent adoption")

    num = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
           "Q13_SpendLast12Months"]
    cat = ["Q2_Gender", "Q3_Country", "Q14_FreqBuyOnline",
           "Q17_BrandsBought", "Q25_AppealSingleCart"]

    X, y = df[num + cat], df["HighIntent"]

    test_size = st.slider("Test-set size", 0.1, 0.5, 0.2, 0.05)
    stratify  = st.checkbox("Stratify split", value=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_stat_
