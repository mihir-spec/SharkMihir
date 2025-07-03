"""
Streamlit application for exploring the synthetic Premium-Sportswear
survey dataset and running quick-start analytics:

▪ Overview  – high-level stats & distributions  
▪ Segments  – unsupervised clustering (k-means)  
▪ Willingness – classification of High-Intent vs Low/Medium  
▪ Spend Model – linear regression on annual spend  
▪ Market Basket – association-rule mining on multi-select columns  
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

# ------------------------------------------------------------------ #
# 1. Sidebar – load data
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Sportswear-App Feasibility Lab", layout="wide")

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload survey CSV (or leave empty to use example)",
    type=["csv"]
)

@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

if uploaded:
    df = load_data(uploaded)
else:
    st.sidebar.info("Using bundled synthetic sample (1 000 rows).")
    df = load_data("sportswear_survey_synthetic.csv")

st.title("🏃‍♀️ Premium Sportswear App – Feasibility Workbench")

# ------------------------------------------------------------------ #
# 2. Overview tab
# ------------------------------------------------------------------ #
tab_overview, tab_segment, tab_cls, tab_reg, tab_rules = st.tabs(
    ["Overview", "Segments", "Willingness Classifier", "Spend Regression", "Association Rules"]
)

with tab_overview:
    st.subheader("Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("High-Intent Users", f"{df['HighIntent'].sum():,}")
    col3.metric("Countries", df["Q3_Country"].nunique())

    st.markdown("#### Age distribution")
    fig_age = px.histogram(df, x="Q1_Age", nbins=30)
    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("#### Sports practised (multi-select exploded)")
    exploded = (
        df["Q9_SportsPractised"].str.split(",", expand=True)
          .stack().reset_index(level=1, drop=True)
          .rename("Sport").replace("", np.nan).dropna()
    )
    fig_sport = px.histogram(exploded, x="Sport")
    st.plotly_chart(fig_sport, use_container_width=True)

# ------------------------------------------------------------------ #
# 3. Clustering tab
# ------------------------------------------------------------------ #
with tab_segment:
    st.subheader("Consumer Segments (k-means)")

    features = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
                "Q13_SpendLast12Months", "Q8_SportsCount"]
    k = st.slider("Clusters (k)", 2, 8, 4)

    scaler = StandardScaler()
    X_clust = scaler.fit_transform(df[features])

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X_clust)
    df["Cluster"] = labels

    fig_clust = px.scatter(
        df, x="Q1_Age", y="Q5_MonthlyIncome",
        color="Cluster", hover_data=["Q13_SpendLast12Months"]
    )
    st.plotly_chart(fig_clust, use_container_width=True)

    st.markdown("**Cluster summary**")
    st.dataframe(df.groupby("Cluster")[features].mean().round(1))

# ------------------------------------------------------------------ #
# 4. Classification tab
# ------------------------------------------------------------------ #
with tab_cls:
    st.subheader("Predict High-Intent Adoption")

    num   = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
             "Q13_SpendLast12Months"]
    cat   = ["Q2_Gender", "Q3_Country", "Q14_FreqBuyOnline",
             "Q17_BrandsBought", "Q25_AppealSingleCart"]

    X = df[num + cat]
    y = df["HighIntent"]

    ohe = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", ohe, cat)
    ])

    clf = Pipeline([
        ("prep", pre),
        ("logreg", LogisticRegression(max_iter=1000))
    ])
    clf.fit(X, y)

    y_pred = clf.predict(X)
    st.code(classification_report(y, y_pred), language="text")

# ------------------------------------------------------------------ #
# 5. Regression tab
# ------------------------------------------------------------------ #
with tab_reg:
    st.subheader("Predict Annual Sportswear Spend")

    target = "Q13_SpendLast12Months"
    base   = ["Q5_MonthlyIncome", "Q8_SportsCount", "Q10_HoursPerWeek",
              "Q25_AppealSingleCart"]

    Xr = df[base]
    yr = df[target]

    pipe_reg = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LinearRegression())
    ]).fit(Xr, yr)

    yr_pred = pipe_reg.predict(Xr)
    st.write(f"R² on training: **{r2_score(yr, yr_pred):.2f}**")

    fig_pred = px.scatter(x=yr, y=yr_pred, labels={"x":"Actual", "y":"Predicted"})
    st.plotly_chart(fig_pred, use_container_width=True)

# ------------------------------------------------------------------ #
# 6. Association-rule tab
# ------------------------------------------------------------------ #
with tab_rules:
    st.subheader("Market-Basket Insights")

    multi_cols = {
        "Sports": "Q9_SportsPractised",
        "Brands": "Q17_BrandsBought"
    }
    col_to_use = st.selectbox("Multi-select column", list(multi_cols.keys()))
    series = df[multi_cols[col_to_use]].str.split(",")

    # explode and one-hot encode
    basket = (
        series.apply(lambda items: pd.Series(1, index=items))
              .fillna(0).astype(int)
    )
    frequent = apriori(basket, min_support=0.05, use_colnames=True)
    rules    = association_rules(frequent, metric="lift", min_threshold=1.2)
    st.dataframe(rules.sort_values("lift", ascending=False).head(15))
