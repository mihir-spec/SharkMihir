"""
Streamlit dashboard for exploring the synthetic Premium-Sportswear
survey dataset.

New charts (July 2025 update):
• Overview: gender pie, brand bar, spend-vs-income scatter, corr heat-map
• Segments: scatter-matrix of clustering vars
• Classifier tab: confusion-matrix heat-map + ROC curve
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, r2_score)
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------------------------------- #
# Sidebar – load data
# ----------------------------------------------------------- #
st.set_page_config(page_title="Sportswear-App Feasibility Lab", layout="wide")

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload survey CSV (or leave empty to use bundled sample)", type=["csv"]
)

@st.cache_data(show_spinner=False)
def load_csv(path): return pd.read_csv(path)

df = load_csv(uploaded) if uploaded else load_csv("sportswear_survey_synthetic.csv")
st.title("🏅 Premium Sportswear – Analytics Workbench")

# ----------------------------------------------------------- #
# Tabs
# ----------------------------------------------------------- #
tab_ov, tab_seg, tab_cls, tab_reg, tab_rule = st.tabs(
    ["Overview", "Segments", "Willingness Classifier",
     "Spend Regression", "Association Rules"]
)

# ----------------------------------------------------------- #
# 1 Overview
# ----------------------------------------------------------- #
with tab_ov:
    st.subheader("Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("High-Intent users", df["HighIntent"].sum())
    c3.metric("Countries", df["Q3_Country"].nunique())

    # --- Age histogram
    st.plotly_chart(px.histogram(df, x="Q1_Age", nbins=30,
                                 title="Age distribution"),
                    use_container_width=True)

    # --- Gender pie
    st.plotly_chart(
        px.pie(df, names="Q2_Gender", title="Gender split",
               hole=.4).update_traces(textposition="inside", textinfo="percent+label"),
        use_container_width=True
    )

    # --- Brand popularity bar
    brands_ex = (df["Q17_BrandsBought"].str.split(",", expand=True)
                   .stack().str.strip().replace("", np.nan).dropna())
    st.plotly_chart(px.histogram(brands_ex, x=0, title="Brands bought (count)"),
                    use_container_width=True)

    # --- Income vs Spend scatter coloured by HighIntent
    st.plotly_chart(
        px.scatter(df, x="Q5_MonthlyIncome", y="Q13_SpendLast12Months",
                   color=df["HighIntent"].map({0:"Low/Med",1:"High"}),
                   labels={"color":"HighIntent"},
                   title="Monthly income vs Annual sportswear spend"),
        use_container_width=True
    )

    # --- Correlation heat-map for numeric columns
    num_cols = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
                "Q13_SpendLast12Months", "Q8_SportsCount"]
    corr = df[num_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=True,
                              title="Correlation matrix (numeric features)"),
                    use_container_width=True)

    # --- Sports practised histogram (kept from previous version)
    sports_ex = (df["Q9_SportsPractised"].str.split(",", expand=True)
                   .stack().str.strip().replace("", np.nan).dropna())
    st.plotly_chart(px.histogram(sports_ex, x=0, title="Sports practised"),
                    use_container_width=True)

# ----------------------------------------------------------- #
# 2 Segments – k-means
# ----------------------------------------------------------- #
with tab_seg:
    st.subheader("Consumer Segments – k-means")
    vars_km = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
               "Q13_SpendLast12Months", "Q8_SportsCount"]
    k = st.slider("k (clusters)", 2, 8, 4)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    df["Cluster"] = km.fit_predict(StandardScaler().fit_transform(df[vars_km]))

    # Scatter (age × income)
    st.plotly_chart(
        px.scatter(df, x="Q1_Age", y="Q5_MonthlyIncome", color="Cluster",
                   hover_data=["Q13_SpendLast12Months"]),
        use_container_width=True
    )

    # NEW – scatter-matrix
    st.plotly_chart(
        px.scatter_matrix(df, dimensions=vars_km, color="Cluster",
                          height=700).update_traces(diagonal_visible=False),
        use_container_width=True
    )

    st.markdown("**Cluster means**")
    st.dataframe(df.groupby("Cluster")[vars_km].mean().round(1))

# ----------------------------------------------------------- #
# 3 Classifier – predict HighIntent
# ----------------------------------------------------------- #
with tab_cls:
    st.subheader("Predict High-Intent adoption")

    num = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
           "Q13_SpendLast12Months"]
    cat = ["Q2_Gender", "Q3_Country", "Q14_FreqBuyOnline",
           "Q17_BrandsBought", "Q25_AppealSingleCart"]

    X = df[num + cat]
    y = df["HighIntent"]

    prep = ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    logit = Pipeline([("prep", prep),
                      ("clf",  LogisticRegression(max_iter=1000))]).fit(X, y)

    # Text report
    st.code(classification_report(y, logit.predict(X)), language="text")

    # --- Confusion-matrix heat-map
    cm = confusion_matrix(y, logit.predict(X))
    st.plotly_chart(
        px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                  x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                  title="Confusion matrix"),
        use_container_width=True
    )

    # --- ROC curve
    proba = logit.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"AUC = {auc(fpr,tpr):.2f}"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(dash="dash"), showlegend=False))
    roc_fig.update_layout(title="Receiver Operating Characteristic",
                          xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
    st.plotly_chart(roc_fig, use_container_width=True)

# ----------------------------------------------------------- #
# 4 Spend Regression
# ----------------------------------------------------------- #
with tab_reg:
    st.subheader("Predict Annual sportswear spend")

    target = "Q13_SpendLast12Months"
    feats  = ["Q5_MonthlyIncome", "Q8_SportsCount", "Q10_HoursPerWeek",
              "Q25_AppealSingleCart"]

    pipe_reg = Pipeline([("sc", StandardScaler()),
                         ("lr", LinearRegression())]).fit(df[feats], df[target])
    y_hat = pipe_reg.predict(df[feats])
    st.write(f"**R² (in-sample)**: {r2_score(df[target], y_hat):.2f}")

    st.plotly_chart(
        px.scatter(x=df[target], y=y_hat,
                   labels={"x":"Actual spend", "y":"Predicted"},
                   title="Actual vs Predicted"),
        use_container_width=True
    )

# ----------------------------------------------------------- #
# 5 Association Rules
# ----------------------------------------------------------- #
with tab_rule:
    st.subheader("Market-basket insights")

    mapping = {"Sports practised": "Q9_SportsPractised",
               "Brands bought":    "Q17_BrandsBought"}
    choice = st.selectbox("Multi-select column", list(mapping.keys()))
    col    = mapping[choice]

    # parse multi-label strings safely
    series = (df[col].fillna("")
                    .apply(lambda s: [x.strip() for x in s.split(",") if x.strip()]))

    basket = (series.explode()
                     .pipe(pd.get_dummies)
                     .groupby(level=0).max()
                     .astype(int))

    if basket.empty or basket.sum().sum() == 0:
        st.warning("No items in this column.")
    else:
        freq  = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.2)
        if rules.empty:
            st.info("No rules above thresholds.")
        else:
            st.dataframe(rules.sort_values("lift", ascending=False).head(15),
                         height=400)
