"""
Streamlit application for exploring the synthetic Premium-Sportswear
survey dataset and running quick-start analytics.

Tabs
• Overview      – high-level stats & distributions
• Segments      – unsupervised clustering (k-means)
• Willingness   – High-Intent classifier
• Spend Model   – linear regression on annual spend
• Association   – market-basket mining
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
    "Upload survey CSV (or leave empty to use example)", type=["csv"]
)

@st.cache_data(show_spinner=False)
def load_data(_file) -> pd.DataFrame:
    return pd.read_csv(_file)

if uploaded:
    df = load_data(uploaded)
else:
    st.sidebar.info("Using bundled synthetic sample (1 000 rows).")
    df = load_data("sportswear_survey_synthetic.csv")

st.title("🏃‍♀️ Premium Sportswear App – Feasibility Workbench")


# ------------------------------------------------------------------ #
# 2. Overview tab
# ------------------------------------------------------------------ #
tab_overview, tab_segment, tab_cls, tab_reg, tab_rules = st.tabs(
    ["Overview", "Segments", "Willingness Classifier",
     "Spend Regression", "Association Rules"]
)

with tab_overview:
    st.subheader("Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",        f"{len(df):,}")
    c2.metric("High-Intent", f"{df['HighIntent'].sum():,}")
    c3.metric("Countries",   df["Q3_Country"].nunique())

    st.markdown("#### Age distribution")
    st.plotly_chart(px.histogram(df, x="Q1_Age", nbins=30),
                    use_container_width=True)

    st.markdown("#### Sports practised (exploded)")
    exploded = (
        df["Q9_SportsPractised"]
          .str.split(",", expand=True)
          .stack()
          .reset_index(level=1, drop=True)
          .rename("Sport")
          .replace("", np.nan)
          .dropna()
    )
    st.plotly_chart(px.histogram(exploded, x="Sport"),
                    use_container_width=True)


# ------------------------------------------------------------------ #
# 3. Clustering tab
# ------------------------------------------------------------------ #
with tab_segment:
    st.subheader("Consumer Segments – k-means")

    features = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
                "Q13_SpendLast12Months", "Q8_SportsCount"]

    k = st.slider("Clusters (k)", 2, 8, 4)
    scaler = StandardScaler()
    X_km   = scaler.fit_transform(df[features])

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    df["Cluster"] = km.fit_predict(X_km)

    st.plotly_chart(
        px.scatter(df, x="Q1_Age", y="Q5_MonthlyIncome",
                   color="Cluster", hover_data=["Q13_SpendLast12Months"]),
        use_container_width=True
    )

    st.markdown("**Cluster means**")
    st.dataframe(df.groupby("Cluster")[features].mean().round(1))


# ------------------------------------------------------------------ #
# 4. Classification tab
# ------------------------------------------------------------------ #
with tab_cls:
    st.subheader("Predict High-Intent Adoption")

    num = ["Q1_Age", "Q5_MonthlyIncome", "Q10_HoursPerWeek",
           "Q13_SpendLast12Months"]
    cat = ["Q2_Gender", "Q3_Country", "Q14_FreqBuyOnline",
           "Q17_BrandsBought", "Q25_AppealSingleCart"]

    X = df[num + cat]
    y = df["HighIntent"]

    ohe = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", ohe,            cat)])

    pipe = Pipeline([("prep", pre),
                     ("clf",  LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)

    y_pred = pipe.predict(X)
    st.code(classification_report(y, y_pred), language="text")


# ------------------------------------------------------------------ #
# 5. Regression tab
# ------------------------------------------------------------------ #
with tab_reg:
    st.subheader("Predict Annual Sportswear Spend")

    target = "Q13_SpendLast12Months"
    preds  = ["Q5_MonthlyIncome", "Q8_SportsCount", "Q10_HoursPerWeek",
              "Q25_AppealSingleCart"]

    pipe_r = Pipeline([("scale", StandardScaler()),
                       ("lr",    LinearRegression())]).fit(df[preds], df[target])

    y_hat = pipe_r.predict(df[preds])
    st.write(f"**R² on training** = {r2_score(df[target], y_hat):.2f}")

    st.plotly_chart(
        px.scatter(x=df[target], y=y_hat,
                   labels={"x": "Actual spend", "y": "Predicted"}),
        use_container_width=True
    )


# ------------------------------------------------------------------ #
# 6. Association-rule tab  ★ FIXED ★
# ------------------------------------------------------------------ #
with tab_rules:
    st.subheader("Market-Basket Insights")

    mapping = {
        "Sports practised": "Q9_SportsPractised",
        "Brands bought":    "Q17_BrandsBought"
    }
    chosen = st.selectbox("Choose multi-select column", list(mapping.keys()))
    col    = mapping[chosen]

    # --- robust parsing: NaN → "", split, strip, drop empties
    series = (
        df[col].fillna("")
              .apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    )

    # Build one-hot transaction matrix
    basket = (
        series.explode()          # each item on its own row
              .pipe(pd.get_dummies)  # item → 0/1
              .groupby(level=0).max()
              .astype(int)
    )

    if basket.empty or basket.sum().sum() == 0:
        st.warning("No items found – nothing to analyse.")
    else:
        freq  = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.2)

        if rules.empty:
            st.info("No association rules above the thresholds.")
        else:
            st.dataframe(
                rules.sort_values("lift", ascending=False).head(15),
                height=400
            )
