"""
Premium-Sportswear App • Analytics Workbench
-------------------------------------------
• Removes outliers via Robust-Scaling (>3×IQR from median)
• Compares 4 classifiers & 4 regressors
• Shows best-model confusion matrix / scatter + feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── scikit-learn ────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             r2_score, mean_absolute_error)
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
# clustering
from sklearn.cluster import KMeans

# market-basket
from mlxtend.frequent_patterns import apriori, association_rules


# ─── Helpers ─────────────────────────────────────────────────────
def fmt(x, nd=2): return f"{x:.{nd}f}"

def robust_outlier_remove(df: pd.DataFrame, cols, thr=3.0):
    """
    Fit RobustScaler on 'cols', transform, and drop any row where the
    absolute scaled value of ANY column exceeds 'thr'.
    """
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[cols])
    mask = (np.abs(scaled) <= thr).all(axis=1)
    return df.loc[mask].reset_index(drop=True)

def get_ohe_feature_names(ct: ColumnTransformer):
    """Return list of feature names after ColumnTransformer."""
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder": continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols))
        else:
            names.extend(cols)
    return names


# ─── Load & clean data ───────────────────────────────────────────
st.set_page_config(page_title="Sportswear-App Analytics", layout="wide")

upl = st.sidebar.file_uploader("Upload survey CSV (or leave blank for demo)",
                               type=["csv"])
@st.cache_data(show_spinner=False)
def load_csv(p): return pd.read_csv(p)

raw_df = load_csv(upl) if upl else load_csv("sportswear_survey_synthetic.csv")

num_for_clean = ["Q1_Age", "Q5_MonthlyIncome",
                 "Q10_HoursPerWeek", "Q13_SpendLast12Months"]

df = robust_outlier_remove(raw_df, num_for_clean, thr=3.0)
st.sidebar.info(f"Rows after outlier removal: **{len(df)}** "
                f"(dropped {len(raw_df) - len(df)})")

st.title("🏅 Premium-Sportswear — Analytics Workbench")

tab_ov, tab_seg, tab_cls, tab_reg, tab_rule = st.tabs(
    ["Overview", "Segments", "Classifier", "Regression", "Assoc Rules"]
)

# ─── 1. Overview ────────────────────────────────────────────────
with tab_ov:
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("High-Intent", df["HighIntent"].sum())
    c3.metric("Countries", df["Q3_Country"].nunique())

    st.plotly_chart(px.histogram(df, x="Q1_Age", nbins=30,
                                 title="Age distribution"),
                    use_container_width=True)

    st.plotly_chart(
        px.pie(df, names="Q2_Gender", hole=.4,
               title="Gender split").update_traces(textinfo="percent+label"),
        use_container_width=True)

    brands = (df["Q17_BrandsBought"].str.split(",", expand=True)
                .stack().str.strip()
                .replace("", np.nan).dropna())
    st.plotly_chart(px.histogram(brands, x=0, title="Brands bought"),
                    use_container_width=True)

    st.plotly_chart(
        px.scatter(df, x="Q5_MonthlyIncome", y="Q13_SpendLast12Months",
                   color=df["HighIntent"].map({0:"Low/Med",1:"High"}),
                   labels={"color":"HighIntent"},
                   title="Income vs Spend"), use_container_width=True)

# ─── 2. Segments ────────────────────────────────────────────────
with tab_seg:
    st.subheader("k-means segments")
    vars_km = ["Q1_Age","Q5_MonthlyIncome","Q10_HoursPerWeek",
               "Q13_SpendLast12Months","Q8_SportsCount"]
    k = st.slider("k",2,8,4)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    df["Cluster"] = km.fit_predict(RobustScaler().fit_transform(df[vars_km]))
    st.plotly_chart(px.scatter(df, x="Q1_Age", y="Q5_MonthlyIncome",
                               color="Cluster"), use_container_width=True)
    st.dataframe(df.groupby("Cluster")[vars_km].mean().round(1))

# ─── 3. Classifier ──────────────────────────────────────────────
with tab_cls:
    st.subheader("High-Intent Classifier")
    num = ["Q1_Age","Q5_MonthlyIncome","Q10_HoursPerWeek",
           "Q13_SpendLast12Months"]
    cat = ["Q2_Gender","Q3_Country","Q14_FreqBuyOnline",
           "Q17_BrandsBought","Q25_AppealSingleCart"]
    X,y = df[num+cat], df["HighIntent"]
    ts = st.slider("Test size",0.1,0.5,0.2,0.05)
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=ts,
                                           stratify=y, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoost": GradientBoostingClassifier(random_state=42)
    }

    rows, best_auc, best_pipe, best_name = [],0,None,""
    for name, clf in models.items():
        prep = ColumnTransformer([
            ("num",RobustScaler(),num),
            ("cat",OneHotEncoder(handle_unknown="ignore"),cat)
        ])
        pipe = Pipeline([("prep",prep),("clf",clf)]).fit(X_tr,y_tr)
        y_pred = pipe.predict(X_te)
        proba = (pipe.predict_proba(X_te)[:,1]
                 if hasattr(pipe,"predict_proba") else np.zeros_like(y_pred))
        auc = roc_auc_score(y_te, proba)
        rows.append({"Model":name,
                     "Acc":fmt(accuracy_score(y_te,y_pred),3),
                     "F1":fmt(f1_score(y_te,y_pred),3),
                     "ROC-AUC":fmt(auc,3)})
        if auc>best_auc:
            best_auc,best_pipe,best_name = auc,pipe,name

    st.dataframe(pd.DataFrame(rows).set_index("Model"))
    st.markdown(f"**Best:** `{best_name}` (AUC {fmt(best_auc,3)})")

    cm = confusion_matrix(y_te, best_pipe.predict(X_te))
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                              title=f"Confusion – {best_name}"),
                    use_container_width=True)

    # Feature importance
    st.markdown("#### Feature importance")
    prep = best_pipe.named_steps["prep"]
    feat_names = get_ohe_feature_names(prep)
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf,"feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=feat_names)
    elif hasattr(clf,"coef_"):
        imp = pd.Series(np.abs(clf.coef_).ravel(), index=feat_names)
    else:
        imp = pd.Series(dtype=float)

    if imp.empty:
        st.info("Model provides no importance scores.")
    else:
        top = imp.sort_values(ascending=False).head(15)
        st.plotly_chart(px.bar(top, orientation="h",
                               title="Top 15 features")
                        .update_yaxes(categoryorder="total ascending"),
                        use_container_width=True)

# ─── 4. Regression ──────────────────────────────────────────────
with tab_reg:
    st.subheader("Annual Spend Regression")
    target = "Q13_SpendLast12Months"
    feats  = ["Q5_MonthlyIncome","Q8_SportsCount",
              "Q10_HoursPerWeek","Q25_AppealSingleCart"]

    regs = {
        "Linear": LinearRegression(),
        "Ridge":  Ridge(alpha=1.0),
        "Lasso":  Lasso(alpha=0.1),
        "DecTree": DecisionTreeRegressor(random_state=42)
    }

    rows, best_r2,best_pipe,best_name = [],-1,None,""
    for n,r in regs.items():
        pipe = Pipeline([("sc",RobustScaler()),("reg",r)]).fit(df[feats],df[target])
        pred = pipe.predict(df[feats])
        r2  = r2_score(df[target], pred)
        mae = mean_absolute_error(df[target], pred)
        rows.append({"Model":n,"R²":fmt(r2,3),"MAE":fmt(mae,0)})
        if r2>best_r2:
            best_r2,best_pipe,best_name = r2,pipe,n

    st.dataframe(pd.DataFrame(rows).set_index("Model"))
    st.markdown(f"**Best:** `{best_name}` (R² {fmt(best_r2,3)})")

    st.plotly_chart(px.scatter(x=df[target], y=best_pipe.predict(df[feats]),
                   labels={"x":"Actual","y":"Predicted"},
                   title=f"Actual vs Predicted – {best_name}"),
                   use_container_width=True)

    st.markdown("#### Feature importance")
    reg = best_pipe.named_steps["reg"]
    if hasattr(reg,"feature_importances_"):
        imp = pd.Series(reg.feature_importances_, index=feats)
    elif hasattr(reg,"coef_"):
        imp = pd.Series(np.abs(reg.coef_), index=feats)
    else:
        imp = pd.Series(dtype=float)

    if imp.empty:
        st.info("Regressor provides no importance scores.")
    else:
        st.plotly_chart(px.bar(imp.sort_values(), orientation="h",
                               title="Feature importance"),
                        use_container_width=True)

# ─── 5. Association Rules ───────────────────────────────────────
with tab_rule:
    st.subheader("Market-basket rules")
    mapping = {"Sports":"Q9_SportsPractised",
               "Brands":"Q17_BrandsBought"}
    choice = st.selectbox("Column", list(mapping.keys()))
    col = mapping[choice]
    series = (df[col].fillna("")
                    .apply(lambda s:[x.strip() for x in s.split(",") if x.strip()]))
    basket = (series.explode()
                     .pipe(pd.get_dummies)
                     .groupby(level=0).max().astype(int))
    if basket.sum().sum()==0:
        st.info("No items present.")
    else:
        freq  = apriori(basket,min_support=0.05,use_colnames=True)
        rules = association_rules(freq,metric="lift",min_threshold=1.2)
        st.dataframe(rules.sort_values("lift",ascending=False).head(15),
                     height=400)
