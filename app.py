"""
Lead Scoring Dashboard
======================
Interactive Streamlit dashboard showing lead scores,
SHAP explanations, and model performance metrics.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Lead Scoring System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color scheme — warm orange/amber for sales vibes
COLORS = {
    "primary": "#F59E0B",      # amber
    "secondary": "#D97706",    # darker amber
    "success": "#10B981",      # green
    "warning": "#F59E0B",      # amber
    "danger": "#EF4444",       # red
    "bg_dark": "#1F2937",      # dark gray
    "text": "#F9FAFB",         # light
    "high": "#10B981",         # green for high score
    "medium": "#F59E0B",       # amber for medium
    "low": "#EF4444",          # red for low score
}


@st.cache_data
def load_data():
    """Load all cached artifacts."""
    # Test predictions
    predictions = pd.read_csv("shap_cache/test_predictions.csv")
    
    # Feature importance
    importance = pd.read_csv("shap_cache/feature_importance.csv", index_col=0, header=None)
    importance.columns = ["importance"]
    importance.index.name = "feature"
    
    # Metrics
    with open("shap_cache/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    
    # SHAP values
    with open("shap_cache/shap_values.pkl", "rb") as f:
        shap_values = pickle.load(f)
    
    # Model
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    return predictions, importance, metrics, shap_values, model


def score_color(score):
    """Return color based on lead score."""
    if score >= 70:
        return COLORS["high"]
    elif score >= 40:
        return COLORS["medium"]
    return COLORS["low"]


def score_label(score):
    """Return label based on lead score."""
    if score >= 70:
        return "🟢 Hot"
    elif score >= 40:
        return "🟡 Warm"
    return "🔴 Cold"


# ============================================================
# LOAD DATA
# ============================================================
predictions, importance, metrics, shap_values, model = load_data()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🎯 Lead Scoring")
st.sidebar.markdown("**XGBoost + SHAP Explainability**")
st.sidebar.divider()

# Filters
st.sidebar.subheader("Filters")

score_range = st.sidebar.slider(
    "Lead Score Range",
    min_value=0, max_value=100,
    value=(0, 100),
    step=5,
)

# Filter by lead source if column exists
if "Lead Source" in predictions.columns:
    sources = ["All"] + sorted(predictions["Lead Source"].unique().tolist())
    selected_source = st.sidebar.selectbox("Lead Source", sources)
else:
    selected_source = "All"

# Apply filters
filtered = predictions[
    (predictions["lead_score"] >= score_range[0]) &
    (predictions["lead_score"] <= score_range[1])
]
if selected_source != "All":
    filtered = filtered[filtered["Lead Source"] == selected_source]

st.sidebar.divider()
st.sidebar.markdown(f"**Showing {len(filtered):,} of {len(predictions):,} leads**")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ This dashboard uses the [Kaggle Lead Scoring Dataset](https://kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset) for demonstration purposes.")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Lead Scoreboard",
    "🔍 Lead Explainer",
    "📈 Model Performance",
    "💡 Feature Insights",
])


# ============================================================
# TAB 1: LEAD SCOREBOARD
# ============================================================
with tab1:
    st.header("Lead Scoreboard")
    st.markdown("All leads ranked by predicted conversion probability. Higher score = more likely to convert.")
    
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hot_count = len(filtered[filtered["lead_score"] >= 70])
        st.metric("🟢 Hot Leads (70+)", f"{hot_count:,}", 
                  f"{hot_count/len(filtered)*100:.1f}%" if len(filtered) > 0 else "0%")
    with col2:
        warm_count = len(filtered[(filtered["lead_score"] >= 40) & (filtered["lead_score"] < 70)])
        st.metric("🟡 Warm Leads (40-70)", f"{warm_count:,}",
                  f"{warm_count/len(filtered)*100:.1f}%" if len(filtered) > 0 else "0%")
    with col3:
        cold_count = len(filtered[filtered["lead_score"] < 40])
        st.metric("🔴 Cold Leads (<40)", f"{cold_count:,}",
                  f"{cold_count/len(filtered)*100:.1f}%" if len(filtered) > 0 else "0%")
    with col4:
        avg_score = filtered["lead_score"].mean() if len(filtered) > 0 else 0
        st.metric("📊 Avg Score", f"{avg_score:.1f}")
    
    st.divider()
    
    # Score distribution
    fig_dist = px.histogram(
        filtered, x="lead_score", nbins=20,
        color_discrete_sequence=[COLORS["primary"]],
        title="Lead Score Distribution",
        labels={"lead_score": "Lead Score", "count": "Number of Leads"},
    )
    fig_dist.add_vline(x=70, line_dash="dash", line_color=COLORS["high"], annotation_text="Hot threshold")
    fig_dist.add_vline(x=40, line_dash="dash", line_color=COLORS["danger"], annotation_text="Cold threshold")
    fig_dist.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Lead table
    st.subheader("Lead Details")
    display_cols = ["lead_score", "actual_converted", "predicted_converted"]
    
    # Add available feature columns
    feature_cols = ["Lead Source", "Total Time Spent on Website", "TotalVisits",
                    "Last Activity", "What is your current occupation", "Tags", "Lead Quality"]
    for col in feature_cols:
        if col in filtered.columns:
            display_cols.append(col)
    
    display_df = filtered[display_cols].sort_values("lead_score", ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    
    st.dataframe(
        display_df.head(50),
        use_container_width=True,
        height=400,
    )
    st.caption("Showing top 50 leads. Use sidebar filters to narrow down.")


# ============================================================
# TAB 2: LEAD EXPLAINER (SHAP)
# ============================================================
with tab2:
    st.header("Lead Explainer")
    st.markdown("Select a lead to understand **why** it scored the way it did. Each bar shows a feature's push toward conversion (right) or away from it (left).")
    
    # Lead selector
    col_sel, col_info = st.columns([1, 2])
    
    with col_sel:
        lead_idx = st.number_input(
            "Select Lead # (row in test set)",
            min_value=0,
            max_value=len(predictions) - 1,
            value=0,
            step=1,
        )
    
    with col_info:
        lead = predictions.iloc[lead_idx]
        score = lead["lead_score"]
        actual = "✅ Converted" if lead["actual_converted"] == 1 else "❌ Not Converted"
        predicted = score_label(score)
        
        st.markdown(f"""
        **Lead Score:** {score:.1f}/100 {predicted}  
        **Actual Outcome:** {actual}  
        **Prediction:** {"Converted" if lead["predicted_converted"] == 1 else "Not Converted"}
        """)
    
    st.divider()
    
    # SHAP waterfall plot
    st.subheader(f"Why Lead #{lead_idx} scored {score:.1f}")
    
    try:
        # Get SHAP values for this lead
        sv = shap_values[lead_idx]
        
        # Create a manual waterfall-style bar chart with plotly
        feature_names = sv.feature_names if hasattr(sv, 'feature_names') else [f"Feature {i}" for i in range(len(sv.values))]
        shap_vals = sv.values
        
        # Sort by absolute value, show top 12
        indices = np.argsort(np.abs(shap_vals))[::-1][:12]
        top_features = [feature_names[i] for i in indices]
        top_values = [shap_vals[i] for i in indices]
        top_colors = [COLORS["high"] if v > 0 else COLORS["danger"] for v in top_values]
        
        fig_shap = go.Figure(go.Bar(
            x=top_values,
            y=top_features,
            orientation="h",
            marker_color=top_colors,
            text=[f"{v:+.3f}" for v in top_values],
            textposition="outside",
        ))
        fig_shap.update_layout(
            title=f"SHAP Feature Contributions for Lead #{lead_idx}",
            xaxis_title="Impact on Prediction (→ more likely to convert)",
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            height=450,
            margin=dict(l=200),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.markdown(f"""
        **How to read this:**  
        - 🟢 Green bars push the prediction **toward conversion** (positive SHAP value)
        - 🔴 Red bars push the prediction **away from conversion** (negative SHAP value)
        - Longer bars = stronger influence on this lead's score
        """)
        
    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}")
        st.info("Try selecting a different lead index.")
    
    # Browse leads
    st.divider()
    st.subheader("Browse Interesting Leads")
    
    col_hot, col_cold, col_wrong = st.columns(3)
    
    with col_hot:
        st.markdown("**🔥 Hottest Lead**")
        hottest_idx = predictions["lead_score"].idxmax()
        st.markdown(f"Lead #{hottest_idx} — Score: {predictions.iloc[hottest_idx]['lead_score']:.1f}")
        if st.button("Explain Hottest", key="hot"):
            st.session_state["lead_idx"] = hottest_idx
    
    with col_cold:
        st.markdown("**🧊 Coldest Lead**")
        coldest_idx = predictions["lead_score"].idxmin()
        st.markdown(f"Lead #{coldest_idx} — Score: {predictions.iloc[coldest_idx]['lead_score']:.1f}")
        if st.button("Explain Coldest", key="cold"):
            st.session_state["lead_idx"] = coldest_idx
    
    with col_wrong:
        st.markdown("**❓ Most Surprising Miss**")
        # Lead with highest score that didn't convert
        false_positives = predictions[(predictions["predicted_converted"] == 1) & (predictions["actual_converted"] == 0)]
        if len(false_positives) > 0:
            worst_fp_idx = false_positives["lead_score"].idxmax()
            st.markdown(f"Lead #{worst_fp_idx} — Score: {predictions.iloc[worst_fp_idx]['lead_score']:.1f} but didn't convert")
            if st.button("Explain Miss", key="miss"):
                st.session_state["lead_idx"] = worst_fp_idx


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
with tab3:
    st.header("Model Performance")
    st.markdown("Comparing XGBoost against Logistic Regression baseline.")
    
    # Metrics cards
    col1, col2, col3, col4, col5 = st.columns(5)
    metric_display = [
        ("Accuracy", metrics["accuracy"], col1),
        ("Precision", metrics["precision"], col2),
        ("Recall", metrics["recall"], col3),
        ("F1 Score", metrics["f1"], col4),
        ("ROC-AUC", metrics["roc_auc"], col5),
    ]
    for name, val, col in metric_display:
        with col:
            st.metric(name, f"{val:.2%}")
    
    st.divider()
    
    # Confusion matrix
    col_cm, col_roc = st.columns(2)
    
    with col_cm:
        st.subheader("Confusion Matrix")
        
        y_actual = predictions["actual_converted"]
        y_pred = predictions["predicted_converted"]
        
        tn = ((y_actual == 0) & (y_pred == 0)).sum()
        fp = ((y_actual == 0) & (y_pred == 1)).sum()
        fn = ((y_actual == 1) & (y_pred == 0)).sum()
        tp = ((y_actual == 1) & (y_pred == 1)).sum()
        
        cm_data = [[tn, fp], [fn, tp]]
        
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Converted", "Converted"],
            y=["Not Converted", "Converted"],
            text_auto=True,
            color_continuous_scale=["#1F2937", COLORS["primary"]],
        )
        fig_cm.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col_roc:
        st.subheader("Score Distribution by Outcome")
        
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=predictions[predictions["actual_converted"] == 1]["lead_score"],
            name="Actually Converted",
            marker_color=COLORS["high"],
            opacity=0.7,
            nbinsx=20,
        ))
        fig_score.add_trace(go.Histogram(
            x=predictions[predictions["actual_converted"] == 0]["lead_score"],
            name="Not Converted",
            marker_color=COLORS["danger"],
            opacity=0.7,
            nbinsx=20,
        ))
        fig_score.update_layout(
            barmode="overlay",
            template="plotly_dark",
            height=350,
            xaxis_title="Lead Score",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_score, use_container_width=True)
    
    # Model comparison
    st.divider()
    st.subheader("XGBoost vs Baseline (Logistic Regression)")
    
    comparison_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Logistic Regression": [0.825, 0.799, 0.730, 0.763, 0.896],
        "XGBoost": [
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["roc_auc"]
        ],
    }
    comp_df = pd.DataFrame(comparison_data)
    comp_df["Improvement"] = comp_df.apply(
        lambda row: f"+{(row['XGBoost'] - row['Logistic Regression'])*100:.1f}pp", axis=1
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4: FEATURE INSIGHTS
# ============================================================
with tab4:
    st.header("Feature Insights")
    st.markdown("Which features matter most for predicting lead conversion, based on SHAP analysis.")
    
    # Global feature importance
    st.subheader("Global Feature Importance (SHAP)")
    
    top_n = st.slider("Show top N features", 5, 25, 15)
    top_features = importance.head(top_n)
    
    fig_imp = go.Figure(go.Bar(
        x=top_features["importance"].values[::-1],
        y=top_features.index[::-1],
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{v:.3f}" for v in top_features["importance"].values[::-1]],
        textposition="outside",
    ))
    fig_imp.update_layout(
        title="Mean |SHAP| Value (Higher = More Important)",
        template="plotly_dark",
        height=max(400, top_n * 30),
        margin=dict(l=250),
        xaxis_title="Mean |SHAP Value|",
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.divider()
    
    # Key insights
    st.subheader("Key Findings")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 🏆 Top Conversion Signals")
        st.markdown("""
        1. **Tags** — The strongest predictor by far. Tags like "Will revert after reading email" (97% conversion) 
           vs "Ringing" (3%) tell you where the lead is in the pipeline.
        
        2. **Time on Website** — Converted leads spend **2.2x more time** on the website. 
           Leads spending 300+ seconds are significantly more likely to buy.
        
        3. **Last Activity** — "SMS Sent" leads convert at 63%, while "Olark Chat" converts at just 9%.
           The outreach channel matters enormously.
        
        4. **Occupation** — Working professionals convert at 92% vs 44% for unemployed.
           Ability to pay strongly predicts conversion.
        """)
    
    with col_b:
        st.markdown("### 📋 Recommendations for Sales Teams")
        st.markdown("""
        1. **Prioritize leads tagged "Will revert after reading email"** — these are your highest-probability 
           conversions. Follow up within 24 hours.
        
        2. **Focus on leads who spent 5+ minutes on the website** — they're actively evaluating. 
           A well-timed call can close them.
        
        3. **Use SMS over email for outreach** — SMS-contacted leads convert at nearly 2x the rate 
           of email-only leads.
        
        4. **Working professionals are your best segment** — tailor messaging around career advancement 
           and ROI of education investment.
        
        5. **Deprioritize leads tagged "Ringing" or "Interested in other courses"** — conversion rates 
           below 3%. Don't waste sales time.
        """)
    
    st.divider()
    st.caption("⚠️ This analysis is based on the Kaggle Lead Scoring Dataset (online education company, ~9,240 leads). "
               "Patterns may differ for other industries. Always validate with client-specific data before making business decisions.")
