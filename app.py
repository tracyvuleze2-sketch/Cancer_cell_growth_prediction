import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import io
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(
    page_title="Cancer TF Discovery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f3d2e; }
[data-testid="stSidebar"] * { color: #d4edda !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label { color: #a8d5b5 !important; }
.metric-card {
    background: #f8fffe;
    border: 1px solid #d0e8df;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-label { font-size: 13px; color: #5a7a6e; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 600; color: #0f3d2e; }
.metric-sub   { font-size: 11px; color: #8aab9a; margin-top: 2px; }
.section-title { font-size: 13px; font-weight: 600; color: #0f3d2e;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
.pred-cancer { background:#fef2f2; border:1px solid #fca5a5;
    border-radius:8px; padding:0.75rem 1rem; color:#991b1b; font-weight:600; }
.pred-normal { background:#f0fdf4; border:1px solid #86efac;
    border-radius:8px; padding:0.75rem 1rem; color:#14532d; font-weight:600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("model_bundle.pkl")


@st.cache_data
def load_defaults():
    gene_df  = pd.read_csv("gene_expression.csv",    index_col="cell_id")
    prot_df  = pd.read_csv("protein_expression.csv", index_col="cell_id")
    label_df = pd.read_csv("cell_labels.csv",        index_col="cell_id")
    tf_annot = pd.read_csv("tf_annotation.csv")
    return gene_df, prot_df, label_df, tf_annot


def prepare_features(prot_df, gene_df, feature_names):
    X = pd.concat([np.log1p(prot_df), np.log1p(gene_df)], axis=1)
    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[feature_names]


def plot_tf_importance(model, feature_names, tf_symbols, top_n=15):
    imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    tf_imp = imp[imp["feature"].isin(tf_symbols)].sort_values("importance", ascending=False).head(top_n)
    tf_imp = tf_imp.sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E24B4A" if v > tf_imp["importance"].median() else "#378ADD"
              for v in tf_imp["importance"]]
    bars = ax.barh(tf_imp["feature"], tf_imp["importance"], color=colors, edgecolor="none")
    ax.set_xlabel("Mean decrease in impurity", fontsize=11)
    ax.set_title(f"Top {top_n} transcription factors", fontsize=12, fontweight="600")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, tf_imp["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#444")
    high_patch = mpatches.Patch(color="#E24B4A", label="Above median importance")
    low_patch  = mpatches.Patch(color="#378ADD", label="Below median importance")
    ax.legend(handles=[high_patch, low_patch], fontsize=9, frameon=False)
    plt.tight_layout()
    return fig


def plot_roc(model, scaler, X, y):
    X_sc = scaler.transform(X)
    y_prob = model.predict_proba(X_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#B4B2A9", lw=1)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("ROC curve", fontsize=12, fontweight="600")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_confusion(model, scaler, X, y):
    X_sc = scaler.transform(X)
    y_pred = model.predict(X_sc)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Cancer"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix", fontsize=12, fontweight="600")
    plt.tight_layout()
    return fig


def plot_expression_compare(gene_df, label_df, tf_symbols, top_n=8):
    labels = label_df["label"]
    cancer_mean = gene_df.loc[labels == 1, [c for c in tf_symbols if c in gene_df.columns]].mean()
    normal_mean = gene_df.loc[labels == 0, [c for c in tf_symbols if c in gene_df.columns]].mean()
    compare = pd.DataFrame({"Cancer": cancer_mean, "Normal": normal_mean})
    compare = compare.sort_values("Cancer", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(compare))
    w = 0.38
    ax.bar(x - w / 2, compare["Cancer"], w, label="Cancer", color="#E24B4A", edgecolor="none")
    ax.bar(x + w / 2, compare["Normal"], w, label="Normal", color="#378ADD", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(compare.index, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mean expression", fontsize=11)
    ax.set_title(f"Top {top_n} TF mean expression: cancer vs normal", fontsize=12, fontweight="600")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


bundle = load_model()
model        = bundle["model"]
scaler       = bundle["scaler"]
feature_names = bundle["feature_names"]

gene_df_default, prot_df_default, label_df_default, tf_annot = load_defaults()
tf_symbols = tf_annot["gene_symbol"].tolist()

with st.sidebar:
    st.markdown("## Cancer TF Discovery")
    st.markdown("Random Forest · Protein + Gene Expression")
    st.markdown("---")
    page = st.selectbox("Section", [
        "Overview and model performance",
        "Transcription factor ranking",
        "Expression analysis",
        "Predict new cells"
    ])
    st.markdown("---")
    top_n = st.slider("TFs to display", min_value=5, max_value=20, value=15)
    st.markdown("---")
    st.markdown("**Dataset summary**")
    st.markdown(f"- 1,200 cells (600 cancer, 600 normal)")
    st.markdown(f"- 80 protein features")
    st.markdown(f"- 68 gene features (20 TFs)")
    st.markdown(f"- 5-fold cross-validation")


if page == "Overview and model performance":
    st.title("Cancer Transcription Factor Discovery")
    st.markdown(
        "This dashboard presents a Random Forest model trained to classify cancer vs normal cells "
        "using protein and gene expression data, and to rank transcription factors by their association "
        "with cancer cell growth."
    )

    X_all = prepare_features(prot_df_default, gene_df_default, feature_names)
    y_all = label_df_default["label"]
    X_all, y_all = X_all.align(y_all, join="inner", axis=0)
    X_all_sc = scaler.transform(X_all)
    y_prob_all = model.predict_proba(X_all_sc)[:, 1]
    y_pred_all = model.predict(X_all_sc)
    auc = roc_auc_score(y_all, y_prob_all)
    acc = (y_pred_all == y_all).mean()
    n_cancer = int((y_all == 1).sum())
    n_normal = int((y_all == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{acc*100:.1f}%</div>
            <div class="metric-sub">on full dataset</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">AUC-ROC</div>
            <div class="metric-value">{auc:.3f}</div>
            <div class="metric-sub">binary classification</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Cancer cells</div>
            <div class="metric-value">{n_cancer}</div>
            <div class="metric-sub">in dataset</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Normal cells</div>
            <div class="metric-value">{n_normal}</div>
            <div class="metric-sub">in dataset</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">ROC curve</div>', unsafe_allow_html=True)
        st.pyplot(plot_roc(model, scaler, X_all, y_all))
    with col2:
        st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
        st.pyplot(plot_confusion(model, scaler, X_all, y_all))

    st.markdown("---")
    st.markdown("**Model configuration**")
    cfg = pd.DataFrame({
        "Parameter": ["n_estimators", "max_depth", "min_samples_split", "max_features", "class_weight", "random_state"],
        "Value":     ["200", "15", "5", "0.12 (12%)", "balanced", "42"]
    })
    st.dataframe(cfg, use_container_width=True, hide_index=True)


elif page == "Transcription factor ranking":
    st.title("Transcription factor ranking")
    st.markdown(
        "Features are ranked by mean decrease in impurity (Gini importance). "
        "Only genes annotated as transcription factors in the TF annotation file are shown."
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-title">Feature importance chart</div>', unsafe_allow_html=True)
        st.pyplot(plot_tf_importance(model, feature_names, tf_symbols, top_n=top_n))

    with col2:
        st.markdown('<div class="section-title">Ranked TF table</div>', unsafe_allow_html=True)
        imp = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        tf_imp = (imp[imp["Feature"].isin(tf_symbols)]
                  .sort_values("Importance", ascending=False)
                  .head(top_n)
                  .reset_index(drop=True))
        tf_imp.index += 1
        tf_imp["Importance"] = tf_imp["Importance"].round(5)
        st.dataframe(tf_imp, use_container_width=True)

        buf = io.BytesIO()
        tf_imp.to_csv(buf, index=True)
        st.download_button("Download TF ranking as CSV", buf.getvalue(),
                           "tf_ranking.csv", "text/csv")


elif page == "Expression analysis":
    st.title("Expression analysis")
    st.markdown(
        "Compare transcription factor expression levels between cancer and normal cells "
        "to understand the biological signal the model is learning from."
    )

    st.markdown('<div class="section-title">Mean TF expression: cancer vs normal</div>', unsafe_allow_html=True)
    st.pyplot(plot_expression_compare(gene_df_default, label_df_default, tf_symbols, top_n=top_n))

    st.markdown("---")
    st.markdown('<div class="section-title">Violin plot: single TF distribution</div>', unsafe_allow_html=True)
    available_tfs = [c for c in tf_symbols if c in gene_df_default.columns]
    selected_tf = st.selectbox("Select transcription factor", available_tfs)

    if selected_tf:
        plot_df = pd.DataFrame({
            "Expression": np.log1p(gene_df_default[selected_tf]),
            "Group": label_df_default["label"].map({0: "Normal", 1: "Cancer"})
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=plot_df, x="Group", y="Expression",
                       palette={"Cancer": "#E24B4A", "Normal": "#378ADD"},
                       inner="box", ax=ax, linewidth=0.8)
        ax.set_title(f"{selected_tf} expression distribution", fontsize=12, fontweight="600")
        ax.set_ylabel("log1p(Expression)", fontsize=11)
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        cancer_med = np.log1p(gene_df_default.loc[label_df_default["label"] == 1, selected_tf]).median()
        normal_med = np.log1p(gene_df_default.loc[label_df_default["label"] == 0, selected_tf]).median()
        fold = np.expm1(cancer_med) / max(np.expm1(normal_med), 1e-9)
        col1, col2, col3 = st.columns(3)
        col1.metric("Cancer median (log1p)", f"{cancer_med:.3f}")
        col2.metric("Normal median (log1p)", f"{normal_med:.3f}")
        col3.metric("Fold change (raw)", f"{fold:.2f}x")


elif page == "Predict new cells":
    st.title("Predict new cells")
    st.markdown(
        "Upload protein and gene expression CSVs for new cells to get cancer/normal predictions "
        "and per-cell probability scores. Files must have a `cell_id` column and the same "
        "feature columns as the training data."
    )

    st.info(
        "No files yet? The default dataset is loaded below so you can explore predictions immediately.",
        icon="i"
    )

    use_default = st.checkbox("Use default dataset", value=True)

    if use_default:
        prot_upload = prot_df_default.copy()
        gene_upload = gene_df_default.copy()
        labels_known = label_df_default["label"]
    else:
        col1, col2 = st.columns(2)
        with col1:
            prot_file = st.file_uploader("Protein expression CSV", type="csv")
        with col2:
            gene_file = st.file_uploader("Gene expression CSV", type="csv")

        if not prot_file or not gene_file:
            st.stop()

        prot_upload = pd.read_csv(prot_file, index_col="cell_id")
        gene_upload = pd.read_csv(gene_file, index_col="cell_id")
        labels_known = None

    X_new = prepare_features(prot_upload, gene_upload, feature_names)
    X_new_sc = scaler.transform(X_new)
    predictions = model.predict(X_new_sc)
    probabilities = model.predict_proba(X_new_sc)[:, 1]

    results = pd.DataFrame({
        "cell_id": X_new.index,
        "prediction": ["Cancer" if p == 1 else "Normal" for p in predictions],
        "cancer_probability": probabilities.round(4)
    })

    n_pred_cancer = int((predictions == 1).sum())
    n_pred_normal = int((predictions == 0).sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total cells</div>
            <div class="metric-value">{len(predictions)}</div>
            <div class="metric-sub">submitted</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Predicted cancer</div>
            <div class="metric-value" style="color:#991b1b">{n_pred_cancer}</div>
            <div class="metric-sub">{n_pred_cancer/len(predictions)*100:.1f}% of total</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Predicted normal</div>
            <div class="metric-value" style="color:#14532d">{n_pred_normal}</div>
            <div class="metric-sub">{n_pred_normal/len(predictions)*100:.1f}% of total</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction results</div>', unsafe_allow_html=True)

    col_filter = st.selectbox("Filter by prediction", ["All", "Cancer", "Normal"])
    display = results if col_filter == "All" else results[results["prediction"] == col_filter]
    st.dataframe(display.reset_index(drop=True), use_container_width=True, height=320)

    buf = io.BytesIO()
    results.to_csv(buf, index=False)
    st.download_button("Download predictions as CSV", buf.getvalue(),
                       "predictions.csv", "text/csv")

    st.markdown("---")
    st.markdown('<div class="section-title">Probability distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    cancer_probs = probabilities[predictions == 1]
    normal_probs = probabilities[predictions == 0]
    ax.hist(normal_probs, bins=30, color="#378ADD", alpha=0.7, label="Predicted normal", edgecolor="none")
    ax.hist(cancer_probs, bins=30, color="#E24B4A", alpha=0.7, label="Predicted cancer", edgecolor="none")
    ax.axvline(0.5, color="#888", lw=1, linestyle="--", label="Decision boundary (0.5)")
    ax.set_xlabel("Cancer probability score", fontsize=11)
    ax.set_ylabel("Cell count", fontsize=11)
    ax.set_title("Distribution of cancer probability scores", fontsize=12, fontweight="600")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
