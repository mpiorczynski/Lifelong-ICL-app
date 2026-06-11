"""Interactive analysis of 0-shot vs few-shot single-task ICL baselines.

Self-contained: needs only the baseline results directory
(<baseline_dir>/<model>/nshot<k>/results.csv with the n/pred_dist/label_dist
columns). Task metadata (class set, test size, label distribution) is derived
from the CSVs — no repo data/ or tokenizer required, so the directory can be
downloaded and analyzed locally.

Usage:
    streamlit run app_zs_fs.py                     # uses ./baseline, ../output/default/baseline, or sidebar path
    BASELINE_DIR=~/Downloads/baseline streamlit run app_zs_fs.py
"""
import json
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import ttest_1samp

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRS = [
    os.environ.get("BASELINE_DIR", ""),
    os.path.join(os.getcwd(), "baseline"),
    os.path.join(HERE, "baseline"),
    os.path.abspath(os.path.join(HERE, "..", "output", "default", "baseline")),
]

st.set_page_config(page_title="0-shot vs few-shot ICL", layout="wide")


def _scale_key(name):
    m = re.search(r"(\d+(?:\.\d+)?)([mb])$", name)
    return float(m.group(1)) * (1000 if m.group(2) == "b" else 1) if m else float("inf")


def _parse_dist(value):
    """JSON Counter string -> Counter (empty on missing/malformed)."""
    try:
        return Counter(json.loads(value))
    except (TypeError, ValueError):
        return Counter()


def _dist_fields(total):
    """Counter -> (normalized 'label: pct' text, max share). Normalizing makes
    few-shot distributions aggregated across rounds comparable to 0-shot/label dists."""
    if not total:
        return "", np.nan
    n = sum(total.values())
    text = ", ".join(f"{k}: {v / n:.1%}" for k, v in total.most_common())
    return text, max(total.values()) / n


@st.cache_data
def discover_models(baseline_dir):
    """{model: [n_shot, ...]} for models with nshot0 plus at least one few-shot run."""
    models = {}
    if not os.path.isdir(baseline_dir):
        return models
    for model in sorted(os.listdir(baseline_dir), key=lambda m: (_scale_key(m), m)):
        mdir = os.path.join(baseline_dir, model)
        if not os.path.isdir(mdir):
            continue
        shots = sorted(
            int(d[5:]) for d in os.listdir(mdir)
            if d.startswith("nshot") and d[5:].isdigit()
            and os.path.isfile(os.path.join(mdir, d, "results.csv"))
        )
        if 0 in shots and len(shots) > 1:
            models[model] = shots
    return models


@st.cache_data
def load_data(baseline_dir, n_shot):
    models = discover_models(baseline_dir)
    rows = []
    for model, shots in models.items():
        if n_shot not in shots:
            continue
        zs = pd.read_csv(os.path.join(baseline_dir, model, "nshot0", "results.csv"), index_col=0)
        fs = pd.read_csv(os.path.join(baseline_dir, model, f"nshot{n_shot}", "results.csv"), index_col=0)
        for task, g in fs.groupby("task_name"):
            zrow = zs[zs.task_name == task]
            if zrow.empty:
                continue
            fs_preds = sum((_parse_dist(s) for s in g.get("pred_dist", pd.Series(dtype=object)).dropna()), Counter())
            zs_preds = _parse_dist(zrow.get("pred_dist", pd.Series(dtype=object)).head(1).squeeze())
            labels = _parse_dist(zrow.get("label_dist", pd.Series(dtype=object)).head(1).squeeze())
            fs_pred_dist, fs_max_share = _dist_fields(fs_preds)
            zs_pred_dist, zs_max_share = _dist_fields(zs_preds)
            label_dist, majority_frac = _dist_fields(labels)
            options = sorted((set(labels) | set(fs_preds) | set(zs_preds)) - {"NO_PREDICTION"})
            n_rounds = len(g)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # near-constant rounds trigger precision warnings
                t, p = (ttest_1samp(g.accuracy.values, zrow.accuracy.iloc[0]) if n_rounds > 1 else (np.nan, np.nan))
            rows.append({
                "model": model,
                "scale": model.rsplit("-", 1)[-1],
                "arch": model.rsplit("-", 1)[0],
                "task_name": task,
                "zs_accuracy": zrow.accuracy.iloc[0],
                "fs_accuracy": g.accuracy.mean(),
                "fs_accuracy_std": g.accuracy.std(),
                "zs_macro_f1": zrow.macro_f1.iloc[0],
                "fs_macro_f1": g.macro_f1.mean(),
                "fs_macro_f1_std": g.macro_f1.std(),
                "n_rounds": n_rounds,
                "test_n": int(zrow["n"].iloc[0]) if "n" in zrow else sum(labels.values()) or np.nan,
                "n_class": len(options) or np.nan,
                "options": " / ".join(options),
                "majority_frac": round(majority_frac, 3) if majority_frac == majority_frac else np.nan,
                "fs_pred_dist": fs_pred_dist,
                "zs_pred_dist": zs_pred_dist,
                "label_dist": label_dist,
                "fs_max_pred_share": fs_max_share,
                "zs_max_pred_share": zs_max_share,
                "ttest_p": p,
                "ttest_t": t,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        for m in ["accuracy", "macro_f1"]:
            df[f"delta_{m}"] = df[f"fs_{m}"] - df[f"zs_{m}"]
    return df


# ---------------- sidebar: data source ----------------
st.sidebar.header("Data")
default_dir = next((d for d in DEFAULT_DIRS if d and os.path.isdir(d)), DEFAULT_DIRS[-1])
baseline_dir = st.sidebar.text_input("Baseline results directory", value=default_dir,
                                     help="Directory containing <model>/nshot<k>/results.csv")
models = discover_models(baseline_dir)
if not models:
    st.error(f"No runs found: expected <model>/nshot0/results.csv plus a few-shot run under {baseline_dir!r}. "
             "Set the path in the sidebar or via the BASELINE_DIR environment variable.")
    st.stop()
fewshot_choices = sorted({s for shots in models.values() for s in shots if s > 0})
n_shot = st.sidebar.selectbox("Few-shot n_shot (vs 0-shot)", fewshot_choices,
                              index=fewshot_choices.index(8) if 8 in fewshot_choices else 0)

df = load_data(baseline_dir, n_shot)
if df.empty:
    st.error(f"No models with both nshot0 and nshot{n_shot} results in {baseline_dir!r}.")
    st.stop()
model_order = [m for m in models if m in set(df.model)]
has_dists = df.label_dist.str.len().gt(0).any()
if not has_dists:
    st.warning("Results lack the pred_dist/label_dist columns (older pipeline version) — "
               "distribution hovers, collapse view, and imbalance filters are disabled.")

# ---------------- sidebar: filters ----------------
st.sidebar.header("Filters")
scales = st.sidebar.multiselect("Scale", sorted(df.scale.unique(), key=_scale_key),
                                default=sorted(df.scale.unique(), key=_scale_key))
archs = st.sidebar.multiselect("Architecture", sorted(df.arch.unique()), default=sorted(df.arch.unique()))
metric = st.sidebar.radio("Metric", ["accuracy", "macro_f1"], horizontal=True)
y_mode = st.sidebar.radio("Y axis", [f"{n_shot}-shot", f"delta ({n_shot}-shot − 0-shot)"])
color_options = ["model"] + (["fs_max_pred_share", "majority_frac", "n_class"] if has_dists else [])
color_by = st.sidebar.selectbox(
    "Color by", color_options,
    format_func=lambda c: {
        "model": "model", "fs_max_pred_share": f"{n_shot}-shot prediction collapse (max pred share)",
        "majority_frac": "test majority fraction", "n_class": "number of classes"}[c],
)

sel = df[df.scale.isin(scales) & df.arch.isin(archs)]
if has_dists:
    st.sidebar.header("Task filters")
    maj_range = st.sidebar.slider("Test majority fraction", 0.0, 1.0, (0.0, 1.0), step=0.05,
                                  help="1/n_class = balanced; near 1.0 = nearly constant test labels.")
    nclass_range = st.sidebar.slider("Number of classes", int(df.n_class.min()), int(df.n_class.max()),
                                     (int(df.n_class.min()), int(df.n_class.max())))
    sel = sel[sel.majority_frac.between(*maj_range) & sel.n_class.between(*nclass_range)]
if st.sidebar.checkbox("Only significant deltas (t-test p < 0.05)", value=False):
    sel = sel[sel.ttest_p < 0.05]

st.title(f"0-shot vs {n_shot}-shot ICL across tasks and models")
st.caption(f"{baseline_dir} | {len(sel)} (model, task) points | {sel.task_name.nunique()} tasks "
           f"| {sel.model.nunique()} models")
if sel.empty:
    st.warning("No points match the current filters.")
    st.stop()

x_col, y_col = f"zs_{metric}", (f"fs_{metric}" if y_mode.endswith("-shot") else f"delta_{metric}")

tab_names = ["Scatter", "Delta heatmap"] + (["Prediction collapse"] if has_dists else []) + ["Tables"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ---------------- scatter ----------------
with tabs["Scatter"]:
    fig = px.scatter(
        sel, x=x_col, y=y_col,
        color=color_by,
        symbol="scale" if len(scales) > 1 and color_by == "model" else None,
        color_continuous_scale="Viridis" if color_by != "model" else None,
        custom_data=["model", "task_name", "options", "test_n", "majority_frac",
                     f"zs_{metric}", f"fs_{metric}", f"fs_{metric}_std", f"delta_{metric}", "ttest_p",
                     "zs_pred_dist", "fs_pred_dist", "label_dist"],
        height=650,
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.8),
        hovertemplate=(
            "<b>%{customdata[0]}</b> | %{customdata[1]}<br>"
            "options: %{customdata[2]}<br>"
            "test n: %{customdata[3]} | majority frac: %{customdata[4]}<br>"
            f"0-shot {metric}: %{{customdata[5]:.3f}} | {n_shot}-shot {metric}: %{{customdata[6]:.3f}} ± %{{customdata[7]:.3f}}<br>"
            "delta: %{customdata[8]:+.3f} (p=%{customdata[9]:.3f})<br>"
            "test labels: %{customdata[12]}<br>"
            "0-shot preds: %{customdata[10]}<br>"
            f"{n_shot}-shot preds (all rounds): %{{customdata[11]}}<extra></extra>"
        ),
    )
    lo = float(min(sel[x_col].min(), sel[y_col].min())) - 0.02
    hi = float(max(sel[x_col].max(), sel[y_col].max())) + 0.02
    if y_mode.endswith("-shot"):
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="gray"))
        fig.add_annotation(x=hi, y=hi, text="y = x (no ICL effect)", showarrow=False, yshift=12, font=dict(color="gray"))
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="no ICL effect", annotation_font_color="gray")
    fig.update_layout(xaxis_title=f"0-shot {metric}", yaxis_title=f"{y_mode} {metric}",
                      legend_title=color_by)
    st.plotly_chart(fig, width="stretch")
    st.caption("Points above the dashed line gain from few-shot ICL; below lose. "
               "Try coloring by prediction collapse: points where few-shot predicts almost one label.")

# ---------------- heatmap ----------------
with tabs["Delta heatmap"]:
    piv = sel.pivot_table(index="model", columns="task_name", values=f"delta_{metric}")
    piv = piv.reindex(index=[m for m in model_order if m in piv.index])
    piv = piv[piv.mean().sort_values().index]  # tasks ordered by mean delta
    lim = float(np.nanmax(np.abs(piv.values))) if piv.size else 1.0
    hfig = px.imshow(piv, color_continuous_scale="RdBu", zmin=-lim, zmax=lim, aspect="auto", height=420,
                     labels=dict(color=f"Δ {metric}"))
    hfig.update_xaxes(tickangle=60, tickfont=dict(size=9))
    st.plotly_chart(hfig, width="stretch")
    st.caption(f"Per-task delta ({n_shot}-shot − 0-shot), tasks sorted by mean delta across the selected models. "
               "Red columns hurt everywhere; blue columns help everywhere; mixed columns are model-specific.")

# ---------------- collapse ----------------
if has_dists:
    with tabs["Prediction collapse"]:
        st.markdown(
            "**Prediction collapse**: share of the most-predicted label among all test predictions. "
            "1.0 = the model predicts a single label for every example; 1/n_class ≈ uniform usage.")
        cfig = px.scatter(
            sel, x="zs_max_pred_share", y="fs_max_pred_share", color="model",
            custom_data=["model", "task_name", "zs_pred_dist", "fs_pred_dist", f"delta_{metric}"],
            height=600,
        )
        cfig.update_traces(
            marker=dict(size=9, opacity=0.8),
            hovertemplate=("<b>%{customdata[0]}</b> | %{customdata[1]}<br>"
                           "0-shot preds: %{customdata[2]}<br>"
                           f"{n_shot}-shot preds: %{{customdata[3]}}<br>"
                           f"Δ {metric}: %{{customdata[4]:+.3f}}<extra></extra>"))
        cfig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        cfig.update_layout(xaxis_title="0-shot max pred share", yaxis_title=f"{n_shot}-shot max pred share")
        st.plotly_chart(cfig, width="stretch")
        st.caption("Below the diagonal: demos diversify predictions (ICL engaging with inputs). "
                   "Top-right corner: collapsed both ways. Above the diagonal: demos *cause* collapse.")

# ---------------- tables ----------------
with tabs["Tables"]:
    st.subheader("Model summary (selected tasks)")
    summary = sel.groupby("model").agg(
        n_tasks=("task_name", "nunique"),
        zs=(f"zs_{metric}", "mean"),
        fs=(f"fs_{metric}", "mean"),
        delta=(f"delta_{metric}", "mean"),
        icl_effective=("ttest_p", lambda p: int(((p < 0.05) & (sel.loc[p.index, f"delta_{metric}"] > 0)).sum())),
        icl_harmful=("ttest_p", lambda p: int(((p < 0.05) & (sel.loc[p.index, f"delta_{metric}"] < 0)).sum())),
        fs_collapse=("fs_max_pred_share", "mean"),
    ).round(3).reindex([m for m in model_order if m in sel.model.unique()])
    st.dataframe(summary, width="stretch")
    st.caption("icl_effective / icl_harmful: tasks with significant (p<0.05) positive / negative delta. "
               "fs_collapse: mean few-shot max-pred-share (1.0 = constant-label predictor).")

    st.subheader("All (model, task) rows")
    show_cols = ["model", "task_name", "n_class", "test_n", "majority_frac",
                 f"zs_{metric}", f"fs_{metric}", f"fs_{metric}_std", f"delta_{metric}", "ttest_p",
                 "zs_max_pred_share", "fs_max_pred_share", "label_dist", "zs_pred_dist", "fs_pred_dist"]
    table = sel[show_cols].sort_values(f"delta_{metric}")
    st.dataframe(table, width="stretch", height=420)
    st.download_button("Download filtered CSV", table.to_csv(index=False), "zs_fs_filtered.csv", "text/csv")
