# -*- coding: utf-8 -*-
"""
3D Facial Mapping — Dementia Behavior Deviation (Research Only)

What this build addresses:
- NumPy 2.x: use np.ptp(...) (no ndarray.ptp)
- Face mesh always faces the viewer like the video (auto-mirror if needed)
- Landmark + pose smoothing (less jitter), fixed axis ranges & camera
- No flashing: stable Streamlit keys, no placeholder .empty() churn
- Live cluster plot uses a FIXED PCA basis (fluid updates, no re-layout)
  and only runs UMAP+HDBSCAN for the final snapshot
- DuplicateElementKey: distinct, stable keys per chart
- Larger charts for readability
"""

from __future__ import annotations
import os, sys, glob, json, time, math, shutil, logging
import random
import uuid

from typing import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Deque
from collections import deque, defaultdict

# --- 3rd-party (graceful) ---
try: import cv2
except Exception: cv2 = None
try: import numpy as np
except Exception: np = None
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    st = None; _HAS_STREAMLIT = False
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None; _HAS_MEDIAPIPE = False
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    go = None; _HAS_PLOTLY = False
try:
    from sklearn.ensemble import IsolationForest, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import StackingClassifier
    from sklearn.decomposition import PCA
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    
    
    
    _HAS_SKLEARN = True
except Exception:
    IsolationForest = StandardScaler = LogisticRegression = GradientBoostingClassifier = RandomForestClassifier = SVC = cross_val_score = Pipeline = None
    _HAS_SKLEARN = False
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False
try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False
try:
    import insightface
    _HAS_INSIGHTFACE = True
except Exception:
    _HAS_INSIGHTFACE = False
    
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

    
try:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#000000",
        "axes.facecolor":   "#000000",
        "savefig.facecolor":"#000000",
        "text.color":       "#e8ecf3",
        "axes.labelcolor":  "#e8ecf3",
        "xtick.color":      "#e8ecf3",
        "ytick.color":      "#e8ecf3",
        "grid.color":       "#222222"
    })
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Streamlit deprecation notices (yellow boxes) you don't want to see:
if _HAS_STREAMLIT:
    # Works across Streamlit versions
    try:
        st.set_option('client.showErrorDetails', False)
    except Exception:
        pass
    # Old option (pre-1.3x). Safe to ignore if missing.
    try:
        st.set_option('deprecation.showPyplotGlobalUse', False)
    except Exception:
        pass





try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False
import csv as _csv

# ---- Plotly/Streamlit config (single place) ----
PLOTLY_CFG_NO_BAR = {
    "displaylogo": False,
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}

PLOTLY_CFG_BAR = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["zoom", "pan", "select", "lasso2d", "toImage"],
    "responsive": True,
    "scrollZoom": False,
}

HIST_MAX = 5000  

# Optional: lightweight deep-learning for landmarks (autoencoder)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# ---------- CVPR-grade Batch Dashboard (Streamlit + Plotly) ----------
from dataclasses import dataclass
import numpy as np, pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, brier_score_loss, accuracy_score
)
import plotly.graph_objects as go
import plotly.express as px

@dataclass
class _Summary:
    n:int=0; tp:int=0; tn:int=0; fp:int=0; fn:int=0
    acc:float=np.nan; auc_roc:float=np.nan; auc_pr:float=np.nan
    ece:float=np.nan; mce:float=np.nan; brier:float=np.nan

class BatchDashboard:
    def __init__(self, uirev: str="global_dashboard_v1", n_boot: int=400, pos_label: int=1):
    
        self._ph = {}
        self.uirev = uirev
        self.n_boot = n_boot
        self.pos_label = pos_label
        self._laid_out = False
        self._rev = 0
        # NEW: stable, per-instance id (persists across reruns in this session)
        import streamlit as st
        sid_key = f"__dash_uid_{self.uirev}"
        if sid_key not in st.session_state:
            st.session_state[sid_key] = f"{self.uirev}-{uuid.uuid4().hex[:8]}"
        self._uid = st.session_state[sid_key]
        # Prebuild unique keys for every chart
        self._keys = {name: f"{self._uid}:{name}" for name in
                      ["conf","roc","pr","reliab","th_sweep","dist","learn","cal_tbl","det"]}

    def layout(self):
        import streamlit as st
        st.subheader("Performance & Calibration — Global Dashboard")
        top = st.container()
        row1 = st.columns([1.1, 1.1, 0.9])
        row2 = st.columns([1.0, 1.0, 1.0])
        row3 = st.columns([1.0, 1.0, 1.0])

        self._ph["conf"] = row1[0].empty()
        self._ph["roc"]  = row1[1].empty()
        self._ph["pr"]   = row1[2].empty()

        self._ph["reliab"]  = row2[0].empty()
        self._ph["th_sweep"]= row2[1].empty()
        self._ph["dist"]    = row2[2].empty()

        self._ph["learn"]   = row3[0].empty()
        self._ph["cal_tbl"] = row3[1].empty()
        self._ph["det"]     = row3[2].empty()

        self._laid_out = True


    def update(self, perf, threshold: float = 0.5):
    
        """perf must expose arrays: perf.y_true (ints 0/1), perf.y_score (floats 0..1)"""
        if not self._laid_out:
                self.layout()
        self._rev += 1
        y = np.asarray(perf.y_true, dtype=int)
        s = np.asarray(perf.y_score, dtype=float)
        if y.size == 0:
            return
        y_hat = (s >= threshold).astype(int)

        smry = self._summary(y, y_hat, s)
        self._plot_confusion(y, y_hat, smry)
        self._plot_roc_with_ci(y, s)
        self._plot_pr(y, s)
        self._plot_reliability(y, s, bins=15)
        self._plot_threshold_sweep(y, s)
        self._plot_distributions(y, s, threshold)
        self._plot_learning_curve(perf)
        self._render_calibration_table(smry)
        self._plot_det(y, s)

    def _draw(self, slot_name: str, fig: "go.Figure"):
        slot = self._ph[slot_name]
        slot.empty()  # clear previous element in this run
        slot.plotly_chart(
            fig,
            use_container_width=True,
            key=f"{self._keys[slot_name]}:{self._rev}"
        )
    

    # ---------- helpers: metrics ----------
    def _ece(self, y, s, bins=15):
        """Expected Calibration Error with adaptive bin edges by quantiles."""
        if len(s) < 2:
            return np.nan, np.nan, (np.array([0,1]), np.array([0,1]))
        q = np.linspace(0, 1, bins+1)
        edges = np.quantile(s, q)
        edges[0] = 0.0; edges[-1] = 1.0
        idx = np.clip(np.digitize(s, edges) - 1, 0, bins-1)
        ece = 0.0; mce = 0.0
        for b in range(bins):
            m = (idx == b)
            if not np.any(m): 
                continue
            conf = s[m].mean()
            acc  = (y[m] == self.pos_label).mean()
            w = m.mean()
            gap = abs(acc - conf)
            ece += w * gap
            mce = max(mce, gap)
        return float(ece), float(mce), (edges, idx)

    def _summary(self, y, y_hat, s)->_Summary:
        cm = confusion_matrix(y, y_hat, labels=[1,0])
        tp, fn, fp, tn = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        acc = (tp+tn)/max(1, tp+tn+fp+fn)
        # AUC ROC/PR (guard for degenerate sets)
        auc_roc = np.nan; auc_pr = np.nan
        if len(np.unique(y)) > 1:
            fpr, tpr, _ = roc_curve(y, s, pos_label=self.pos_label)
            auc_roc = auc(fpr, tpr)
        try:
            auc_pr = average_precision_score(y, s, pos_label=self.pos_label)
        except Exception:
            pass
        # Brier
        try:
            brier = brier_score_loss(y, s, pos_label=self.pos_label)
        except Exception:
            brier = np.nan
        ece, mce, _ = self._ece(y, s, bins=15)
        return _Summary(n=len(y), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                        acc=float(acc), auc_roc=float(auc_roc),
                        auc_pr=float(auc_pr), ece=float(ece), mce=float(mce),
                        brier=float(brier))


    # --- Confusion matrix (Plotly) ---
    def render_confusion_plotly(y_true, y_pred,
                                labels=("Normotypical", "Dementia")):
        # Keep this order so x-axis shows [Dementia, Normotypical]
        order_true = [0, 1]   # 0=Normotypical, 1=Dementia
        order_pred = [1, 0]   # x-axis: Dementia first, then Normotypical
    
        cm = confusion_matrix(y_true, y_pred, labels=order_true)[:, order_pred]
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct = (cm / row_sums)
    
        text = [[f"{cm[r, c]:d}<br>({cm_pct[r, c]*100:.0f}%)"
                 for c in range(cm.shape[1])] for r in range(cm.shape[0])]
    
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[labels[1], labels[0]],  # Predicted axis
                y=[labels[0], labels[1]],  # True axis
                text=text,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Blues",
                showscale=True,
                hovertemplate="True=%{y}<br>Pred=%{x}<br>Count=%{z}<extra></extra>",
                zmin=0,
                zmax=int(cm.max()) if cm.size and cm.max() else 1
            )
        )
        acc = accuracy_score(y_true, y_pred)
        fig.update_layout(
            title=f"Confusion  (n={len(y_true)} | acc={acc:.2f})",
            xaxis_title="Predicted",
            yaxis_title="True",
            template="plotly_dark",
            margin=dict(l=70, r=30, t=60, b=50),
        )
        return fig
    
    
    # --- Reliability diagram (Plotly) ---
    def _wilson_interval(k, n, z=1.96):
        if n == 0: return (0.0, 0.0)
        phat = k / n
        denom = 1 + z**2/n
        centre = phat + z*z/(2*n)
        rad = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)  # <- math.sqrt
        low  = (centre - rad)/denom
        high = (centre + rad)/denom
        return max(0.0, low), min(1.0, high)
    
    
    def _quantile_bins(p, n_bins=12):
        qs = np.linspace(0, 1, n_bins+1)
        edges = np.quantile(p, qs)
        return np.unique(edges)
    
    def render_reliability_plotly(y_true, y_prob, n_bins=12, seed=7):
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        n = len(y_true)
    
        # Build bins on quantiles; fallback to uniform if collapsed
        edges = _quantile_bins(y_prob, n_bins=n_bins)
        if len(edges) < 3:
            edges = np.linspace(0, 1, n_bins+1)
    
        bins = np.digitize(y_prob, edges[1:-1], right=True)
        bin_stats = []
        ece = 0.0
        for b in range(len(edges)-1):
            mask = (bins == b)
            nb = int(mask.sum())
            if nb == 0:
                continue
            conf = float(y_prob[mask].mean())
            tr = float(y_true[mask].mean())
            lo, hi = _wilson_interval(int(y_true[mask].sum()), nb)
            bin_stats.append((conf, tr, nb, lo, hi))
            ece += (nb/n) * abs(tr - conf)
    
        # Calibration cloud (all points), jitter 0/1 so they don’t overlap
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.035, 0.035, size=n)
        y_cloud = np.clip(y_true + jitter, 0, 1)
    
        fig = go.Figure()
    
        # 1) all datapoints
        fig.add_trace(go.Scattergl(
            x=y_prob, y=y_cloud,
            mode="markers",
            name=f"{n} samples",
            opacity=0.25,
            marker=dict(size=6),
            hovertemplate="p̂=%{x:.3f}<br>y=%{customdata}<extra></extra>",
            customdata=y_true
        ))
    
        # 2) perfect diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="perfect",
            line=dict(dash="dash", width=2)
        ))
    
        # 3) binned curve with Wilson CIs and marker size ∝ count
        if bin_stats:
            xs  = [c for c,_,_,_,_ in bin_stats]
            ys  = [t for _,t,_,_,_ in bin_stats]
            ns  = [nb for *_, nb, _lo, _hi in bin_stats]
            los = [lo for *_, lo, _hi in bin_stats]
            his = [hi for *_, _lo, hi in bin_stats]
            err_minus = [y - lo for y, lo in zip(ys, los)]
            err_plus  = [hi - y for y, hi in zip(ys, his)]
            msize = (np.array(ns) / max(ns)) * 14 + 8
    
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=f"quantile bins (n_bins={len(xs)})",
                marker=dict(size=msize),
                error_y=dict(
                    type="data",
                    array=err_plus,
                    arrayminus=err_minus,
                    visible=True,
                    thickness=1
                ),
                hovertemplate=("mean p̂=%{x:.3f}<br>true rate=%{y:.3f}"
                               "<br>bin n=%{customdata}"),
                customdata=ns
            ))
    
        fig.update_layout(
            title=f"Reliability Diagram (quantile bins) — ECE={ece:.3f}",
            xaxis_title="Mean confidence",
            yaxis_title="True rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template="plotly_dark",
            margin=dict(l=70, r=30, t=60, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.02)
        )
        return fig
    
    def _plot_confusion(self, y, y_hat, smry:_Summary):
        # Order rows/cols as ["Dementia","Normotypical"]
        labels = ["Dementia", "Normotypical"]
        cm = confusion_matrix(y, y_hat, labels=[1, 0]).astype(int)
    
        # Row-wise percentages (recall-style)
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct = cm / row_sums
    
        # Centered text for each cell: "count\n(percent)"
        text = [
            [f"{cm[r, c]:d}<br>({cm_pct[r, c]*100:.0f}%)" for c in range(cm.shape[1])]
            for r in range(cm.shape[0])
        ]
    
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,               # Predicted
                y=labels,               # True
                colorscale="Blues",
                zmin=0,
                zmax=int(cm.max()) if cm.size and cm.max() else 1,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=16),
                hovertemplate="True=%{y}<br>Pred=%{x}<br>Count=%{z}<extra></extra>",
                showscale=True,
            )
        )
        fig.update_layout(
            title=f"Confusion  (n={smry.n} | acc={smry.acc:.2f})",
            xaxis_title="Predicted",
            yaxis_title="True",
            template="plotly_dark",
            uirevision=self.uirev,  # stable; prevents re-layout "flash"
            margin=dict(l=40, r=10, t=45, b=40),
            height=380,
        )
    
        # single, stable draw slot (prebuilt in layout())
        self._draw("conf", fig)
    
    

    def _bootstrap_ci_xy(self, y, s, curve="roc"):
        if len(np.unique(y)) < 2:
            return None
        rng = np.random.RandomState(7)
        xs, ys = [], []
        for _ in range(self.n_boot):
            idx = rng.randint(0, len(y), len(y))
            yb, sb = y[idx], s[idx]
            if curve == "roc":
                x, yv, _ = roc_curve(yb, sb, pos_label=self.pos_label)
            else:
                yv, x, _ = precision_recall_curve(yb, sb, pos_label=self.pos_label)
            xs.append(x); ys.append(yv)
        # Interpolate to common grid
        grid = np.linspace(0,1,101)
        interp_y = []
        for x, yv in zip(xs, ys):
            interp_y.append(np.interp(grid, x, yv))
        lo = np.percentile(interp_y, 2.5, axis=0)
        hi = np.percentile(interp_y,97.5, axis=0)
        return grid, lo, hi

    def _plot_roc_with_ci(self, y, s):
        if len(np.unique(y)) < 2:
            self._ph["roc"].write("ROC: need both classes.")
            return
        fpr, tpr, _ = roc_curve(y, s, pos_label=self.pos_label)
        au = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={au:.3f}"))
        # CI band
        ci = self._bootstrap_ci_xy(y, s, curve="roc")
        if ci is not None:
            grid, lo, hi = ci
            fig.add_traces([
                go.Scatter(x=grid, y=hi, line=dict(width=0), showlegend=False),
                go.Scatter(x=grid, y=lo, fill="tonexty", line=dict(width=0),
                           name="95% CI", opacity=0.25)
            ])
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 name="chance", line=dict(dash="dash")))
        fig.update_layout(title="ROC (with 95% CI)", xaxis_title="FPR",
                          yaxis_title="TPR", uirevision=self.uirev,
                          margin=dict(l=40,r=10,t=45,b=40), height=380)
        self._draw("roc", fig)
        
        

    def _plot_pr(self, y, s):
        if len(np.unique(y)) < 2:
            self._ph["pr"].write("PR: need both classes.")
            return
        p, r, _ = precision_recall_curve(y, s, pos_label=self.pos_label)
        ap = average_precision_score(y, s, pos_label=self.pos_label)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"AP={ap:.3f}"))
        fig.update_layout(title="Precision–Recall", xaxis_title="Recall",
                          yaxis_title="Precision", uirevision=self.uirev,
                          margin=dict(l=40,r=10,t=45,b=40), height=380)
        self._draw("pr", fig)
        
        

    def _plot_reliability(self, y, s, bins=15):
        # quantile bins
        if len(s) < 2:
            self._ph["reliab"].write("Reliability: not enough points.")
            return
        q = np.linspace(0,1,bins+1)
        edges = np.quantile(s, q); edges[0]=0.0; edges[-1]=1.0
        idx = np.clip(np.digitize(s, edges)-1, 0, bins-1)
        confs = []; accs = []; sizes=[]
        for b in range(bins):
            m = (idx==b)
            if not np.any(m): continue
            confs.append(s[m].mean())
            accs.append((y[m]==self.pos_label).mean())
            sizes.append(int(m.sum()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=confs, y=accs, mode="markers+lines",
                                 name="empirical"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 name="perfect", line=dict(dash="dash")))
        fig.update_layout(title="Reliability Diagram (quantile bins)",
                          xaxis_title="Mean confidence", yaxis_title="True rate",
                          uirevision=self.uirev, margin=dict(l=40,r=10,t=45,b=40),
                          height=380)
        self._draw("reliab", fig)
        
        

    def _plot_threshold_sweep(self, y, s):
        if len(np.unique(y)) < 2:
            self._ph["th_sweep"].write("Threshold sweep: need both classes.")
            return
        th = np.linspace(0,1,101)
        ba, f1, mcc = [], [], []
        for t in th:
            yh = (s>=t).astype(int)
            tp = np.sum((y==1)&(yh==1)); tn = np.sum((y==0)&(yh==0))
            fp = np.sum((y==0)&(yh==1)); fn = np.sum((y==1)&(yh==0))
            sens = tp/max(1,tp+fn); spec = tn/max(1,tn+fp)
            ba.append(0.5*(sens+spec))
            p = tp/max(1,tp+fp); r = sens
            f1.append(0.0 if (p+r)==0 else 2*p*r/(p+r))
            denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            mcc.append(0.0 if denom==0 else ((tp*tn)-(fp*fn))/denom)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=th, y=ba, name="Balanced Acc", mode="lines"))
        fig.add_trace(go.Scatter(x=th, y=f1, name="F1", mode="lines"))
        fig.add_trace(go.Scatter(x=th, y=mcc, name="MCC", mode="lines"))
        fig.update_layout(title="Metric vs Threshold", xaxis_title="Threshold",
                          yaxis_title="Score", uirevision=self.uirev,
                          margin=dict(l=40,r=10,t=45,b=40), height=380)
        self._draw("th_sweep", fig)
        
        

    def _plot_distributions(self, y, s, threshold):
        if len(s)==0:
            return
        df = pd.DataFrame({"score":s, "label":np.where(y==1,"Dementia","Normotypical")})
        fig = px.histogram(df, x="score", color="label", barmode="overlay",
                           nbins=40, histnorm="probability density")
        fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"t={threshold:.2f}")
        fig.update_layout(title="Score Distributions (class-conditional)",
                          uirevision=self.uirev, margin=dict(l=40,r=10,t=45,b=40),
                          height=380)
        self._draw("dist", fig)
        
        

    def _plot_learning_curve(self, perf):
        # expects perf.history as list of dicts with {"n":i, "acc":..., "auc":..., "ece":...}
        hist = getattr(perf, "history", [])
        if not hist:
            # build a minimal cumulative curve from current rows
            y = np.asarray(perf.y_true, int)
            s = np.asarray(perf.y_score, float)
            if len(y) == 0: return
            accs=[]; aucs=[]; eces=[]; ns=[]
            for i in range(1, len(y)+1):
                yi, si = y[:i], s[:i]
                yh = (si>=0.5).astype(int)
                accs.append((yi==yh).mean())
                if len(np.unique(yi))>1:
                    fpr, tpr, _ = roc_curve(yi, si, pos_label=1)
                    aucs.append(auc(fpr,tpr))
                else:
                    aucs.append(np.nan)
                eces.append(self._ece(yi, si, bins=10)[0])
                ns.append(i)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ns, y=accs, name="Acc", mode="lines"))
            fig.add_trace(go.Scatter(x=ns, y=aucs, name="AUC", mode="lines"))
            fig.add_trace(go.Scatter(x=ns, y=eces, name="ECE", mode="lines"))
        else:
            df = pd.DataFrame(hist)
            fig = go.Figure()
            for k in [c for c in df.columns if c!="n"]:
                fig.add_trace(go.Scatter(x=df["n"], y=df[k], name=k, mode="lines"))
        fig.update_layout(title="Learning Curve (cumulative)", xaxis_title="# processed",
                          yaxis_title="Score", uirevision=self.uirev,
                          margin=dict(l=40,r=10,t=45,b=40), height=380)
        self._draw("learn", fig)
        
        

    def _render_calibration_table(self, smry:_Summary):
        df = pd.DataFrame({
            "Metric":["AUC-ROC","AP","ECE","MCE","Brier","Acc","n"],
            "Value":[smry.auc_roc, smry.auc_pr, smry.ece, smry.mce, smry.brier, smry.acc, smry.n]
        })
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df["Metric"], [f"{v:.3f}" if isinstance(v,float) else v for v in df["Value"]]])
        )])
        fig.update_layout(title="Calibration / Summary", height=380, margin=dict(l=10,r=10,t=45,b=10))
        self._draw("cal_tbl", fig)
        
        

    def _plot_det(self, y, s):
        if len(np.unique(y)) < 2:
            self._ph["det"].write("DET: need both classes.")
            return
        fpr, tpr, _ = roc_curve(y, s, pos_label=self.pos_label)
        fnr = 1 - tpr
        # Normal deviate transform
        def _nd(p): 
            p = np.clip(p, 1e-6, 1-1e-6)
            from math import sqrt, log
            return np.sqrt(2)*self._erfinv(2*p-1)
        x = np.array(list(map(_nd, fpr)))
        yv= np.array(list(map(_nd, fnr)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=yv, mode="lines", name="DET"))
        fig.update_layout(title="DET Curve", xaxis_title="FPR (norm deviate)",
                          yaxis_title="FNR (norm deviate)",
                          uirevision=self.uirev, height=380,
                          margin=dict(l=40,r=10,t=45,b=40))
        self._draw("det", fig)
        
        

    # quick erf^-1 to avoid SciPy
    @staticmethod
    def _erfinv(x):
        # Winitzki approximation
        a = 0.147
        ln = np.log(1 - x*x)
        first = 2/(np.pi*a) + ln/2
        return np.sign(x) * np.sqrt( np.sqrt(first**2 - ln/a) - first )
# --------------------------------------------------------------------


# --- Logging ---
LOG = logging.getLogger("neurocv3d")
if not LOG.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

def _smooth_pose(R: Optional["np.ndarray"], alpha: float = 0.15) -> Optional["np.ndarray"]:
    if R is None or st is None: return R
    e = rotation_to_euler(R)
    key = "pose_euler_smooth"
    if key not in st.session_state:
        st.session_state[key] = e
        return _euler_to_R(e)
    s = st.session_state[key]
    s = (1.0 - alpha) * s + alpha * e
    st.session_state[key] = s
    return _euler_to_R(s)

def _smooth_landmarks(lms: Optional["np.ndarray"], alpha: float = 0.25) -> Optional["np.ndarray"]:
    if lms is None or st is None: return lms
    key = "lms_smooth"
    lms = lms.astype(np.float32)
    if key not in st.session_state or not isinstance(st.session_state[key], np.ndarray) or st.session_state[key].shape != lms.shape:
        st.session_state[key] = lms
    else:
        st.session_state[key] = (1.0 - alpha) * st.session_state[key] + alpha * lms
    return st.session_state[key]


def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def ts() -> str: return time.strftime("%Y%m%d-%H%M%S")

# --- Config ---
@dataclass
class AppConfig:
    cache_dir: str = "./cache"
    face_detector: str = "mediapipe"
    use_isolation_forest: bool = True
    # speed knobs
    process_downscale: float = 1.0
    preview_downscale: float = 1.0
    skip_every_n: int = 0
    update_preview_every_n: int = 2
    skip_live_3d: bool = False
    # windows
    baseline_s: float = 15.0
    window_s: float = 6.0
    step_s: float = 2.0
    max_frames_debug: int = 0

# --- Detection / mesh ---
class FaceDetector:
    def detect(self, img): raise NotImplementedError

class MediaPipeDetector(FaceDetector):
    def __init__(self):
        if not _HAS_MEDIAPIPE: raise RuntimeError("mediapipe not installed")
        self.fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    def detect(self, img):
        H,W = img.shape[:2]
        res = self.fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out=[]
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x,y,w,h = int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H)
                s = float(d.score[0]) if d.score else 0.0
                out.append((x,y,w,h,s))
        return out

class InsightFaceDetector(FaceDetector):
    def __init__(self, fallback: Optional[FaceDetector]=None):
        if not _HAS_INSIGHTFACE: raise RuntimeError("insightface not installed")
        self.app = insightface.app.FaceAnalysis(name="buffalo_l"); self.app.prepare(ctx_id=-1)
        self.fallback=fallback
    def detect(self, img):
        try:
            faces = self.app.get(img)
            out=[]
            for f in faces:
                x1,y1,x2,y2 = f.bbox.astype(int); out.append((x1,y1,x2-x1,y2-y1,float(getattr(f,"det_score",1.0))))
            if out: return out
        except Exception: pass
        return self.fallback.detect(img) if self.fallback else []

class FaceMesh:
    def __init__(self, refine=True):
        if not _HAS_MEDIAPIPE: raise RuntimeError("mediapipe not installed")
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,
                                                    refine_landmarks=refine,min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
    def __call__(self, img) -> Optional["np.ndarray"]:
        res = self.mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: return None
        pts = [(p.x,p.y,p.z) for p in res.multi_face_landmarks[0].landmark]
        return np.asarray(pts, dtype=np.float32)

# --- Pose ---
MP_IDX = dict(nose=1, chin=152, re=33, le=263, rm=61, lm=291)
OBJ3D = np.float32([[0,0,0],[0,-50,-30],[35,35,-30],[-35,35,-30],[30,-30,-30],[-30,-30,-30]])

def rotation_to_euler(R: "np.ndarray") -> "np.ndarray":
    sy = math.sqrt(R[0,0]**2+R[1,0]**2)
    if sy>=1e-6:
        x = math.degrees(math.atan2(R[2,1],R[2,2]))
        y = math.degrees(math.atan2(-R[2,0],sy))
        z = math.degrees(math.atan2(R[1,0],R[0,0]))
    else:
        x = math.degrees(math.atan2(-R[1,2],R[1,1])); y = math.degrees(math.atan2(-R[2,0],sy)); z = 0.0
    return np.array([z,y,x], dtype=np.float32)

def _euler_to_R(e: "np.ndarray") -> "np.ndarray":
    z, y, x = [math.radians(float(v)) for v in e]
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def _smooth_pose(R: Optional["np.ndarray"], alpha: float = 0.15) -> Optional["np.ndarray"]:
    if R is None or st is None: return R
    e = rotation_to_euler(R)
    key = "pose_euler_smooth"
    if key not in st.session_state:
        st.session_state[key] = e
        return _euler_to_R(e)
    s = st.session_state[key]
    s = (1.0 - alpha) * s + alpha * e
    st.session_state[key] = s
    return _euler_to_R(s)

def _smooth_landmarks(lms: Optional["np.ndarray"], alpha: float = 0.25) -> Optional["np.ndarray"]:
    if lms is None or st is None: return lms
    key = "lms_smooth"
    lms = lms.astype(np.float32)
    if key not in st.session_state or not isinstance(st.session_state[key], np.ndarray) or st.session_state[key].shape != lms.shape:
        st.session_state[key] = lms
    else:
        st.session_state[key] = (1.0 - alpha) * st.session_state[key] + alpha * lms
    return st.session_state[key]

def head_pose(img, lms) -> Optional[Tuple["np.ndarray","np.ndarray","np.ndarray"]]:
    H,W = img.shape[:2]
    ids = [MP_IDX["nose"],MP_IDX["chin"],MP_IDX["re"],MP_IDX["le"],MP_IDX["rm"],MP_IDX["lm"]]
    pts = lms[ids,:2]; pts = np.column_stack([pts[:,0]*W, pts[:,1]*H]).astype(np.float32)
    f = 1.2*max(W,H); K = np.array([[f,0,W/2],[0,f,H/2],[0,0,1]], dtype=np.float32)
    ok,rvec,tvec = cv2.solvePnP(OBJ3D, pts, K, np.zeros((5,1),np.float32), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    R,_ = cv2.Rodrigues(rvec); e = rotation_to_euler(R)
    return R,tvec,e

# --- Features ---
class Feat:
    def __init__(self, fps: float):
        self.fps = max(1.0,float(fps))
        self.prev_e: Optional["np.ndarray"] = None
        self.h_blink: Deque[float] = deque(maxlen=int(self.fps*10))
        self.h_ear:   Deque[float] = deque(maxlen=int(self.fps*10))
        self.h_jaw:   Deque[float] = deque(maxlen=int(self.fps*10))
        self.h_gaze:  Deque[float] = deque(maxlen=int(self.fps*10))
        self.h_brow:  Deque[float] = deque(maxlen=int(self.fps*10))
        # --- Robust blink state ---
        self.ear_hist: Deque[float] = deque(maxlen=int(self.fps * 6))  # ~6s history
        self.blink_state: int = 0          # 0=open, 1=closed
        self.closed_len: int = 0           # frames spent closed
        self.t: float = 0.0                # time in seconds
        self.blink_times: Deque[float] = deque()  # timestamps (s) of blink events
        
    @staticmethod
    def _d(a,b): return float(np.linalg.norm(a-b))
    def ear(self, l):
        # 6-point EAR per eye (MediaPipe indices)
        # Left eye: 33-133 (horiz), vertical pairs (159,145) and (160,144)
        # Right eye: 263-362 (horiz), vertical pairs (386,374) and (385,373)
        def _eye_ear(hl, hr, v1a, v1b, v2a, v2b):
            v1 = self._d(l[v1a, :2], l[v1b, :2])
            v2 = self._d(l[v2a, :2], l[v2b, :2])
            h  = self._d(l[hl,  :2], l[hr,  :2]) + 1e-6
            return (v1 + v2) / (2.0 * h)
    
        L = _eye_ear(33, 133, 159, 145, 160, 144)
        R = _eye_ear(263, 362, 386, 374, 385, 373)
        return (L + R) / 2.0, L, R
    
    
    def _update_blink(self, ear_val: float) -> Tuple[int, float]:
        """
        Hysteresis over a robust EAR baseline:
          - close when EAR <= (median - 2.2*MAD)
          - open  when EAR >= (median - 1.2*MAD)
          - count a blink on closed→open if closed ≥ ~60ms
          - rate computed over a 12s sliding window
        Returns (blink_event, blink_rate_hz)
        """
        self.ear_hist.append(float(ear_val))
        self.t += 1.0 / self.fps
    
        if len(self.ear_hist) < max(6, int(self.fps * 0.3)):
            # not enough context yet
            return 0, 0.0
    
        arr = np.asarray(self.ear_hist, dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)) + 1e-6)
    
        close_thr = med - 2.2 * mad
        open_thr  = med - 1.2 * mad
        blink_event = 0
    
        if self.blink_state == 0:
            if ear_val <= close_thr:
                self.blink_state = 1
                self.closed_len = 1
        else:
            self.closed_len += 1
            if ear_val >= open_thr:
                if self.closed_len >= max(2, int(0.06 * self.fps)):  # ≥ ~60 ms
                    self.blink_times.append(self.t)
                    blink_event = 1
                self.blink_state = 0
                self.closed_len = 0
    
        # Compute live rate in Hz over a 12s window
        win = 12.0
        while self.blink_times and (self.t - self.blink_times[0]) > win:
            self.blink_times.popleft()
        rate_hz = float(len(self.blink_times) / win)
        return blink_event, rate_hz
    
    
    def mouth(self,l):
        u,lw=l[13,:2],l[14,:2]; lc,rc=l[61,:2],l[291,:2]
        return self._d(u,lw)/(self._d(lc,rc)+1e-6)
    def brow_asym(self,l):
        left=np.mean(l[[70,63,105],1]); right=np.mean(l[[336,296,334],1]); return float(abs(left-right))
    def gaze_var(self,l):
        iris=[468,469,470,471,472] if l.shape[0]>472 else []
        if not iris: return 0.0
        pts=l[iris,:2]; c=np.mean(pts,axis=0); return float(np.mean(np.linalg.norm(pts-c,axis=1)))
    def head_jitter(self,R):
        if R is None: return 0.0
        e = rotation_to_euler(R)
        if self.prev_e is None: self.prev_e=e; return 0.0
        d = np.abs(e-self.prev_e); self.prev_e=e; return float(np.sum(d))
    def per_frame(self, l, R):
        ear, el, er = self.ear(l)
        # Robust blink event + live rate (Hz)
        blink_evt, rate_hz = self._update_blink(ear)
    
        jaw   = self.mouth(l)
        brow  = self.brow_asym(l)
        gaze  = self.gaze_var(l)
        jitter= self.head_jitter(R)
    
        # histories for sparklines
        self.h_blink.append(blink_evt)   # event stream (0 except 1 on blink)
        self.h_ear.append(ear)
        self.h_jaw.append(jaw)
        self.h_gaze.append(gaze)
        self.h_brow.append(brow)
    
        return dict(
            ear=ear, ear_l=el, ear_r=er,
            blink=blink_evt,                 # 1 only on closed→open (event)
            blink_rate_hz_live=rate_hz,      # true live frequency
            mouth_open=jaw, brow_asym=brow,
            gaze_disp=gaze, head_jitter=jitter
        )
    
    
    def aggs(self):
        def var(dq): return float(np.var(np.asarray(dq,dtype=np.float32))) if dq else 0.0
        def rng(dq):
            if not dq: return 0.0
            a=np.asarray(dq,dtype=np.float32); return float(a.max()-a.min())
        def blink_rate(dq):
            if not dq: return 0.0
            a=np.asarray(dq,dtype=np.float32); rising=np.clip(a[1:]-a[:-1],0,1).sum()
            return float(rising/max(1.0,len(a)/self.fps))
        return dict(blink_rate_hz=blink_rate(self.h_blink), ear_var=var(self.h_ear),
                    jaw_var=var(self.h_jaw), gaze_var=var(self.h_gaze), brow_asym_amp=rng(self.h_brow))

# --- Baseline/DBDI ---
@dataclass
class NormStats:
    mean: Dict[str,float]=field(default_factory=dict)
    std: Dict[str,float]=field(default_factory=dict)
    def z(self,f):
        out={}
        for k,v in f.items():
            m=self.mean.get(k,0.0); s=self.std.get(k,1.0); s=1.0 if s<1e-6 else s
            out[k]=(v-m)/s
        return out

class Baseline:
    def __init__(self,use_iso=True):
        self.use_iso=bool(use_iso and _HAS_SKLEARN); self.stats=NormStats()
        self.names: List[str]=[]; self.scaler=None; self.iso=None
    def fit(self, windows: List[Dict[str,float]]):
        if not windows: return
        keys=sorted(windows[0].keys()); self.names=keys
        X=np.asarray([[w[k] for k in keys] for w in windows],dtype=np.float32)
        self.stats.mean={k:float(X[:,i].mean()) for i,k in enumerate(keys)}
        self.stats.std ={k:float(X[:,i].std(ddof=1)+1e-6) for i,k in enumerate(keys)}
        if self.use_iso:
            self.scaler=StandardScaler().fit(X); Xn=self.scaler.transform(X)
            self.iso=IsolationForest(n_estimators=256, contamination=0.06, random_state=0).fit(Xn)
    def score(self, feats: Dict[str,float]) -> Tuple[float,Dict[str,float]]:
        z=self.stats.z(feats)
        if self.use_iso and self.iso is not None and self.scaler is not None and self.names:
            x=np.asarray([[feats[k] for k in self.names]],dtype=np.float32); xn=self.scaler.transform(x)
            raw=-float(self.iso.score_samples(xn)[0]); dbdi=float(np.clip(33.0*(raw-0.0),0.0,100.0)); return dbdi,z
        return float(np.clip(15.0*np.mean(np.abs(list(z.values()))),0.0,100.0)), z

# --- Research classifier ---
class Vectorizer:
    KEYS=["ear","ear_l","ear_r","blink","mouth_open","brow_asym","gaze_disp","head_jitter",
          "blink_rate_hz","ear_var","jaw_var","gaze_var","brow_asym_amp"]
    def __call__(self, frames: List[Dict[str,float]]):
        if not frames: return [], []
        present=sorted({k for f in frames for k in f if k in self.KEYS}); vec=[]; names=[]
        for k in present:
            a=np.asarray([f.get(k,0.0) for f in frames],dtype=np.float32)
            stats=[float(np.mean(a)), float(np.std(a)+1e-6), float(np.median(a)), float(np.percentile(a,10)), float(np.percentile(a,90))]
            vec.extend(stats); names.extend([f"{k}_mean",f"{k}_std",f"{k}_median",f"{k}_p10",f"{k}_p90"])
        return vec,names

class ResearchClassifier:
    """
    Supervised classifier with selectable algorithms + 'replace model' workflow.
    Trains from cache/features_db.json and overwrites cache/research_model.joblib
    """
    MODEL_REGISTRY = {
        "GradientBoosting": lambda: GradientBoostingClassifier(),
        "LogisticRegression": lambda: LogisticRegression(max_iter=2000),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
        "SVC (prob)": lambda: SVC(kernel="rbf", probability=True)
    }

    def __init__(self, cache_dir: str):
        ensure_dir(cache_dir)
        self.cache_dir = cache_dir
        self.db  = os.path.join(cache_dir, "features_db.json")
        self.m   = os.path.join(cache_dir, "research_model.joblib")     # calibrated stack
        self.meta_path = os.path.join(cache_dir, "model_meta.json")     # threshold + cv acc
        self.thr_path  = os.path.join(cache_dir, "decision_threshold.json")
        self.online    = os.path.join(cache_dir, "online_model.joblib") # SGD partial_fit
        self.vec = Vectorizer()

    # ----- tiny DB helpers -----
    def _rows(self):
        if os.path.exists(self.db):
            try:
                with open(self.db, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self, rows):
        with open(self.db, "w") as f:
            json.dump(rows, f, indent=2)

    def add(self, path: str, label: str, frames: List[Dict[str, float]]):
        v, n = self.vec(frames)
        rows = self._rows()
        rows.append(dict(path=path, label=label, vec=v, names=n, ts=ts()))
        self._save(rows)

    # ----- meta / thresholds -----
    def _write_meta(self, cv_acc: Optional[float], n_rows: int, thr: float, model_name: str):
        try:
            with open(self.meta_path, "w") as f:
                json.dump(
                    dict(model=model_name, n=n_rows, cv=cv_acc, ts=ts(), thr=thr),
                    f, indent=2
                )
        except Exception:
            pass
        try:
            with open(self.thr_path, "w") as f:
                json.dump(dict(threshold=thr, ts=ts()), f, indent=2)
        except Exception:
            pass

    def _load_threshold(self, default: float = 0.5) -> float:
        try:
            if os.path.exists(self.thr_path):
                with open(self.thr_path, "r") as f:
                    return float((json.load(f) or {}).get("threshold", default))
        except Exception:
            pass
        return float(default)

    @staticmethod
    def _optimal_threshold(y: "np.ndarray", p: "np.ndarray") -> float:
        best_t, best_j = 0.5, -9e9
        for t in np.linspace(0.05, 0.95, 181):
            pred = (p >= t).astype(np.int32)
            tp = int(np.sum((pred==1)&(y==1)))
            tn = int(np.sum((pred==0)&(y==0)))
            fp = int(np.sum((pred==1)&(y==0)))
            fn = int(np.sum((pred==0)&(y==1)))
            sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
            spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
            J = sens + spec - 1.0
            if J > best_j:
                best_j, best_t = J, t
        return float(best_t)

    def get_current_model_name(self) -> str:
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r") as f:
                    meta = json.load(f) or {}
                    name = str(meta.get("model", "")).strip()
                    if name:
                        return name
        except Exception:
            pass
        return "GradientBoosting"

    # ----- train/replace -----
    def train(self, model_choice: Optional[str] = None) -> Dict[str, Any]:
        rows = self._rows()
        if not rows or len({r["label"] for r in rows}) < 2:
            raise RuntimeError("Need at least two classes in features_db.json (Dementia vs Normotypical).")

        X = np.asarray([r["vec"] for r in rows], dtype=np.float32)
        y = np.asarray([1 if r["label"].strip().lower().startswith("dem") else 0 for r in rows], dtype=np.int32)

        # Robust pipeline: Standardize → PCA → Stacked (SVC + RF + LR) → Calibrated probabilities
        base_svc = SVC(kernel="rbf", probability=True, class_weight="balanced")
        base_rf  = RandomForestClassifier(
            n_estimators=800, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=0, n_jobs=-1
        )
        base_lr  = LogisticRegression(max_iter=5000, class_weight="balanced")

        stack = StackingClassifier(
            estimators=[("svc", base_svc), ("rf", base_rf), ("lr", base_lr)],
            final_estimator=LogisticRegression(max_iter=5000, class_weight="balanced"),
            passthrough=False, n_jobs=-1
        )

        n_comp = int(min(48, X.shape[1]))
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=n_comp, random_state=0)),
            ("stack", stack),
        ])

        param_dist = {
            "pca__n_components": [min(24, X.shape[1]), min(32, X.shape[1]), n_comp],
            "stack__svc__C": np.logspace(-1, 2, 8),
            "stack__svc__gamma": np.logspace(-3, 0, 8),
            "stack__rf__n_estimators": [600, 800, 1000, 1200],
            "stack__rf__max_depth": [10, 12, 14, 16],
        }
        cv = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=0)
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=20, cv=cv,
            scoring="roc_auc", n_jobs=-1, refit=True, random_state=0, verbose=0
        )
        search.fit(X, y)
        best = search.best_estimator_

        # Probability calibration (sigmoid) on the tuned stack
        calib = CalibratedClassifierCV(base_estimator=best, cv=3, method="sigmoid")
        calib.fit(X, y)

        # Persist
        if _HAS_JOBLIB:
            try:
                joblib.dump(calib, self.m)
            except Exception:
                pass

        # Compute & store optimal threshold on training
        p = calib.predict_proba(X)[:, 1]
        thr = self._optimal_threshold(y, p)

        # Save meta
        self._write_meta(cv_acc=float(search.best_score_), n_rows=len(rows), thr=thr,
                         model_name=(model_choice or "Stacked+PCA+Calibrated"))

        return dict(n=len(rows), cv_auc=float(search.best_score_), thr=float(thr),
                    model=(model_choice or "Stacked+PCA+Calibrated"))

    # ----- inference -----
    def predict(self, frames: List[Dict[str, float]]) -> float:
        v, _ = self.vec(frames)
        if not v: return 0.5
        model = None
        if _HAS_JOBLIB and os.path.exists(self.m):
            try:
                model = joblib.load(self.m)
            except Exception:
                model = None
        if model is None:
            return 0.5
        x = np.asarray([v], dtype=np.float32)
        cls_idx = int(np.where(getattr(model, "classes_", np.array([0,1])) == 1)[0][0]) if hasattr(model, "classes_") else 1
        proba = float(model.predict_proba(x)[0, cls_idx])
        

        # Optional blend with online model if present (learn fast)
        if os.path.exists(self.online) and _HAS_JOBLIB:
            try:
                onl = joblib.load(self.online)
                proba_onl = float(onl["pipe"].predict_proba(x)[0, 1])
                proba = 0.6 * proba + 0.4 * proba_onl
            except Exception:
                pass
        return proba

    # ----- online partial-fit (fast learning from labeled videos) -----
    def online_update(self, frames: List[Dict[str, float]], label: str):
        v, _ = self.vec(frames)
        if not v: return
        X = np.asarray([v], dtype=np.float32)
        y = np.asarray([1 if label.strip().lower().startswith("dem") else 0], dtype=np.int32)

        pipe = None
        if os.path.exists(self.online) and _HAS_JOBLIB:
            try:
                pipe = joblib.load(self.online)
            except Exception:
                pipe = None
        if pipe is None:
            scaler = StandardScaler(with_mean=True, with_std=True)
            clf = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal",
                                penalty="l2", random_state=0)
            X0 = scaler.fit_transform(X)
            clf.partial_fit(X0, y, classes=np.array([0, 1], dtype=np.int32))
            pipe = {"scaler": scaler, "clf": clf}
        else:
            X0 = pipe["scaler"].partial_fit(X).transform(X)
            pipe["clf"].partial_fit(X0, y)

        if _HAS_JOBLIB:
            pipe["pipe"] = Pipeline([("scaler", pipe["scaler"]), ("sgd", pipe["clf"])])
            joblib.dump(pipe, self.online)


# --- Processor ---
@dataclass
class FrameOut:
    frame_bgr: "np.ndarray"
    lms: Optional["np.ndarray"]
    R: Optional["np.ndarray"]
    per_frame: Dict[str,float]
    dbdi: Optional[float]
    z: Optional[Dict[str,float]]

class Processor:
    def __init__(self, cfg: AppConfig):
        if cv2 is None or np is None: raise RuntimeError("OpenCV+NumPy required")
        self.cfg=cfg
        self.det = InsightFaceDetector(MediaPipeDetector()) if (cfg.face_detector=="insightface" and _HAS_INSIGHTFACE) else MediaPipeDetector()
        self.mesh = FaceMesh(True)
        self.base = Baseline(cfg.use_isolation_forest)
        self.fe: Optional[Feat] = None

    def _down(self, img, factor: float):
        if factor<=1.01: return img
        H,W=img.shape[:2]; return cv2.resize(img,(int(W/factor),int(H/factor)),interpolation=cv2.INTER_AREA)

    def stream(self, path: str) -> Iterable[FrameOut]:
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): return
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.cfg.max_frames_debug>0: total=min(total,self.cfg.max_frames_debug)
        self.fe=Feat(fps)
        wlen=max(1,int(self.cfg.window_s*fps)); step=max(1,int(self.cfg.step_s*fps))
        baseline_frames=int(self.cfg.baseline_s*fps)
        buf: Deque[Dict[str,float]] = deque()
        i=0
        while True:
            if total>0 and i>=total: break
            ok,frm=cap.read()
            if not ok: break
            i+=1
            if self.cfg.skip_every_n>0 and (i % (self.cfg.skip_every_n+1))!=1: continue
            proc=self._down(frm,self.cfg.process_downscale)

            lms=self.mesh(proc)
            R=None
            if lms is not None:
                hp=head_pose(proc,lms)
                if hp is not None: R=hp[0]

            pf={}
            if lms is not None: pf=self.fe.per_frame(lms,R)

            dbdi=None; z=None
            if pf:
                buf.append(pf)
                if len(buf)==baseline_frames and self.base.stats.mean=={}:
                    wins=[]
                    arr=list(buf)
                    if len(arr)>=wlen:
                        for s in range(0,len(arr)-wlen+1,step):
                            sl=arr[s:s+wlen]
                            agg={k:float(np.mean([x[k] for x in sl])) for k in sl[0].keys()}
                            agg.update(self.fe.aggs()); wins.append(agg)
                    if wins: self.base.fit(wins)
                if self.base.stats.mean!={} and len(buf)>=wlen and ((len(buf)-wlen)%step==0):
                    sl=list(buf)[-wlen:]
                    agg={k:float(np.mean([x[k] for x in sl])) for k in sl[0].keys()}
                    agg.update(self.fe.aggs()); dbdi,z=self.base.score(agg)

            yield FrameOut(frame_bgr=frm, lms=lms, R=R, per_frame=pf, dbdi=dbdi, z=z)
        cap.release()

# --- Visuals ---
REST_STRIDE = 2  # downsample background cloud → smoother rendering

FACE_PARTS = {
    "eyes":  ([33,133,362,263], (0,200,255)),
    "brows": ([70,63,105,336,296,334], (0,255,0)),
    "mouth": ([61,291,13,14], (255,0,0)),
    "rest":  ([], (180,180,255)),
}
def _maybe_mirror_x_after_pose(X: "np.ndarray", lms: "np.ndarray") -> "np.ndarray":
    try:
        re_id, le_id = MP_IDX["re"], MP_IDX["le"]
        ok_2d = float(lms[le_id,0]) < float(lms[re_id,0])
        ok_3d = float(X[le_id])     < float(X[re_id])
        if ok_2d != ok_3d:
            X = -X
    except Exception:
        pass
    return X


def draw_axes(img, R, scale=80.0):
    H,W=img.shape[:2]; f=1.2*max(W,H)
    K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1]],dtype=np.float32)
    rvec,_=cv2.Rodrigues(R)
    origin=np.float32([[0,0,0]]); axes=np.float32([[scale,0,0],[0,scale,0],[0,0,scale]])
    pts,_=cv2.projectPoints(np.vstack([origin,axes]), rvec, np.zeros((3,1),np.float32), K, np.zeros((5,1),np.float32))
    pts=pts.reshape(-1,2).astype(int); o=tuple(pts[0])
    cv2.line(img,o,tuple(pts[1]),(0,0,255),2); cv2.line(img,o,tuple(pts[2]),(0,255,0),2); cv2.line(img,o,tuple(pts[3]),(255,0,0),2)

def draw_face_overlay(img, lms, R):
    if lms is None: return img
    H, W = img.shape[:2]
    lms_vis = _smooth_landmarks(lms, alpha=0.25)
    if lms_vis is None: return img
    pts = (lms_vis[:, :2] * np.array([W, H], dtype=np.float32)).astype(int)
    out = img.copy()

    def dot(p, r_outer, r_inner, color):
        cv2.circle(out, p, r_outer, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, p, r_inner, color, -1, lineType=cv2.LINE_AA)

    for name, (idxs, color) in FACE_PARTS.items():
        for idx in idxs:
            x, y = pts[int(idx)]
            dot((x, y), 5, 3, color)

    for k in range(0, pts.shape[0], 6):
        x, y = pts[k]
        dot((x, y), 4, 2, FACE_PARTS["rest"][1])

    x0, y0, dy = W - 170, 18, 18
    for i, (name, (_, color)) in enumerate(FACE_PARTS.items()):
        cv2.circle(out, (x0, y0 + i * dy), 6, color, -1)
        cv2.putText(out, name, (x0 + 14, y0 + 5 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def _axis_off():
    return dict(visible=False, showgrid=False, showline=False, zeroline=False,
                showticklabels=False, showspikes=False)

def _maybe_mirror_x_after_pose(X: "np.ndarray", lms: "np.ndarray") -> "np.ndarray":
    """
    Ensure left/right in 3D matches the original video view.
    We compare the x-ordering of left-eye(263) vs right-eye(33) in both spaces.
    If they disagree, mirror X.
    """
    try:
        re_id, le_id = MP_IDX["re"], MP_IDX["le"]   # 33 (viewer right), 263 (viewer left)
        ok_2d = float(lms[le_id,0]) < float(lms[re_id,0])  # left eye should have smaller x in image
        ok_3d = float(X[le_id])   < float(X[re_id])
        if ok_2d != ok_3d:
            X = -X
    except Exception:
        pass
    return X

def plot_face_mesh_3d(lms, R):
    if not _HAS_PLOTLY or lms is None: return None
    L = _smooth_landmarks(lms, alpha=0.25)
    if L is None: return None
    P = L.astype(np.float32) - L.mean(axis=0, keepdims=True)
    span = float(np.linalg.norm(np.ptp(P[:, :2], axis=0)))
    if span > 1e-6:
        P /= span
    if R is not None:
        P = P @ R.T
    X, Y, Z = P[:, 0].copy(), -P[:, 1].copy(), -P[:, 2].copy()
    X = _maybe_mirror_x_after_pose(X, lms)

    idx_rest = np.arange(X.shape[0])[::REST_STRIDE]
    traces = [go.Scatter3d(
        x=X[idx_rest], y=Y[idx_rest], z=Z[idx_rest], mode="markers", name="rest",
        marker=dict(size=3, opacity=0.9, color="rgba(180,180,255,0.85)")
    )]
    palette = {"eyes": "rgb(0,200,255)", "brows": "rgb(0,255,0)", "mouth": "rgb(255,0,0)"}
    groups = {"eyes": [33, 133, 362, 263], "brows": [70, 63, 105, 336, 296, 334], "mouth": [61, 291, 13, 14]}
    for name, idxs in groups.items():
        ii = np.asarray(idxs, dtype=int)
        traces.append(go.Scatter3d(
            x=X[ii], y=Y[ii], z=Z[ii], mode="markers+lines", name=name,
            marker=dict(size=5, color=palette[name]),
            line=dict(color=palette[name], width=3)
        ))

    rng = 0.8
    ax = dict(visible=False, showgrid=False, showline=False, zeroline=False,
              showticklabels=False, showspikes=False)
    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(**ax, range=[-rng, rng]),
            yaxis=dict(**ax, range=[-rng, rng]),
            zaxis=dict(**ax, range=[-rng, rng]),
            bgcolor="#000000",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.3), up=dict(x=0, y=1, z=0)),
            aspectmode="cube"
        ),
        template="graphite",
        paper_bgcolor="#000000",
        uirevision="mesh_live_v1",
        legend=dict(x=1.02, y=1.0),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        height=500
    )
    return fig



# ---------- Live behavior embedding with a FIXED PCA basis (no flashing) ----------
def _pca_fit(X: "np.ndarray") -> Tuple["np.ndarray","np.ndarray"]:
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt, principal axes are Vt
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return X.mean(axis=0, keepdims=True), Vt

def _pca_transform(X: "np.ndarray", mean: "np.ndarray", Vt: "np.ndarray") -> "np.ndarray":
    Xc = X - mean
    Z = (Xc @ Vt.T)[:, :3]
    if Z.shape[1] < 3:
        Z = np.pad(Z, ((0,0),(0,3-Z.shape[1])), mode="constant", constant_values=0.0)
    return Z

def _ensure_live_pca_model(X: "np.ndarray"):
    if st is None: return None
    if "live_pca_mean" not in st.session_state or "live_pca_Vt" not in st.session_state:
        m, Vt = _pca_fit(X)
        st.session_state["live_pca_mean"] = m
        st.session_state["live_pca_Vt"] = Vt

def plot_behavior_live_fixed_pca(wins: List[Dict[str, float]]):
    if not _HAS_PLOTLY or not wins: return None
    keys = sorted(wins[0].keys())
    X = np.asarray([[w[k] for k in keys] for w in wins], dtype=np.float32)
    _ensure_live_pca_model(X)
    m = st.session_state.get("live_pca_mean", X.mean(axis=0, keepdims=True))
    Vt = st.session_state.get("live_pca_Vt", _pca_fit(X)[1])
    U = _pca_transform(X, m, Vt)

    fig = go.Figure([go.Scatter3d(
        x=U[:,0], y=U[:,1], z=U[:,2], mode="markers", name="windows",
        marker=dict(size=3, opacity=0.9)
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="Behavior axis 1 (fixed PCA)",
            yaxis_title="Behavior axis 2",
            zaxis_title="Behavior axis 3",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="#000000",
            aspectmode="cube"
        ),
        template="graphite",
        paper_bgcolor="#000000",
        uirevision="umap_live_fixedpca_v1",
        margin=dict(l=0, r=0, t=0, b=0),
        height=500
    )
    return fig



# ---------- Final (snapshot) behavior embedding with UMAP + optional HDBSCAN ----------
def _pca3(X: "np.ndarray") -> "np.ndarray":
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :3] * S[:3]
    if Z.shape[1] < 3:
        Z = np.pad(Z, ((0,0),(0, 3-Z.shape[1])), mode="constant", constant_values=0.0)
    return Z

def plot_behavior_umap3d(wins: List[Dict[str, float]]):
    if not _HAS_PLOTLY or not wins:
        return None, None
    keys = sorted(wins[0].keys())
    X = np.asarray([[w[k] for k in keys] for w in wins], dtype=np.float32)
    n = X.shape[0]

    # Fast first pass with PCA (guaranteed), then try UMAP for nicer geometry
    def _pca3(X_):
        Xc = X_ - X_.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :3] * S[:3]
        if Z.shape[1] < 3:
            Z = np.pad(Z, ((0,0),(0,3-Z.shape[1])), mode="constant")
        return Z

    U = _pca3(X)
    if _HAS_UMAP and n >= 4:
        try:
            n_components = 3
            n_neighbors  = max(5, min(20, n // 3))
            U = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.10,
                random_state=0
            ).fit_transform(X)
            if U.shape[1] < 3:
                U = np.pad(U, ((0,0),(0,3-U.shape[1])), mode="constant")
        except Exception:
            pass

    # Optional clustering
    labels = np.zeros(n, dtype=int)
    if _HAS_HDBSCAN and n >= 12:
        try:
            labels = hdbscan.HDBSCAN(min_cluster_size=max(8, n//10)).fit_predict(U)
        except Exception:
            pass

    uniq = sorted(set(labels.tolist()))
    palette = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    traces = []
    for i, c in enumerate(uniq):
        m = labels == c
        color = palette[i % len(palette)]
        name  = "noise" if c == -1 else f"cluster {c} (n={int(m.sum())})"
        traces.append(go.Scatter3d(
            x=U[m,0], y=U[m,1], z=U[m,2],
            mode="markers",
            name=name,
            marker=dict(size=3, color=color, opacity=0.9)
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="Behavior axis 1",
            yaxis_title="Behavior axis 2",
            zaxis_title="Behavior axis 3",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="#000000",
            aspectmode="cube"
        ),
        template="graphite",
        paper_bgcolor="#000000",
        margin=dict(l=0, r=0, t=6, b=0),
        height=500,
        title="3D Behavior Map (snapshot)"
    )
    # Provide a simple legend mapping dict in case you want to reuse colors
    legend = {int(c): traces[i].marker.color for i, c in enumerate(uniq)}
    return fig, legend



def plot_decision_surface3d(components: Dict[str, float], sev_score: float,
                            dbdi: Optional[float], p_live: Optional[float] = None,
                            thr: Optional[float] = None):
    """Interactive 3D decision surface: z = P(Dementia) as a function of Severity (x) and DBDI (y)."""
    if not _HAS_PLOTLY:
        return None

    # Normalize inputs
    sev_u = float(np.clip(sev_score, 0.0, 1.0))
    dbdi_u = float(np.clip((dbdi or 0.0) / 100.0, 0.0, 1.0))

    # Blend rule: use your live mix by default (can swap to a sigmoid if you prefer curvature)
    def mix(sev, d):
        return 0.65 * sev + 0.35 * d

    # Surface grid
    n = 60
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys)
    Z = mix(X, Y)

    # Current point
    p_u = float(np.clip(p_live if p_live is not None else mix(sev_u, dbdi_u), 0.0, 1.0))

    data = [
        go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.95, name="P(Dementia)"),
        go.Scatter3d(
            x=[sev_u], y=[dbdi_u], z=[p_u],
            mode="markers+text",
            text=[f"• ({sev_u:.2f}, {dbdi_u:.2f}, {p_u:.2f})"],
            textposition="top center",
            marker=dict(size=7, symbol="diamond", line=dict(width=1), color="#ff7f50"),
            name="current"
        )
    ]

    # Optional threshold plane (flat z = thr)
    if isinstance(thr, (float, int)) and 0.0 <= float(thr) <= 1.0:
        Zthr = np.full_like(Z, float(thr))
        data.append(go.Surface(
            x=X, y=Y, z=Zthr, showscale=False, opacity=0.25, name=f"threshold={thr:.2f}"
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        template="graphite",
        paper_bgcolor="#000000",
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Decision Surface — P(Dementia)=0.65·Severity + 0.35·DBDI  |  now ≈ {p_u:.2f}",
        scene=dict(
            xaxis=dict(
                title="Severity (0–1)", gridcolor="#222222", showspikes=False,
                backgroundcolor="#000000", range=[0,1]
            ),
            yaxis=dict(
                title="DBDI (0–1)", gridcolor="#222222", showspikes=False,
                backgroundcolor="#000000", range=[0,1]
            ),
            zaxis=dict(
                title="P(Dementia)", gridcolor="#222222", showspikes=False,
                backgroundcolor="#000000", range=[0,1]
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        uirevision="decision_surface_live_v1"
    )
    return fig

def _pca_fit(X: "np.ndarray") -> Tuple["np.ndarray","np.ndarray"]:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return X.mean(axis=0, keepdims=True), Vt

def _pca_transform(X: "np.ndarray", mean: "np.ndarray", Vt: "np.ndarray") -> "np.ndarray":
    Xc = X - mean
    Z = (Xc @ Vt.T)[:, :3]
    if Z.shape[1] < 3:
        Z = np.pad(Z, ((0,0),(0,3-Z.shape[1])), mode="constant", constant_values=0.0)
    return Z

def _ensure_live_pca_model(X: "np.ndarray"):
    if st is None: return None
    if "live_pca_mean" not in st.session_state or "live_pca_Vt" not in st.session_state:
        m, Vt = _pca_fit(X)
        st.session_state["live_pca_mean"] = m
        st.session_state["live_pca_Vt"] = Vt


def zero_shot_severity(z: Dict[str,float], dbdi: Optional[float]) -> Tuple[float,str,Dict[str,float]]:
    import numpy as _np, math
    def pick(*keys): 
        for k in keys:
            v = float(z.get(k,0.0))
            if abs(v) > 1e-3: return v
        return float(z.get(keys[0],0.0))
    def norm_unit(v: float) -> float:
        if not math.isfinite(v): v = 0.0
        return float(_np.clip(abs(v)/3.0, 0.02, 1.0))
    blink_z = pick("blink_rate_hz","blink")
    jaw_z   = pick("jaw_var","mouth_open")
    gaze_z  = pick("gaze_var","gaze_disp")
    brow_z  = pick("brow_asym_amp","brow_asym")
    jitter_z= pick("head_jitter")
    ear_var_z = float(z.get("ear_var",0.0))
    if abs(ear_var_z) <= 1e-3: ear_var_z = abs(float(z.get("ear",0.0)))
    eye_stab = float(_np.clip(1.0 - abs(ear_var_z)/3.0, 0.02, 1.0))
    comp = dict(blink=norm_unit(blink_z), eye=eye_stab, jaw=norm_unit(jaw_z),
                gaze=norm_unit(gaze_z), brow=norm_unit(brow_z), jitter=norm_unit(jitter_z))
    w = dict(blink=0.15, eye=0.20, jaw=0.15, gaze=0.15, brow=0.20, jitter=0.15)
    sev = sum(w[k]*comp[k] for k in comp)
    if dbdi is not None: sev = 0.7*sev + 0.3*float(_np.clip(dbdi/100.0,0.0,1.0))
    if   sev<0.2: lab="CDR 0 — None"
    elif sev<0.4: lab="CDR 0.5 — Very Mild"
    elif sev<0.6: lab="CDR 1 — Mild"
    elif sev<0.8: lab="CDR 2 — Moderate"
    else:         lab="CDR 3 — Severe"
    return float(_np.clip(sev,0.0,1.0)), lab, comp

def infer_state(z: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
    import numpy as _np, math
    def pick(*keys):
        for k in keys:
            v = float(z.get(k,0.0))
            if abs(v) > 1e-3: return v
        return float(z.get(keys[0],0.0))
    def norm_unit(v: float) -> float:
        if not math.isfinite(v): v = 0.0
        return float(_np.clip(abs(v)/3.0, 0.0, 1.0))
    blink  = norm_unit(pick("blink_rate_hz","blink"))
    jaw    = norm_unit(pick("jaw_var","mouth_open"))
    gaze   = norm_unit(pick("gaze_var","gaze_disp"))
    brow   = norm_unit(pick("brow_asym_amp","brow_asym"))
    jitter = norm_unit(pick("head_jitter"))
    ear_v  = float(z.get("ear_var",0.0)) if abs(float(z.get("ear_var",0.0)))>1e-3 else float(z.get("ear",0.0))
    eye_stab = float(_np.clip(1.0 - abs(ear_v)/3.0, 0.0, 1.0))
    comp = dict(blink=blink, eye=eye_stab, jaw=jaw, gaze=gaze, brow=brow, jitter=jitter)
    drowsy  = 0.35*blink + 0.30*(1.0-eye_stab) + 0.15*(1.0-min(jitter,1.0)) + 0.10*(1.0-jaw) + 0.10*(1.0-gaze)
    anxious = 0.30*jitter + 0.25*jaw + 0.25*gaze + 0.20*brow
    focused = 0.35*eye_stab + 0.25*(1.0-blink) + 0.20*(1.0-gaze) + 0.20*(1.0-jaw)
    candidates = [("Drowsy / Fatigued", drowsy), ("Anxious / Agitated", anxious), ("Focused / Engaged", focused)]
    candidates.sort(key=lambda t: t[1], reverse=True)
    (label, best), (_, second) = candidates[0], candidates[1]
    return label, float(_np.clip(best-second, 0.0, 1.0)), comp



def plot_reasoning_sankey(components: Dict[str, float], sev_score: float,
                          dbdi: Optional[float], p_live: Optional[float] = None):
    """Live 'reasoning' graph: components → Severity → P(Dementia), with DBDI as a parallel path."""
    if not _HAS_PLOTLY:
        return None

    # Weights used in zero_shot_severity (kept consistent)
    w = dict(blink=0.15, eye=0.20, jaw=0.15, gaze=0.15, brow=0.20, jitter=0.15)

    # Normalize inputs to [0,1]
    def nz(x): 
        try:
            v = float(x)
            if not math.isfinite(v): v = 0.0
        except Exception:
            v = 0.0
        return float(np.clip(v, 0.0, 1.0))

    comp_keys = ["blink", "eye", "jaw", "gaze", "brow", "jitter"]
    comp_vals = [nz(components.get(k, 0.0)) for k in comp_keys]
    comp_contrib = [w[k] * v for k, v in zip(comp_keys, comp_vals)]
    dbdi_u = nz((dbdi or 0.0) / 100.0)
    sev_u  = nz(sev_score)
    p_u    = nz(p_live if p_live is not None else (0.65 * sev_u + 0.35 * dbdi_u))

    # Node order
    nodes = [ "Blinking", "Eyelid stability", "Mouth / jaw", "Gaze variability", "Brow asymmetry", "Head jitter",
              "DBDI", "Severity", "P(Dementia)" ]
    colors = ["#fca311", "#19c37d", "#e76f51", "#9b5de5", "#577590", "#2a9d8f",
              "#264653", "#e9c46a", "#e63946"]

    # Links: components → Severity
    src = []; tgt = []; val = []; link_colors = []
    for i, c in enumerate(comp_contrib):
        src.append(i); tgt.append(7); val.append(max(1e-3, float(c))); link_colors.append("rgba(100,100,255,0.35)")

    # Link: DBDI → Severity
    src.append(6); tgt.append(7); val.append(max(1e-3, 0.5 * dbdi_u)); link_colors.append("rgba(255,120,80,0.45)")

    # Links: Severity → P(Dementia) and DBDI → P(Dementia)
    src.append(7); tgt.append(8); val.append(max(1e-3, 0.75 * sev_u));   link_colors.append("rgba(255,200,0,0.55)")
    src.append(6); tgt.append(8); val.append(max(1e-3, 0.35 * dbdi_u)); link_colors.append("rgba(255,120,80,0.45)")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, color=colors, pad=16, thickness=16),
        link=dict(source=src, target=tgt, value=val, color=link_colors)
    )])

    # Title with live values
    fig.update_layout(
        template="graphite",
        paper_bgcolor="#000000",
        margin=dict(l=20, r=20, t=30, b=20),
        title=f"Reasoning (live) — Severity: {sev_u:.2f} · DBDI: {dbdi_u:.2f} · P(Dementia)~{p_u:.2f}"
    )
    return fig

# ========= Complementary analysers & robust zero-shot DX =========

def _robust_unit_from_series(arr: List[float], invert: bool=False) -> float:
    """
    Map a 1D series to [0,1] in a robust, dataset-free way using median & MAD.
    High value means 'more evidence for risk' unless invert=True (then high=good).
    """
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size < 6:
        return 0.5  # not enough evidence
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)) + 1e-6)
    # signed distance in MADs; use magnitude for risk
    z = float(np.median(np.abs(a - med)) / (mad + 1e-6))
    # clip + saturate
    u = float(np.tanh(z / 3.0))  # 0..~1
    if invert:
        u = 1.0 - u
    return float(np.clip(u, 0.0, 1.0))

def _trend_consistency(arr: List[float]) -> Tuple[float, float]:
    """
    Returns (trend, consistency) ∈ [0,1]^2
    - trend: how strongly the last third is elevated vs first third
    - consistency: fraction of windows above robust threshold
    """
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size < 12:
        return 0.0, 0.0
    n3 = max(4, a.size // 3)
    first, last = a[:n3], a[-n3:]
    med = float(np.median(a)); mad = float(np.median(np.abs(a - med)) + 1e-6)
    thr = med + 1.5 * mad  # robust "high" marker
    consistency = float(np.mean(a > thr))
    # signed lift of last vs first, robustly scaled to ~[0,1]
    lift = float((np.median(last) - np.median(first)) / (mad + 1e-6))
    trend = float(np.clip(np.tanh(max(0.0, lift) / 3.0), 0.0, 1.0))
    return trend, consistency

def _expressivity_from_landmarks(lms_series: List["np.ndarray"]) -> float:
    """
    Frame-to-frame landmark motion magnitude (normalized to [0,1]).
    Lower motion → hypomimia; we return 'deficit' so higher means risk.
    """
    if not lms_series or not isinstance(lms_series[0], np.ndarray):
        return 0.5
    # use a subset: eyes, brows, mouth
    idx = np.array([33,133,362,263, 70,63,105, 336,296,334, 61,291,13,14], dtype=int)
    mags = []
    for i in range(1, len(lms_series)):
        L0 = lms_series[i-1]; L1 = lms_series[i]
        if not (isinstance(L0, np.ndarray) and isinstance(L1, np.ndarray)):
            continue
        if L0.shape != L1.shape:
            continue
        d = np.linalg.norm((L1[idx,:2] - L0[idx,:2]), axis=1)  # in normalized image coords
        mags.append(float(np.mean(d)))
    if len(mags) < 6:
        return 0.5
    # Robust normalize by its own scale
    med = float(np.median(mags))
    mad = float(np.median(np.abs(np.asarray(mags) - med)) + 1e-8)
    expr = float(med / (mad + 1e-6))
    # more motion => healthier → risk is inverse
    healthy = float(np.clip(np.tanh(expr / 6.0), 0.0, 1.0))
    deficit = 1.0 - healthy
    return float(np.clip(deficit, 0.0, 1.0))

class _AudioAnalyzer:
    """Optional tiny audio prosody analysis; safe no-op if librosa can't load."""
    def __init__(self, path: str):
        self.path = path
        self.ok = _HAS_LIBROSA

    def features(self) -> Dict[str, float]:
        if not self.ok:
            return {}
        try:
            # librosa can usually read mp4/mov via audioread; keep it light
            y, sr = librosa.load(self.path, sr=16000, mono=True)
            if y.size < sr * 2:
                return {}
            # Short-time energy & voicing
            hop = 256
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop).flatten()
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=hop).flatten()
            # speech activity (rms above robust threshold)
            med = float(np.median(rms)); mad = float(np.median(np.abs(rms - med)) + 1e-6)
            act = (rms > (med + 1.5 * mad)).mean()
            # prosody "flatness" ~ low variance of energy & pitch proxy
            flat = 1.0 - float(np.clip(np.tanh(np.std(rms) * 8.0), 0.0, 1.0))
            # pauses: long low-energy spans ratio
            pauses = float(np.clip((rms < (med + 0.3 * mad)).mean(), 0.0, 1.0))
            return dict(voice_activity=float(act), prosody_flat=flat, pause_ratio=pauses)
        except Exception:
            return {}

class ZeroShotDX:
    """
    Robust zero-shot dementia likelihood using:
    - self-normalized time-series (z-like windows you already compute),
    - trends/consistency over time,
    - landmark expressivity deficit,
    - optional audio prosody (pause ratio / prosody flatness).
    Produces p ∈ [0,1], severity ∈ [0,1], and an abstain flag when evidence is weak.
    """
    def __init__(self, windows_z: List[Dict[str, float]], dbdi_series: List[float], lms_series: List["np.ndarray"], video_path: Optional[str]):
        self.wz = windows_z or []
        self.dbdi = [float(d) for d in (dbdi_series or [])]
        self.lms_series = lms_series or []
        self.audio = _AudioAnalyzer(video_path) if video_path else None

    def _series(self, key: str) -> List[float]:
        return [float(w.get(key, 0.0)) for w in self.wz if key in w]

    def run(self) -> Dict[str, Any]:
        # Pull robust series (z-like): higher magnitude → larger deviation
        ear_v   = self._series("ear_var")     or self._series("ear")
        jaw_v   = self._series("jaw_var")     or self._series("mouth_open")
        gaze_v  = self._series("gaze_var")    or self._series("gaze_disp")
        brow_a  = self._series("brow_asym_amp") or self._series("brow_asym")
        jitter  = self._series("head_jitter")
        blink_h = self._series("blink_rate_hz")

        # Unitized components in [0,1]
        comp = dict(
            eye   = _robust_unit_from_series(ear_v,  invert=True),  # stability high = good
            jaw   = _robust_unit_from_series(jaw_v),
            gaze  = _robust_unit_from_series(gaze_v),
            brow  = _robust_unit_from_series(brow_a),
            jitter= _robust_unit_from_series(jitter),
            blink = _robust_unit_from_series(blink_h)
        )

        # Expressivity deficit from landmarks
        comp["expressivity_deficit"] = _expressivity_from_landmarks(self.lms_series)

        # Optional audio
        aud = self.audio.features() if self.audio else {}
        if aud:
            # Higher pauses / flatter prosody → riskier
            comp["pause_ratio"]   = float(np.clip(aud.get("pause_ratio", 0.0), 0.0, 1.0))
            comp["prosody_flat"]  = float(np.clip(aud.get("prosody_flat", 0.0), 0.0, 1.0))
        else:
            comp["pause_ratio"]  = 0.5
            comp["prosody_flat"] = 0.5

        # Trend/consistency bonuses
        t_blink, c_blink = _trend_consistency(blink_h)
        t_gaze,  c_gaze  = _trend_consistency(gaze_v)
        t_jit,   c_jit   = _trend_consistency(jitter)

        # Combine (weights informed by clinical priors + your existing emphasis)
        w = dict(
            blink=0.10, eye=0.18, jaw=0.12, gaze=0.14, brow=0.12, jitter=0.12,
            expressivity_deficit=0.11, pause_ratio=0.06, prosody_flat=0.05
        )
        core = sum(w[k] * comp[k] for k in w if k in comp)

        # Add DBDI (scaled) and trend/consistency
        dbdi_u = float(np.clip((np.median(self.dbdi) if self.dbdi else 0.0) / 100.0, 0.0, 1.0))
        enrich = 0.15 * dbdi_u + 0.05 * (t_blink + t_gaze + t_jit)/3.0 + 0.05 * (c_blink + c_gaze + c_jit)/3.0
        severity = float(np.clip(core + enrich, 0.0, 1.0))

        # Calibrated squashing to probability; center ~0.55 with sharper slope
        # (Acts like a prior-informed logistic without needing labels.)
        a, b = 3.2, 0.55
        p = float(1.0 / (1.0 + np.exp(-a * (severity - b))))

        # Simple abstain when weak: low consistency AND close to mid
        margin = abs(severity - b)
        consistency = (c_blink + c_gaze + c_jit) / 3.0
        abstain = bool((margin < 0.07) and (consistency < 0.15))

        return dict(
            p=p, severity=severity, components=comp,
            consistency=float(consistency), abstain=abstain, audio=aud
        )


def risk_score(z: Dict[str, float],
               dbdi: Optional[float],
               blink_rate_hz_override: Optional[float] = None
              ) -> Tuple[float, Dict[str, float]]:
    """
    Returns a unit risk score in [0,1] for Dementia (no severity tiers) and the per-component contributions.
    """
    def pick(*keys) -> float:
        for k in keys:
            v = float(z.get(k, 0.0))
            if abs(v) > 1e-3:
                return v
        return float(z.get(keys[0], 0.0))

    def norm_unit(v: float) -> float:
        if not math.isfinite(v): v = 0.0
        return float(np.clip(abs(v) / 3.0, 0.02, 1.0))

    # Components ∈ [0,1]
    if blink_rate_hz_override is not None:
        blink_u = float(np.clip(blink_rate_hz_override / 0.35, 0.0, 1.0))  # ~0–21 blinks/min
    else:
        blink_u = norm_unit(pick("blink_rate_hz", "blink"))

    jaw_u   = norm_unit(pick("jaw_var", "mouth_open"))
    gaze_u  = norm_unit(pick("gaze_var", "gaze_disp"))
    brow_u  = norm_unit(pick("brow_asym_amp", "brow_asym"))
    jit_u   = norm_unit(pick("head_jitter"))

    ear_v   = float(z.get("ear_var", 0.0))
    if abs(ear_v) <= 1e-3:
        ear_v = abs(float(z.get("ear", 0.0)))
    eye_u = float(np.clip(1.0 - abs(ear_v) / 3.0, 0.02, 1.0))  # stability (higher = steadier lids)

    comp = dict(blink=blink_u, eye=eye_u, jaw=jaw_u, gaze=gaze_u, brow=brow_u, jitter=jit_u)

    # Weights (kept from the old heuristic to maintain behavior)
    w = dict(blink=0.15, eye=0.20, jaw=0.15, gaze=0.15, brow=0.20, jitter=0.15)
    core = sum(w[k] * comp[k] for k in comp)  # 0..1
    if dbdi is not None:
        core = 0.7 * core + 0.3 * float(np.clip(dbdi / 100.0, 0.0, 1.0))

    return float(np.clip(core, 0.0, 1.0)), comp


# ---- Simple affective state inference (fast, heuristic) ----
def infer_state(z: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
    # helpers (mirror zero_shot_severity’s scaling)
    def pick(*keys) -> float:
        for k in keys:
            v = float(z.get(k, 0.0))
            if abs(v) > 1e-3:
                return v
        return float(z.get(keys[0], 0.0))
    def norm_unit(v: float) -> float:
        if not math.isfinite(v):
            v = 0.0
        return float(np.clip(abs(v) / 3.0, 0.0, 1.0))

    # normalized components in [0,1] (higher = “more of the thing”)
    blink  = norm_unit(pick("blink_rate_hz", "blink"))
    jaw    = norm_unit(pick("jaw_var", "mouth_open"))
    gaze   = norm_unit(pick("gaze_var", "gaze_disp"))
    brow   = norm_unit(pick("brow_asym_amp", "brow_asym"))
    jitter = norm_unit(pick("head_jitter"))
    ear_v  = float(z.get("ear_var", 0.0)) if abs(float(z.get("ear_var", 0.0))) > 1e-3 else float(z.get("ear", 0.0))
    eye_stab = float(np.clip(1.0 - abs(ear_v) / 3.0, 0.0, 1.0))  # high = stable eyelids

    comp = dict(blink=blink, eye=eye_stab, jaw=jaw, gaze=gaze, brow=brow, jitter=jitter)

    # state scores (tuned for quick, sensible behavior)
    drowsy  = 0.35*blink + 0.30*(1.0-eye_stab) + 0.15*(1.0 - min(jitter, 1.0)) + 0.10*(1.0-jaw) + 0.10*(1.0-gaze)
    anxious = 0.30*jitter + 0.25*jaw + 0.25*gaze + 0.20*brow
    focused = 0.35*eye_stab + 0.25*(1.0-blink) + 0.20*(1.0-gaze) + 0.20*(1.0-jaw)

    candidates = [("Drowsy / Fatigued", drowsy), ("Anxious / Agitated", anxious), ("Focused / Engaged", focused)]
    candidates.sort(key=lambda t: t[1], reverse=True)
    (label, best), (_, second) = candidates[0], candidates[1]
    confidence = float(np.clip(best - second, 0.0, 1.0))  # gap to runner-up

    return label, confidence, comp

# --- Real-time text readout (compact KPI board with colored chips + sparklines) ---
KPI_META = {
    "blink":  dict(label="Blinking",          invert=False, explain="Higher → more frequent blinking (fatigue/stress)"),
    "eye":    dict(label="Eyelid stability",  invert=True,  explain="Higher → steadier eyelids (generally good)"),
    "jaw":    dict(label="Mouth / jaw motion",invert=False, explain="Higher → more mouth activity (restlessness/tension)"),
    "gaze":   dict(label="Gaze variability",  invert=False, explain="Higher → less stable gaze (distraction/agitation)"),
    "brow":   dict(label="Brow asymmetry",    invert=False, explain="Higher → left/right imbalance (tension/strain)"),
    "jitter": dict(label="Head jitter",       invert=False, explain="Higher → more head micro-movement (instability)"),
}

def _sev_color(v: float, invert: bool=False) -> str:
    import numpy as _np
    s = 1.0 - float(_np.clip(v, 0, 1)) if invert else float(_np.clip(v, 0, 1))
    if s >= 0.66: return "#19c37d"
    if s >= 0.33: return "#f4a261"
    return "#e76f51"

def _chip(text: str, value=None, invert: bool=False) -> str:
    bg = _sev_color(0.0 if value is None else float(value), invert)
    vs = "" if value is None else f'<span style="opacity:.8">({float(value):.2f})</span>'
    return (f'<span style="display:inline-flex;align-items:center;gap:.45rem;'
            f'padding:.18rem .5rem;border-radius:.6rem;background:{bg};color:#0b0f19;'
            f'font-weight:600;font-size:.86rem;line-height:1.2">{text}{vs}</span>')

def _sparkline(name, series, invert=False):
    import numpy as _np
    if not series: return None
    y = _np.asarray(series[-240:], dtype=float); x = _np.arange(len(y))
    if invert:
        bands = [(0.70,1.00,"rgba(25,195,125,0.18)"),(0.40,0.70,"rgba(244,162,97,0.18)"),(0.00,0.40,"rgba(231,111,81,0.18)")]
    else:
        bands = [(0.00,0.30,"rgba(25,195,125,0.18)"),(0.30,0.60,"rgba(244,162,97,0.18)"),(0.60,1.00,"rgba(231,111,81,0.18)")]
    shapes = [dict(type="rect", xref="x", yref="y", x0=-1, x1=len(y)+1, y0=lo, y1=hi, fillcolor=col, line=dict(width=0))
              for (lo,hi,col) in bands]
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(width=1.4), marker=dict(size=4),
                             name=name, showlegend=False))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers",
                             marker=dict(size=8, line=dict(width=1,color="#111")), showlegend=False))
    fig.update_layout(template="graphite", paper_bgcolor="#1d1f27", plot_bgcolor="#1d1f27",
                      margin=dict(l=8,r=8,t=6,b=8), height=90, xaxis=dict(visible=False),
                      yaxis=dict(range=[0,1],zeroline=False,showgrid=False,
                                 tickmode="array",tickvals=[0,.5,1],ticktext=["0","0.5","1"],ticks="outside",ticklen=3),
                      shapes=shapes, uirevision=f"spark_{name}")
    return fig

def render_live_readout(container, components, sev_lbl, dbdi, state, state_conf, sev_score, p_live):
    import time
    import numpy as _np
    import streamlit as st
    import plotly.graph_objects as go

    container = container.empty()
    st.session_state["live_render_nonce"] = st.session_state.get("live_render_nonce", 0) + 1
    nonce = f"{int(time.time()*1000)%10_000_000}_{st.session_state['live_render_nonce']}"

    # update history
    hist = st.session_state.setdefault("live_hist", {k: [] for k in KPI_META})
    for k in KPI_META:
        v = float(components.get(k, 0.0))
        hist[k].append(v)
        if len(hist[k]) > 240:
            del hist[k][0:len(hist[k])-240]

    with container.container():
        chips = []
        chips.append(_chip(f"Overall: {sev_lbl}"))
        if dbdi is not None: chips.append(_chip("DBDI", (dbdi or 0)/100.0))
        chips.append(_chip(f"State: {state}", state_conf, True))
        st.markdown('<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin:.25rem 0 .5rem 0">'
                    + " ".join(chips) + "</div>", unsafe_allow_html=True)

        keys = ["blink","eye","jaw","gaze","brow","jitter"]
        for row in (keys[:3], keys[3:]):
            cols = st.columns(3, vertical_alignment="center")
            for c, k in zip(cols, row):
                meta = KPI_META[k]
                with c:
                    st.markdown(_chip(meta["label"], hist[k][-1] if hist[k] else 0.0, meta["invert"]),
                                unsafe_allow_html=True)
                    fig = _sparkline(k, hist[k], invert=meta["invert"])
                    if fig is not None:
                        fig.update_layout(paper_bgcolor="#1d1f27", plot_bgcolor="#1d1f27")
                        st.plotly_chart(fig, config=PLOTLY_CFG_NO_BAR, key=f"live_plot_{k}_{nonce}", theme=None)

        # decision surface (kept from your old version)
        fig_surface = plot_decision_surface3d(components, sev_score, dbdi, p_live=p_live,
                                              thr=st.session_state.get("live_thr", 0.5))
        if fig_surface is not None:
            st.plotly_chart(fig_surface, config=PLOTLY_CFG_BAR, key=f"decision_surface_{nonce}", theme=None)


# ---------- Performance logging / analytics (no Plotly) ----------
_CDR_MAP = {
    "CDR 0 — None": 0.0,
    "CDR 0.5 — Very Mild": 0.5,
    "CDR 1 — Mild": 1.0,
    "CDR 2 — Moderate": 2.0,
    "CDR 3 — Severe": 3.0,
}
def _cdr_to_num(s: str) -> float:
    return _CDR_MAP.get(s, 0.0)

import numpy as np

def _roc_points(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p, pos_label=1)
    return fpr, tpr, thr

def _auc_trap(x, y): return float(np.trapz(y, x))

def _bootstrap_ci(y, p, stat_fn, n=400, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2: return (np.nan, np.nan, np.nan)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        vals.append(float(stat_fn(y[idx], p[idx])))
    vals = np.sort(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 50)), float(np.nanpercentile(vals, 97.5))

def _auc_roc(y, p):
    f, t, _ = _roc_points(y, p); return _auc_trap(f, t)

def _pr_points(y, p):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, p, pos_label=1)
    return rec, prec  # callers expect r, p

def _ap(r, p): return float(np.trapz(p, r))

def _ece_adaptive(y, p, bins=15):
    y = np.asarray(y); p = np.asarray(p)
    if len(p) < 2: return 0.0, 0.0, []
    q = np.linspace(0,1,bins+1)
    edges = np.quantile(p, q); edges[0]=0.0; edges[-1]=1.0
    idx = np.clip(np.digitize(p, edges)-1, 0, bins-1)
    ece = 0.0; mce = 0.0; info=[]
    for b in range(bins):
        m = (idx==b)
        if not np.any(m):
            info.append((0.0, 0.0, float(edges[b+1]-edges[b]), 0)); continue
        conf = float(p[m].mean()); acc = float((y[m]==1).mean()); w = m.mean()
        ece += w*abs(acc-conf); mce = max(mce, abs(acc-conf))
        info.append((conf, acc, float(edges[b+1]-edges[b]), int(m.sum())))
    return float(ece), float(mce), info

def _kde1d(x, grid, bw=0.08):
    x = np.asarray(x, float); grid = np.asarray(grid, float)
    if x.size == 0: return np.zeros_like(grid)
    K = np.exp(-0.5*((grid[:,None]-x[None,:])/bw)**2)/(np.sqrt(2*np.pi)*bw)
    return K.mean(axis=1)

def _det_points(y, p):
    f, t, _ = _roc_points(y, p); fnr = 1.0 - t
    def _erfinv(z):
        a=0.147; ln=np.log(1 - z*z); first=2/(np.pi*a)+ln/2
        return np.sign(z)*np.sqrt(np.sqrt(first**2 - ln/a) - first)
    def _nd(q):
        q = np.clip(q, 1e-6, 1-1e-6)
        return np.sqrt(2.0)*_erfinv(2*q-1)
    return _nd(f), _nd(fnr)

def _brier_decomp(y, p, bins=15):
    y = np.asarray(y); p = np.asarray(p)
    if p.size == 0: return 0.0, 0.0, 0.0, 0.0
    brier = float(np.mean((p - y)**2))
    ece, mce, info = _ece_adaptive(y, p, bins=bins)
    rel = res = 0.0; pi = float(np.mean(y)); unc = pi*(1-pi)
    for conf, acc, _, cnt in info:
        if cnt==0: continue
        w = cnt/len(p); rel += w*(acc-conf)**2; res += w*(acc-pi)**2
    return brier, rel, res, unc


class PerfStore:
    def __init__(self, cache_dir=None, threshold=0.5):
        self.rows = []          # list of dicts {gt_label, pred_label, pred_prob, ...}
        self.threshold = threshold
        self.history = []       # optional cumulative metrics over time

        # NEW: persistent path for the JSON perf log
        self.path = os.path.join(cache_dir, "perf_log.json") if cache_dir else "perf_log.json"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except Exception:
            pass

        # Optional warm start (don’t crash if file missing/corrupt)
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    self.rows = json.load(f) or []
        except Exception:
            self.rows = []

    @property
    def y_true(self):
            out = []
            for r in self.rows:
                if "y_true" in r:
                    out.append(int(r["y_true"]))
                elif "gt_label" in r:
                    out.append(1 if str(r["gt_label"]).startswith("Dementia") else 0)
            return out

    @property
    def y_score(self):
            out = []
            for r in self.rows:
                if "y_score" in r:
                    out.append(float(r["y_score"]))
                elif "pred_prob" in r:
                    out.append(float(r["pred_prob"]))
            return out

    # RENAMED: this used to be `add(self, y_true, y_score)` and was being shadowed.
    def add_point(self, y_true: int, y_score: float):
        """Append a single (y_true, y_score) pair for lightweight dashboards."""
        self.rows.append({"y_true": int(y_true), "y_score": float(y_score)})
        # keep tiny running history (unchanged)
        import numpy as np
        y = np.array(self.y_true); s = np.array(self.y_score)
        yh = (s >= self.threshold).astype(int)
        acc = (y == yh).mean()
        try:
            from sklearn.metrics import roc_curve, auc
            au = auc(*roc_curve(y, s, pos_label=1)[:2]) if len(np.unique(y)) > 1 else np.nan
        except Exception:
            au = np.nan
        # simple ECE
        ece = 0.0
        if len(s) > 1:
            q = np.linspace(0, 1, 11)
            edges = np.quantile(s, q); edges[0] = 0; edges[-1] = 1
            idx = np.clip(np.digitize(s, edges) - 1, 0, 9)
            for b in range(10):
                m = (idx == b)
                if not np.any(m): 
                    continue
                conf = s[m].mean()
                accb = (y[m] == 1).mean()
                ece += (m.mean()) * abs(accb - conf)
        self.history.append({"n": len(self.rows), "acc": float(acc), "auc": float(au), "ece": float(ece)})

    # ---------- persistent log helpers ----------
    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except Exception:
            pass
        with open(self.path, "w") as f:
            json.dump(self.rows, f, indent=2)

    # logging-style add (the one your app calls later)
    def add(self, **row):
        self.rows.append(dict(ts=ts(), **row))
        self._save()


    def labels(self):
        labs = sorted(set([r["gt_label"] for r in self.rows] + [r["pred_label"] for r in self.rows]))
        return labs

    def confusion(self):
        labs = self.labels()
        if not labs:
            return labs, []
        idx = {l: i for i, l in enumerate(labs)}
        M = [[0] * len(labs) for _ in labs]
        for r in self.rows:
            gi, pi = idx.get(r["gt_label"]), idx.get(r["pred_label"])
            if gi is not None and pi is not None:
                M[gi][pi] += 1
        return labs, M

    def accuracy(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r["gt_label"] == r["pred_label"]) / len(self.rows)

    def brier(self) -> float:
        if not self.rows:
            return 0.0
        s = 0.0
        for r in self.rows:
            y = 1.0 if r["gt_label"].startswith("Dementia") else 0.0
            p = float(r.get("pred_prob", 0.5))
            s += (p - y) ** 2
        return s / len(self.rows)

    def severity_mae(self) -> float:
        vals = [abs(r.get("sev_gt_num", 0.0) - r.get("sev_pred_num", 0.0)) for r in self.rows if "sev_gt_num" in r and "sev_pred_num" in r]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def optimal_threshold(self):
        """Grid-search a probability cut for balanced accuracy; default 0.5."""
        if not self.rows:
            return 0.5, 0.0
        ys = [1 if r["gt_label"].startswith("Dementia") else 0 for r in self.rows]
        ps = [float(r.get("pred_prob", 0.5)) for r in self.rows]
        best_t, best_bacc = 0.5, -1.0
        for t in [i / 100 for i in range(5, 96)]:
            tp = tn = fp = fn = 0
            for y, p in zip(ys, ps):
                pred = 1 if p >= t else 0
                if pred == 1 and y == 1: tp += 1
                elif pred == 0 and y == 0: tn += 1
                elif pred == 1 and y == 0: fp += 1
                else: fn += 1
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            bacc = (sens + spec) / 2.0
            if bacc > best_bacc:
                best_bacc, best_t = bacc, t
        return best_t, best_bacc

    # --- Matplotlib visualizations (quiet; no Streamlit deprecation spam) ---
    def plot_trend(self):
        if not _HAS_MPL or not self.rows:
            return None
        fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=120)
        acc, correct = [], 0
        for i, r in enumerate(self.rows, 1):
            correct += int(r["gt_label"] == r["pred_label"])
            acc.append(correct / i)
        ax.plot(range(1, len(acc) + 1), acc, marker="o", linewidth=1)
        ax.set_ylim(0, 1); ax.set_xlabel("Run"); ax.set_ylabel("Accuracy")
        ax.set_title("Cumulative accuracy"); ax.grid(True, alpha=0.3)
        fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
        fig.tight_layout()
        return fig
    


    
    def _ys_ps(self):
        ys = [1 if r["gt_label"].startswith("Dementia") else 0 for r in self.rows]
        ps = [float(r.get("pred_prob", 0.5)) for r in self.rows]
        return ys, ps
    
    def logloss(self) -> float:
        if not self.rows: return 0.0
        vals = [float(r.get("xent", 0.0)) for r in self.rows if "xent" in r]
        return float(sum(vals) / len(vals)) if vals else 0.0
    
    def plot_roc(self):
        if not _HAS_MPL:
            return None
        ys, ps = self._ys_ps()
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
        ax.plot([0, 1], [0, 1], '--', linewidth=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
        ax.grid(True, alpha=0.3)
        fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
        if len(ps) >= 3 and (1 in ys) and (0 in ys):
            thr = [i/100 for i in range(0,101)]
            P = sum(ys); N = len(ys) - P
            fpr, tpr = [], []
            for t in thr:
                tp = sum(1 for y,p in zip(ys,ps) if p>=t and y==1)
                fp = sum(1 for y,p in zip(ys,ps) if p>=t and y==0)
                fn = P - tp; tn = N - fp
                tpr.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
                fpr.append(fp/(fp+tn) if (fp+tn)>0 else 0.0)
            ax.plot(fpr, tpr, marker='.', linewidth=1)
        else:
            ax.text(0.5, 0.5, "Need ≥3 runs\nwith both classes", ha='center', va='center', fontsize=9, alpha=0.85)
        fig.tight_layout()
        return fig
    
    
    def plot_reliability(self, bins: int = 10):
        if not _HAS_MPL or len(self.rows) < 1:
            fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
            ax.plot([0, 1], [0, 1], '--', linewidth=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Predicted prob"); ax.set_ylabel("Empirical rate")
            ax.set_title("Reliability"); ax.grid(True, alpha=0.3)
            fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
            fig.tight_layout(); return fig
    
        ys, ps = self._ys_ps()
        n = len(ps)
        B = max(3, min(bins, n))
        edges = np.linspace(0.0, 1.0, B + 1)
        xs, obs = [], []
        for i in range(B):
            lo, hi = edges[i], edges[i + 1]
            idx = [k for k, p in enumerate(ps)
                   if (p >= lo and (p < hi or (i == B - 1 and p <= hi)))]
            if not idx: continue
            xs.append(sum(ps[k] for k in idx) / len(idx))
            obs.append(sum(ys[k] for k in idx) / len(idx))
        if len(xs) < 2:
            xs = [float(np.mean(ps))]; obs = [float(np.mean(ys))]
    
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
        ax.plot(xs, obs, marker='o', linewidth=1)
        ax.plot([0, 1], [0, 1], '--', linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted prob"); ax.set_ylabel("Empirical rate")
        ax.set_title("Reliability"); ax.grid(True, alpha=0.3)
        fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
        fig.tight_layout()
        return fig
    
    
    
    def plot_threshold_sweep(self):
        if not _HAS_MPL:
            return None
        ys, ps = self._ys_ps()
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
        ax.set_ylim(0, 1); ax.set_xlabel("Threshold"); ax.set_ylabel("Balanced acc.")
        ax.set_title("Threshold sweep"); ax.grid(True, alpha=0.3)
        fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
        if len(ps) >= 3 and (1 in ys) and (0 in ys):
            thr = [i/100 for i in range(5,96)]
            baccs = []
            for t in thr:
                tp = tn = fp = fn = 0
                for y,p in zip(ys,ps):
                    pred = 1 if p>=t else 0
                    if pred==1 and y==1: tp+=1
                    elif pred==0 and y==0: tn+=1
                    elif pred==1 and y==0: fp+=1
                    else: fn+=1
                sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
                spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
                baccs.append((sens+spec)/2.0)
            ax.plot(thr, baccs, linewidth=1)
            best_t, _ = self.optimal_threshold()
            ax.axvline(best_t, linestyle='--', linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "Need ≥3 runs\nwith both classes", ha='center', va='center', fontsize=9, alpha=0.85)
        fig.tight_layout()
        return fig
    
    
    
    
    def plot_prob_hist(self):
        if not _HAS_MPL:
            return None
        ys, ps = self._ys_ps()
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
        ax.set_xlim(0,1); ax.set_xlabel("Pred prob"); ax.set_ylabel("Count")
        ax.set_title("Score histogram")
        fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
        if len(ps) == 0:
            ax.text(0.5, 0.5, "No runs yet", ha='center', va='center', fontsize=9, alpha=0.85)
        elif (1 in ys) and (0 in ys) and len(ps) >= 3:
            pos = [p for y,p in zip(ys,ps) if y==1]
            neg = [p for y,p in zip(ys,ps) if y==0]
            ax.hist(neg, bins=12, alpha=0.6, label='Normo')
            ax.hist(pos, bins=12, alpha=0.6, label='Dem.')
            ax.legend(fontsize=8)
        else:
            ax.hist(ps, bins=12, alpha=0.7)
            ax.text(0.5, 0.9, "Need ≥3 runs and both classes\nfor split hist", ha='center', va='center',
                    fontsize=8, alpha=0.85, transform=ax.transAxes)
        fig.tight_layout()
        return fig
    
    
    
    def plot_confusion(self):
        if not _HAS_MPL: return None
        labs, M = self.confusion()
        if not labs or not M: return None
        n = len(labs)
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(1, max(max(r) for r in M)))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labs, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labs, fontsize=8)
        vmax = im.get_array().max()
        for i in range(n):
            for j in range(n):
                val = M[i][j]
                # high-contrast text color
                color = "#111" if val < 0.55 * (vmax if vmax else 1) else "#fff"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
        fig.tight_layout()
        return fig
    
    
    
    
    def plot_severity_error(self):
        if not _HAS_MPL: return None
        rows = [r for r in self.rows if "sev_gt_num" in r and "sev_pred_num" in r]
        if not rows: return None
        buckets = defaultdict(list)
        for r in rows:
            buckets[r["sev_gt_num"]].append(abs(r["sev_pred_num"] - r["sev_gt_num"]))
        xs = sorted(buckets.keys())
        ys = [sum(buckets[x])/len(buckets[x]) if buckets[x] else 0.0 for x in xs]
        fig, ax = plt.subplots(figsize=(3.2,2.6), dpi=140)
        ax.bar([str(x) for x in xs], ys)
        ax.set_xlabel("CDR ground-truth"); ax.set_ylabel("MAE")
        ax.set_title("Severity error by CDR")
        fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
        fig.tight_layout(); return fig
        
    
    # =========================
    # Global, in-place dashboard
    # =========================
    class BatchDashboard:
        """One dashboard that gets updated after each video — no duplicates."""
        def __init__(self):
            st.markdown("## Performance & Calibration — Global Dashboard")
            r1 = st.columns(3)
            r2 = st.columns(3)
            r3 = st.columns(3)
            self.ph_conf = r1[0].empty()
            self.ph_roc  = r1[1].empty()
            self.ph_pr   = r1[2].empty()
            self.ph_rel  = r2[0].empty()
            self.ph_thr  = r2[1].empty()
            self.ph_hist = r2[2].empty()
            self.ph_det  = r3[0].empty()
            self.ph_cost = r3[1].empty()
            self.ph_tbl  = r3[2].empty()
            self.rev = 0
    
        def _ys_ps(self, perf: "PerfStore"):
            ys, ps = perf._ys_ps()
            ys = np.asarray(ys, dtype=np.int32)
            ps = np.asarray(ps, dtype=np.float32)
            return ys, ps
    
        def update(self, perf: "PerfStore"):
            self.rev += 1
            ys, ps = self._ys_ps(perf)
    
            # --- Confusion (existing utility) ---
            fig = perf.plot_confusion()
            if fig is not None: self.ph_conf.pyplot(fig, clear_figure=True, use_container_width=True)
    
            # --- ROC (AUC with bootstrap CI + TPR@FPR=0.05) ---
            fig_roc = None
            try:
                f, t, _ = _roc_points(ys, ps)
                auc = _auc_trap(f, t)
                lo, med, hi = _bootstrap_ci(ys, ps, _auc_roc, n=400, seed=42)
                # TPR@FPR=0.05
                want = 0.05
                idx = np.argmin(np.abs(f - want))
                tpr_at = t[idx]
                fig_roc, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
                ax.plot([0,1],[0,1],'--',linewidth=.8)
                ax.plot(f, t, linewidth=1.4)
                ax.scatter([want],[tpr_at], s=18)
                ax.set_title(f"ROC  AUC={auc:.3f}  CI[{lo:.3f},{hi:.3f}] | TPR@FPR=0.05 → {tpr_at:.2f}")
                ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.grid(True, alpha=.3)
                fig_roc.tight_layout()
            except Exception:
                pass
            if fig_roc is not None: self.ph_roc.pyplot(fig_roc, clear_figure=True, use_container_width=True)
    
            # --- PR (AP + bootstrap CI) ---
            fig_pr = None
            try:
                r, p = _pr_points(ys, ps)
                ap = _ap(r, p)
                lo, med, hi = _bootstrap_ci(ys, ps, lambda y_, p_: _ap(*_pr_points(y_, p_)), n=400, seed=43)
                fig_pr, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
                ax.plot(r, p, linewidth=1.4)
                ax.set_xlim(0,1); ax.set_ylim(0,1)
                ax.set_title(f"Precision–Recall  AP={ap:.3f}  CI[{lo:.3f},{hi:.3f}]")
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(True, alpha=.3)
                fig_pr.tight_layout()
            except Exception:
                pass
            if fig_pr is not None: self.ph_pr.pyplot(fig_pr, clear_figure=True, use_container_width=True)
    
            # --- Reliability (ECE/MCE + binomial bands) ---
            fig_rel = None
            try:
                ece, mce, bins_info = _ece_adaptive(ys, ps, bins=15)
                xs = [b[0] for b in bins_info if b[3] > 0]
                acc= [b[1] for b in bins_info if b[3] > 0]
                fig_rel, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
                ax.plot([0,1], [0,1], '--', linewidth=.8)
                ax.scatter(xs, acc, s=26)
                ax.set_xlim(0,1); ax.set_ylim(0,1)
                ax.set_title(f"Reliability  ECE={ece:.3f}  MCE={mce:.3f}")
                ax.set_xlabel("Predicted prob."); ax.set_ylabel("Empirical rate")
                ax.grid(True, alpha=.3); fig_rel.tight_layout()
            except Exception:
                pass
            if fig_rel is not None: self.ph_rel.pyplot(fig_rel, clear_figure=True, use_container_width=True)
    
            # --- Threshold sweep (Balanced Acc, F1, MCC, J) ---
            fig_thr = None
            try:
                if len(ps) >= 3 and (1 in ys) and (0 in ys):
                    th = np.linspace(0.01, 0.99, 199)
                    baccs=[]; f1s=[]; mccs=[]; js=[]
                    P = (ys==1).sum(); N = (ys==0).sum()
                    for t in th:
                        pred = (ps >= t).astype(np.int32)
                        tp = int(np.sum((pred==1)&(ys==1)))
                        tn = int(np.sum((pred==0)&(ys==0)))
                        fp = int(np.sum((pred==1)&(ys==0)))
                        fn = int(np.sum((pred==0)&(ys==1)))
                        sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
                        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
                        baccs.append((sens+spec)/2.0)
                        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                        rec  = sens
                        f1s.append(2*prec*rec/(prec+rec) if prec+rec>0 else 0.0)
                        num = tp*tn - fp*fn
                        den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-9
                        mccs.append(num/den)
                        js.append(sens + spec - 1.0)
                    fig_thr, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
                    ax.plot(th, baccs, label="Balanced Acc")
                    ax.plot(th, f1s,   label="F1")
                    ax.plot(th, mccs,  label="MCC")
                    ax.plot(th, js,    label="Youden's J")
                    ax.set_ylim(0,1); ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
                    ax.set_title("Threshold sweep (multi-metric)"); ax.grid(True, alpha=.3); ax.legend(fontsize=8)
                    fig_thr.tight_layout()
            except Exception:
                pass
            if fig_thr is not None: self.ph_thr.pyplot(fig_thr, clear_figure=True, use_container_width=True)
    
            # --- Score distributions: class KDE + decision line ---
            fig_hist = None
            try:
                if len(ps) >= 1:
                    xs = np.linspace(0,1,400)
                    pos = ps[ys==1]; neg = ps[ys==0]
                    kde_pos = _kde1d(pos, xs) if len(pos)>0 else np.zeros_like(xs)
                    kde_neg = _kde1d(neg, xs) if len(neg)>0 else np.zeros_like(xs)
                    fig_hist, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
                    ax.plot(xs, kde_neg, label="Normotypical")
                    ax.plot(xs, kde_pos, label="Dementia")
                    thr, _ = perf.optimal_threshold()
                    ax.axvline(thr, linestyle="--", linewidth=.8)
                    ax.set_title("Score KDEs + optimal threshold")
                    ax.set_xlabel("Pred prob."); ax.set_ylabel("Density")
                    ax.grid(True, alpha=.3); ax.legend(fontsize=8); fig_hist.tight_layout()
            except Exception:
                pass
            if fig_hist is not None: self.ph_hist.pyplot(fig_hist, clear_figure=True, use_container_width=True)
    
            # --- DET curve ---
            fig_det = None
            try:
                x, y = _det_points(ys, ps)
                fig_det, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
                ax.plot(x, y, linewidth=1.4)
                ax.set_xlabel("FPR (probit)"); ax.set_ylabel("FNMR (probit)")
                ax.set_title("DET"); ax.grid(True, alpha=.3); fig_det.tight_layout()
            except Exception:
                pass
            if fig_det is not None: self.ph_det.pyplot(fig_det, clear_figure=True, use_container_width=True)
    
            # --- Risk/cost curve (tunable prior & costs) ---
            fig_cost = None
            try:
                prior = 0.25  # π: prior P(Dementia) — adjust if desired
                C_fn, C_fp = 4.0, 1.0
                f, t, th = _roc_points(ys, ps)
                cost = prior*(1.0 - t)*C_fn + (1.0 - prior)*f*C_fp
                i_best = int(np.argmin(cost))
                fig_cost, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
                ax.plot(th, cost, linewidth=1.2)
                ax.scatter([th[i_best]], [cost[i_best]], s=22)
                ax.set_xlim(0,1)
                ax.set_title(f"Risk curve (π={prior:.2f}, Cfn={C_fn:.0f}, Cfp={C_fp:.0f})  best t={th[i_best]:.2f}")
                ax.set_xlabel("Threshold"); ax.set_ylabel("Expected cost")
                ax.grid(True, alpha=.3); fig_cost.tight_layout()
            except Exception:
                pass
            if fig_cost is not None: self.ph_cost.pyplot(fig_cost, clear_figure=True, use_container_width=True)
    
            # --- Summary table ---
            try:
                acc = perf.accuracy()
                brier, rel, res, unc = _brier_decomp(ys, ps, bins=15)
                auc = _auc_roc(ys, ps)
                ap  = _ap(*_pr_points(ys, ps))
                ece, mce, _ = _ece_adaptive(ys, ps, bins=15)
                thr, bacc = perf.optimal_threshold()
                df = pd.DataFrame([dict(
                    n=len(ps),
                    accuracy=acc,
                    auc=auc,
                    ap=ap,
                    brier=brier,
                    ece=ece,
                    mce=mce,
                    refinement=res,
                    uncertainty=unc,
                    best_threshold=thr,
                    best_bal_acc=bacc
                )]).T
                df.columns = ["value"]
                self.ph_tbl.dataframe(df, use_container_width=True, height=360)
            except Exception:
                self.ph_tbl.empty()
    
    
   


# --- Real-time text readout (compact KPI board with colored chips + sparklines) ---

# Meaning + polarity for each component
KPI_META = {
    "blink":  dict(label="Blinking",          invert=False, explain="Higher → more frequent blinking (fatigue/stress)"),
    "eye":    dict(label="Eyelid stability",  invert=True,  explain="Higher → steadier eyelids (generally good)"),
    "jaw":    dict(label="Mouth / jaw motion",invert=False, explain="Higher → more mouth activity (restlessness/tension)"),
    "gaze":   dict(label="Gaze variability",  invert=False, explain="Higher → less stable gaze (distraction/agitation)"),
    "brow":   dict(label="Brow asymmetry",    invert=False, explain="Higher → left/right imbalance (tension/strain)"),
    "jitter": dict(label="Head jitter",       invert=False, explain="Higher → more head micro-movement (instability)"),
}

def _sev_color(v: float, invert: bool=False) -> str:
    """Return a chip color (green→amber→red). For invert=True, high is good."""
    s = 1.0 - float(np.clip(v, 0, 1)) if invert else float(np.clip(v, 0, 1))
    # s≈1 => green (good), s≈0 => red (bad)
    if s >= 0.66: return "#19c37d"   # green
    if s >= 0.33: return "#f4a261"   # amber
    return "#e76f51"                 # red

def _chip(text: str, value: Optional[float] = None, invert: bool = False) -> str:
    bg = _sev_color(0.0 if value is None else value, invert)
    # Build the optional value snippet separately to avoid nested f-strings with backslashes
    value_snippet = ""
    if value is not None:
        value_snippet = f'<span style="opacity:.8">({value:.2f})</span>'
    return (
        f'<span style="display:inline-flex;align-items:center;gap:.45rem;'
        f'padding:.18rem .5rem;border-radius:.6rem;background:{bg};color:#0b0f19;'
        f'font-weight:600;font-size:.86rem;line-height:1.2">{text}{value_snippet}</span>'
    )


def _sparkline(name: str, series: List[float], invert: bool=False):
    """Small history chart (last ~240 pts), shaded good/ok/bad bands."""
    if not series: return None
    y = np.asarray(list(series), dtype=float) 
    x = np.arange(len(y))
    if invert:
        bands = [(0.70, 1.00, "rgba(25,195,125,0.18)"),
                 (0.40, 0.70, "rgba(244,162,97,0.18)"),
                 (0.00, 0.40, "rgba(231,111,81,0.18)")]
    else:
        bands = [(0.00, 0.30, "rgba(25,195,125,0.18)"),
                 (0.30, 0.60, "rgba(244,162,97,0.18)"),
                 (0.60, 1.00, "rgba(231,111,81,0.18)")]
    shapes = [dict(type="rect", xref="x", yref="y",
                   x0=-1, x1=len(y)+1, y0=lo, y1=hi, fillcolor=col, line=dict(width=0))
              for (lo, hi, col) in bands]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                             line=dict(width=1.4),
                             marker=dict(size=4),
                             name=name, showlegend=False))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers",
                             marker=dict(size=8, line=dict(width=1,color="#111")),
                             showlegend=False))
    fig.update_layout(
        template="graphite",
        paper_bgcolor="#1d1f27",
        plot_bgcolor="#1d1f27",
        margin=dict(l=8, r=8, t=6, b=8),
        height=90,
        xaxis=dict(visible=False),
        yaxis=dict(range=[0, 1], zeroline=False, showgrid=False, tickmode="array",
                   tickvals=[0, .5, 1], ticktext=["0", "0.5", "1"], ticks="outside", ticklen=3),
        shapes=shapes,
        uirevision=f"spark_{name}",
    )
    return fig


def plot_kpi_heatmap_live(hist: Dict[str, List[float]]):
    if not _HAS_PLOTLY or not hist:
        return None
    keys = ["blink", "eye", "jaw", "gaze", "brow", "jitter"]
    labels = [KPI_META[k]["label"] for k in keys]
    N = max((len(hist[k]) for k in keys), default=0)
    N = min(HIST_MAX, N)
    
    if N < 2:
        return None

    M = np.zeros((len(keys), N), dtype=float)
    for i, k in enumerate(keys):
        s = np.asarray(list(hist[k])[-N:], dtype=float)
        if s.size < N:
            s = np.pad(s, (N - s.size, 0), constant_values=np.nan)
        M[i, :] = np.clip(s, 0.0, 1.0)

    fig = go.Figure(data=go.Heatmap(
        z=M, x=list(range(N)), y=labels,
        zmin=0.0, zmax=1.0, colorscale="Viridis",
        colorbar=dict(title="Intensity", thickness=12)
    ))
    fig.update_layout(
        template="graphite",
        paper_bgcolor="#000000",
        margin=dict(l=10, r=10, t=30, b=10),
        title="Spatiotemporal Stability Map (live)",
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(size=10))
    )
    return fig


def render_live_readout(container, components: Dict[str,float], sev_lbl: str,
                        dbdi: Optional[float], state: str, state_conf: float,
                        sev_score: float, p_live: Optional[float]):

    """
    Rich, compact live panel:
      - Top chips (Severity, DBDI, State)
      - 6 tiny sparklines updated in real-time
      - One-line meaning hints below each chart
    """
    # Clear previous child so we don’t accumulate duplicate-key elements
    container = container.empty()

    # Nonce so keys are guaranteed unique per rerun (prevents DuplicateElementKey)
    nonce = int(time.time() * 1000) % 10_000_000
    st.session_state["live_render_nonce"] = st.session_state.get("live_render_nonce", 0) + 1
    nonce = f"{nonce}_{st.session_state['live_render_nonce']}"

    # --- Update histories in session_state ---
    hist = st.session_state.setdefault("live_hist", {k: [] for k in KPI_META})
    for k in KPI_META:
        v = float(components.get(k, 0.0))
        hist[k].append(v)
        if len(hist[k]) > 240:
            del hist[k][0:len(hist[k])-240]

    # --- Render ---
    with container.container():
        # Header chips
        chips = []
        chips.append(_chip(f"Overall: {sev_lbl}", None, False))
        if dbdi is not None: chips.append(_chip("DBDI", dbdi/100.0, False))
        chips.append(_chip(f"State: {state}", state_conf, True))
        st.markdown(
            '<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin:.25rem 0 0.5rem 0">'
            + " ".join(chips) + "</div>", unsafe_allow_html=True
        )

        # KPI grid (2 rows × 3 cols)
        keys = ["blink", "eye", "jaw", "gaze", "brow", "jitter"]
        for row in (keys[:3], keys[3:]):
            cols = st.columns(3, vertical_alignment="center")
            for c, k in zip(cols, row):
                meta = KPI_META[k]
                label = meta["label"]
                invert = bool(meta["invert"])
                series = hist[k]
                with c:
                    # mini title chip with current value + color
                    st.markdown(_chip(label, series[-1] if series else 0.0, invert), unsafe_allow_html=True)
                    fig = _sparkline(k, series, invert=invert)
                    if fig is not None:
                        fig.update_layout(paper_bgcolor="#1d1f27", plot_bgcolor="#1d1f27")
                        
                        st.plotly_chart(
                            fig,
                            config=PLOTLY_CFG_NO_BAR,
                            key=f"live_plot_{k}_{nonce}",
                            theme=None,
                        )
                        
                        
                    st.caption(meta["explain"])
                    
        # --- Spatiotemporal KPI Heatmap (CVPR-leaning) ---
        fig_heat = plot_kpi_heatmap_live(hist)
        if fig_heat is not None:
            st.plotly_chart(
                fig_heat,
                config=PLOTLY_CFG_BAR,
                key=f"kpi_heat_{nonce}",
                theme=None,
            )
        


# --- Stable Streamlit keys for each chart (no duplicate collisions) ---
MESH_KEY_LIVE  = "mesh_chart_live"
UMAP_KEY_LIVE  = "umap_chart_live"
RADAR_KEY_LIVE = "radar_chart_live"


def _normalize_gt(x) -> str:
    """Return canonical 'Dementia' or 'Normotypical' from messy labels."""
    if x is None:
        return ""
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")

    # canonical sets
    DEM = {
        "dementia", "dem", "demented", "alz", "alzheimer", "alzheimers",
        "ad", "mci->dementia", "amnestic dementia"
    }
    NOR = {
        "normotypical", "neurotypical", "control", "normal", "healthy",
        "non dementia", "non-dementia", "no dementia"
    }

    if s in DEM or "dementia" in s or "alzheimer" in s or s.startswith("dem"):
        return "Dementia"
    if s in NOR or "normo" in s or "control" in s or "healthy" in s or "neurotypical" in s:
        return "Normotypical"

    # Last-resort: assume anything starting with 'non ' means non-dementia
    if s.startswith("non "):
        return "Normotypical"

    # If truly unknown, keep title-cased so you can spot it in UI/logs
    logging.warning(f"Unrecognized label '{x}', leaving as-is.")
    return s.title()

# ---- Reproducible, (optionally) stratified shuffle for eval order ----
from typing import Sequence, Union

def _infer_label_from_str(x: str) -> Optional[int]:
    s = x.lower()
    if "dem" in s: return 1           # Dementia
    if "normo" in s or "control" in s or "healthy" in s: return 0
    return None

def randomize_items(
    items: Sequence[Union[str, Tuple[str,str], Dict[str, Any]]],
    seed: int = 42,
    stratify: bool = True
) -> List:
    """
    Accepts a list of paths OR (path,label) tuples OR dicts with 'path'/'label'.
    Returns a new list in randomized order. If `stratify=True`, alternates classes when labels are known.
    Label convention: 1=Dementia, 0=Normotypical.
    """
    rng = np.random.default_rng(seed)

    # Normalize to (path, label_or_None, original_obj)
    norm = []
    for obj in items:
        if isinstance(obj, str):
            y = _infer_label_from_str(obj)
            norm.append((obj, y, obj))
        elif isinstance(obj, tuple) and len(obj) >= 2:
            p, lab = obj[0], obj[1]
            y = (1 if str(lab).lower().startswith("dem") else 0) if isinstance(lab, str) else int(lab)
            norm.append((p, y, obj))
        elif isinstance(obj, dict):
            # also support CSV rows shaped like {'url':..., 'gt_label':...}
            p = obj.get("path", obj.get("file", obj.get("p", obj.get("url", ""))))
            lab = obj.get("label", obj.get("y", obj.get("gt_label", None)))
        
            y = None
            if lab is not None:
                y = (1 if str(lab).lower().startswith("dem") else 0) if isinstance(lab, str) else int(lab)
            else:
                y = _infer_label_from_str(str(p))
            norm.append((p, y, obj))
        else:
            # unknown; keep but without label
            norm.append((str(obj), None, obj))

    if not stratify:
        out = norm[:]
        rng.shuffle(out)
        return [o for *_ignore, o in out]

    # Stratified interleave when we have labels
    pos = [o for o in norm if o[1] == 1]
    neg = [o for o in norm if o[1] == 0]
    unl = [o for o in norm if o[1] is None]

    if len(pos) == 0 or len(neg) == 0:
        # Not enough labels → plain shuffle of everything
        out = norm[:]
        rng.shuffle(out)
        return [o for *_ignore, o in out]

    rng.shuffle(pos); rng.shuffle(neg); rng.shuffle(unl)

    # Interleave POS/NEG roughly 1:1, then sprinkle unlabeled
    i = j = 0
    out = []
    toggle = True  # start with Dementia
    while i < len(pos) or j < len(neg):
        if toggle and i < len(pos):
            out.append(pos[i]); i += 1
        elif (not toggle) and j < len(neg):
            out.append(neg[j]); j += 1
        # Flip; if one side exhausted, keep taking from the other
        toggle = not toggle
        if i >= len(pos): toggle = False
        if j >= len(neg): toggle = True

    # Insert unlabeled evenly
    if unl:
        step = max(1, len(out) // (len(unl) + 1))
        k = 0
        for u in unl:
            idx = min(len(out), k * step)
            out.insert(idx, u)
            k += 1

    return [o for *_ignore, o in out]


def load_csv_rows(file_like) -> List[Dict[str, str]]:
    """
    Accept CSV/XLSX. Columns supported (any case):
      URL: url, video, link, youtube_url, youtube, yt_url, href, source, src, path,
           file, video_path, local_path, OR video_id/id (builds a YT URL).
      Label: gt, label, ground_truth, diagnosis, class, category, group, condition,
             status, cohort, type, OR a boolean/0-1 column like dementia/is_dementia.
      Optional severity: severity, cdr, cdr_global, 'cdr global', 'cdr score', cdrscore.
    Returns rows like: {'url': ..., 'gt_label': ..., 'severity': ...?}
    """
    rows = []
    try:
        name = str(getattr(file_like, "name", "") or "")
        ext = os.path.splitext(name)[1].lower()

        if _HAS_PANDAS:
            # Read the table
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(file_like)
            else:
                df = pd.read_csv(file_like)

            # Normalize headers once
            df.columns = [str(c).strip().lower() for c in df.columns]

            url_cols = ["url","video","link","youtube_url","youtube","yt_url","href",
                        "source","src","path","file","video_path","local_path"]
            id_cols  = ["id","video_id","youtube_id","ytid","yt_id"]
            lab_cols = ["gt","label","ground_truth","diagnosis","class","category",
                        "group","condition","status","cohort","type"]
            dem_cols = ["dementia","is_dementia","dem","alz","alzheimer","alzheimers"]
            sev_cols = ["severity","cdr","cdr_global","cdr global","cdr score","cdrscore"]

            def _get_str(val: Any) -> str:
                # robust str() that treats NaN/None as ""
                if val is None: return ""
                s = str(val).strip()
                if s.lower() in {"nan", "none"}: return ""
                return s

            for _, r in df.iterrows():
                # ----- URL -----
                url = ""
                for c in url_cols:
                    if c in df.columns:
                        s = _get_str(r.get(c, ""))
                        if s:
                            url = s
                            break
                if not url:
                    vid = ""
                    for c in id_cols:
                        if c in df.columns:
                            s = _get_str(r.get(c, ""))
                            if s:
                                vid = s
                                break
                    if vid:
                        url = f"https://www.youtube.com/watch?v={vid}"

                # ----- Label -----
                label_val = None
                for c in lab_cols:
                    if c in df.columns:
                        s = _get_str(r.get(c, ""))
                        if s:
                            label_val = s
                            break

                if label_val is None:
                    for c in dem_cols:
                        if c in df.columns:
                            s = _get_str(r.get(c, ""))
                            if not s:
                                continue
                            low = s.lower()
                            if low in {"1","true","yes","y","t"}:
                                label_val = "Dementia"
                            elif low in {"0","false","no","n","f"}:
                                label_val = "Normotypical"
                            else:
                                # try numeric, else normalize text
                                try:
                                    label_val = "Dementia" if float(low) > 0 else "Normotypical"
                                except Exception:
                                    label_val = _normalize_gt(low)
                            break

                gt_label = _normalize_gt(label_val) if label_val else "Normotypical"

                # ----- Severity (optional, carried through if present) -----
                sev = None
                for c in sev_cols:
                    if c in df.columns:
                        sev = r.get(c)
                        break

                if url:
                    out = dict(url=url, gt_label=gt_label)
                    if sev is not None:
                        out["severity"] = _get_str(sev)
                    rows.append(out)

        else:
            # CSV-only path, now with more header synonyms
            file_like.seek(0)
            reader = _csv.DictReader(
                (line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else line for line in file_like)
            )
            for r in reader:
                # URL
                url = ""
                for k in ["url","video","link","youtube_url","youtube","yt_url","href","source","src","path","file",
                          "video_path","local_path"]:
                    if k in r and r[k].strip():
                        url = r[k].strip(); break
                if not url:
                    for k in ["id","video_id","youtube_id","ytid","yt_id"]:
                        if k in r and r[k].strip():
                            url = f"https://www.youtube.com/watch?v={r[k].strip()}"; break

                # Label
                gt = ""
                for k in ["gt","label","ground_truth","diagnosis","class","category","group","condition","status","cohort","type"]:
                    if k in r and r[k].strip():
                        gt = r[k].strip(); break
                if not gt:
                    for k in ["dementia","is_dementia","dem","alz","alzheimer","alzheimers"]:
                        if k in r and r[k].strip():
                            v = r[k].strip().lower()
                            gt = "Dementia" if v in {"1","true","yes","y","t"} else "Normotypical"
                            break
                gt_label = _normalize_gt(gt) if gt else "Normotypical"

                if url:
                    rows.append(dict(url=url, gt_label=gt_label))

    except Exception as e:
        LOG.warning(f"CSV/XLSX parse failed: {e}")

    return rows


# ---------- Deep landmark autoencoder (ad-hoc, fast) ----------
class LandmarkAE(nn.Module):
    def __init__(self, d_in: int, d_lat: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, d_lat)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, d_in)
        )
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

def _prep_landmark_matrix(lms_series: List["np.ndarray"]) -> Optional["np.ndarray"]:
    """T×(468×3) matrix from list of landmarks; normalized per series."""
    frames = [L for L in lms_series if isinstance(L, np.ndarray)]
    if not frames:
        return None
    T = len(frames)
    P = frames[0].shape[0]
    X = np.stack([f.reshape(-1) for f in frames], axis=0).astype(np.float32)  # (T, P*3)
    # Center & scale for stability
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    return Xn

def train_landmark_ae(lms_series: List["np.ndarray"], epochs: int = 6, latent_dim: int = 6):
    Xn = _prep_landmark_matrix(lms_series)
    if Xn is None or not _HAS_TORCH:
        return None, None, None, "PyTorch not installed or no landmarks collected."

    device = "cpu"
    x = torch.from_numpy(Xn).to(device)
    model = LandmarkAE(d_in=x.shape[1], d_lat=latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_hist = []
    model.train()
    for e in range(epochs):
        opt.zero_grad()
        xhat, _ = model(x)
        loss = ((xhat - x) ** 2).mean()
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.detach().cpu().item()))
    model.eval()
    with torch.no_grad():
        xhat, z = model(x)
        # Per-landmark reconstruction error (T, P, 3) -> mean over time&coords → (P,)
        err = (x - xhat).pow(2).cpu().numpy()  # (T, D)
    D = lms_series[0].shape[0] * 3
    T = err.shape[0]
    err = err.reshape(T, -1, 3).mean(axis=(0, 2))  # (P,)
    err = (err - err.min()) / (err.max() - err.min() + 1e-8)
    Z = z.cpu().numpy()  # (T, latent_dim)
    return Z, err, loss_hist, None

def plot_latent3d(Z: "np.ndarray"):
    if not _HAS_PLOTLY or Z is None or Z.shape[1] < 3:
        return None
    fig = go.Figure([go.Scatter3d(
        x=Z[:,0], y=Z[:,1], z=Z[:,2], mode="markers",
        marker=dict(size=3, opacity=0.9)
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="z1", yaxis_title="z2", zaxis_title="z3",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False),
            bgcolor="#000000", aspectmode="cube"
        ),
        template="graphite",
        paper_bgcolor="#000000",
        margin=dict(l=0, r=0, t=6, b=0),
        height=500,
        title="Landmark AE — Latent space"
    )
    return fig

def overlay_error_heatmap(frame_bgr: "np.ndarray", lms: "np.ndarray", err: "np.ndarray"):
    """Draw per-landmark error as colored dots on the frame."""
    if frame_bgr is None or lms is None or err is None:
        return frame_bgr
    img = frame_bgr.copy()
    H, W = img.shape[:2]
    pts = (lms[:, :2] * np.array([W, H], dtype=np.float32)).astype(int)
    # radius 2..7, color from green->red
    for i, (x, y) in enumerate(pts):
        e = float(np.clip(err[i], 0, 1))
        r = int(2 + 5 * e)
        color = (int(255 * e), int(255 * (1 - e)), 0)  # (R,G,0)
        cv2.circle(img, (int(x), int(y)), r+1, (0,0,0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), r, color, -1, lineType=cv2.LINE_AA)
    return img

def plot_loss_curve(loss_hist: List[float]):
    if not _HAS_MPL or not loss_hist:
        return None
    fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=140)
    ax.plot(range(1, len(loss_hist)+1), loss_hist, marker="o", linewidth=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("AE reconstruction loss")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig


# --- App ---
class App:
    def __init__(self, cfg: AppConfig):
        if not _HAS_STREAMLIT: raise RuntimeError("streamlit not installed")
        if cv2 is None or np is None: raise RuntimeError("OpenCV+NumPy required")
        self.cfg=cfg; ensure_dir(self.cfg.cache_dir); self.rc=ResearchClassifier(self.cfg.cache_dir); self.perf=PerfStore(self.cfg.cache_dir)
        

    def _resolve(self, s: str)->Optional[str]:
        if os.path.exists(s): return s
        if s.startswith("http"):
            exe=shutil.which("yt-dlp") or shutil.which("youtube-dl")
            if not exe: st.error("yt-dlp not found"); return None
            out_dir=os.path.join(self.cfg.cache_dir,"videos"); ensure_dir(out_dir)
            tmpl=os.path.join(out_dir,"%(id)s.%(ext)s")
            rc=os.system(f"{exe} -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' -o '{tmpl}' '{s}'")
            if rc!=0: st.error("yt-dlp failed"); return None
            mp4s=sorted(glob.glob(os.path.join(out_dir,"*.mp4")), key=os.path.getmtime); return mp4s[-1] if mp4s else None
        st.error("Path not found"); return None
    
    def _reset_cache(self, wipe_videos: bool):
        # delete files
        try:
            if wipe_videos:
                shutil.rmtree(self.cfg.cache_dir, ignore_errors=True)
                ensure_dir(self.cfg.cache_dir)
            else:
                for fn in ["features_db.json", "perf_log.json", "research_model.joblib"]:
                    p = os.path.join(self.cfg.cache_dir, fn)
                    if os.path.exists(p): os.remove(p)
                # clear per-run folders but keep the tree
                for sub in ["uploads", "videos"]:
                    d = os.path.join(self.cfg.cache_dir, sub)
                    shutil.rmtree(d, ignore_errors=True)
                    ensure_dir(d)
        except Exception as e:
            st.warning(f"Reset encountered: {e}")
    
        # clear Streamlit session state noise
        for k in list(st.session_state.keys()):
            if k.startswith(("live_", "pose_", "lms_smooth", "mesh_", "umap_", "ae_")) or \
               k in {"live_pca_mean","live_pca_Vt","label_choice","severity_choice"}:
                try: del st.session_state[k]
                except Exception: pass
    
        st.rerun()
    
    def _maybe_delete_downloaded(self, src: str):
        """Delete video if it lives under cache/videos (downloaded via yt-dlp)."""
        try:
            vids_dir = os.path.realpath(os.path.join(self.cfg.cache_dir, "videos"))
            p = os.path.realpath(src or "")
            if os.path.isfile(p) and p.startswith(vids_dir + os.sep):
                os.remove(p)
                LOG.info(f"Deleted downloaded video to free space: {p}")
                if st is not None:
                    st.caption(f"🧹 Deleted downloaded video to free space: {os.path.basename(p)}")
        except Exception as e:
            LOG.warning(f"Could not delete downloaded video '{src}': {e}")
    
    
    def _process_single(self, src: str, gt_label: str, auto_train_model_name: Optional[str] = None, dash: Optional["BatchDashboard"]=None):
    
    
        vcol, rcol = st.columns([3,2])
        with vcol:
            st.markdown(f"**Ground truth:** {gt_label}")
            vph = st.empty(); prog = st.progress(0.0)
        with rcol:
            live_readout_ph = st.empty()
            final_readout_ph = st.empty()
    
        proc = Processor(self.cfg)
        perframe: List[Dict[str,float]] = []
        landmarks_series: List[np.ndarray] = []
        last_frame_for_heatmap = None
        windows_for_final: List[Dict[str,float]] = []
        frame_count=0; last_prev=-999
    
        for out in proc.stream(src):
            frame_count += 1
            if out.lms is not None:
                landmarks_series.append(out.lms.copy())
            
            # keep a reasonably fresh frame for heatmap preview
            last_frame_for_heatmap = out.frame_bgr
            if out.per_frame: perframe.append(out.per_frame)
            if frame_count - last_prev >= self.cfg.update_preview_every_n:
                frm = out.frame_bgr
                if self.cfg.preview_downscale > 1.01:
                    h, w = frm.shape[:2]
                    frm = cv2.resize(frm, (int(w/self.cfg.preview_downscale), int(h/self.cfg.preview_downscale)), interpolation=cv2.INTER_AREA)
                frm = draw_face_overlay(frm, out.lms, out.R)
                vph.image(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                last_prev = frame_count
                
                
            if out.z is not None:
                sev_s, sev_lbl, components = zero_shot_severity(out.z, out.dbdi)
                state_lbl, state_conf, _ = infer_state(out.z)
                p_live = 0.65*float(np.clip(sev_s, 0.0, 1.0)) + 0.35*float(np.clip((out.dbdi or 0.0)/100.0, 0.0, 1.0))
                render_live_readout(live_readout_ph, components, sev_lbl, out.dbdi, state_lbl, state_conf, sev_s, p_live)
                windows_for_final.append({**out.z, "dbdi": float(out.dbdi or 0.0)})
            
            
            
            if frame_count % 60 == 0:
                prog.progress(min(0.99, frame_count/600.0))
        prog.progress(1.0)
    
        final_dbdi, final_z = None, None
        if windows_for_final:
            last_w = windows_for_final[-1]
            final_dbdi = float(last_w.get("dbdi", 0.0))
            final_z    = {k: v for k, v in last_w.items() if k != "dbdi"}
        elif perframe:
            agg = defaultdict(list)
            for f in perframe:
                for k, v in f.items():
                    agg[k].append(v)
            avg = {k: float(np.mean(v)) for k, v in agg.items()}
            base = Baseline(use_iso=self.cfg.use_isolation_forest)
            base.fit([avg])
            dbdi, z = base.score(avg)
            final_dbdi, final_z = dbdi, z
        
        # ---- Robust Zero-Shot DX (uses the full time series) ----
        # Build a windows_z timeline from what you already buffered
        win_z = []
        for w in windows_for_final:
            # keep only z-keys + carry dbdi
            win_z.append({k: float(v) for k, v in w.items() if k != "dbdi"})
        dbdi_series = [float(w.get("dbdi", 0.0)) for w in windows_for_final]
        
        zdx = ZeroShotDX(win_z, dbdi_series, landmarks_series, src)
        zres = zdx.run()  # {'p', 'severity', 'components', 'consistency', 'abstain', 'audio'}
        if zres.get("audio"):
            aud = zres["audio"]
            st.caption(
                f"🎧 Audio used · pause_ratio={aud.get('pause_ratio', 0.0):.2f} · prosody_flat={aud.get('prosody_flat', 0.0):.2f}"
            )
            
        p_zero = float(zres["p"])
        sev_zero = float(zres["severity"])
        
        # Supervised (if present)
        p_sup = (self.rc.predict(perframe) if perframe else 0.5)
        has_sup = os.path.exists(self.rc.m) and not math.isclose(p_sup, 0.5, abs_tol=1e-6)
        
        # Blend with light supervision if available; otherwise pure zero-shot
        sup_w = 0.45 if has_sup else 0.0
        final_p = sup_w * float(p_sup) + (1.0 - sup_w) * p_zero
        
        # Abstain → keep probability but mark it and avoid overconfident UI tweaks
        abstain = bool(zres.get("abstain", False))
        
        
        with rcol:
            final_readout_ph.markdown(
                (f"**Final summary** — " +
                 (f"**DBDI:** {final_dbdi:.0f}  |  " if final_dbdi is not None else "") +
                 f"**P(Dementia):** {final_p*100:.1f}%"
                 + ("  _(abstained: low evidence)_" if abstain else "")
                )
            )
        
        
        # --- If requested, add this run to the features DB and retrain selected model ---
        try:
            if st.session_state.get("__request_add") and perframe:
                self.rc.add(src, gt_label, perframe)
                self.rc.online_update(perframe, gt_label)
                
                
                # reset the flag so subsequent runs don't retrain unintentionally
                st.session_state["__request_add"] = False
                if auto_train_model_name:
                    info = self.rc.train(auto_train_model_name)
                    note = f"Trained {info['model']} on {info['n']} rows"
                    if info.get("cv_acc") is not None:
                        note += f" (CV={info['cv_acc']:.3f})"
                    st.success(f"📦 Added to DB & 🔄 {note}")
        except Exception as e:
            st.warning(f"Added to DB, but retrain failed: {e}")
        
        
        st.markdown("### Outcome")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("DBDI", f"{final_dbdi:.0f}" if final_dbdi is not None else "—")
        with m2:
            st.metric("P(Dementia)", f"{final_p*100:.1f}%")
        st.caption(f"Zero-shot severity (0–1): **{sev_zero:.2f}** · consistency: **{zres['consistency']:.2f}**")
        
            
        
        
        # Auto-train + log
        thr, bacc = self.perf.optimal_threshold()
        pred_class = "Dementia" if final_p >= thr else "Normotypical"
        pred_label = pred_class
        pp = float(np.clip(final_p, 1e-6, 1 - 1e-6))
        y = 1.0 if gt_label == "Dementia" else 0.0
        xent = float(-(y * math.log(pp) + (1.0 - y) * math.log(1.0 - pp)))
        self.perf.add(
            path=src, gt_label=gt_label, pred_label=pred_label,
            pred_prob=float(final_p), threshold=float(thr),
            dbdi=float(final_dbdi or 0.0), xent=xent
        )
        
        
        
        # Update the single global dashboard
        if dash is not None:
            dash.update(self.perf, threshold=self.perf.optimal_threshold()[0])
            
        
    
        
    def _process_quick(self, src: str, gt_label: str, dash: Optional["BatchDashboard"]=None):
        proc = Processor(self.cfg)
        perframe: List[Dict[str,float]] = []
        windows_for_final: List[Dict[str,float]] = []
    
        for out in proc.stream(src):
            if out.per_frame:
                perframe.append(out.per_frame)
            if out.z is not None:
                windows_for_final.append({**out.z, "dbdi": float(out.dbdi or 0.0)})
    
        # NEW: define final_dbdi for logging/UI
        if windows_for_final:
            final_dbdi = float(windows_for_final[-1].get("dbdi", 0.0))
        else:
            # Fallback: estimate from perframe aggregates if we never populated windows_for_final
            final_dbdi = 0.0
            try:
                if perframe:
                    from collections import defaultdict
                    agg = defaultdict(list)
                    for f in perframe:
                        for k, v in f.items():
                            agg[k].append(v)
                    avg = {k: float(np.mean(v)) for k, v in agg.items()}
                    base = Baseline(use_iso=self.cfg.use_isolation_forest)
                    base.fit([avg])
                    final_dbdi, _ = base.score(avg)
            except Exception:
                pass
    
    
        # Build windows_z & dbdi series (quick mode recorded them)
        win_z = []
        for w in windows_for_final:
            win_z.append({k: float(v) for k, v in w.items() if k != "dbdi"})
        dbdi_series = [float(w.get("dbdi", 0.0)) for w in windows_for_final]
        
        zdx = ZeroShotDX(win_z, dbdi_series, [], src)
        zres = zdx.run()
        p_zero = float(zres["p"])
        p_sup  = (self.rc.predict(perframe) if perframe else 0.5)
        has_sup = os.path.exists(self.rc.m) and not math.isclose(p_sup, 0.5, abs_tol=1e-6)
        sup_w = 0.45 if has_sup else 0.0
        sup_prob = sup_w * float(p_sup) + (1.0 - sup_w) * p_zero
        
        st.markdown("### Performance & Calibration")
        g1c1, g1c2, g1c3 = st.columns(3)
        with g1c1:
            fig = self.perf.plot_confusion()
            if fig: st.pyplot(fig, clear_figure=True, width="stretch")
        with g1c2:
            fig = self.perf.plot_trend()
            if fig: st.pyplot(fig, clear_figure=True, width="stretch")
        with g1c3:
            fig = self.perf.plot_roc()
            if fig: st.pyplot(fig, clear_figure=True, width="stretch")
        
        g2c1, g2c2, g2c3 = st.columns(3)
        with g2c1:
            fig = self.perf.plot_reliability()
            if fig: st.pyplot(fig, clear_figure=True, width="stretch")
        with g2c2:
            fig = self.perf.plot_threshold_sweep()
            if fig: st.pyplot(fig, clear_figure=True, width="stretch")
        with g2c3:
            fig = self.perf.plot_prob_hist()
            if fig: st.pyplot(fig, clear_figure=True, width="stretch")
        
        
        thr, _ = self.perf.optimal_threshold()
        pred_label = "Dementia" if sup_prob >= thr else "Normotypical"
        y = 1.0 if gt_label == "Dementia" else 0.0
        pp = float(np.clip(sup_prob, 1e-6, 1.0 - 1e-6))
        xent = float(-(y * math.log(pp) + (1.0 - y) * math.log(1.0 - pp)))
    
        self.perf.add(
            path=src, gt_label=gt_label, pred_label=pred_label,
            pred_prob=float(sup_prob), threshold=float(thr),
            dbdi=float(final_dbdi or 0.0), xent=xent
        )
    
        if dash is not None:
            dash.update(self.perf, threshold=self.perf.optimal_threshold()[0])
            
    
        st.success(f"Quick analysis completed — P(Dementia) = {sup_prob*100:.1f}% | DBDI = {(final_dbdi or 0):.0f}")
    

    def run(self):
        st.set_page_config(page_title="3D Facial Mapping (Research)", layout="wide")
        # Hide any Plotly notifier banners (belt-and-suspenders)
        st.markdown("<style>.plotly-notifier{display:none!important}</style>", unsafe_allow_html=True)
        
        # --- White background for the whole app ---
        try:
            import plotly.io as pio
            from copy import deepcopy
            graphite = deepcopy(pio.templates["plotly_dark"])
            # Full black theme
            graphite.layout.paper_bgcolor = "#000000"
            graphite.layout.plot_bgcolor  = "#000000"
            graphite.layout.font.color    = "#e8ecf3"
            graphite.layout.colorway      = pio.templates["plotly_dark"].layout.colorway
            graphite.layout.scene = dict(
                xaxis=dict(backgroundcolor="#000000", gridcolor="#222222", zerolinecolor="#222222", showbackground=True),
                yaxis=dict(backgroundcolor="#000000", gridcolor="#222222", zerolinecolor="#222222", showbackground=True),
                zaxis=dict(backgroundcolor="#000000", gridcolor="#222222", zerolinecolor="#222222", showbackground=True),
            )
            pio.templates["graphite"] = graphite
            pio.templates.default = "graphite"
        except Exception:
            pass
        
        
        
        st.markdown("""
        <style>
          :root{
            --bg:#000000;        /* full black */
            --panel:#0a0a0a;     /* near black panels */
            --text:#e8ecf3;      /* near-white text */
            --muted:#aeb7c7;
            --accent:#ff7f50;    /* coral accent */
            --ring:#222222;      /* borders */
          }
          .stApp, .stApp [data-testid="stHeader"], .stApp [data-testid="stSidebar"],
          .block-container { background:#000000 !important; color:var(--text) !important; }
        
          /* Inputs / uploaders / radios / selects */
          [data-testid="stFileUploader"] div, .stTextInput>div>div>input,
            .stNumberInput>div>div>input, .stRadio>div, .stSelectbox>div>div {
              background:var(--panel) !important; color:var(--text) !important;
              border:1px solid var(--ring) !important; border-radius:8px;
          }
          
          /* Push cloud icon away from rounded border */
          [data-testid="stFileUploader"] svg { position: relative; left: 8px !important; }
          
          /* Add left padding to the dropzone/button (covers Streamlit variants) */
          [data-testid="stFileUploader"] div[role="button"],
          [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
          [data-testid="stFileUploader"] section[tabindex]{
            padding-left: 46px !important;
          }
          
        
          /* Buttons */
          .stButton>button {
            background:var(--panel) !important; color:var(--text) !important;
            border:1px solid var(--ring) !important; border-radius:8px;
          }
          .stButton>button:hover { filter:brightness(1.08); }
        
          /* Progress bar accent */
          .stProgress>div>div { background:var(--accent) !important; }
        
          /* Generic text */
          .stMarkdown p, .stMarkdown span, .stMetric label, .stMetric div {
            color:var(--text) !important;
          }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("3D Facial Mapping — Dementia Behavior Deviation (Research Only)")

        with st.sidebar:
            st.header("Speed & Processing")
            self.cfg.process_downscale = float(st.slider("Process downscale", 1.0, 4.0, self.cfg.process_downscale, 0.25))
            self.cfg.preview_downscale = float(st.slider("Preview downscale", 1.0, 4.0, self.cfg.preview_downscale, 0.25))
            self.cfg.update_preview_every_n = int(st.slider("Update preview every N frames", 1, 12, self.cfg.update_preview_every_n))
            self.cfg.skip_every_n = int(st.slider("Skip every N frames", 0, 5, self.cfg.skip_every_n))
            self.cfg.skip_live_3d = st.checkbox("Skip live 3D during processing", value=self.cfg.skip_live_3d)
            st.header("Analysis")
            self.cfg.face_detector = st.selectbox("Face detector", ["mediapipe"]+(["insightface"] if _HAS_INSIGHTFACE else []),
                                                  index=0 if self.cfg.face_detector=="mediapipe" else 1)
            self.cfg.use_isolation_forest = st.checkbox("Use IsolationForest (DBDI)", value=self.cfg.use_isolation_forest)
            self.cfg.baseline_s = float(st.number_input("Baseline seconds", 5.0, 120.0, self.cfg.baseline_s, 1.0))
            self.cfg.window_s = float(st.number_input("Window length (s)", 2.0, 30.0, self.cfg.window_s, 0.5))
            self.cfg.step_s = float(st.number_input("Step (s)", 0.5, 10.0, self.cfg.step_s, 0.5))
            self.cfg.max_frames_debug = int(st.number_input("Debug: Max frames (0=all)", 0, 50000, self.cfg.max_frames_debug, 100))
            
            # --- Classifier selection / replacement ---
            st.header("Classifier")
            model_options = ["GradientBoosting", "LogisticRegression", "RandomForest", "SVC (prob)"]
            
            # Preselect current on-disk model if present
            current_saved = self.rc.get_current_model_name()
            if "clf_choice" not in st.session_state:
                st.session_state["clf_choice"] = current_saved if current_saved in model_options else "GradientBoosting"
            
            st.session_state["clf_choice"] = st.selectbox(
                "Training model",
                model_options,
                index=model_options.index(st.session_state["clf_choice"]) if st.session_state["clf_choice"] in model_options else 0,
                help="Pick the algorithm used for supervised P(Dementia)."
            )
            
            if st.button("Replace model (train from DB)", help="Retrain from features_db.json and overwrite the saved model."):
                try:
                    info = self.rc.train(st.session_state["clf_choice"])
                    msg = f"✅ Replaced with {info['model']} · rows={info['n']}"
                    if info.get("cv_acc") is not None:
                        msg += f" · CV={info['cv_acc']:.3f}"
                    st.success(msg)
                except Exception as e:
                    st.error(f"Training failed: {e}")
            

        st.subheader("Source")
        c1,c2 = st.columns([3,2])
        with c1:
            url = st.text_input("Video URL or local path (optional)")
            up  = st.file_uploader("…or upload a video", type=["mp4","mov","avi","mkv","mpg","mpeg","m4v"], accept_multiple_files=False)
            
        st.divider()
        st.caption("Reset")
        cA, cB = st.columns(2)
        with cA:
            if st.button("Reset logs/model", help="Clears features_db, perf_log, model; keeps folders"):
                self._reset_cache(wipe_videos=False)
        with cB:
            if st.button("Reset EVERYTHING", help="Also deletes downloaded/uploads videos"):
                self._reset_cache(wipe_videos=True)
        
        
        # --- Batch CSV (url,label[,severity] or url,gt) ---
        # --- Batch table (CSV/XLSX): url,label[,severity]  OR  url,gt ---
        batch_csv = st.file_uploader(
            "Batch CSV/XLSX (url,label[,severity] or url,gt)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
            key="csv_batch"
        )
        run_batch = False
        run_batch_quick = False
        if batch_csv is not None:
            st.caption("Table detected. Choose how to run:")
            bc1, bc2 = st.columns(2)
            with bc1:
                run_batch = st.button("Run batch (auto-train & log)")
            with bc2:
                run_batch_quick = st.button("Quick Analyze batch (live 6 graphs)")
        
        
        
        
        
        if "label_choice" not in st.session_state:
            st.session_state["label_choice"] = "Normotypical"
        if "severity_choice" not in st.session_state:
            st.session_state["severity_choice"] = "CDR 2 — Moderate"
        
        with c2:
            st.markdown("**Ground-truth label**")
            b1, b2 = st.columns(2)
            with b1:
                if st.button(("✓ " if st.session_state["label_choice"] == "Normotypical" else "") + "Normotypical"):
                    st.session_state["label_choice"] = "Normotypical"
            with b2:
                if st.button(("✓ " if st.session_state["label_choice"] == "Dementia" else "") + "Dementia"):
                    st.session_state["label_choice"] = "Dementia"
            st.caption(f"Selected ground truth: **{st.session_state['label_choice']}**")
        
            
            # Single-run action
            act_visualize = st.button("Visualize & Analyze", use_container_width=True)
            
            
            # Remember if the user wants to add + auto-train this run
            act_add = st.checkbox(
                "Add this run to DB (and auto-train)",
                value=False,
                help="Store features to cache/features_db.json; if a model is selected, retrain after the run."
            )
            st.session_state["__request_add"] = bool(act_add)
            
                
            

        src=None
        if up is not None:
            src=os.path.join(self.cfg.cache_dir,"uploads", f"{ts()}_{up.name}"); ensure_dir(os.path.dirname(src))
            with open(src,"wb") as h: h.write(up.getbuffer()); st.caption(f"Saved upload to: {src}")
        elif url.strip():
            src=self._resolve(url.strip());  # may download
            if src: st.caption(f"Resolved to: {src}")
            
        # Handle batch first (returns after finishing)
        if run_batch or run_batch_quick:
            rows = load_csv_rows(batch_csv)
            if not rows:
                st.error("CSV parsed but found no usable rows/columns. Expect 'url' and either 'label'+optional 'severity' or a combined 'gt' column.")
                return
            
            # Randomize order (stratified by label) with a fresh seed each run
            seed = int(time.time() * 1000) % 2_147_483_647
            order = randomize_items([(r["url"], r["gt_label"]) for r in rows if r.get("url") and r.get("gt_label")],
                                    seed=seed, stratify=True)
            
            st.write(f"Found **{len(order)}** rows (randomized).")
            dl_cache = {}
            dash = BatchDashboard()
            for i, (url_i, gt_i) in enumerate(order, 1):
                st.markdown(f"### [{i}/{len(order)}] {gt_i} — {url_i}")
            
                if url_i.startswith("http"):
                    if url_i not in dl_cache:
                        path_i = self._resolve(url_i)
                        if not path_i:
                            st.warning(f"Could not download/resolve: {url_i}")
                            continue
                        dl_cache[url_i] = path_i
                    src_i = dl_cache[url_i]
                else:
                    src_i = url_i if os.path.exists(url_i) else None
                    if not src_i:
                        st.warning(f"Local path not found: {url_i}")
                        continue
        
                if run_batch_quick:
                    # Fast live charts per item (no auto-train)
                    self._process_quick(src_i, gt_i, dash=dash)
                
                    
                else:
                    # Full run per item (respects Add-to-DB + auto-train)
                    auto_model = st.session_state.get("clf_choice", "GradientBoosting")
                    self._process_single(
                        src_i,
                        gt_i,
                        auto_train_model_name=auto_model if st.session_state.get("__request_add") else None
                    )
        
                # Free disk for this item
                self._maybe_delete_downloaded(src_i)
                if url_i in dl_cache:
                    del dl_cache[url_i]
        
            st.success("Batch (Quick) completed." if run_batch_quick else "Batch completed.")
            return
        
        

        # Always define before use (prevents UnboundLocalError)
        gt_label = st.session_state.get("label_choice", "Normotypical")
        
        if act_visualize and not src:
            st.error("Please provide a URL or upload a video.")
            return
        
        if act_visualize and src:
            auto_model = st.session_state.get("clf_choice", "GradientBoosting")
            self._process_single(
                src,
                gt_label,
                auto_train_model_name=auto_model if st.session_state.get("__request_add") else None
            )
            self._maybe_delete_downloaded(src)
            return
        
        
        
            

def main():
    app = App(AppConfig()); app.run()

if __name__ == "__main__":
    main()