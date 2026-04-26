# -*- coding: utf-8 -*-
"""CICIoT2023 IoT Intrusion Detection System â€” Command Center Dashboard.

Run from project root (with the FastAPI service already up on port 8000):
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR  = PROJECT_ROOT / "data" / "artifacts"
DRIFT_DIR      = PROJECT_ROOT / "data" / "drift" / "step7"
DEFAULT_API    = "http://127.0.0.1:8000"

# â”€â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.set_page_config(
    page_title="IDS Command Center",
    page_icon="IDS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€â”€ Global CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown("""
<style>
/* â”€â”€ layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stAppViewContainer"] { background: #0a0e1a; }
[data-testid="stSidebar"]          { background: #060a14; border-right: 1px solid #1a2744; }
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

/* â”€â”€ typography â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
h1, h2, h3 { letter-spacing: 0.04em; }

/* â”€â”€ page header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.ids-header {
    background: linear-gradient(135deg, #060a14 0%, #0d1b3e 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
}
.ids-header h1 {
    color: #00d4ff;
    margin: 0 0 4px 0;
    font-size: 1.6rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.ids-header .sub {
    color: #64748b;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* â”€â”€ section titles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.section-title {
    color: #00d4ff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 1.4rem 0 0.8rem 0;
}

/* â”€â”€ stat cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.stat-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    text-align: center;
    min-height: 80px;
}
.stat-card .val  { color: #e2e8f0; font-size: 1.55rem; font-weight: 700; line-height: 1.2; }
.stat-card .lbl  { color: #64748b; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; }
.stat-card .val.good   { color: #10b981; }
.stat-card .val.warn   { color: #f59e0b; }
.stat-card .val.bad    { color: #ef4444; }
.stat-card .val.cyan   { color: #00d4ff; }

/* â”€â”€ alert / severity badges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    vertical-align: middle;
}
.badge-critical { background:#7f1d1d; color:#fca5a5; border:1px solid #ef4444; }
.badge-high     { background:#7c2d12; color:#fdba74; border:1px solid #f97316; }
.badge-warning  { background:#78350f; color:#fcd34d; border:1px solid #f59e0b; }
.badge-safe     { background:#064e3b; color:#6ee7b7; border:1px solid #10b981; }
.badge-info     { background:#0c4a6e; color:#7dd3fc; border:1px solid #38bdf8; }
.badge-none     { background:#1e293b; color:#94a3b8; border:1px solid #475569; }

/* â”€â”€ alert bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.alert-bar {
    border-radius: 6px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-bar.critical { background:#1c0808; border:1px solid #ef4444; color:#fca5a5; }
.alert-bar.warning  { background:#1c1208; border:1px solid #f59e0b; color:#fcd34d; }
.alert-bar.none     { background:#071c12; border:1px solid #10b981; color:#6ee7b7; }

/* â”€â”€ connection dot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.dot-online  { display:inline-block; width:9px; height:9px; border-radius:50%; background:#10b981; margin-right:6px; box-shadow:0 0 5px #10b981; }
.dot-offline { display:inline-block; width:9px; height:9px; border-radius:50%; background:#ef4444; margin-right:6px; }

/* â”€â”€ prediction result card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.pred-result {
    background: #0d1b3e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin: 0.8rem 0;
}

/* â”€â”€ psi table row colours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.psi-critical { background: #1c0808 !important; }
.psi-warning  { background: #1c1208 !important; }
.psi-stable   { background: #071c12 !important; }

/* â”€â”€ sidebar labels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.sidebar-label {
    color: #475569;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1rem 0 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)

# â”€â”€â”€ Colour palette for matplotlib (dark theme) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_DARK_BG   = "#0a0e1a"
_CARD_BG   = "#111827"
_CYAN      = "#00d4ff"
_GREEN     = "#10b981"
_RED       = "#ef4444"
_AMBER     = "#f59e0b"
_TEXT      = "#e2e8f0"
_MUTED     = "#475569"
_GRID      = "#1e3a5f"

def _dark_fig(w: float = 10, h: float = 5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT, labelsize=7)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_CYAN)
    ax.yaxis.set_tick_params(which="both", color=_GRID)
    ax.grid(color=_GRID, linewidth=0.5, alpha=0.6)
    return fig, ax

# â”€â”€â”€ Threat severity helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _threat_info(class_name: str) -> tuple[str, str, str]:
    """Returns (badge_class, level_label, bar_colour) for a predicted class."""
    n = class_name.lower()
    if n == "benign":
        return "badge-safe", "BENIGN", _GREEN
    if any(x in n for x in ["ddos", "flood"]):
        return "badge-critical", "DDoS", _RED
    if "dos" in n:
        return "badge-critical", "DoS", _RED
    if any(x in n for x in ["mirai", "botnet"]):
        return "badge-critical", "BOTNET", _RED
    if any(x in n for x in ["injection", "sqli", "xss", "backdoor", "web"]):
        return "badge-critical", "WEB ATTACK", _RED
    if any(x in n for x in ["brute", "force", "password"]):
        return "badge-high", "BRUTE FORCE", "#f97316"
    if any(x in n for x in ["recon", "scan", "port"]):
        return "badge-warning", "RECON", _AMBER
    return "badge-high", "ATTACK", "#f97316"


# â”€â”€â”€ Data helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data(show_spinner=False)
def fetch_json(url: str) -> dict | list:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def fetch_png(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


@st.cache_data(show_spinner=False)
def load_test_sample(n: int = 500) -> tuple[pd.DataFrame, pd.Series] | None:
    X_path = PROCESSED_DIR / "X_test_raw.csv"
    y_path = PROCESSED_DIR / "y_test_encoded.csv"
    if not X_path.exists() or not y_path.exists():
        return None
    X = pd.read_csv(X_path, nrows=n)
    y = pd.read_csv(y_path, nrows=n).iloc[:, 0]
    return X, y


def _api_status(api_url: str) -> tuple[bool, dict]:
    try:
        info = fetch_json(f"{api_url}/")
        return True, info
    except Exception as e:
        return False, {"error": str(e)}


# â”€â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 8px 0;'>
        <div style='color:#00d4ff; font-weight:800; letter-spacing:0.16em; font-size:1.4rem;'>IDS</div>
        <div style='color:#00d4ff; font-weight:700; letter-spacing:0.1em; font-size:0.85rem;'>IDS COMMAND CENTER</div>
        <div style='color:#475569; font-size:0.65rem; letter-spacing:0.08em;'>CICIoT2023 | IoT Network Security</div>
    </div>
    <hr style='border-color:#1a2744; margin:8px 0 12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">API Endpoint</div>', unsafe_allow_html=True)
    api_url = st.text_input(
        "API Endpoint", DEFAULT_API, label_visibility="collapsed"
    ).rstrip("/")

    online, info = _api_status(api_url)

    if online:
        st.markdown(
            f'<p style="font-size:0.78rem;"><span class="dot-online"></span>'
            f'CONNECTED &nbsp;|&nbsp; {len(info.get("available_models",[]))} model(s) online</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:0.78rem;"><span class="dot-offline"></span>SERVICE OFFLINE</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:#1c0808;border:1px solid #ef4444;border-radius:6px;'
            'padding:8px 12px;font-size:0.75rem;color:#fca5a5;">'
            'WARNING: Start the service:<br>'
            '<code style="color:#fcd34d;">uvicorn src.serve.main:app --reload</code>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # â”€â”€ Model / feature / class data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        models_resp   = fetch_json(f"{api_url}/models")
        features_resp = fetch_json(f"{api_url}/features")
        classes_resp  = fetch_json(f"{api_url}/classes")
    except Exception as e:
        st.error(f"Cannot fetch service metadata: {e}")
        st.stop()

    available_models   = models_resp["available"]
    unavailable_models = models_resp["unavailable"]

    if not available_models:
        st.warning("No model artifacts found on this machine.")
        st.stop()

    st.markdown('<div class="sidebar-label">Detection Engine</div>', unsafe_allow_html=True)
    model_labels = {
        m["key"]: f"{m['display_name']}  [{m['variant']}]" for m in available_models
    }
    selected_key = st.selectbox(
        "Detection Engine",
        options=list(model_labels.keys()),
        format_func=lambda k: model_labels[k],
        label_visibility="collapsed",
    )
    selected_meta = next(m for m in available_models if m["key"] == selected_key)

    st.markdown('<hr style="border-color:#1a2744; margin:12px 0;">', unsafe_allow_html=True)
    n_feat  = features_resp["n_features"]
    n_cls   = classes_resp["n_classes"]
    st.markdown(
        f'<div style="font-size:0.72rem; color:#64748b; line-height:2;">'
        f'<span style="color:#e2e8f0;">&#x2023; Feature vector</span> &nbsp;{n_feat} dims<br>'
        f'<span style="color:#e2e8f0;">&#x2023; Threat classes</span> &nbsp;{n_cls} categories<br>'
        f'<span style="color:#e2e8f0;">&#x2023; Preprocessing</span> &nbsp;{selected_meta["variant"]}<br>'
        f'<span style="color:#e2e8f0;">&#x2023; Loader</span> &nbsp;{selected_meta["loader_type"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if unavailable_models:
        st.markdown('<hr style="border-color:#1a2744; margin:12px 0;">', unsafe_allow_html=True)
        with st.expander("Offline engines", expanded=False):
            for m in unavailable_models:
                st.markdown(
                    f'<div style="color:#ef4444;font-size:0.72rem;">&#x2717; {m["display_name"]} ({m["variant"]})</div>',
                    unsafe_allow_html=True,
                )

# â”€â”€â”€ Page header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

FEATURE_NAMES = features_resp["feature_names"]
CLASS_NAMES   = [classes_resp["classes"][str(i)] for i in range(classes_resp["n_classes"])]

st.markdown(f"""
<div class="ids-header">
    <h1>IDS &nbsp; Intrusion Detection System</h1>
    <div class="sub">
        CICIoT2023 Dataset &nbsp;|&nbsp;
        Active Engine: <span style="color:#00d4ff;">{selected_meta['display_name']}</span>
        &nbsp;|&nbsp; {n_cls} Threat Classes &nbsp;|&nbsp; {n_feat} Network Features
    </div>
</div>
""", unsafe_allow_html=True)

# â”€â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

tab_intel, tab_detect, tab_bench, tab_soc = st.tabs([
    "Threat Intelligence",
    "Live Detection",
    "Model Benchmark",
    "SOC Monitor",
])

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1 â€” THREAT INTELLIGENCE (model metrics)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

with tab_intel:
    try:
        m = fetch_json(f"{api_url}/models/{selected_key}/metrics")
    except Exception as e:
        st.error(f"Could not load metrics: {e}")
        st.stop()

    st.markdown('<div class="section-title">Detection Performance</div>', unsafe_allow_html=True)

    def _card(val: str, label: str, cls: str = "") -> str:
        return (
            f'<div class="stat-card">'
            f'<div class="val {cls}">{val}</div>'
            f'<div class="lbl">{label}</div>'
            f'</div>'
        )

    f1      = m["test_macro_f1"]
    acc     = m.get("test_accuracy", float("nan"))
    logloss = m.get("test_log_loss")
    fpr     = m["test_benign_fpr"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_card(f"{f1:.4f}", "Test Macro-F1", "good" if f1 > 0.85 else "warn" if f1 > 0.70 else "bad"), unsafe_allow_html=True)
    with c2:
        st.markdown(_card(f"{acc:.4f}", "Test Accuracy", "good" if acc > 0.90 else "warn"), unsafe_allow_html=True)
    with c3:
        if logloss is None:
            st.markdown(_card("N/A", "Log-Loss", "warn"), unsafe_allow_html=True)
        else:
            st.markdown(_card(f"{logloss:.4f}", "Log-Loss", "good" if logloss < 0.3 else "warn" if logloss < 1.0 else "bad"), unsafe_allow_html=True)
    with c4:
        st.markdown(_card(f"{fpr:.4f}", "Benign False-Positive Rate", "good" if fpr < 0.05 else "warn" if fpr < 0.15 else "bad"), unsafe_allow_html=True)

    st.markdown("")

    c5, c6, c7 = st.columns(3)
    train_f1 = m["train_macro_f1"]
    val_f1   = m["val_macro_f1"]
    gap      = train_f1 - val_f1
    with c5:
        st.markdown(_card(f"{train_f1:.4f}", "Train Macro-F1", "cyan"), unsafe_allow_html=True)
    with c6:
        st.markdown(_card(f"{val_f1:.4f}",   "Validation Macro-F1", "cyan"), unsafe_allow_html=True)
    with c7:
        g_cls = "good" if abs(gap) < 0.02 else "warn" if abs(gap) < 0.05 else "bad"
        st.markdown(_card(f"{gap:+.4f}", "Train / Val Gap", g_cls), unsafe_allow_html=True)

    with st.expander("Full metrics JSON", expanded=False):
        st.json(m)

    # â”€â”€ Per-class detection rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown('<div class="section-title">Per-Class Detection Rate (Recall)</div>', unsafe_allow_html=True)

    per_class = pd.DataFrame({
        "class": CLASS_NAMES,
        "recall": m["test_per_class_recall"],
    }).sort_values("recall", ascending=True)

    bar_colors = [_RED if v < 0.5 else _AMBER if v < 0.8 else _GREEN for v in per_class["recall"]]

    fig, ax = _dark_fig(w=11, h=9)
    ax.barh(per_class["class"], per_class["recall"], color=bar_colors, edgecolor="none", height=0.72)
    ax.axvline(0.5, color=_RED,   linestyle="--", linewidth=0.8, alpha=0.7, label="0.50 threshold")
    ax.axvline(0.8, color=_AMBER, linestyle="--", linewidth=0.8, alpha=0.7, label="0.80 threshold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall", color=_TEXT)
    ax.set_title(f"Per-Class Detection Rate - {selected_meta['display_name']}", color=_CYAN, fontsize=10, pad=10)
    ax.legend(fontsize=7, framealpha=0.2, labelcolor=_TEXT)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # â”€â”€ Confusion matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    normalize = st.toggle("Row-normalise", value=True, key="cm_norm")
    try:
        png = fetch_png(
            f"{api_url}/models/{selected_key}/confusion_matrix.png"
            f"?normalize={str(normalize).lower()}"
        )
        st.image(png, width="stretch")
    except Exception as e:
        st.warning(f"Confusion matrix unavailable: {e}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 2 â€” LIVE DETECTION (predict)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

with tab_detect:
    st.markdown(
        '<div class="section-title">Network Traffic Classifier</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;font-size:0.82rem;">'
        "Submit a network flow feature vector. The active detection engine will "
        "classify it across all 34 threat categories and return confidence scores.</p>",
        unsafe_allow_html=True,
    )

    test_sample = load_test_sample()
    if test_sample is None:
        st.warning("Test split not found in `data/processed/`. Cannot load sample rows.")
        st.stop()
    X_test, y_test = test_sample

    mode = st.radio(
        "Input mode",
        ["Pick a test-set sample", "Paste a JSON feature vector"],
        horizontal=True,
    )

    row_dict: dict[str, float] | None = None
    true_label_id: int | None = None

    if mode == "Pick a test-set sample":
        col_a, col_b, col_c = st.columns([2, 2, 4])
        with col_a:
            row_idx = st.number_input(
                "Sample index (0-based)",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
                step=1,
            )
        with col_b:
            true_label_id = int(y_test.iloc[int(row_idx)])
            true_name     = CLASS_NAMES[true_label_id]
            badge_cls, lvl_lbl, _ = _threat_info(true_name)
            st.markdown(
                f'<div style="padding-top:28px;">'
                f'Ground truth: &nbsp;<span class="badge {badge_cls}">{lvl_lbl}</span>'
                f'&nbsp;<span style="color:#e2e8f0;font-size:0.82rem;">{true_name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        row_dict = X_test.iloc[int(row_idx)].to_dict()
        with st.expander("Feature vector", expanded=False):
            feat_df = pd.Series(row_dict, name="value").to_frame()
            st.dataframe(feat_df, width="stretch", height=300)
    else:
        default_json = json.dumps(X_test.iloc[0].to_dict(), indent=2)
        raw = st.text_area("Feature vector (JSON)", value=default_json, height=300)
        try:
            row_dict = json.loads(raw)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            row_dict = None

    if row_dict is not None and st.button("Classify Traffic", type="primary"):
        payload = {"rows": row_dict}
        try:
            r = requests.post(
                f"{api_url}/predict?model={selected_key}", json=payload, timeout=30
            )
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.stop()

        pred = resp["predictions"][0]
        badge_cls, lvl_lbl, bar_col = _threat_info(pred["predicted_class"])
        match_flag = (
            true_label_id is not None and pred["predicted_class_id"] == true_label_id
        )
        match_html = ""
        if true_label_id is not None:
            match_html = (
                '<span style="color:#10b981;">&#x2713; MATCH</span>'
                if match_flag
                else '<span style="color:#ef4444;">&#x2717; MISMATCH</span>'
            )

        pred_html = (
            f'<div class="pred-result">'
            f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
            f'<div>'
            f'<div style="color:#475569;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;">Classification</div>'
            f'<div style="font-size:1.35rem;font-weight:700;color:#e2e8f0;margin:4px 0;">{pred["predicted_class"]}</div>'
            f'</div>'
            f'<span class="badge {badge_cls}" style="font-size:0.85rem;padding:5px 16px;">{lvl_lbl}</span>'
            f'{match_html}'
            f'</div>'
            f'<div style="display:flex;gap:2.5rem;margin-top:14px;flex-wrap:wrap;">'
            f'<div>'
            f'<div style="color:#475569;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;">Detection Confidence</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#00d4ff;">{pred["confidence"]*100:.2f}%</div>'
            f'</div>'
            f'<div>'
            f'<div style="color:#475569;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;">Class ID</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#e2e8f0;">{pred["predicted_class_id"]}</div>'
            f'</div>'
            f'<div>'
            f'<div style="color:#475569;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;">Engine</div>'
            f'<div style="font-size:1rem;font-weight:700;color:#94a3b8;">{resp["model"]}</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(pred_html, unsafe_allow_html=True)

        # â”€â”€ Probability chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown('<div class="section-title">Threat Probability Distribution</div>', unsafe_allow_html=True)
        probs    = pred["probabilities"]
        prob_df  = (
            pd.DataFrame({"class": list(probs.keys()), "probability": list(probs.values())})
            .sort_values("probability", ascending=True)
        )

        bar_colors_prob = []
        for cls_name in prob_df["class"]:
            if cls_name == pred["predicted_class"]:
                bar_colors_prob.append(bar_col)
            elif cls_name.lower() == "benign":
                bar_colors_prob.append(_GREEN)
            else:
                bar_colors_prob.append(_MUTED)

        fig, ax = _dark_fig(w=10, h=10)
        ax.barh(prob_df["class"], prob_df["probability"], color=bar_colors_prob, edgecolor="none", height=0.72)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", color=_TEXT)
        ax.set_title(f"All-Class Threat Scores - {resp['model']} [{resp['variant']}]", color=_CYAN, fontsize=10, pad=10)
        for y_pos, (cls_name, p) in enumerate(zip(prob_df["class"], prob_df["probability"])):
            if p >= 0.01:
                ax.text(p + 0.005, y_pos, f"{p:.3f}", va="center", fontsize=6.5, color=_TEXT)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # â”€â”€ Top-5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with st.expander("Top-5 threat candidates", expanded=True):
            top5 = (
                prob_df.sort_values("probability", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            top5.index = top5.index + 1
            top5["probability"] = top5["probability"].map(lambda v: f"{v:.5f}")
            st.table(top5)

        with st.expander("Raw API response", expanded=False):
            st.json(resp)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3 â€” MODEL BENCHMARK
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

with tab_bench:
    st.markdown('<div class="section-title">Detection Engine Benchmark</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b;font-size:0.82rem;">'
        "Stored test-set metrics for every available engine.</p>",
        unsafe_allow_html=True,
    )

    rows = []
    for m_info in available_models:
        try:
            metrics = fetch_json(f"{api_url}/models/{m_info['key']}/metrics")
        except Exception:
            continue
        rows.append({
            "Engine":        m_info["display_name"],
            "Variant":       m_info["variant"],
            "Accuracy":      metrics.get("test_accuracy"),
            "Precision":     metrics.get("test_macro_precision"),
            "Recall":        metrics.get("test_macro_recall"),
            "F1":            metrics.get("test_macro_f1"),
            "Log-Loss":      metrics.get("test_log_loss"),
            "Benign FPR":    metrics["test_benign_fpr"],
            "Train/Val Gap": metrics["train_macro_f1"] - metrics["val_macro_f1"],
        })

    if not rows:
        st.info("No metric files found for the available models.")
    else:
        df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

        def _fmt_optional(v):
            return f"{v:.4f}" if v is not None else "N/A"

        has_logloss = df["Log-Loss"].notna().any()
        style = df.style.format({
            "Accuracy":      _fmt_optional,
            "Precision":     _fmt_optional,
            "Recall":        _fmt_optional,
            "F1":            _fmt_optional,
            "Log-Loss":      _fmt_optional,
            "Benign FPR":    "{:.4f}",
            "Train/Val Gap": "{:+.4f}",
        }).background_gradient(
            subset=["Accuracy", "Precision", "Recall", "F1"], cmap="Blues"
        )
        if has_logloss:
            style = style.background_gradient(subset=["Log-Loss", "Benign FPR"], cmap="Reds_r")
        st.dataframe(style, width="stretch")

        st.markdown('<div class="section-title">Metric Comparison</div>', unsafe_allow_html=True)

        selected_metric = st.selectbox(
            "Metric to compare",
            options=["Accuracy", "Precision", "Recall", "F1"],
            index=3,
            key="benchmark_metric",
        )
        plot_df = df.dropna(subset=[selected_metric]).sort_values(
            selected_metric, ascending=False
        )

        colors_bench = [
            _CYAN if eng == selected_meta["display_name"] else _MUTED
            for eng in plot_df["Engine"]
        ]
        fig, ax = _dark_fig(w=9, h=4)
        bars = ax.bar(plot_df["Engine"], plot_df[selected_metric], color=colors_bench, edgecolor="none", width=0.5)
        ax.set_ylabel(f"Test {selected_metric}", color=_TEXT)
        ax.set_ylim(0, 1)
        ax.set_title(f"Test {selected_metric} - All Available Engines", color=_CYAN, fontsize=10, pad=10)
        for bar, v in zip(bars, plot_df[selected_metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                fontsize=8,
                color=_TEXT,
            )
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 4 â€” SOC MONITOR (drift)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_ALERT_COLORS = {"critical": "critical", "warning": "warning", "none": "none"}


def _alert_bar_html(level: str, msg: str, source: str) -> str:
    labels = {"critical": "CRITICAL", "warning": "WARNING", "none": "OK"}
    badge_classes = {"critical": "badge-critical", "warning": "badge-warning", "none": "badge-safe"}
    return (
        f'<div class="alert-bar {level}">'
        f'<span class="badge {badge_classes.get(level,"badge-info")}">{labels.get(level,"INFO")}</span>'
        f'<strong>{source.replace("_"," ").upper()}</strong> - {msg}'
        f'</div>'
    )

def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _stream_csv_to_api(
    *,
    api_url: str,
    model_key: str,
    x_path: Path,
    y_path: Path | None,
    batch_size: int,
    limit: int,
    sleep_s: float,
    feature_names: list[str],
    class_names: list[str],
    progress,
    status,
) -> dict:
    if not x_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {x_path}")
    if y_path is not None and not y_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {y_path}")

    read_kwargs = {} if limit == 0 else {"nrows": limit}
    X = pd.read_csv(x_path, **read_kwargs)
    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise ValueError(f"Feature CSV is missing required columns: {missing}")
    X = X[feature_names]

    y = None
    if y_path is not None:
        y = pd.read_csv(y_path, **read_kwargs).iloc[:, 0]
        if len(y) != len(X):
            raise ValueError(f"Feature/label row-count mismatch: {len(X)} X rows, {len(y)} y rows.")

    total = len(X)
    if total == 0:
        raise ValueError("Selected CSV contains no rows to stream.")

    sent = 0
    correct = 0
    pred_counts = {name: 0 for name in class_names}
    t0 = pd.Timestamp.now()

    for start in range(0, total, batch_size):
        batch_X = X.iloc[start:start + batch_size]
        batch_y = y.iloc[start:start + batch_size].to_numpy() if y is not None else None
        payload = {"rows": batch_X.to_dict(orient="records")}

        r = requests.post(f"{api_url}/predict?model={model_key}", json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json()

        for i, pred in enumerate(resp["predictions"]):
            pred_counts[pred["predicted_class"]] = pred_counts.get(pred["predicted_class"], 0) + 1
            if batch_y is not None and pred["predicted_class_id"] == int(batch_y[i]):
                correct += 1

        sent += len(batch_X)
        progress.progress(sent / total)
        if y is not None:
            status.write(f"Streamed {sent:,}/{total:,} rows | rolling accuracy {correct / sent * 100:.2f}%")
        else:
            top = sorted(pred_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            summary = ", ".join(f"{name}={count}" for name, count in top if count)
            status.write(f"Streamed {sent:,}/{total:,} rows | top predictions: {summary}")

        if sleep_s > 0:
            import time
            time.sleep(sleep_s)

    dt = (pd.Timestamp.now() - t0).total_seconds()
    return {
        "sent": sent,
        "seconds": dt,
        "rate": sent / dt if dt > 0 else 0.0,
        "accuracy": correct / sent if y is not None and sent else None,
        "pred_counts": {k: v for k, v in pred_counts.items() if v},
    }


with tab_soc:
    st.markdown('<div class="section-title">Security Operations - Drift Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b;font-size:0.82rem;">'
        "Monitor statistical drift in the live prediction stream. Send traffic first, "
        "then refresh to see drift signals.</p>",
        unsafe_allow_html=True,
    )

    demo_sources = {
        "Default test split": {
            "x": PROCESSED_DIR / "X_test_raw.csv",
            "y": PROCESSED_DIR / "y_test_encoded.csv",
        },
        "Step 7 labeled drift": {
            "x": DRIFT_DIR / "X_drift_test.csv",
            "y": DRIFT_DIR / "y_drift_test.csv",
        },
        "Backdoor malware traffic": {
            "x": DRIFT_DIR / "Backdoor_Malware.pcap.csv",
            "y": None,
        },
        "Custom path": {"x": None, "y": None},
    }

    st.markdown('<div class="section-title">Traffic Replay</div>', unsafe_allow_html=True)
    src_col, batch_col, limit_col, sleep_col = st.columns([2.2, 1, 1, 1])
    with src_col:
        replay_source = st.selectbox(
            "Traffic source",
            options=list(demo_sources.keys()),
            index=0,
            key="drift_replay_source",
        )
    with batch_col:
        replay_batch = st.number_input("Batch size", min_value=1, max_value=1000, value=64, step=1)
    with limit_col:
        replay_limit = st.number_input("Limit", min_value=0, max_value=100000, value=1000, step=100)
    with sleep_col:
        replay_sleep = st.number_input("Sleep", min_value=0.0, max_value=5.0, value=0.05, step=0.01, format="%.2f")

    selected_source = demo_sources[replay_source]
    if replay_source == "Custom path":
        custom_x = st.text_input("Feature CSV path", value=str(DRIFT_DIR / "Backdoor_Malware.pcap.csv"))
        custom_y = st.text_input("Optional label CSV path", value="")
        replay_x_path = _resolve_project_path(custom_x)
        replay_y_path = _resolve_project_path(custom_y) if custom_y.strip() else None
    else:
        replay_x_path = selected_source["x"]
        replay_y_path = selected_source["y"]

    cmd = (
        f"python -m src.client.stream_test --model {selected_key} "
        f"--x \"{replay_x_path.relative_to(PROJECT_ROOT) if replay_x_path.is_relative_to(PROJECT_ROOT) else replay_x_path}\" "
    )
    if replay_y_path is not None:
        cmd += f"--y \"{replay_y_path.relative_to(PROJECT_ROOT) if replay_y_path.is_relative_to(PROJECT_ROOT) else replay_y_path}\" "
    cmd += f"--batch-size {replay_batch} --limit {replay_limit} --sleep {replay_sleep}"
    st.code(cmd, language="bash")

    run_col, note_col = st.columns([1, 4])
    with run_col:
        do_stream = st.button("Run Stream", type="primary", width="stretch")
    with note_col:
        st.caption("This runs inside Streamlit and sends the selected CSV to the FastAPI /predict endpoint.")

    if do_stream:
        prog = st.progress(0.0)
        live_status = st.empty()
        try:
            result = _stream_csv_to_api(
                api_url=api_url,
                model_key=selected_key,
                x_path=replay_x_path,
                y_path=replay_y_path,
                batch_size=int(replay_batch),
                limit=int(replay_limit),
                sleep_s=float(replay_sleep),
                feature_names=FEATURE_NAMES,
                class_names=CLASS_NAMES,
                progress=prog,
                status=live_status,
            )
            fetch_json.clear()
            msg = f"Streamed {result['sent']:,} rows in {result['seconds']:.2f}s ({result['rate']:.0f} rows/s)."
            if result["accuracy"] is not None:
                msg += f" Accuracy: {result['accuracy'] * 100:.2f}%."
            st.success(msg)
            with st.expander("Prediction distribution", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [{"class": k, "count": v} for k, v in result["pred_counts"].items()]
                    ).sort_values("count", ascending=False),
                    width="stretch",
                    hide_index=True,
                )
        except Exception as e:
            st.error(f"Stream failed: {e}")

    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
    with ctrl1:
        window_size = st.slider(
            "Rolling window (predictions)", 50, 2000, 500, 50, key="drift_window"
        )
    with ctrl2:
        do_refresh = st.button("Refresh", type="primary", width="stretch")
    with ctrl3:
        do_reset = st.button("Reset Log", type="secondary", width="stretch")

    bucket_size = st.select_slider(
        "Confidence history bucket size",
        options=[50, 100, 200, 500],
        value=100,
        key="drift_bucket",
    )

    if do_reset:
        try:
            r = requests.post(f"{api_url}/drift/reset?model={selected_key}", timeout=10)
            r.raise_for_status()
            st.success(f"Log cleared - {r.json()['rows_deleted']} rows deleted.")
            fetch_json.clear()
        except Exception as e:
            st.error(f"Reset failed: {e}")

    if do_refresh:
        fetch_json.clear()

    # â”€â”€ Fetch drift status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        drift_data = fetch_json(
            f"{api_url}/drift/status?model={selected_key}&window={window_size}"
        )
    except Exception as e:
        st.warning(f"Could not fetch drift status: {e}")
        drift_data = None

    if drift_data is not None:
        total_logged = drift_data.get("total_logged", 0)
        wsize        = drift_data.get("window_size", 0)
        alerts       = drift_data.get("alerts", [])

        # â”€â”€ Summary cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        s1, s2, s3, s4 = st.columns(4)
        alert_level = "bad" if any(a["level"] == "critical" for a in alerts) \
                      else "warn" if alerts else "good"
        status_lbl  = "ALERT" if any(a["level"] == "critical" for a in alerts) \
                      else "WARNING" if alerts else "NOMINAL"
        with s1:
            st.markdown(_card(f"{total_logged:,}", "Total Events Logged", "cyan"), unsafe_allow_html=True)
        with s2:
            st.markdown(_card(f"{wsize:,}", f"Window Analysed", "cyan"), unsafe_allow_html=True)
        with s3:
            st.markdown(_card(str(len(alerts)), "Active Alerts", alert_level), unsafe_allow_html=True)
        with s4:
            st.markdown(_card(status_lbl, "Threat Status", alert_level), unsafe_allow_html=True)

        st.markdown("")

        if total_logged == 0:
            st.markdown(
                '<div class="alert-bar none">'
                '<strong>NO TRAFFIC</strong> - '
                'Run the stream-test client to generate prediction events.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            # â”€â”€ Alert panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.markdown('<div class="section-title">Alert Summary</div>', unsafe_allow_html=True)
            if alerts:
                for a in alerts:
                    st.markdown(
                        _alert_bar_html(a["level"], a["message"], a["source"]),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    _alert_bar_html("none", "All drift signals within normal range.", "system"),
                    unsafe_allow_html=True,
                )

            # â”€â”€ Confidence trend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.markdown('<div class="section-title">Confidence Trend Over Traffic Stream</div>', unsafe_allow_html=True)
            try:
                hist_data = fetch_json(
                    f"{api_url}/drift/confidence_history"
                    f"?model={selected_key}&bucket_size={bucket_size}"
                )
            except Exception as e:
                hist_data = None
                st.warning(f"Confidence history unavailable: {e}")

            if hist_data and hist_data.get("n_buckets", 0) > 0:
                history = hist_data["history"]
                buckets = [h["bucket_index"] for h in history]
                means   = [h["mean_confidence"] for h in history]
                mins_   = [h["min_confidence"]  for h in history]
                maxs_   = [h["max_confidence"]  for h in history]

                fig, ax = _dark_fig(w=11, h=3.5)
                ax.fill_between(buckets, mins_, maxs_, alpha=0.15, color=_CYAN)
                ax.plot(buckets, means, color=_CYAN, linewidth=1.8, label="Mean confidence")
                if len(means) > 1:
                    ax.axhline(means[0], color=_MUTED, linestyle="--", linewidth=0.8, label="Baseline (bucket 0)")
                ax.set_ylim(0, 1)
                ax.set_xlabel(f"Bucket  (1 bucket = {bucket_size} events)", color=_TEXT)
                ax.set_ylabel("Confidence", color=_TEXT)
                ax.set_title(f"Detection Confidence Over Time - {selected_meta['display_name']}", color=_CYAN, fontsize=10, pad=10)
                ax.legend(fontsize=7.5, framealpha=0.15, labelcolor=_TEXT)
                fig.tight_layout(pad=1.5)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            # â”€â”€ Class distribution drift â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.markdown('<div class="section-title">Threat Class Distribution Shift</div>', unsafe_allow_html=True)
            cls_report = drift_data.get("class_drift", {})

            if cls_report.get("available"):
                p_val = cls_report.get("pvalue", 1.0)
                kl    = cls_report.get("kl_divergence", 0.0)
                cls_alert = cls_report.get("alert", "none")
                badge_map = {"critical": "badge-critical", "warning": "badge-warning", "none": "badge-safe"}

                ca1, ca2, ca3 = st.columns(3)
                with ca1:
                    st.markdown(
                        _card(f"{p_val:.5f}", "Chi-squared p-value",
                              "bad" if p_val < 0.01 else "warn" if p_val < 0.05 else "good"),
                        unsafe_allow_html=True,
                    )
                with ca2:
                    st.markdown(
                        _card(f"{kl:.5f}", "KL Divergence",
                              "bad" if kl > 0.5 else "warn" if kl > 0.1 else "good"),
                        unsafe_allow_html=True,
                    )
                with ca3:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div style="margin-top:8px;"><span class="badge {badge_map.get(cls_alert,"badge-none")}">'
                        f'{cls_alert.upper()}</span></div>'
                        f'<div class="lbl">Distribution Status</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("")
                ref_probs = cls_report["reference_probs"]
                obs_probs = cls_report["observed_probs"]
                x = np.arange(len(ref_probs))
                w = 0.4

                fig, ax = _dark_fig(w=14, h=4)
                ax.bar(x - w/2, ref_probs, w, label="Training reference", color=_CYAN,  alpha=0.7, edgecolor="none")
                ax.bar(x + w/2, obs_probs, w, label=f"Live window (n={wsize})", color=_AMBER, alpha=0.8, edgecolor="none")
                ax.set_xticks(x)
                ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=5.5)
                ax.set_ylabel("Proportion", color=_TEXT)
                ax.set_title(
                    f"Class Distribution - Training vs Live Window  |  "
                    f"chi2 p={p_val:.4f}  KL={kl:.4f}",
                    color=_CYAN, fontsize=9, pad=10,
                )
                ax.legend(fontsize=7.5, framealpha=0.15, labelcolor=_TEXT)
                fig.tight_layout(pad=1.5)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            # â”€â”€ Confidence drift detail â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            conf_report = drift_data.get("confidence_drift", {})
            if conf_report.get("available"):
                st.markdown('<div class="section-title">Confidence Drift Detail</div>', unsafe_allow_html=True)
                baseline = conf_report["baseline_mean_confidence"]
                win_mean = conf_report["window_mean_confidence"]
                rel_drop = conf_report["relative_drop"]
                c_alert  = conf_report["alert"]

                cd1, cd2, cd3 = st.columns(3)
                with cd1:
                    st.markdown(_card(f"{baseline:.4f}", "Baseline Mean Confidence", "cyan"), unsafe_allow_html=True)
                with cd2:
                    cd_cls = "bad" if c_alert == "critical" else "warn" if c_alert == "warning" else "good"
                    st.markdown(_card(f"{win_mean:.4f}", "Window Mean Confidence", cd_cls), unsafe_allow_html=True)
                with cd3:
                    drop_cls = "bad" if rel_drop > 0.20 else "warn" if rel_drop > 0.10 else "good"
                    st.markdown(_card(f"{rel_drop*100:.1f}%", "Relative Drop", drop_cls), unsafe_allow_html=True)

            # â”€â”€ Alert-rate spike â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ar = drift_data.get("alert_rate", {})
            if ar.get("available"):
                st.markdown('<div class="section-title">Alert Rate Spike Detection</div>', unsafe_allow_html=True)
                ar_alert    = ar["alert"]
                win_attack  = ar["window_attack_rate"]
                ref_attack  = ar["reference_attack_rate"]
                abs_rise    = ar["absolute_rise"]
                n_attacks   = ar["n_attacks"]
                n_benign_w  = ar["n_benign"]

                badge_map_ar = {"critical": "badge-critical", "warning": "badge-warning", "none": "badge-safe"}
                label_map_ar = {"critical": "SPIKE DETECTED", "warning": "ELEVATED", "none": "NOMINAL"}

                ar1, ar2, ar3, ar4 = st.columns(4)
                with ar1:
                    ar_cls = "bad" if ar_alert == "critical" else "warn" if ar_alert == "warning" else "good"
                    st.markdown(_card(f"{win_attack*100:.1f}%", "Window Attack Rate", ar_cls), unsafe_allow_html=True)
                with ar2:
                    st.markdown(_card(f"{ref_attack*100:.1f}%", "Baseline Attack Rate", "cyan"), unsafe_allow_html=True)
                with ar3:
                    rise_cls = "bad" if abs_rise > 0.20 else "warn" if abs_rise > 0.10 else "good"
                    st.markdown(_card(f"{abs_rise*100:+.1f}%", "Absolute Rise", rise_cls), unsafe_allow_html=True)
                with ar4:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div style="margin-top:8px;"><span class="badge {badge_map_ar.get(ar_alert,"badge-none")}">'
                        f'{label_map_ar.get(ar_alert,"UNKNOWN")}</span></div>'
                        f'<div class="lbl">Alert Rate Status</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Mini bar chart: benign vs attack in window
                fig, ax = _dark_fig(w=5, h=2.2)
                labels = ["Benign", "Attack"]
                counts = [n_benign_w, n_attacks]
                colors = [_GREEN, _RED]
                ax.bar(labels, counts, color=colors, edgecolor="none", width=0.4)
                ax.set_ylabel("Predictions", color=_TEXT)
                ax.set_title(f"Window Traffic Breakdown (n={wsize})", color=_CYAN, fontsize=9, pad=8)
                for i, v in enumerate(counts):
                    ax.text(i, v + max(counts) * 0.02, str(v), ha="center", fontsize=9, color=_TEXT)
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

    # â”€â”€ Feature PSI (static) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown('<hr style="border-color:#1a2744;margin:20px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live Feature PSI Analysis - Training vs Recent Traffic</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b;font-size:0.78rem;">'
        "Uses the selected rolling window, comparing training features against recently streamed traffic. &nbsp; "
        "Population Stability Index per feature: &nbsp;"
        '<span style="color:#10b981;">&lt; 0.10 = stable</span> &nbsp;|&nbsp; '
        '<span style="color:#f59e0b;">0.10-0.25 = slight drift</span> &nbsp;|&nbsp; '
        '<span style="color:#ef4444;">&gt; 0.25 = significant drift</span>'
        "</p>",
        unsafe_allow_html=True,
    )
    try:
        psi_data = fetch_json(
            f"{api_url}/drift/feature_analysis?model={selected_key}&window={window_size}&bins=10"
        )
    except Exception as e:
        psi_data = None
        st.warning(f"Feature PSI unavailable: {e}")

    if psi_data and psi_data.get("available"):
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(_card(str(psi_data["n_stable"]),   "Stable Features",   "good"), unsafe_allow_html=True)
        with p2:
            st.markdown(_card(str(psi_data["n_warning"]),  "Slight Drift",      "warn" if psi_data["n_warning"] else "good"), unsafe_allow_html=True)
        with p3:
            st.markdown(_card(str(psi_data["n_critical"]), "Significant Drift", "bad" if psi_data["n_critical"] else "good"), unsafe_allow_html=True)

        st.caption(psi_data.get("note", ""))

        # Bar chart: top-20 by PSI
        feat_df = pd.DataFrame(psi_data["features"])
        top20   = feat_df.head(20)
        psi_colors = [
            _RED if lvl == "critical" else _AMBER if lvl == "warning" else _GREEN
            for lvl in top20["level"]
        ]

        fig, ax = _dark_fig(w=10, h=5)
        ax.barh(top20["feature"][::-1], top20["psi"][::-1], color=psi_colors[::-1], edgecolor="none", height=0.72)
        ax.axvline(0.10, color=_AMBER, linestyle="--", linewidth=0.8, alpha=0.8, label="Warning (0.10)")
        ax.axvline(0.25, color=_RED,   linestyle="--", linewidth=0.8, alpha=0.8, label="Critical (0.25)")
        ax.set_xlabel("PSI Score", color=_TEXT)
        ax.set_title("Feature PSI - Top 20 by Drift Score", color=_CYAN, fontsize=10, pad=10)
        ax.legend(fontsize=7.5, framealpha=0.15, labelcolor=_TEXT)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        with st.expander("Full feature PSI table", expanded=False):
            def _highlight_psi(row):
                if row["level"] == "critical":
                    return ["background-color:#1c0808; color:#fca5a5"] * len(row)
                if row["level"] == "warning":
                    return ["background-color:#1c1208; color:#fcd34d"] * len(row)
                return ["background-color:#071c12; color:#6ee7b7"] * len(row)

            styled_psi = feat_df.style.apply(_highlight_psi, axis=1).format({"psi": "{:.5f}"})
            st.dataframe(styled_psi, width="stretch", height=500)
    elif psi_data:
        st.info(psi_data.get("message", "No live feature PSI data is available yet."))


