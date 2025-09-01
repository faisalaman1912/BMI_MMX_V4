# Home.py
import os
import time
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Config paths (consistent with other modules)
# -----------------------------------------------------------------------------
DATA_DIR = os.environ.get("SAVED_DATA_DIR", "data/curated")
MODELS_DIR = os.environ.get("SAVED_MODELS_DIR", "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

st.set_page_config(page_title="Home", layout="wide")

# -----------------------------------------------------------------------------
# Styles: pearl-white banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .hero {
        background: #fafaf7; /* pearl-ish white */
        border: 1px solid #eee;
        border-radius: 16px;
        padding: 32px 16px;
        text-align: center;
        margin-bottom: 18px;
      }
      .hero h1 {
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.2px;
      }
      .meta {
        color: #666;
        font-size: 0.92rem;
      }
      .smallcaps {
        font-variant: all-small-caps;
        letter-spacing: 0.6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero"><h1>Marketing Mix Modelling by BlueMatter</h1></div>', unsafe_allow_html=True)
st.caption(
    f"Data root: `{os.path.abspath(DATA_DIR)}` · Models root: `{os.path.abspath(MODELS_DIR)}` "
    "· Set env vars `SAVED_DATA_DIR` / `SAVED_MODELS_DIR` to override."
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(max(0, n))
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:,.0f} {u}"
        s /= 1024.0

def _list_recent_files(root: str, exts: Optional[Tuple[str, ...]], limit: int = 5, include_subfolders: bool = True) -> List[Dict]:
    """
    List most recent files under root. If exts is None, include all.
    Returns list of dicts with name, path, size, ts (mtime), type, mtime_str.
    """
    if not os.path.isdir(root):
        return []
    out: List[Dict] = []
    if include_subfolders:
        walker = os.walk(root)
        for dirpath, _dirs, files in walker:
            for fn in files:
                if exts and not fn.lower().endswith(exts):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    stt = os.stat(full)
                except Exception:
                    continue
                ext = os.path.splitext(fn)[1].lower().strip(".")
                out.append({
                    "name": fn,
                    "path": full,
                    "size": _human_size(stt.st_size),
                    "ts": stt.st_mtime,
                    "type": ext.upper() if ext else "",
                    "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stt.st_mtime)),
                })
    else:
        try:
            for fn in os.listdir(root):
                if exts and not fn.lower().endswith(exts):
                    continue
                full = os.path.join(root, fn)
                if not os.path.isfile(full):
                    continue
                try:
                    stt = os.stat(full)
                except Exception:
                    continue
                ext = os.path.splitext(fn)[1].lower().strip(".")
                out.append({
                    "name": fn,
                    "path": full,
                    "size": _human_size(stt.st_size),
                    "ts": stt.st_mtime,
                    "type": ext.upper() if ext else "",
                    "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stt.st_mtime)),
                })
        except Exception:
            pass

    out.sort(key=lambda d: d["ts"], reverse=True)
    return out[:limit]

# -----------------------------------------------------------------------------
# Layout: left (datasets) | right (models)
# -----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Recent uploads")
    # Same extensions as your ingestion/editor pages
    dataset_exts = (".csv", ".xlsx", ".parquet")
    recent_ds = _list_recent_files(DATA_DIR, exts=dataset_exts, limit=5, include_subfolders=True)

    if not recent_ds:
        st.info("No datasets found yet. Go to **Ingestion & Curation Desk** to upload.")
    else:
        df = pd.DataFrame([
            {"File": d["name"], "Type": d["type"], "Size": d["size"], "Uploaded": d["mtime_str"]}
            for d in recent_ds
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Latest saved models")
    # Common “model” extensions; adjust as needed for your stack
    model_exts = (".pkl", ".pickle", ".joblib", ".sav", ".pt", ".pth", ".onnx", ".json")
    recent_models = _list_recent_files(MODELS_DIR, exts=model_exts, limit=5, include_subfolders=True)

    if not recent_models:
        st.info("No models found yet. This panel will list your most recently saved model files.")
    else:
        dfm = pd.DataFrame([
            {"Model file": m["name"], "Type": m["type"], "Size": m["size"], "Saved": m["mtime_str"]}
            for m in recent_models
        ])
        st.dataframe(dfm, use_container_width=True, hide_index=True)

# Optional: quick hints
with st.expander("Where do these come from?", expanded=False):
    st.markdown(
        f"""
- **Uploads**: Any **CSV / XLSX / Parquet** saved under `{DATA_DIR}` (including subfolders).
- **Models**: Any of {", ".join(model_exts)} under `{MODELS_DIR}` (including subfolders).
- You can change these folders with environment variables:
  - `SAVED_DATA_DIR` → datasets
  - `SAVED_MODELS_DIR` → models
        """
    )
