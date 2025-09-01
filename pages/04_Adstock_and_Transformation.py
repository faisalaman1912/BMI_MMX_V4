# 05_Adstock_and_Transformation.py
# Streamlit page: Adstock and Transformation
# -----------------------------------------------------------------------------
# Features
# - Pick dataset from the latest "saved" folder (auto-detected, with manual override)
# - Select ID, Time, Dependent Variable, and metrics to transform
# - Configure per-metric: Lag (step 1), Adstock (0..1 step 0.05), Transform (Log/NegExp),
#   Curvature C (with suggested range), Scaling (None/MinMax/ZScore)
# - Apply pipeline (Lag -> Adstock -> Transform -> Scale) within each ID group
# - Append transformed columns to original data
# - Save CSV + Parquet into the same folder for downstream modules
# -----------------------------------------------------------------------------

import os
import re
import json
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------- Utilities -------------------------------- #

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_base_dir_candidates() -> List[str]:
    """
    Returns sensible default search paths for saved datasets.
    The first existing directory is used; otherwise we create 'saved_datasets/'.
    """
    env_path = os.environ.get("SAVED_DATA_DIR", "").strip()
    candidates = []
    if env_path:
        candidates.append(env_path)

    candidates.extend([
        "./saved_datasets",
        "./data/saved",
        "./data/curated",
        "./datasets",
        "./data",
    ])
    return candidates


@st.cache_data(show_spinner=False)
def list_subdirs_sorted(base_dir: str) -> List[str]:
    """List subdirectories in base_dir sorted by modified time DESC."""
    if not os.path.isdir(base_dir):
        return []
    subs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            subs.append((full, os.path.getmtime(full)))
    subs.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in subs]


@st.cache_data(show_spinner=False)
def list_datasets(folder: str) -> List[str]:
    """List dataset files (csv/parquet) in a folder."""
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith((".csv", ".parquet")):
            files.append(os.path.join(folder, name))
    files.sort()
    return files


@st.cache_data(show_spinner=False)
def read_dataset(path: str, nrows_preview: Optional[int] = None) -> pd.DataFrame:
    """Read CSV/Parquet with optional preview row limit."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, nrows=nrows_preview)
    elif path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")


def _try_to_datetime(series: pd.Series) -> pd.Series:
    """Attempt to parse a wide range of time formats to datetime."""
    s = series.astype(str).str.strip()

    # 1) General inference
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt.notna().all():
        return dt

    # 2) Explicit common patterns
    patterns = [
        "%b-%y",      # Jan-24
        "%b-%Y",      # Jan-2024
        "%Y-%b",      # 2024-Jan
        "%d-%b-%y",   # 05-Jan-24
        "%d-%b-%Y",   # 05-Jan-2024
        "%Y-%m-%d",   # 2024-01-05
        "%d-%m-%Y",   # 05-01-2024
        "%m/%d/%Y",   # 01/05/2024
        "%d/%m/%Y",   # 05/01/2024
    ]
    for fmt in patterns:
        dt2 = pd.to_datetime(s, format=fmt, errors="coerce")
        if dt2.notna().all():
            return dt2

    # 3) Quarter strings like "Q1-2024" or "q2 2023"
    # Convert to end-of-quarter timestamp
    qmatch = s.str.extract(r'(?i)\bQ([1-4])\s*[-_/ ]\s*(\d{2,4})\b')
    if qmatch.notna().all().all():
        q = qmatch.iloc[:, 0].astype(int)
        y = qmatch.iloc[:, 1].astype(int)
        y = np.where(y < 100, 2000 + y, y)
        per = pd.PeriodIndex(year=y, quarter=q, freq="Q")
        return per.asfreq("Q").to_timestamp(how="end")

    # 4) Fallback to NaT
    return pd.to_datetime(s, errors="coerce")


def order_by_time(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Adds an internal sortable datetime column and returns (sorted_df, dt_series).
    """
    dt = _try_to_datetime(df[time_col])
    if dt.isna().any():
        st.warning(
            f"⚠️ Some values in '{time_col}' could not be parsed as dates. "
            "Those rows will be placed at the end for ordering."
        )
        # Place NaTs at the end by filling with max + 1 day
        safe_dt = dt.fillna(dt.max() + pd.Timedelta(days=1))
    else:
        safe_dt = dt
    df_sorted = df.copy()
    df_sorted["___dt__"] = safe_dt
    df_sorted = df_sorted.sort_values(["__id__", "_\__dt__".replace("\\", "")])  # placeholder, will be re-sorted later
    return df_sorted, safe_dt


def apply_adstock_series(x: pd.Series, adstock: float) -> pd.Series:
    """
    Apply classic geometric adstock within a single group's time order:
      y[t] = x[t] + adstock * y[t-1]
    """
    if len(x) == 0:
        return x
    y = np.zeros(len(x), dtype=float)
    xv = x.fillna(0.0).astype(float).to_numpy()
    y[0] = xv[0]
    for t in range(1, len(xv)):
        y[t] = xv[t] + adstock * y[t - 1]
    return pd.Series(y, index=x.index)


def transform_series(x: pd.Series, transform: str, C: float) -> pd.Series:
    """
    Apply saturation transform:
      - "Log":   log(1 + C * max(x,0))
      - "NegExp": 1 - exp(-C * max(x,0))
    """
    xv = x.fillna(0.0).astype(float)
    xv = xv.clip(lower=0.0)  # ensure non-negative for saturations
    C = float(C)
    if transform == "Log":
        return np.log1p(C * xv)
    else:  # NegExp
        return 1.0 - np.exp(-C * xv)


def scale_series(x: pd.Series, mode: str) -> pd.Series:
    xv = x.astype(float)
    if mode == "MinMax":
        mn, mx = float(xv.min()), float(xv.max())
        rng = mx - mn
        if rng == 0:
            return pd.Series(0.0, index=x.index)
        return (xv - mn) / rng
    elif mode == "ZScore":
        mu, sd = float(xv.mean()), float(xv.std(ddof=0))
        if sd == 0:
            return pd.Series(0.0, index=x.index)
        return (xv - mu) / sd
    else:
        return xv


def suggest_C_bounds(series: pd.Series) -> Tuple[float, float]:
    """Suggest C in [1/mean, 4/mean] (guarding for zeros)."""
    mean_val = float(series.replace([np.inf, -np.inf], np.nan).dropna().mean())
    if mean_val <= 0 or np.isnan(mean_val):
        # Fallback suggestion
        return (0.001, 0.010)
    lo = 1.0 / mean_val
    hi = 4.0 / mean_val
    return (lo, hi)


def make_output_name(metric: str, lag: int, adstock: float, transform: str, C: float, scaling: str) -> str:
    safe = lambda s: re.sub(r"[^A-Za-z0-9_]+", "_", str(s))
    ad = f"{adstock:.2f}".replace(".", "p")
    cstr = f"{C:.6g}".replace(".", "p")
    parts = [metric, f"lag{lag}", f"ads{ad}", transform, f"C{cstr}"]
    if scaling and scaling != "None":
        parts.append(scaling)
    return "__".join(safe(p) for p in parts)


# ------------------------------- Page Layout -------------------------------- #

st.set_page_config(page_title="Adstock & Transformation", layout="wide")

st.title("Adstock and Transformation")
st.caption(
    "Configure lag, adstock, saturation transform, and scaling per metric. "
    "Pipeline: **Lag → Adstock → Transform → Scale** (within each ID group)."
)

with st.expander("How it works (formulas)", expanded=False):
    st.markdown(
        """
- **Adstock:**  \n
  \( y_t = x_t + \alpha \cdot y_{t-1} \), where \( \alpha \in [0,1] \) is the adstock (carryover) rate.

- **Log transform:**  \n
  \( f(x) = \log(1 + C \cdot \max(x, 0)) \)

- **Negative Exponential:**  \n
  \( f(x) = 1 - e^{-C \cdot \max(x, 0)} \)

- **Scaling:**  \n
  MinMax: \( (x - \min)/( \max - \min) \)  \n
  ZScore: \( (x - \mu)/\sigma \)
        """
    )

# -------------------------- Folder & File Selection ------------------------- #

# Choose base saved dir
default_base_dir = None
for cand in find_base_dir_candidates():
    if os.path.isdir(cand):
        default_base_dir = cand
        break
if default_base_dir is None:
    default_base_dir = "./saved_datasets"
    ensure_dir(default_base_dir)

st.sidebar.header("Storage")
base_dir = st.sidebar.text_input(
    "Saved data root folder",
    value=default_base_dir,
    help="Root folder that contains dated refresh subfolders.",
)

subdirs = list_subdirs_sorted(base_dir)
auto_latest = st.sidebar.checkbox("Auto-pick latest subfolder", value=True)
if subdirs:
    default_folder = subdirs[0] if auto_latest else subdirs[0]
    folder = st.sidebar.selectbox(
        "Select refresh subfolder",
        options=subdirs,
        index=0,
        format_func=lambda p: os.path.relpath(p, base_dir),
        help="The latest (most recently modified) folder is shown first.",
    )
else:
    st.sidebar.info("No subfolders found. Using root folder.")
    folder = base_dir

datasets = list_datasets(folder)
if not datasets:
    st.warning(
        f"No datasets found in: `{os.path.relpath(folder)}`. "
        "Place CSV or Parquet files there."
    )
    st.stop()

dataset_path = st.selectbox(
    "Select dataset file",
    options=datasets,
    index=0,
    format_func=lambda p: os.path.basename(p),
)

# Load full dataset (no preview truncation so transforms are accurate)
df = read_dataset(dataset_path, nrows_preview=None)

if df.empty:
    st.error("Selected dataset is empty.")
    st.stop()

st.success(f"Loaded: **{os.path.basename(dataset_path)}**  ·  {df.shape[0]:,} rows × {df.shape[1]:,} cols")

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ---------------------------- Column Selections ----------------------------- #

all_cols = list(df.columns)

# Heuristics for defaults
time_guess = None
for guess in ["Month", "month", "Date", "date", "Period", "period", "Time", "time", "Quarter", "quarter"]:
    if guess in df.columns:
        time_guess = guess
        break

id_col = st.selectbox("ID column", options=all_cols, index=0)
time_col = st.selectbox("Time period column", options=all_cols, index=(all_cols.index(time_guess) if time_guess in all_cols else 0))

# Create a temp sort key per ID for stable ordering
df = df.copy()
df["__id__"] = df[id_col].astype(str)

# Derive time ordering
dt_series = _try_to_datetime(df[time_col])
if dt_series.isna().all():
    st.warning(f"Couldn't fully parse '{time_col}' as dates. Ordering may be imperfect.")
safe_dt = dt_series.fillna(dt_series.max() + pd.Timedelta(days=1))
df["__dt__"] = safe_dt

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude ID/time from dependent and metrics candidates
dependent_candidates = [c for c in numeric_cols if c not in [id_col, time_col]]

dep_col = st.selectbox(
    "Dependent variable (Y)",
    options=dependent_candidates if dependent_candidates else all_cols,
    index=0 if dependent_candidates else 0,
    help="Used for reference; not modified on this page."
)

metric_candidates = [c for c in numeric_cols if c not in [dep_col]]
metric_candidates = [c for c in metric_candidates if c not in [id_col, time_col]]

selected_metrics = st.multiselect(
    "Metrics to transform",
    options=metric_candidates,
    default=metric_candidates[: min(6, len(metric_candidates))],
)

if not selected_metrics:
    st.info("Select one or more metric columns to configure transformations.")
    st.stop()

# Lag bound: based on distinct time periods (global)
distinct_periods = int(df["__dt__"].nunique())
max_lag_allowed = max(0, distinct_periods - 1)
st.caption(f"Detected **{distinct_periods}** distinct time periods → max allowable lag = **{max_lag_allowed}**.")

# Build editable config table
def init_config_df(metrics: List[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        loC, hiC = suggest_C_bounds(df[m])
        defaultC = loC  # start at lower bound (1/mean)
        rows.append({
            "metric": m,
            "lag": 0,
            "adstock": 0.50,
            "transform": "NegExp",
            "C": round(float(defaultC), 6),
            "C_hint": f"{loC:.6g} .. {hiC:.6g}",
            "scaling": "None",
            "output_name": make_output_name(m, 0, 0.5, "NegExp", defaultC, "None"),
        })
    return pd.DataFrame(rows)

cfg_key = f"adstock_cfg::{os.path.basename(dataset_path)}"
if cfg_key not in st.session_state:
    st.session_state[cfg_key] = init_config_df(selected_metrics)
else:
    # Keep config aligned with current selection
    existing = st.session_state[cfg_key]
    # Drop rows for unselected metrics
    existing = existing[existing["metric"].isin(selected_metrics)].copy()
    # Add rows for newly selected metrics
    for m in selected_metrics:
        if m not in existing["metric"].tolist():
            loC, hiC = suggest_C_bounds(df[m])
            defaultC = loC
            new_row = {
                "metric": m,
                "lag": 0,
                "adstock": 0.50,
                "transform": "NegExp",
                "C": round(float(defaultC), 6),
                "C_hint": f"{loC:.6g} .. {hiC:.6g}",
                "scaling": "None",
                "output_name": make_output_name(m, 0, 0.5, "NegExp", defaultC, "None"),
            }
            existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    st.session_state[cfg_key] = existing

cfg_df = st.session_state[cfg_key]

st.markdown("#### Configure per-metric parameters")
edited_cfg = st.data_editor(
    cfg_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "metric": st.column_config.TextColumn("metric", disabled=True),
        "lag": st.column_config.NumberColumn(
            "lag (periods)",
            min_value=0, max_value=max_lag_allowed, step=1, help="Shift right by this many periods before adstock."
        ),
        "adstock": st.column_config.NumberColumn(
            "adstock (0–1)",
            min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
            help="Carryover rate α in y[t] = x[t] + α·y[t-1]"
        ),
        "transform": st.column_config.SelectboxColumn(
            "transform", options=["Log", "NegExp"], help="Saturation function"
        ),
        "C": st.column_config.NumberColumn(
            "C (curvature)",
            min_value=0.0, step=0.0001, format="%.6f",
            help="Suggested between lower/upper bounds shown"
        ),
        "C_hint": st.column_config.TextColumn("C suggestion (1/mean .. 4/mean)", disabled=True),
        "scaling": st.column_config.SelectboxColumn(
            "scaling", options=["None", "MinMax", "ZScore"], help="Final re-scaling"
        ),
        "output_name": st.column_config.TextColumn(
            "output column name",
            help="Optional custom name. Leave as-is to auto-name."
        ),
    },
    key=f"editor::{cfg_key}",
)

# Auto-refresh output names if parameters changed
def refresh_output_names(df_cfg: pd.DataFrame) -> pd.DataFrame:
    df_cfg = df_cfg.copy()
    for i, row in df_cfg.iterrows():
        # If user left default-ish name or blank, update; else respect their custom name
        default_name = make_output_name(row["metric"], int(row["lag"]), float(row["adstock"]),
                                        row["transform"], float(row["C"]), row["scaling"])
        current = str(row.get("output_name", "")).strip()
        if (not current) or (current.startswith(row["metric"] + "__")):
            df_cfg.at[i, "output_name"] = default_name
    return df_cfg

edited_cfg = refresh_output_names(edited_cfg)
st.session_state[cfg_key] = edited_cfg  # persist

# ------------------------------- Apply Changes ------------------------------ #

st.markdown("---")
st.subheader("Apply & Save")

default_new_name = os.path.splitext(os.path.basename(dataset_path))[0] + "__adstocked_transformed"
suggested_file_base = f"{default_new_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
new_filename = st.text_input(
    "Base filename for new dataset (no extension)",
    value=suggested_file_base,
    help="Will save both .csv and .parquet with this base name in the selected folder."
)

go = st.button("Apply transformations and create new dataset", type="primary")

if go:
    # Work on a copy, sorted by ID + time
    work = df.copy()
    work = work.sort_values(["__id__", "__dt__"], kind="mergesort")

    out_cols = []
    problems = []

    for _, row in edited_cfg.iterrows():
        metric = str(row["metric"])
        lag = int(row["lag"])
        ad = float(row["adstock"])
        transform = str(row["transform"])
        C = float(row["C"])
        scaling = str(row["scaling"])
        out_name = str(row["output_name"]).strip() or make_output_name(metric, lag, ad, transform, C, scaling)

        if metric not in work.columns:
            problems.append(f"Metric '{metric}' not found.")
            continue

        # Pipeline within each ID group in time order
        def pipeline(group: pd.DataFrame) -> pd.Series:
            x = group[metric]
            if lag > 0:
                x = x.shift(lag)
            y = apply_adstock_series(x, ad)
            y = transform_series(y, transform, C)
            y = scale_series(y, scaling)
            return y

        series = work.groupby("__id__", sort=False, group_keys=False).apply(pipeline)
        work[out_name] = series
        out_cols.append(out_name)

    if problems:
        st.error("Some issues occurred:\n- " + "\n- ".join(problems))
        st.stop()

    # Remove helper cols
    result = work.drop(columns=["__id__", "__dt__"], errors="ignore")

    st.success(f"Created {len(out_cols)} transformed columns: {', '.join(out_cols)}")

    with st.expander("Preview of appended columns (top 20 rows)", expanded=False):
        st.dataframe(result[[time_col, id_col] + out_cols].head(20), use_container_width=True)

    # Save artifacts
    base = os.path.join(folder, new_filename)
    csv_path = base + ".csv"
    parquet_path = base + ".parquet"

    try:
        result.to_csv(csv_path, index=False)
        result.to_parquet(parquet_path, index=False)
    except Exception as e:
        st.error(f"Failed to save files: {e}")
        st.stop()

    st.success(f"Saved:\n- `{os.path.relpath(csv_path)}`\n- `{os.path.relpath(parquet_path)}`")

    st.download_button("Download CSV", data=result.to_csv(index=False).encode("utf-8"),
                       file_name=os.path.basename(csv_path), mime="text/csv")
    try:
        import io
        import pyarrow as _pa  # noqa: F401
        import pyarrow.parquet as _papq  # noqa: F401
        # If pyarrow exists, offer direct parquet download as well
        buf = io.BytesIO()
        result.to_parquet(buf, index=False)
        st.download_button("Download Parquet", data=buf.getvalue(),
                           file_name=os.path.basename(parquet_path),
                           mime="application/octet-stream")
    except Exception:
        pass

    # Update a lightweight index manifest (optional, used by other modules if desired)
    manifest_path = os.path.join(folder, "datasets_index.json")
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {"datasets": []}
        manifest["datasets"].append({
            "name": os.path.basename(new_filename),
            "path_csv": csv_path,
            "path_parquet": parquet_path,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": os.path.basename(dataset_path),
            "module": "Adstock_and_Transformation",
            "columns_appended": out_cols,
            "id_col": id_col,
            "time_col": time_col,
            "dependent_var": dep_col,
        })
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        st.info(f"Updated manifest: `{os.path.relpath(manifest_path)}`")
    except Exception as e:
        st.warning(f"Could not update manifest: {e}")

# ------------------------------- End of Page -------------------------------- #
