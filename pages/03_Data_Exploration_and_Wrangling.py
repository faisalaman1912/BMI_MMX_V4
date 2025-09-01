import os
from pathlib import Path
import io
import pandas as pd
import numpy as np
import streamlit as st
import datetime

# Attempt to import plotly but fail gracefully if not installed
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ------------------------------- #
# Paths
# ------------------------------- #
# Resolve BASE_DIR robustly even when Streamlit changes working dir
BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- #
# Helpers
# ------------------------------- #
def _list_datasets():
    files = []
    for fn in sorted(os.listdir(DATASETS_DIR)):
        full = DATASETS_DIR / fn
        if not full.is_file():
            continue
        ext = full.suffix.lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            continue
        files.append({"name": fn, "path": str(full), "type": ext.strip(".")})
    return files


def _read_dataset(path: str, ext: str) -> pd.DataFrame:
    if ext == "csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _save_dataset(df: pd.DataFrame, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return str(out_path)


def _df_info_to_string(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


# ------------------------------- #
# Streamlit UI
# ------------------------------- #
st.set_page_config(page_title="Data Exploration & Wrangling", layout="wide")
st.title("🧹 Data Exploration & Wrangling")

# Upload area (optional) so user can push new datasets into storage
with st.expander("Upload dataset (CSV or Excel)"):
    uploaded = st.file_uploader("Upload CSV / Excel to add to datasets", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
    if uploaded is not None:
        # Use original filename
        fname = uploaded.name
        out_path = DATASETS_DIR / fname
        # Save file to datasets dir
        with open(out_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved uploaded dataset to {out_path}")

# List available datasets
datasets = _list_datasets()
if not datasets:
    st.warning("No datasets available. Upload a dataset above or add files to storage/datasets.")
    st.stop()

sel_dataset = st.selectbox("Select dataset", options=[d["name"] for d in datasets])
# find dataset meta
dataset = next(d for d in datasets if d["name"] == sel_dataset)

# Session state: store loaded dataframe + history for undo
if "current_dataset_name" not in st.session_state or st.session_state.current_dataset_name != sel_dataset:
    st.session_state.current_dataset_name = sel_dataset
    st.session_state.df_original = _read_dataset(dataset["path"], dataset["type"])
    st.session_state.df = st.session_state.df_original.copy()
    st.session_state.history = []  # stack of previous df states (as copies)

# convenience
df = st.session_state.df

# Top row: quick stats
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)
with col2:
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
    st.write("---")
    st.write("**Last loaded:**", datetime.datetime.fromtimestamp(Path(dataset['path']).stat().st_mtime))

# ------------------------------- #
# Data Exploration
# ------------------------------- #
st.subheader("🔍 Data Exploration")

with st.expander("Dataset Info"):
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Data Types:")
    st.write(df.dtypes)
    st.text_area("info() output", value=_df_info_to_string(df), height=200)

with st.expander("Summary Statistics"):
    st.dataframe(df.describe(include='all').T, use_container_width=True)

with st.expander("Missing Values"):
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["% Missing"] = 100 * missing["Missing Count"] / len(df)
    st.dataframe(missing.sort_values("% Missing", ascending=False), use_container_width=True)
    if HAS_PLOTLY:
        fig = px.bar(missing.sort_values("% Missing", ascending=False), x="Column", y="% Missing", title="Missing Value % by Column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("plotly not available — install `plotly` to see interactive charts (add to requirements.txt)")

# Unique value explorer for categorical columns
with st.expander("Value counts (select a column)"):
    col_for_values = st.selectbox("Column", options=list(df.columns), key="values_col")
    topn = st.number_input("Top N", min_value=1, max_value=100, value=20)
    try:
        vc = df[col_for_values].value_counts(dropna=False).head(topn)
        st.dataframe(vc)
    except Exception as e:
        st.error(f"Cannot compute value counts: {e}")

# ------------------------------- #
# Data Wrangling
# ------------------------------- #
st.subheader("🛠️ Data Wrangling")

# helper to push state
def push_history():
    st.session_state.history.append(st.session_state.df.copy())

# Reset / undo
undo_col1, undo_col2 = st.columns([1, 1])
with undo_col1:
    if st.button("Undo last change"):
        if st.session_state.history:
            st.session_state.df = st.session_state.history.pop()
            st.success("Reverted last change")
        else:
            st.info("No history to undo")
with undo_col2:
    if st.button("Reset to original"):
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.history = []
        st.success("Reset dataset to original loaded file")

# Missing value handling
with st.expander("Handle missing values"):
    col_missing = st.selectbox("Select column to handle missing values", options=df.columns, key="missing_col")
    strategy = st.radio("Imputation Strategy", options=["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Custom Value"])
    custom_val = None
    if strategy == "Custom Value":
        custom_val = st.text_input("Enter custom value (will attempt to coerce to numeric if column numeric)")

    if st.button("Apply Missing Value Handling"):
        push_history()
        if strategy == "Drop Rows":
            st.session_state.df = st.session_state.df.dropna(subset=[col_missing])
        elif strategy == "Fill with Mean":
            if pd.api.types.is_numeric_dtype(st.session_state.df[col_missing]):
                st.session_state.df[col_missing] = st.session_state.df[col_missing].fillna(st.session_state.df[col_missing].mean())
            else:
                st.error("Mean fill requires numeric column")
        elif strategy == "Fill with Median":
            if pd.api.types.is_numeric_dtype(st.session_state.df[col_missing]):
                st.session_state.df[col_missing] = st.session_state.df[col_missing].fillna(st.session_state.df[col_missing].median())
            else:
                st.error("Median fill requires numeric column")
        elif strategy == "Fill with Mode":
            try:
                modev = st.session_state.df[col_missing].mode()[0]
                st.session_state.df[col_missing] = st.session_state.df[col_missing].fillna(modev)
            except Exception as e:
                st.error(f"Cannot fill with mode: {e}")
        elif strategy == "Custom Value":
            if pd.api.types.is_numeric_dtype(st.session_state.df[col_missing]):
                val = pd.to_numeric(custom_val, errors='coerce')
                st.session_state.df[col_missing] = st.session_state.df[col_missing].fillna(val)
            else:
                st.session_state.df[col_missing] = st.session_state.df[col_missing].fillna(custom_val)
        st.success(f"Applied {strategy} on {col_missing}")

# Column operations
with st.expander("Column operations"):
    col_ops = st.selectbox("Select operation", ["Rename Column", "Drop Column", "Create New Column"], key="col_ops")
    if col_ops == "Rename Column":
        col_to_rename = st.selectbox("Select column to rename", options=st.session_state.df.columns, key="rename_col")
        new_name = st.text_input("New column name")
        if st.button("Rename Column"):
            push_history()
            st.session_state.df = st.session_state.df.rename(columns={col_to_rename: new_name})
            st.success(f"Renamed {col_to_rename} → {new_name}")
    elif col_ops == "Drop Column":
        col_to_drop = st.multiselect("Select columns to drop", options=st.session_state.df.columns, key="drop_cols")
        if st.button("Drop Selected Columns"):
            push_history()
            st.session_state.df = st.session_state.df.drop(columns=col_to_drop)
            st.success(f"Dropped {len(col_to_drop)} column(s)")
    elif col_ops == "Create New Column":
        new_col_name = st.text_input("New column name", key="new_col_name")
        st.caption("Write an expression using column names (e.g. `A + B` or `` `col name` * 2 ``). Do NOT include `df[...]` — use bare column names.")
        formula = st.text_area("Enter a formula using pandas.eval syntax (e.g., `A + B`)", key="create_formula")
        if st.button("Create Column"):
            try:
                push_history()
                # Using DataFrame.eval — safer than eval
                expr = f"`{new_col_name}` = {formula}"
                st.session_state.df.eval(expr, inplace=True, engine='python')
                st.success(f"Created column {new_col_name}")
            except Exception as e:
                st.error(f"Error creating column: {e}")

# Create lag for target variable (useful for time series)
with st.expander("Target & lag (for modeling)"):
    maybe_time_cols = [c for c in st.session_state.df.columns if 'date' in c.lower() or 'time' in c.lower() or pd.api.types.is_datetime64_any_dtype(st.session_state.df[c])]
    target_col = st.selectbox("Select target variable (optional)", options=[None] + list(st.session_state.df.columns), index=0)
    create_lag = st.checkbox("Create lag for target")
    if create_lag and target_col:
        lag_period = st.number_input("Lag periods (integer)", min_value=1, value=1)
        time_col = st.selectbox("If you have a time column, select it (optional)", options=[None] + maybe_time_cols, index=0)
        if st.button("Apply Lag"):
            push_history()
            if time_col:
                # ensure time column is datetime
                try:
                    st.session_state.df[time_col] = pd.to_datetime(st.session_state.df[time_col])
                    st.session_state.df = st.session_state.df.sort_values(time_col)
                except Exception:
                    st.warning("Could not convert time column to datetime — lag will use current ordering")
            st.session_state.df[f"{target_col}_lag_{lag_period}"] = st.session_state.df[target_col].shift(lag_period)
            st.success(f"Created lag column {target_col}_lag_{lag_period}")

# ------------------------------- #
# Visualization quick tools
# ------------------------------- #
st.subheader("📈 Quick Visualizations")
vis_col1, vis_col2, vis_col3 = st.columns([1,1,1])
with vis_col1:
    num_cols = list(st.session_state.df.select_dtypes(include=['number']).columns)
    cat_cols = list(st.session_state.df.select_dtypes(include=['object', 'category']).columns)
    hist_col = st.selectbox("Histogram column (numeric)", options=[None] + num_cols, key="hist_col")
    if st.button("Plot Histogram"):
        if not hist_col:
            st.error("Pick a numeric column")
        else:
            if HAS_PLOTLY:
                fig = px.histogram(st.session_state.df, x=hist_col, nbins=50, title=f"Histogram of {hist_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(st.session_state.df[hist_col].dropna())
with vis_col2:
    box_col = st.selectbox("Boxplot column (numeric)", options=[None] + num_cols, key="box_col")
    if st.button("Plot Boxplot"):
        if not box_col:
            st.error("Pick a numeric column")
        else:
            if HAS_PLOTLY:
                fig = px.box(st.session_state.df, y=box_col, title=f"Boxplot of {box_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(st.session_state.df[box_col].describe())
with vis_col3:
    scatter_x = st.selectbox("Scatter X (numeric)", options=[None] + num_cols, key="scatter_x")
    scatter_y = st.selectbox("Scatter Y (numeric)", options=[None] + num_cols, key="scatter_y")
    if st.button("Plot Scatter"):
        if not scatter_x or not scatter_y:
            st.error("Pick two numeric columns")
        else:
            if HAS_PLOTLY:
                fig = px.scatter(st.session_state.df, x=scatter_x, y=scatter_y, title=f"Scatter: {scatter_x} vs {scatter_y}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(st.session_state.df[[scatter_x, scatter_y]].describe())

# Correlation
with st.expander("Correlation matrix"):
    corr_method = st.selectbox("Method", options=["pearson", "spearman", "kendall"], index=0)
    corr = st.session_state.df.corr(method=corr_method)
    st.dataframe(corr, use_container_width=True)
    if HAS_PLOTLY:
        try:
            fig = px.imshow(corr, text_auto=True, title=f"Correlation ({corr_method})")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Could not render correlation heatmap with plotly")

# ------------------------------- #
# Save Wrangled Dataset
# ------------------------------- #
st.subheader("💾 Save Processed Dataset")
save_name = st.text_input("Enter filename to save (e.g., wrangled_data.csv)", value=f"wrangled_{sel_dataset}")
if st.button("Save Dataset"):
    out_path = DATASETS_DIR / save_name
    if out_path.exists():
        if not st.checkbox("Overwrite existing file?", key="overwrite_confirm"):
            st.warning("File exists — check the box to confirm overwrite")
        else:
            path = _save_dataset(st.session_state.df, out_path)
            st.success(f"Dataset saved at {path}")
    else:
        path = _save_dataset(st.session_state.df, out_path)
        st.success(f"Dataset saved at {path}")
    # update datasets list in memory
    st.experimental_rerun()

# Provide immediate download
with st.expander("Download current dataframe"):
    csv_bytes = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv_bytes, file_name=f"{Path(sel_dataset).stem}_wrangled_{int(datetime.datetime.now().timestamp())}.csv", mime="text/csv")

# Integration with Modeling page: store selected features
with st.expander("Modeling export (quick)"):
    model_name = st.text_input("Model name (optional)")
    indep_vars = st.multiselect("Select independent variables", options=list(st.session_state.df.columns))
    if st.button("Export selection to Modeling"):
        # store in a simple json-like memory (session_state) for modeling page to pick up
        st.session_state.modeling = {
            "model_name": model_name,
            "dataset": sel_dataset,
            "independent_vars": indep_vars,
            "target": target_col
        }
        st.success("Saved modeling metadata to session state. Go to the Modeling page to continue.")

# show final preview and update session
st.subheader("✅ Current dataset (final preview)")
st.dataframe(st.session_state.df.head(50), use_container_width=True)

# persist current df in session state (already is)


# End of file
