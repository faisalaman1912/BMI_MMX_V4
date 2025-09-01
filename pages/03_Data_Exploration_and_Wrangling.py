import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from pathlib import Path

# ------------------------------- #
# Paths
# ------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
CURATED_DIR = BASE_DIR / "data" / "curated"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
CURATED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- #
# Helpers
# ------------------------------- #
def _list_datasets(directory: Path):
    files = []
    for fn in sorted(os.listdir(directory)):
        full = directory / fn
        if not full.is_file():
            continue
        ext = full.suffix.lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            continue
        files.append({"name": fn, "path": str(full), "type": ext.strip(".")})
    return files

@st.cache_data
def _load_dataset(path, ftype):
    if ftype == "csv":
        return pd.read_csv(path)
    elif ftype in ("xlsx", "xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ftype}")

# ------------------------------- #
# Streamlit App
# ------------------------------- #
st.title("📊 Data Exploration and Wrangling")

# Upload dataset
st.subheader("Upload Dataset")
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if file:
    path = DATASETS_DIR / file.name
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    st.success(f"Uploaded {file.name} → {path}")

# List available datasets (raw + curated)
datasets = _list_datasets(DATASETS_DIR) + _list_datasets(CURATED_DIR)
if not datasets:
    st.warning("No datasets available. Upload a dataset above or add files to storage/datasets or data/curated.")
    st.stop()

sel_dataset = st.selectbox("Select a dataset", [d["name"] for d in datasets])
sel = next(d for d in datasets if d["name"] == sel_dataset)

# Load dataset
df = _load_dataset(sel["path"], sel["type"])

# Keep copy in session_state
if "original" not in st.session_state:
    st.session_state.original = df.copy()
if "working" not in st.session_state:
    st.session_state.working = df.copy()
if "history" not in st.session_state:
    st.session_state.history = []

df = st.session_state.working

st.write("### Preview")
st.dataframe(df.head())

# ------------------------------- #
# Wrangling options
# ------------------------------- #
st.sidebar.header("Wrangling Options")

# Drop columns
cols = st.sidebar.multiselect("Drop columns", df.columns)
if cols and st.sidebar.button("Apply Drop"):
    st.session_state.history.append(df.copy())
    st.session_state.working = df.drop(columns=cols)
    st.experimental_rerun()

# Rename column
rename_col = st.sidebar.selectbox("Rename column", [None] + list(df.columns))
if rename_col:
    new_name = st.sidebar.text_input("New name", rename_col)
    if st.sidebar.button("Apply Rename"):
        st.session_state.history.append(df.copy())
        st.session_state.working = df.rename(columns={rename_col: new_name})
        st.experimental_rerun()

# Create new column
new_col = st.sidebar.text_input("New column name")
expr = st.sidebar.text_input("Formula (e.g., col1 + col2)")
if new_col and expr and st.sidebar.button("Create Column"):
    try:
        st.session_state.history.append(df.copy())
        st.session_state.working[new_col] = df.eval(expr)
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# Handle missing values
st.sidebar.subheader("Missing Values")
method = st.sidebar.selectbox("Fill method", [None, "drop", "mean", "median", "mode", "custom"])
if method:
    if method == "custom":
        fill_val = st.sidebar.text_input("Fill value")
    if st.sidebar.button("Apply Fill"):
        st.session_state.history.append(df.copy())
        if method == "drop":
            st.session_state.working = df.dropna()
        elif method == "mean":
            st.session_state.working = df.fillna(df.mean(numeric_only=True))
        elif method == "median":
            st.session_state.working = df.fillna(df.median(numeric_only=True))
        elif method == "mode":
            st.session_state.working = df.fillna(df.mode().iloc[0])
        elif method == "custom":
            val = pd.to_numeric(fill_val, errors="ignore")
            st.session_state.working = df.fillna(val)
        st.experimental_rerun()

# Reset / Undo
if st.sidebar.button("Reset to Original"):
    st.session_state.working = st.session_state.original.copy()
    st.session_state.history = []
    st.experimental_rerun()
if st.sidebar.button("Undo Last") and st.session_state.history:
    st.session_state.working = st.session_state.history.pop()
    st.experimental_rerun()

# ------------------------------- #
# EDA
# ------------------------------- #
st.subheader("Exploration")

with st.expander("Info"):
    buf = []
    df.info(buf=buf.append)
    st.text("\n".join(buf))

with st.expander("Summary Stats"):
    st.write(df.describe(include="all"))

with st.expander("Missing Values"):
    st.write(df.isna().sum())

with st.expander("Value Counts"):
    col = st.selectbox("Column", df.columns)
    st.write(df[col].value_counts())

# ------------------------------- #
# Visualizations
# ------------------------------- #
st.subheader("Visualizations")

chart = st.selectbox("Chart Type", ["Histogram", "Boxplot", "Scatter", "Correlation Heatmap"])
if chart == "Histogram":
    col = st.selectbox("Column", df.columns)
    fig = px.histogram(df, x=col)
    st.plotly_chart(fig, use_container_width=True)
elif chart == "Boxplot":
    col = st.selectbox("Column", df.columns)
    fig = px.box(df, y=col)
    st.plotly_chart(fig, use_container_width=True)
elif chart == "Scatter":
    x = st.selectbox("X-axis", df.columns)
    y = st.selectbox("Y-axis", df.columns)
    fig = px.scatter(df, x=x, y=y)
    st.plotly_chart(fig, use_container_width=True)
elif chart == "Correlation Heatmap":
    corr = df.corr(numeric_only=True)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        colorscale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------- #
# Save wrangled dataset
# ------------------------------- #
st.subheader("Save Processed Dataset")
save_name = st.text_input("Enter filename", value=f"wrangled_{sel_dataset}")
if st.button("Save Dataset"):
    out_path = CURATED_DIR / save_name
    if out_path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    st.success(f"Saved to {out_path}")
    st.download_button(
        label="Download now",
        data=open(out_path, "rb").read(),
        file_name=out_path.name
    )
