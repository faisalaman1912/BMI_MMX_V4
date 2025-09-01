import os
import time
import pandas as pd
import numpy as np
import streamlit as st
pip install plotly
import plotly.express as px

# ------------------------------- #
# Paths
# ------------------------------- #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")

os.makedirs(DATASETS_DIR, exist_ok=True)

# ------------------------------- #
# Helpers
# ------------------------------- #
def _list_datasets():
    files = []
    for fn in os.listdir(DATASETS_DIR):
        full = os.path.join(DATASETS_DIR, fn)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".csv", ".xlsx"):
            continue
        files.append({"name": fn, "path": full, "type": ext.strip(".")})
    files.sort(key=lambda x: x["name"])
    return files

def _read_dataset(path: str, ext: str) -> pd.DataFrame:
    if ext == "csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

def _save_dataset(df: pd.DataFrame, name: str):
    out_path = os.path.join(DATASETS_DIR, name)
    df.to_csv(out_path, index=False)
    return out_path

# ------------------------------- #
# Streamlit UI
# ------------------------------- #
st.set_page_config(page_title="Data Exploration & Wrangling", layout="wide")
st.title("🧹 Data Exploration & Wrangling")

datasets = _list_datasets()
if not datasets:
    st.warning("No datasets available. Please upload data in **Ingestion & Curation Desk** first.")
    st.stop()

sel_dataset = st.selectbox("Select dataset", options=[d["name"] for d in datasets])
dataset = next(d for d in datasets if d["name"] == sel_dataset)
df = _read_dataset(dataset["path"], dataset["type"])

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

# ------------------------------- #
# Data Exploration
# ------------------------------- #
st.subheader("🔍 Data Exploration")

with st.expander("Dataset Info"):
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Data Types:")
    st.write(df.dtypes)

with st.expander("Summary Statistics"):
    st.dataframe(df.describe(include="all").T, use_container_width=True)

with st.expander("Missing Values"):
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["% Missing"] = 100 * missing["Missing Count"] / len(df)
    st.dataframe(missing, use_container_width=True)

    fig = px.bar(missing, x="Column", y="% Missing", title="Missing Value % by Column")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------- #
# Data Wrangling
# ------------------------------- #
st.subheader("🛠️ Data Wrangling")

# Handle missing values
col_missing = st.selectbox("Select column to handle missing values", options=df.columns)
strategy = st.radio("Imputation Strategy", options=["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Custom Value"])
if strategy == "Custom Value":
    custom_val = st.text_input("Enter custom value")

if st.button("Apply Missing Value Handling"):
    if strategy == "Drop Rows":
        df = df.dropna(subset=[col_missing])
    elif strategy == "Fill with Mean":
        if pd.api.types.is_numeric_dtype(df[col_missing]):
            df[col_missing] = df[col_missing].fillna(df[col_missing].mean())
    elif strategy == "Fill with Median":
        if pd.api.types.is_numeric_dtype(df[col_missing]):
            df[col_missing] = df[col_missing].fillna(df[col_missing].median())
    elif strategy == "Fill with Mode":
        df[col_missing] = df[col_missing].fillna(df[col_missing].mode()[0])
    elif strategy == "Custom Value":
        df[col_missing] = df[col_missing].fillna(custom_val)

    st.success(f"Applied {strategy} on {col_missing}")

# Column operations
st.subheader("🔧 Column Operations")
col_ops = st.selectbox("Select operation", ["Rename Column", "Drop Column", "Create New Column"])
if col_ops == "Rename Column":
    col_to_rename = st.selectbox("Select column to rename", options=df.columns)
    new_name = st.text_input("New column name")
    if st.button("Rename Column"):
        df = df.rename(columns={col_to_rename: new_name})
        st.success(f"Renamed {col_to_rename} → {new_name}")

elif col_ops == "Drop Column":
    col_to_drop = st.multiselect("Select columns to drop", options=df.columns)
    if st.button("Drop Selected Columns"):
        df = df.drop(columns=col_to_drop)
        st.success(f"Dropped {len(col_to_drop)} column(s)")

elif col_ops == "Create New Column":
    new_col_name = st.text_input("New column name")
    formula = st.text_area("Enter a formula using pandas syntax (e.g., `df['A'] + df['B']`)")
    if st.button("Create Column"):
        try:
            df[new_col_name] = pd.eval(formula)
            st.success(f"Created column {new_col_name}")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------------------- #
# Save Wrangled Dataset
# ------------------------------- #
st.subheader("💾 Save Processed Dataset")
save_name = st.text_input("Enter filename to save (e.g., wrangled_data.csv)", value=f"wrangled_{sel_dataset}")
if st.button("Save Dataset"):
    path = _save_dataset(df, save_name)
    st.success(f"Dataset saved at {path}")
    st.write("Now available in **Ingestion & Curation Desk** and **Modeling**")
