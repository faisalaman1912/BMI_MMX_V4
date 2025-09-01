# Home.py
import os
import pandas as pd
import streamlit as st

DATA_DIR = os.environ.get("SAVED_DATA_DIR", "data/curated")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Home", layout="wide")
st.title("🏠 Home")

st.caption(f"All modules read/write from **{DATA_DIR}**. Set env `SAVED_DATA_DIR` to override.")

# List files
exts = (".csv", ".xlsx", ".parquet")
files = []
for root, _, names in os.walk(DATA_DIR):
    for n in names:
        if n.lower().endswith(exts):
            files.append(os.path.join(root, n))
files.sort()

if not files:
    st.info(f"No datasets found in `{DATA_DIR}` yet. Go to **Ingestion & Curation** to upload.")
else:
    sel = st.selectbox("Select a dataset", files, index=0)
    low = sel.lower()

    if low.endswith(".csv"):
        df_head = pd.read_csv(sel, nrows=2000)
        st.write("**Columns:**")
        st.code(", ".join(map(str, df_head.columns)))
        with st.expander("Preview (top 15 rows)"):
            st.dataframe(df_head.head(15), use_container_width=True)

    elif low.endswith(".parquet"):
        try:
            df_head = pd.read_parquet(sel)
            st.write("**Columns:**")
            st.code(", ".join(map(str, df_head.columns)))
            with st.expander("Preview (top 15 rows)"):
                st.dataframe(df_head.head(15), use_container_width=True)
        except Exception as e:
            st.error(f"Parquet preview failed: {e}")

    else:  # .xlsx
        try:
            xl = pd.ExcelFile(sel)
            sheet = st.selectbox("Sheet", xl.sheet_names, index=0)
            df_head = xl.parse(sheet, nrows=2000)
            st.write("**Columns:**")
            st.code(", ".join(map(str, df_head.columns)))
            with st.expander("Preview (top 15 rows)"):
                st.dataframe(df_head.head(15), use_container_width=True)
        except Exception as e:
            st.error(f"Excel preview failed: {e}")
