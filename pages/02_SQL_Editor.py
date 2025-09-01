# pages/02_SQL_Editor.py
# -----------------------------------------------------------------------------
# SQL Editor for local datasets
# - Discovers datasets under a chosen folder (defaults to data/curated)
# - Registers each file as a SQL table (DuckDB if available; fallback SQLite)
# - Table catalog, column browser, SQL runner, save & download results
# -----------------------------------------------------------------------------

import os
import re
import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Prefer DuckDB; fallback to SQLite if not installed
_ENGINE = "duckdb"
try:
    import duckdb  # type: ignore
except Exception:
    import sqlite3  # type: ignore
    _ENGINE = "sqlite3"

st.set_page_config(page_title="SQL Editor", layout="wide")

# ------------------------------- Utilities ---------------------------------- #

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def default_data_root() -> str:
    """Single source of truth for the app's data root."""
    return os.environ.get("SAVED_DATA_DIR", "data/curated")

def find_base_dir_candidates() -> List[str]:
    env_path = os.environ.get("SAVED_DATA_DIR", "").strip()
    cands: List[str] = []
    if env_path:
        cands.append(env_path)
    cands.extend([
        "./data/curated",         # <- prefer curated
        "./saved_datasets",
        "./data/saved",
        "./datasets",
        "./data",
    ])
    return cands

@st.cache_data(show_spinner=False)
def list_dataset_files(folder: str, include_subfolders: bool = True) -> List[str]:
    """Return dataset file paths (csv / parquet / xlsx) under a folder."""
    if not os.path.isdir(folder):
        return []
    exts = (".csv", ".parquet", ".xlsx")
    out: List[str] = []
    if include_subfolders:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.lower().endswith(exts):
                out.append(os.path.join(folder, f))
    out.sort()
    return out

def _safe_table_name(path: str) -> str:
    """Create a SQL-safe table name from filename (lowercase, [a-z0-9_])."""
    base = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    if not re.match(r"^[A-Za-z_]", name):
        name = "t_" + name
    return name.lower()

@st.cache_data(show_spinner=False)
def excel_sheet_names(path: str) -> List[str]:
    try:
        xls = pd.ExcelFile(path)
        return xls.sheet_names
    except Exception:
        return []

def read_full_dataframe(path: str, sheet: Optional[str]) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path)
    if low.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            st.error(f"Failed to read Parquet: {e}")
            raise
    if low.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError("Unsupported file: " + path)

def register_tables_from_files(
    files: List[str],
    excel_sheet_choice: Dict[str, Optional[str]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Load all datasets to pandas DataFrames and return:
      - table_name -> DataFrame
      - table_name -> source path
    """
    table_dfs: Dict[str, pd.DataFrame] = {}
    table_src: Dict[str, str] = {}
    name_counts: Dict[str, int] = {}

    for p in files:
        tname = _safe_table_name(p)
        # disambiguate duplicates
        cnt = name_counts.get(tname, 0)
        name_counts[tname] = cnt + 1
        if cnt > 0:
            tname = f"{tname}_{cnt+1}"

        sheet = excel_sheet_choice.get(p)
        df = read_full_dataframe(p, sheet=sheet)
        table_dfs[tname] = df
        table_src[tname] = p

    return table_dfs, table_src

def run_sql_duckdb(query: str, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        for name, df in tables.items():
            con.register(name, df)
        return con.execute(query).df()
    finally:
        con.close()

def run_sql_sqlite(query: str, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    con = sqlite3.connect(":memory:")
    try:
        for name, df in tables.items():
            df.to_sql(name, con, index=False, if_exists="replace")
        return pd.read_sql_query(query, con)
    finally:
        con.close()

def run_sql(query: str, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return run_sql_duckdb(query, tables) if _ENGINE == "duckdb" else run_sql_sqlite(query, tables)

def suggest_select_star(table: str, limit: int = 100) -> str:
    return f"SELECT * FROM {table} LIMIT {limit};"

def suggest_join_example(left: str, right: str, key: Optional[str] = None) -> str:
    k = key or "id"
    return f"""SELECT l.*, r.*
FROM {left} l
LEFT JOIN {right} r
  ON l.{k} = r.{k}
LIMIT 100;"""

# ------------------------------- UI: Header --------------------------------- #

st.title("SQL Editor")
st.caption(
    ("Engine: **DuckDB**" if _ENGINE == "duckdb" else "Engine: **SQLite (fallback)**")
    + " · Query CSV/Parquet/Excel files as tables. Joins supported."
)

# ------------------------------- Sidebar: Data ------------------------------- #

st.sidebar.header("Datasets")

# Prefer ./data/curated (or SAVED_DATA_DIR), else fallback to first existing candidate
preferred = default_data_root()
if os.path.isdir(preferred):
    default_base_dir = preferred
else:
    default_base_dir = None
    for c in find_base_dir_candidates():
        if os.path.isdir(c):
            default_base_dir = c
            break
if default_base_dir is None:
    default_base_dir = preferred  # create if missing
ensure_dir(default_base_dir)

base_dir = st.sidebar.text_input(
    "Saved data root folder",
    value=default_base_dir,
    help="Root folder that contains your saved/curated datasets.",
)

include_subfolders = st.sidebar.checkbox("Include subfolders", value=True)

# Gather files from the chosen folder...
files_primary = list_dataset_files(base_dir, include_subfolders=include_subfolders)
# ...and also ensure data/curated is included (de-duped)
curated_root = default_data_root()
files_curated = list_dataset_files(curated_root, include_subfolders=include_subfolders) if os.path.isdir(curated_root) else []
# Merge & de-duplicate (preserve order)
files = list(dict.fromkeys(files_primary + files_curated))

if not files:
    st.warning(
        f"No datasets found in: `{os.path.relpath(base_dir)}` or `{os.path.relpath(curated_root)}`. "
        "Place **CSV, Parquet, or Excel (.xlsx)** files there "
        f"{'(including subfolders)' if include_subfolders else ''}."
    )
    st.stop()

# Optional: per-Excel sheet picker (only shown for .xlsx)
st.sidebar.subheader("Excel Sheets")
excel_sheet_choice: Dict[str, Optional[str]] = {}
for p in files:
    if p.lower().endswith(".xlsx"):
        sheets = excel_sheet_names(p)
        if len(sheets) > 1:
            excel_sheet_choice[p] = st.sidebar.selectbox(
                f"Sheet for {os.path.basename(p)}",
                options=sheets,
                index=0,
                key=f"sheet::{p}",
            )
        else:
            excel_sheet_choice[p] = sheets[0] if sheets else None
    else:
        excel_sheet_choice[p] = None

# Load dataframes and register table mapping
with st.spinner("Loading datasets..."):
    tables_df, tables_src = register_tables_from_files(files, excel_sheet_choice)

# --------------------------- Catalog & Columns Pane -------------------------- #

left, right = st.columns([1, 2])

with left:
    st.subheader("Tables")
    table_names = sorted(tables_df.keys())
    st.dataframe(
        pd.DataFrame({
            "table": table_names,
            "rows": [len(tables_df[t]) for t in table_names],
            "cols": [len(tables_df[t].columns) for t in table_names],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Select a table to view columns**")
    selected_table = st.selectbox(
        "Table",
        options=["(choose)"] + table_names,
        index=0,
        key="table_select",
    )

    if selected_table != "(choose)":
        cols = list(tables_df[selected_table].columns)
        st.markdown(f"**Columns in `{selected_table}`** ({len(cols)}):")
        st.code(", ".join(cols), language="text")

        # Quick paste snippets
        st.markdown("**Quick SQL snippets**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("SELECT *", key="btn_select_star"):
                st.session_state["sql"] = suggest_select_star(selected_table)
        with c2:
            # Show a join suggestion if there is another table
            other = [t for t in table_names if t != selected_table]
            if other and st.button("JOIN example", key="btn_join_example"):
                st.session_state["sql"] = suggest_join_example(selected_table, other[0])

with right:
    st.subheader("SQL Editor")
    default_sql = st.session_state.get("sql") or (
        "/* Example\n"
        f"{suggest_select_star(sorted(tables_df.keys())[0])}\n"
        "*/"
    )
    sql = st.text_area(
        "Write SQL here",
        value=default_sql,
        height=220,
        key="sql",
        help=(
            "Use table names shown on the left. "
            "DuckDB syntax preferred; SQLite fallback supports most basics."
        ),
    )

    run = st.button("Run Query", type="primary")
    if run:
        try:
            with st.spinner("Running query..."):
                df_out = run_sql(sql, tables_df)

            st.success(f"Query returned {len(df_out):,} rows × {len(df_out.columns)} columns")
            st.dataframe(df_out.head(1000), use_container_width=True)

            # -------------------- Save / Download -------------------- #
            st.markdown("---")
            st.subheader("Save or Download Result")

            base_name = f"sql_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_name = st.text_input("Base filename (no extension)", value=base_name)

            # Save to a chosen folder (default to curated)
            default_target = curated_root if os.path.isdir(curated_root) else (os.path.dirname(files[0]) if files else base_dir)
            target_folder = st.text_input(
                "Save to folder",
                value=default_target,
                help="Choose any existing folder. Files saved as both CSV & Parquet."
            )
            ensure_dir(target_folder)

            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                if st.button("Save to CSV"):
                    csv_path = os.path.join(target_folder, out_name + ".csv")
                    df_out.to_csv(csv_path, index=False)
                    st.success(f"Saved CSV → `{os.path.relpath(csv_path)}`")

            with c2:
                # Parquet save (if pyarrow/fastparquet available)
                try:
                    import pyarrow  # noqa: F401
                    parquet_ok = True
                except Exception:
                    parquet_ok = False

                if st.button("Save to Parquet"):
                    if parquet_ok:
                        pq_path = os.path.join(target_folder, out_name + ".parquet")
                        df_out.to_parquet(pq_path, index=False)
                        st.success(f"Saved Parquet → `{os.path.relpath(pq_path)}`")
                    else:
                        st.error("Parquet requires `pyarrow` or `fastparquet` installed.")

            with c3:
                # Downloads (no disk write)
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name=out_name + ".csv",
                    mime="text/csv",
                )

                buf = io.BytesIO()
                try:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_out.to_excel(writer, sheet_name="result", index=False)
                    st.download_button(
                        "Download Excel",
                        data=buf.getvalue(),
                        file_name=out_name + ".xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception:
                    # If xlsxwriter/openpyxl not available, just skip Excel download
                    pass

            # Optional: update manifest in folder to help other pages discover this result
            try:
                manifest_path = os.path.join(target_folder, "datasets_index.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                else:
                    manifest = {"datasets": []}
                manifest["datasets"].append({
                    "name": out_name,
                    "path_csv": os.path.join(target_folder, out_name + ".csv"),
                    "path_parquet": os.path.join(target_folder, out_name + ".parquet"),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source": "SQL_Editor",
                    "module": "SQL_Editor",
                })
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                st.caption(f"Manifest updated: `{os.path.relpath(manifest_path)}`")
            except Exception:
                pass

        except Exception as e:
            st.error(f"Query failed: {e}")

# ----------------------- Helpful Hints & Quick Help -------------------------- #

with st.expander("Tips", expanded=False):
    st.markdown(
        """
- **Tables = Files**: Every CSV/Parquet/Excel becomes a table (name = filename, normalized).
- **Excel**: If a workbook has multiple sheets, pick the one you want in the sidebar.
- **DuckDB vs SQLite**:
  - DuckDB (preferred): better SQL features, Parquet native support, fast joins.
  - SQLite (fallback): widely available; works well for CSV/XLSX.
- **Snippets**: Use the buttons on the left to paste a `SELECT *` or a sample `JOIN`.
- **Save**: You can save results to disk (CSV/Parquet) or download (CSV/Excel).
        """
    )
