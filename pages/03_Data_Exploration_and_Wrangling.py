import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Data Exploration & Wrangling", layout="wide")

st.title("Data Exploration & Wrangling")
st.write("""
Use this section to inspect the raw data, clean it, engineer features, and visualize distributions and relationships.
""")

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_excel(uploaded_file)
    return df

# Sidebar upload
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Summary
    st.subheader("Data Summary")
    buffer = df.describe(include='all', datetime_is_numeric=True).T
    buffer["missing"] = df.isna().sum()
    buffer["dtype"] = df.dtypes
    st.write(buffer)

    # Columns
    st.subheader("Select Columns for Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    category_cols = df.select_dtypes(include='object').columns.tolist()

    st.write("Numeric columns:", numeric_cols)
    st.write("Categorical columns:", category_cols)

    selected_num = st.multiselect("Choose numeric columns to visualize", numeric_cols)
    selected_cat = st.multiselect("Choose categorical columns to inspect", category_cols)

    # Numeric viz
    if selected_num:
        st.subheader("Numeric Features Distribution")
        for col in selected_num:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig, clear_figure=True)

        st.subheader("Scatter Plot")
        if len(selected_num) >= 2:
            c1, c2 = st.columns(2)
            x_sel = c1.selectbox("X-axis", selected_num, key="scatter_x")
            y_sel = c2.selectbox("Y-axis", selected_num, key="scatter_y")

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_sel, y=y_sel, ax=ax)
            ax.set_title(f"{x_sel} vs. {y_sel}")
            st.pyplot(fig, clear_figure=True)

    # Categorical viz
    if selected_cat:
        st.subheader("Categorical Feature Counts")
        for col in selected_cat:
            fig, ax = plt.subplots()
            df[col].value_counts().head(20).plot(kind='bar', ax=ax)
            ax.set_title(f"Top 20 categories of {col}")
            st.pyplot(fig, clear_figure=True)
