import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Results", layout="wide")
st.title("📈 Model Results Dashboard")

MODEL_DIR = "storage/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# Load available models
# ---------------------------------------------------
saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".json")]

if not saved_models:
    st.warning("No saved models found. Please build a model first.")
    st.stop()

selected_model = st.selectbox("Select a model", saved_models)

# ---------------------------------------------------
# Load model data (stub structure)
# ---------------------------------------------------
try:
    with open(os.path.join(MODEL_DIR, selected_model), "r") as f:
        model_data = json.load(f)
    impacts = pd.DataFrame(model_data["impacts"])  # {channel, impact}
    response_curves = pd.DataFrame(model_data["curves"])  # {channel, spend, response}
except Exception as e:
    st.error("⚠️ Error – Could not load model data. Please reach out to BlueMatter.")
    st.stop()

# ---------------------------------------------------
# Visualization 1: % Impact Pie Chart
# ---------------------------------------------------
st.subheader("Channel % Impact Distribution")

fig_pie = px.pie(
    impacts,
    names="channel",
    values="impact",
    title="Channel Contribution (%)",
    hole=0.3,
)
st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------
# Visualization 2: 100% Stacked Bar
# ---------------------------------------------------
st.subheader("Impact Distribution (100% Stacked Bar)")

impacts["percent"] = impacts["impact"] / impacts["impact"].sum() * 100
fig_stack = px.bar(
    impacts,
    x=["Total"],
    y="percent",
    color="channel",
    title="Channel Impact (%)",
    text=impacts["percent"].round(1).astype(str) + "%",
)
fig_stack.update_layout(barmode="stack", xaxis_title=None, yaxis_title="Percent (%)")
st.plotly_chart(fig_stack, use_container_width=True)

# ---------------------------------------------------
# Visualization 3: Response Curves
# ---------------------------------------------------
st.subheader("Response Curves")

channels_to_plot = st.multiselect(
    "Select channel(s) to view response curve",
    options=response_curves["channel"].unique(),
    default=[response_curves["channel"].unique()[0]],
)

if channels_to_plot:
    curve_data = response_curves[response_curves["channel"].isin(channels_to_plot)]
    fig_curve = px.line(
        curve_data,
        x="spend",
        y="response",
        color="channel",
        title="Response Curves",
        markers=True,
    )
    st.plotly_chart(fig_curve, use_container_width=True)

# ---------------------------------------------------
# Visualization 4: Clustered Column Chart
# ---------------------------------------------------
st.subheader("Channel Impact Comparison")

fig_cluster = px.bar(
    impacts,
    x="channel",
    y="impact",
    color="channel",
    title="Channel Impactable (Absolute)",
    barmode="group",
    text=impacts["impact"].round(2),
)
st.plotly_chart(fig_cluster, use_container_width=True)

st.success("✅ Results visualization completed!")
