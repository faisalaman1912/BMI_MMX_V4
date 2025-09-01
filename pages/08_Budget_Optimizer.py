
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Budget Optimizer", layout="wide")
st.title("📊 Budget Optimizer")

MODEL_DIR = "storage/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# Load available models
# ---------------------------------------------------
saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

if not saved_models:
    st.warning("No saved models found. Please build a model first.")
    st.stop()

selected_model = st.selectbox("Select a model", saved_models)

# (stub: load model metadata)
try:
    # Replace with joblib.load in real app
    model_channels = ["TV", "Digital", "Print", "OOH"]  # dummy
except Exception as e:
    st.error("⚠️ Error – Could not load model. Please reach out to BlueMatter.")
    st.stop()

# ---------------------------------------------------
# Optimization Scenario
# ---------------------------------------------------
scenario = st.radio(
    "Select Optimization Scenario",
    ["Profit Maximization", "Blue Sky (MROI=1)", "Historically Optimized", "Budget Based Optimization"],
)

st.write(f"### Selected Scenario: {scenario}")

# ---------------------------------------------------
# Constraint Inputs
# ---------------------------------------------------
constraints = {}
if scenario in ["Profit Maximization", "Blue Sky (MROI=1)", "Budget Based Optimization"]:
    st.subheader("Constraints (Optional)")
    for ch in model_channels:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            freeze = st.checkbox(f"Freeze {ch}", key=f"freeze_{ch}")
        with col2:
            min_val = st.number_input(f"{ch} Min", value=0.0, key=f"min_{ch}")
        with col3:
            max_val = st.number_input(f"{ch} Max", value=100.0, key=f"max_{ch}")

        constraints[ch] = {
            "freeze": freeze,
            "min": min_val,
            "max": max_val,
        }

if scenario == "Budget Based Optimization":
    total_budget = st.number_input("Enter total budget", min_value=0.0, value=100.0)

# ---------------------------------------------------
# Run Optimization
# ---------------------------------------------------
if st.button("Run Optimization"):
    try:
        # Dummy setup (in practice use model coefficients)
        n = len(model_channels)
        response = np.random.rand(n) * 10  # dummy coefficients
        cost = np.ones(n)

        # Objective: maximize response → minimize -response
        c = -response

        # Bounds
        bounds = []
        for ch in model_channels:
            if constraints[ch]["freeze"]:
                bounds.append((constraints[ch]["min"], constraints[ch]["min"]))
            else:
                bounds.append((constraints[ch]["min"], constraints[ch]["max"]))

        # Budget constraint (only for budget-based scenario)
        A_eq, b_eq = None, None
        if scenario == "Budget Based Optimization":
            A_eq = [cost]
            b_eq = [total_budget]

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if result.success:
            allocation = dict(zip(model_channels, result.x.round(2)))
            st.success("✅ Optimization Completed")
            st.json(allocation)
        else:
            st.error("⚠️ Error – Optimization did not converge. Please reach out to BlueMatter.")
    except Exception as e:
        st.error("⚠️ Error – This functionality is still building. Please reach out to BlueMatter for next steps.")
