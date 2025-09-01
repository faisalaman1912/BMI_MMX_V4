# tabs/_08_results.py
file_name="impactables.csv", mime="text/csv")


st.divider()


# Response curve
st.subheader("Response Curve")
channel_feats = [f for f in model.features if f in model.channel_map]
if not channel_feats:
st.info("No channel features with transforms found in this model.")
return


ch_feat = st.selectbox("Channel", channel_feats)
c1, c2, c3 = st.columns(3)
with c1:
spend_min = st.number_input("Min spend (raw)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
with c2:
spend_max = st.number_input("Max spend (raw)", min_value=0.0, value=1_000_000.0, step=1000.0, format="%.2f")
with c3:
steps = st.slider("Steps", min_value=10, max_value=80, value=30)


if st.button("Generate Response"):
try:
rc = response_curve(model, ch_feat, spend_min, spend_max, steps)
fig2 = px.line(rc, x="spend", y="y_pred", title=f"Predicted {model.target} vs Spend ({ch_feat})")
st.plotly_chart(fig2, use_container_width=True)


fig3 = px.line(rc, x="spend", y="channel_contribution", title=f"{ch_feat} Contribution vs Spend")
st.plotly_chart(fig3, use_container_width=True)


png2 = _fig_to_png(fig2)
png3 = _fig_to_png(fig3)
cdl1, cdl2, cdl3 = st.columns(3)
with cdl1:
if png2: st.download_button("Download curve PNG", png2, "response_curve.png", "image/png")
with cdl2:
if png3: st.download_button("Download contrib PNG", png3, "contribution_curve.png", "image/png")
with cdl3:
st.download_button("Download curve data (CSV)", rc.to_csv(index=False).encode(), "response_curve.csv", "text/csv")
except Exception:
st.error("Error — This functionality is still building. Please reach out to BlueMatter for next steps.")


st.divider()


# Clustered comparison across models
st.subheader("Compare Models — Impactable Shares (Clustered)")
comp_rows = []
for nm in compare_models:
row = df_show[df_show["name"] == nm]
if row.empty:
continue
m = SavedModel.from_json(load_model_json(row.iloc[0]["path"]))
ct = contribution_table_from_feature_means(m)
ct = ct[ct["feature"] != "intercept_base"].copy()
ct["share_pct_pos"] = 100.0 * ct["contribution"].clip(lower=0) / max(1e-9, ct["contribution"].clip(lower=0).sum())
ct["model"] = nm
comp_rows.append(ct[["model","feature","share_pct_pos"]])
if comp_rows:
comp_df = pd.concat(comp_rows, ignore_index=True)
fig4 = px.bar(comp_df, x="feature", y="share_pct_pos", color="model", barmode="group", title="Impactable Share by Model (Clustered)")
st.plotly_chart(fig4, use_container_width=True)
png4 = _fig_to_png(fig4)
if png4:
st.download_button("Download clustered PNG", png4, "impactables_clustered.png", "image/png")
st.download_button("Download clustered CSV", comp_df.to_csv(index=False).encode(), "impactables_clustered.csv", "text/csv")
else:
st.info("Select at least one model to compare.")


except Exception:
st.error("Error — This functionality is still building. Please reach out to BlueMatter for next steps.")
