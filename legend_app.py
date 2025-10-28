from lmfit import Model

# ---------- 6. Define models ----------
def fourPL(x, A, B, EC50, n):
    return A + (B - A) / (1 + np.exp(n * (x - EC50)))

def fivePL(x, A, B, EC50, n, s):
    return A + (B - A) / ((1 + np.exp(n * (x - EC50))) ** s)

# ---------- 7. Fit curves & plot ----------
fit_results, plots = {}, []
for a in analytes:
    try:
        x = np.log10(reps[a].astype(float))
        y = np.log10(std_rows[a].astype(float))
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if fit_type == "4PL":
            model = Model(fourPL)
            params = model.make_params(A=min(y), B=max(y), EC50=np.median(x), n=-1)
            params["A"].min = 0
            params["n"].max = -1e-3
        else:
            model = Model(fivePL)
            params = model.make_params(A=min(y), B=max(y), EC50=np.median(x), n=-1, s=1)
            params["A"].min = 0
            params["n"].max = -1e-3
            params["s"].min = 0.5
            params["s"].max = 2

        result = model.fit(y, x=x, params=params)
        y_fit = result.best_fit
        r2 = 1 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)

        fit_results[a] = {"result": result, "r2": r2}

        # Plot standards
        x_fit = np.linspace(min(x), max(x), 100)
        y_curve = model.eval(x=x_fit, **result.best_values)
        fig = px.scatter(x=x, y=y, color_discrete_sequence=["red"],
                         title=f"{a} — {fit_type} Fit (R²={r2:.3f})",
                         labels={"x": "log10(Concentration)", "y": "log10(MFI)"})
        fig.add_scatter(x=x_fit, y=y_curve, mode="lines", name="Fit", line=dict(color="orange"))

        # Overlay sample MFIs
        sample_y = np.log10(samples[a].astype(float))
        sample_mask = np.isfinite(sample_y)
        fig.add_scatter(
            x=[None]*len(sample_y[sample_mask]),  # placeholder for legend
            y=[None]*len(sample_y[sample_mask]),
            mode="markers",
            name="Samples (overlay)",
            marker=dict(color="blue", size=6),
        )

        # actually overlay each sample on same MFI axis at arbitrary x
        # since sample concs unknown before interpolation, just stack them
        for val in sample_y[sample_mask]:
            fig.add_scatter(
                x=[max(x_fit) + 0.02],  # small offset to the right
                y=[val],
                mode="markers",
                marker=dict(color="blue", size=6),
                showlegend=False
            )

        if r2 < 0.95:
            st.warning(f"⚠️ {a}: R²={r2:.3f}, curve may be poor.")
        plots.append(fig)
    except Exception as e:
        st.warning(f"⚠️ Fit failed for {a}: {e}")

st.success("✅ Curve fitting complete.")
for fig in plots:
    st.plotly_chart(fig, use_container_width=True)

# ---------- 8. Interpolate samples ----------
samples = df.loc[~df["ID"].str.contains("Standard", case=False, na=False)].copy()
for a in analytes:
    if a not in fit_results:
        continue
    result = fit_results[a]["result"]
    popt = result.best_values
    y = np.log10(samples[a].astype(float))
    mask = np.isfinite(y)
    try:
        if fit_type == "4PL":
            A, B, EC50, n = popt["A"], popt["B"], popt["EC50"], popt["n"]
            x_pred = EC50 + (1 / n) * np.log((B - A)/(y[mask] - A) - 1)
        else:
            A, B, EC50, n, s = popt["A"], popt["B"], popt["EC50"], popt["n"], popt["s"]
            x_pred = EC50 + (1 / n) * np.log(((B - A)**(1/s))/((y[mask] - A)**(1/s)) - 1)
        samples.loc[mask, f"{a}_conc_pgml"] = 10 ** x_pred * samples["ID"].map(sample_dilutions)
    except Exception:
        samples[f"{a}_conc_pgml"] = np.nan

# ---------- 9. Export ----------
st.markdown("### Export results")
output_name = st.text_input("Output file name (no extension):", value="LegendPlex_results")

# QC summary table
qc_summary = pd.DataFrame([
    {"Analyte": a, "R²": fit_results[a]["r2"],
     "Fit_Status": "OK" if fit_results[a]["r2"] >= 0.95 else "Poor"}
    for a in fit_results
])
st.dataframe(qc_summary.style.background_gradient(cmap="RdYlGn", subset=["R²"]))

# build long + wide exports
long_records = []
for _, row in samples.iterrows():
    sample_id = row["ID"]
    dilution = sample_dilutions.get(sample_id, 1)
    for a in analytes:
        mfi = row[a]
        conc = row.get(f"{a}_conc_pgml", np.nan)
        long_records.append({
            "ID": sample_id,
            "Analyte": a,
            "MFI": mfi,
            "Concentration_pg_mL": conc,
            "Dilution_Factor": dilution
        })
long_df = pd.DataFrame(long_records)
existing_cols = [col for col in samples.columns if col.endswith("_conc_pgml")]
wide_df = samples[["ID"] + existing_cols]

# write Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    long_df.to_excel(writer, index=False, sheet_name="Long_Format")
    wide_df.to_excel(writer, index=False, sheet_name="Wide_Format")
    qc_summary.to_excel(writer, index=False, sheet_name="QC_Summary")
output.seek(0)

st.download_button(
    label="⬇️ Download .xlsx file (with QC summary)",
    data=output,
    file_name=f"{output_name}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
