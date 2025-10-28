import streamlit as st
import pandas as pd
import numpy as np
import re
from lmfit import Model
import plotly.express as px
import io

st.title("LegendPlex Analyzer — Interactive 4PL/5PL (Python version)")

# ---------- 1. Upload & clean ----------
uploaded = st.file_uploader("Upload FlowJo export (.csv or .xlsx)", type=["csv", "xlsx"])

def load_file(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

def clean_colnames(df):
    df.columns = df.columns.astype(str)
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)
    def extract_marker(c):
        m = re.search(r".*?/(.*?)\s*\|", c)
        return m.group(1).strip() if m else c
    df.columns = [extract_marker(c) for c in df.columns]
    return df

if uploaded:
    df = clean_colnames(load_file(uploaded))
    st.success("✅ Columns cleaned successfully.")
    st.dataframe(df.head())

    # ---------- 2. Detect standards ----------
    std_rows = df[df["ID"].str.contains("Standard", case=False, na=False)]
    if std_rows.empty:
        st.error("⚠️ No standards detected. Make sure IDs contain 'Standard'.")
        st.stop()
    st.info(f"Detected {len(std_rows)} standards.")

    # ---------- 3. User inputs ----------
    dilution_factor = st.number_input("Serial dilution factor (e.g., 3)", min_value=1.0, value=3.0)
    analytes = [c for c in df.columns if c not in ["ID", "WELL ID"]]
    st.markdown("### Enter top standard concentrations (from CoA, ng/mL)")
    top_conc = {a: st.number_input(f"{a}:", min_value=0.001, value=10.0) for a in analytes}

    st.markdown("### Assign dilution factors to sample IDs (optional)")
    sample_ids = df.loc[~df["ID"].str.contains("Standard", case=False, na=False), "ID"].unique()
    sample_dilutions = {sid: st.number_input(f"{sid} dilution factor:", min_value=1.0, value=1.0)
                        for sid in sample_ids}

    fit_type = st.radio("Choose curve model:", ["4PL", "5PL"])
    proceed = st.button("Run analysis")

    if proceed:
        # ---------- 4. Prepare standard conc table ----------
        conc_pg = {a: [top_conc[a] * 1000] for a in analytes}
        for _ in range(1, len(std_rows)):
            for a in analytes:
                conc_pg[a].append(conc_pg[a][-1] / dilution_factor)
        reps = pd.DataFrame(conc_pg)
        reps.insert(0, "ID", std_rows["ID"].values)

        # ---------- 5. QC ----------
        if df[analytes].isnull().any().any():
            st.error("Missing or invalid MFI values detected.")
            st.stop()
        st.success("✅ Standard MFI values OK.")

        # ---------- 6. Define models ----------
        def fourPL(x, A, B, EC50, n):
            return A + (B - A) / (1 + np.exp(n * (x - EC50)))

        def fivePL(x, A, B, EC50, n, s):
            return A + (B - A) / ((1 + np.exp(n * (x - EC50))) ** s)

        # ---------- 7. Fit curves & plot ----------
        fit_results, plots = {}, []
        samples = df.loc[~df["ID"].str.contains("Standard", case=False, na=False)].copy()

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

                # Plot
                x_fit = np.linspace(min(x), max(x), 100)
                y_curve = model.eval(x=x_fit, **result.best_values)
                fig = px.scatter(x=x, y=y, color_discrete_sequence=["red"],
                                 title=f"{a} — {fit_type} Fit (R²={r2:.3f})",
                                 labels={"x": "log10(Concentration)", "y": "log10(MFI)"})
                fig.add_scatter(x=x_fit, y=y_curve, mode="lines", name="Fit", line=dict(color="orange"))

                # Interpolate samples and overlay
                y_samples = np.log10(samples[a].astype(float))
                mask_s = np.isfinite(y_samples)
                try:
                    popt = result.best_values
                    if fit_type == "4PL":
                        A, B, EC50, n = popt["A"], popt["B"], popt["EC50"], popt["n"]
                        x_pred = EC50 + (1 / n) * np.log((B - A)/(y_samples[mask_s] - A) - 1)
                    else:
                        A, B, EC50, n, s = popt["A"], popt["B"], popt["EC50"], popt["n"], popt["s"]
                        x_pred = EC50 + (1 / n) * np.log(((B - A)**(1/s))/((y_samples[mask_s] - A)**(1/s)) - 1)
                    conc_pred = 10 ** x_pred
                    samples.loc[mask_s, f"{a}_conc_pgml"] = conc_pred * samples["ID"].map(sample_dilutions)
                    # Overlay sample points
                    fig.add_scatter(x=np.log10(conc_pred), y=y_samples[mask_s],
                                    mode="markers", name="Samples", marker=dict(color="blue", size=6))
                except Exception:
                    samples[f"{a}_conc_pgml"] = np.nan

                if r2 < 0.95:
                    st.warning(f"⚠️ {a}: R²={r2:.3f}, curve may be poor.")
                plots.append(fig)
            except Exception as e:
                st.warning(f"⚠️ Fit failed for {a}: {e}")

        st.success("✅ Curve fitting complete.")
        for fig in plots:
            st.plotly_chart(fig, use_container_width=True)

        # ---------- 8. Export ----------
        st.markdown("### Export results")
        output_name = st.text_input("Output file name (no extension):", value="LegendPlex_results")

        qc_summary = pd.DataFrame([
            {"Analyte": a, "R²": fit_results[a]["r2"],
             "Fit_Status": "OK" if fit_results[a]["r2"] >= 0.95 else "Poor"}
            for a in fit_results
        ])
        st.dataframe(qc_summary.style.background_gradient(cmap="RdYlGn", subset=["R²"]))

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
else:
    st.info("Please upload a FlowJo export to begin.")
