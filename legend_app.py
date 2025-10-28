import streamlit as st
import pandas as pd
import numpy as np
import re
from lmfit import Model
import plotly.express as px
import io

st.title("LegendPlex Analyzer — R-equivalent 4PL/5PL Fitting (Expert Version)")

# ---------- 1. Upload & clean ----------
uploaded = st.file_uploader("Upload FlowJo export (.csv or .xlsx)", type=["csv", "xlsx"])

def load_file(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

def clean_colnames(df):
    df.columns = df.columns.astype(str)
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)
    def extract_marker(c):
        m = re.search(r".*?/(.*?)\\s*\\|", c)
        return m.group(1).strip() if m else c
    df.columns = [extract_marker(c) for c in df.columns]
    return df

if uploaded:
    df = clean_colnames(load_file(uploaded))
    st.success("✅ Columns cleaned successfully.")
    st.dataframe(df.head())

    # ---------- 2. Detect standards ----------
    st.markdown("### Step 2: Detect Standards")

    std_candidates = df[df["ID"].str.contains("Standard", case=False, na=False)]
    if std_candidates.empty:
        st.warning("⚠️ No 'Standard' found in ID column. Select wells manually.")
        all_wells = df["ID"].tolist()
        selected_wells = st.multiselect("Select wells containing standards", options=all_wells)
        std_rows = df[df["ID"].isin(selected_wells)].copy()
    else:
        std_rows = std_candidates.copy()

    order_option = st.radio(
        "Order of standards in file:",
        ["Top row = highest concentration", "Top row = lowest concentration (blank first)"]
    )

    std_rows = std_rows.sort_values(
        by="ID",
        key=lambda x: pd.to_numeric(x.str.extract(r"(\\d+)")[0], errors="coerce")
    )
    if order_option == "Top row = lowest concentration (blank first)":
        std_rows = std_rows.iloc[::-1].reset_index(drop=True)

    samples = df.loc[~df.index.isin(std_rows.index)].copy()
    st.info(f"Detected {len(std_rows)} standards and {len(samples)} samples.")

    # ---------- 3. User inputs ----------
    dilution_factor = st.number_input("Serial dilution factor (e.g., 3)", min_value=1.0, value=3.0)
    analytes = [c for c in df.columns if c not in ["ID", "WELL ID"]]
    st.markdown("### Enter top standard concentrations (from CoA, ng/mL)")
    top_conc = {a: st.number_input(f"{a}:", min_value=0.001, value=10.0) for a in analytes}

    st.markdown("### Assign dilution factors to sample IDs (optional)")
    sample_ids = samples["ID"].unique()
    sample_dilutions = {sid: st.number_input(f"{sid} dilution factor:", min_value=1.0, value=1.0)
                        for sid in sample_ids}
    apply_sample_dilution = st.checkbox("Apply dilution factors to final concentrations", value=False)
    fit_type = st.radio("Choose curve model:", ["4PL", "5PL"])
    proceed = st.button("Run analysis")

    if proceed:
        # ---------- 4. Prepare standard conc table ----------
        std_n = len(std_rows)
        powers = np.arange(std_n, dtype=float)
        conc_pg = {a: (top_conc[a] * 1000) / (dilution_factor ** powers) for a in analytes}
        reps = pd.DataFrame(conc_pg)
        reps.insert(0, "ID", std_rows["ID"].values)
        st.write("### Standard concentrations (pg/mL)")
        st.dataframe(reps)

        if df[analytes].isnull().any().any():
            st.error("Missing or invalid MFI values detected.")
            st.stop()
        st.success("✅ Standard MFI values OK.")

        # ---------- 5. Define models ----------
        def fourPL(x, deltaA, log10EC50, n, Amin):
            x = np.clip(x, 1e-6, np.inf)  # avoid log10(0)
            return Amin + (deltaA / (1 + np.exp(n * (np.log10(x) - log10EC50))))

        def fivePL(x, A, B, EC50, n, s):
            x = np.clip(x, 1e-6, np.inf)
            return A + (B - A) / ((1 + np.exp(n * (np.log10(x) - EC50))) ** s)

        # ---------- 6. Fit curves & plot ----------
        fit_results, plots = {}, []

        for a in analytes:
            try:
                x = reps[a].astype(float)  # raw conc
                y = np.log10(std_rows[a].astype(float))  # log10(MFI)
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]

                Amin_fixed = np.percentile(y, 5)
                deltaA_start = np.percentile(y, 95) - Amin_fixed
                log10EC50_start = np.median(np.log10(x))
                n_start = -1

                if fit_type == "4PL":
                    model = Model(fourPL)
                    params = model.make_params(
                        deltaA=deltaA_start,
                        log10EC50=log10EC50_start,
                        n=n_start,
                        Amin=Amin_fixed
                    )
                    params["Amin"].vary = False
                    params["deltaA"].min = 0
                    params["deltaA"].max = 1e6
                    params["log10EC50"].min = min(np.log10(x))
                    params["log10EC50"].max = max(np.log10(x))
                    params["n"].min = -10
                    params["n"].max = -0.001
                else:
                    model = Model(fivePL)
                    params = model.make_params(
                        A=min(y),
                        B=max(y),
                        EC50=np.median(np.log10(x)),
                        n=-1,
                        s=1
                    )
                    params["A"].min = 0
                    params["B"].max = 10
                    params["n"].max = -1e-3
                    params["s"].min = 0.5
                    params["s"].max = 2

                result = model.fit(y, x=x, params=params, method="least_squares", max_nfev=20000)
                y_fit = result.best_fit
                r2 = 1 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)
                fit_results[a] = {"result": result, "r2": r2, "Amin": Amin_fixed}

                # Plot standards
                x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
                y_curve = model.eval(x=x_fit, **result.best_values)
                fig = px.scatter(
                    x=np.log10(x), y=y, color_discrete_sequence=["red"],
                    title=f"{a} — {fit_type} Fit (R²={r2:.3f})",
                    labels={"x": "log10(Concentration)", "y": "log10(MFI)"}
                )
                fig.add_scatter(x=np.log10(x_fit), y=y_curve, mode="lines", name="Fit", line=dict(color="orange"))

                # ---------- Interpolate samples ----------
                y_samples = np.log10(samples[a].astype(float))
                mask_s = np.isfinite(y_samples)
                popt = result.best_values
                try:
                    if fit_type == "4PL":
                        deltaA, log10EC50, n = popt["deltaA"], popt["log10EC50"], popt["n"]
                        Amin = Amin_fixed
                        Amax = Amin + deltaA
                        # R's exact inverse
                        x_pred = 10 ** (
                            (1 / n) * (np.log((Amax - y_samples[mask_s]) / (y_samples[mask_s] - Amin)) + n * log10EC50)
                        )
                    else:
                        A, B, EC50, n, s = popt["A"], popt["B"], popt["EC50"], popt["n"], popt["s"]
                        x_pred = 10 ** (
                            (1 / n) * (np.log(((B - y_samples[mask_s]) / (y_samples[mask_s] - A)) ** (1/s)) + n * EC50)
                        )

                    conc_pred = x_pred
                    if apply_sample_dilution:
                        conc_pred = conc_pred * samples["ID"].map(sample_dilutions)
                    samples.loc[mask_s, f"{a}_conc_pgml"] = conc_pred

                    fig.add_scatter(
                        x=np.log10(conc_pred),
                        y=y_samples[mask_s],
                        mode="markers",
                        name="Samples",
                        marker=dict(color="blue", size=6)
                    )
                except Exception as e:
                    samples[f"{a}_conc_pgml"] = np.nan
                    st.warning(f"⚠️ Interpolation failed for {a}: {e}")

                if r2 < 0.95:
                    st.warning(f"⚠️ {a}: R²={r2:.3f}, curve may be poor.")
                plots.append(fig)
            except Exception as e:
                st.warning(f"⚠️ Fit failed for {a}: {e}")

        st.success("✅ Curve fitting complete.")
        for fig in plots:
            st.plotly_chart(fig, use_container_width=True)

        # ---------- 7. Export ----------
        output_name = st.text_input("Output file name:", value="LegendPlex_results")
        qc_summary = pd.DataFrame([
            {"Analyte": a, "R²": fit_results[a]["r2"],
             "Fit_Status": "OK" if fit_results[a]["r2"] >= 0.95 else "Poor"}
            for a in fit_results
        ])
        st.dataframe(qc_summary)

        long_records = []
        for _, row in samples.iterrows():
            sid = row["ID"]
            dil = sample_dilutions.get(sid, 1)
            for a in analytes:
                mfi = row[a]
                conc = row.get(f"{a}_conc_pgml", np.nan)
                long_records.append({
                    "ID": sid,
                    "Analyte": a,
                    "MFI": mfi,
                    "Concentration_pg_mL": conc,
                    "Dilution_Factor": dil if apply_sample_dilution else 1
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
            label="⬇️ Download .xlsx (with QC summary)",
            data=output,
            file_name=f"{output_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Please upload a FlowJo export to begin.")
