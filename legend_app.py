import streamlit as st
import pandas as pd
import numpy as np
import re, io
from lmfit import Model
import plotly.express as px

st.title("LegendPlex — Strict R-Equivalent 4PL Analyzer")

# ---------- Upload & Clean ----------
uploaded = st.file_uploader("Upload FlowJo export (.csv or .xlsx)", type=["csv", "xlsx"])
if not uploaded:
    st.stop()

def load_file(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

def clean_colnames(df):
    df.columns = df.columns.astype(str)
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)
    def extract_marker(c):
        try:
            m = re.search(r".*?/(.*?)\s*\|", c)
            if m and m.group(1).strip():
                return m.group(1).strip()
            parts = re.split(r"[/|]", c)
            return parts[-2].strip() if len(parts) > 1 else c.strip()
        except Exception:
            return c.strip()
    df.columns = [extract_marker(c) for c in df.columns]
    return df

df = clean_colnames(load_file(uploaded))
if df.shape[0] > 2:
    df = df.iloc[:-2, :].copy()  # drop last 2 rows (like R)
st.success("✅ Columns cleaned.")
st.dataframe(df.head())

# ---------- Standards ----------
std_rows = df[df["ID"].str.contains("Standard", case=False, na=False)].copy()
if std_rows.empty:
    st.warning("No 'Standard' IDs detected; using first 12 rows as standards.")
    std_rows = df.iloc[:12, :].copy()
samples = df.loc[~df.index.isin(std_rows.index)].copy()

analytes = [c for c in df.columns if c not in ["ID", "WELL ID"]]
st.write(f"Detected analytes: {analytes}")

# ---------- User Inputs ----------
dilution_factor = st.number_input("Serial dilution factor", min_value=1.0, value=3.0, step=0.5)
st.markdown("### Enter top standard concentrations (from CoA, ng/mL)")
top_conc = {a: st.number_input(f"{a}", min_value=0.0001, value=10.0) for a in analytes}
run = st.button("Run R-Equivalent 4PL Fit")

# ---------- 4PL model identical to R ----------
def fourPL_R(Concentration, deltaA, log10EC50, n, Amin_fixed):
    return Amin_fixed + (deltaA / (1 + np.exp(n * (Concentration - log10EC50))))

def inverse_fourPL_R(MFI, Amin_fixed, deltaA, log10EC50, n):
    Amax = Amin_fixed + deltaA
    return (1/n) * (np.log((Amax - MFI) / (MFI - Amin_fixed)) + n * log10EC50)

if not run:
    st.stop()

# ---------- Build standard ladder exactly like R ----------
std_n = len(std_rows)
powers = np.arange(std_n, dtype=float)
conc_pg = {a: (top_conc[a] * 1000.0) / (dilution_factor ** powers) for a in analytes}
reps = pd.DataFrame(conc_pg)
reps.insert(0, "ID", std_rows["ID"].values)

# log10 transform concentrations (x in R)
reps_log10 = reps.copy()
for a in analytes:
    reps_log10[a] = np.log10(reps_log10[a])

st.write("Standard ladder preview (log10 pg/mL):")
st.dataframe(reps_log10.head())

fit_results = {}
plots = []
records = []

for a in analytes:
    try:
        # --- get data for analyte ---
        x = np.asarray(reps_log10[a], dtype=float)           # log10(conc)
        y = np.log10(np.asarray(std_rows[a], dtype=float))    # log10(MFI)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # --- starting values like R ---
        Amin_fixed = np.quantile(y, 0.05)
        Amax_start = np.quantile(y, 0.95)
        mid_y = (Amax_start + Amin_fixed) / 2
        idx_mid = int(np.argmin(np.abs(y - mid_y)))
        log10EC50_start = x[idx_mid] if len(x) > 0 else np.median(x)
        n_start = -1

        model = Model(fourPL_R)
        params = model.make_params(
            deltaA=Amax_start - Amin_fixed,
            log10EC50=log10EC50_start,
            n=n_start,
            Amin_fixed=Amin_fixed
        )
        params["Amin_fixed"].vary = False
        params["deltaA"].min, params["deltaA"].max = 1e-6, (Amax_start - Amin_fixed) * 5
        params["log10EC50"].min, params["log10EC50"].max = min(x), max(x)
        params["n"].min, params["n"].max = -10, -1e-3

        result = model.fit(y, Concentration=x, params=params,
                           method="leastsq", max_nfev=20000)
        y_fit = result.best_fit
        r2 = 1 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)

        fit_results[a] = {"result": result, "Amin": Amin_fixed, "r2": r2}

        # --- plot standards + curve ---
        x_fit = np.linspace(min(x), max(x), 200)
        y_curve = model.eval(Concentration=x_fit, **result.best_values)
        fig = px.scatter(x=x, y=y, color_discrete_sequence=["red"],
                         title=f"{a} — 4PL (R²={r2:.3f})",
                         labels={"x": "log10(Conc pg/mL)", "y": "log10(MFI)"})
        fig.add_scatter(x=x_fit, y=y_curve, mode="lines", name="Fit", line=dict(color="orange"))

        # --- interpolate samples ---
        y_samp = np.log10(np.asarray(samples[a], dtype=float))
        mask_s = np.isfinite(y_samp)
        p = result.best_values
        x_pred = inverse_fourPL_R(y_samp[mask_s],
                                  Amin_fixed,
                                  p["deltaA"],
                                  p["log10EC50"],
                                  p["n"])
        conc_pred = 10 ** x_pred  # pg/mL
        samples.loc[mask_s, f"{a}_conc_pgml"] = conc_pred

        fig.add_scatter(x=x_pred, y=y_samp[mask_s],
                        mode="markers", name="Samples", marker=dict(color="blue", size=6))

        plots.append(fig)

        for sid, mfi_val, cval in zip(samples["ID"], samples[a], samples[f"{a}_conc_pgml"]):
            records.append({"ID": sid, "Analyte": a, "MFI": mfi_val, "Conc_pg/mL": cval})
    except Exception as e:
        st.warning(f"⚠️ {a}: fit failed ({e})")

# ---------- Show Plots ----------
for fig in plots:
    st.plotly_chart(fig, use_container_width=True)

# ---------- Export ----------
long_df = pd.DataFrame(records)
wide_df = samples[["ID"] + [f"{a}_conc_pgml" for a in analytes]]
qc = pd.DataFrame([
    {"Analyte": a,
     "Amin_fixed": fit_results[a]["Amin"],
     "R²": fit_results[a]["r2"],
     **fit_results[a]["result"].best_values}
    for a in fit_results
])

out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as w:
    long_df.to_excel(w, index=False, sheet_name="Long")
    wide_df.to_excel(w, index=False, sheet_name="Wide")
    qc.to_excel(w, index=False, sheet_name="QC")
out.seek(0)

st.download_button(
    "⬇️ Download results (.xlsx)",
    data=out,
    file_name="LegendPlex_Rmatch.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
