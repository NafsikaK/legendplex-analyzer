import streamlit as st
import pandas as pd
import numpy as np
import re, io
from lmfit import Model
import plotly.express as px

st.title("LegendPlex Analyzer — 4PL (R-equivalent)")

# ---------- Upload & clean ----------
uploaded = st.file_uploader("Upload FlowJo export (.csv or .xlsx)", type=["csv", "xlsx"])

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

if not uploaded:
    st.info("Upload a FlowJo export to begin.")
    st.stop()

df = clean_colnames(load_file(uploaded))
st.success("✅ Columns cleaned successfully.")
st.dataframe(df.head())

# ---------- Detect standards ----------
std_rows = df[df["ID"].str.contains("Standard", case=False, na=False)].copy()
if std_rows.empty:
    st.warning("⚠️ No standards detected automatically. Select wells manually.")
    std_rows = df[df["ID"].isin(st.multiselect("Select wells with standards", df["ID"]))]

samples = df.loc[~df.index.isin(std_rows.index)].copy()
st.info(f"Detected {len(std_rows)} standards and {len(samples)} samples.")

# ---------- User inputs ----------
dilution_factor = st.number_input("Serial dilution factor (e.g., 3)", min_value=1.0, value=3.0)
analytes = [c for c in df.columns if c not in ["ID", "WELL ID"]]
st.markdown("### Enter top standard concentrations (from CoA, ng/mL)")
top_conc = {a: st.number_input(f"{a}", min_value=0.001, value=10.0) for a in analytes}
apply_dilution = st.checkbox("Apply sample dilution factors?", value=False)
fit_button = st.button("Run R-equivalent 4PL fit")

# ---------- 4PL R-equivalent model ----------
def fourPL_R(Concentration, deltaA, log10EC50, n, Amin_fixed):
    return Amin_fixed + (deltaA / (1 + np.exp(n * (Concentration - log10EC50))))

def inverse_fourPL_R(MFI, Amin_fixed, deltaA, log10EC50, n):
    Amax = Amin_fixed + deltaA
    return (1/n) * (np.log((Amax - MFI) / (MFI - Amin_fixed)) + n * log10EC50)

if not fit_button:
    st.stop()

# ---------- Build concentration ladder ----------
std_n = len(std_rows)
powers = np.arange(std_n, dtype=float)
conc_pg = {a: (top_conc[a]*1000) / (dilution_factor ** powers) for a in analytes}
reps = pd.DataFrame(conc_pg)
reps.insert(0, "ID", std_rows["ID"].values)
st.dataframe(reps.head())

# ---------- Fit ----------
fit_results = {}
plots = []
out_long = []

for a in analytes:
    try:
        # log10 transform for both concentration and MFI
        x = np.log10(np.asarray(reps[a], dtype=float))
        y = np.log10(np.asarray(std_rows[a], dtype=float))

        # drop non-finite
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # Amin fixed
        Amin_fixed = np.quantile(y, 0.05)
        Amax_start = np.quantile(y, 0.95)
        mid_y = (Amax_start + Amin_fixed) / 2
        mid_row = y[np.argmin(np.abs(y - mid_y))]
        log10EC50_start = x[list(y).index(mid_row)] if len(y)>1 else np.median(x)
        n_start = -1

        model = Model(fourPL_R)
        params = model.make_params(
            deltaA=Amax_start - Amin_fixed,
            log10EC50=log10EC50_start,
            n=n_start,
            Amin_fixed=Amin_fixed
        )
        params["Amin_fixed"].vary = False
        params["deltaA"].min, params["deltaA"].max = 1e-6, (Amax_start - Amin_fixed)*5
        params["log10EC50"].min, params["log10EC50"].max = min(x), max(x)
        params["n"].min, params["n"].max = -10, -1e-3

        result = model.fit(y, Concentration=x, params=params,
                           method="leastsq", max_nfev=20000)
        y_fit = result.best_fit
        r2 = 1 - np.sum((y - y_fit)**2)/np.sum((y - np.mean(y))**2)
        fit_results[a] = {"result": result, "Amin": Amin_fixed, "r2": r2}

        # ---------- Plot ----------
        x_fit = np.linspace(min(x), max(x), 200)
        y_curve = model.eval(Concentration=x_fit, **result.best_values)
        fig = px.scatter(x=x, y=y, title=f"{a} — 4PL (R²={r2:.3f})",
                         labels={"x": "log10(Concentration)", "y": "log10(MFI)"},
                         color_discrete_sequence=["red"])
        fig.add_scatter(x=x_fit, y=y_curve, mode="lines", name="Fit", line=dict(color="orange"))

        # ---------- Interpolate samples ----------
        y_samples = np.log10(np.asarray(samples[a], dtype=float))
        mask_s = np.isfinite(y_samples)
        p = result.best_values
        x_pred = inverse_fourPL_R(y_samples[mask_s],
                                  Amin_fixed,
                                  p["deltaA"],
                                  p["log10EC50"],
                                  p["n"])
        conc_pred = 10 ** x_pred
        samples.loc[mask_s, f"{a}_conc_pgml"] = conc_pred

        fig.add_scatter(
            x=x_pred,
            y=y_samples[mask_s],
            mode="markers",
            name="Samples",
            marker=dict(color="blue", size=6)
        )
        plots.append(fig)

        # ---------- Store results ----------
        for sid, mfi_val, cval in zip(samples["ID"], samples[a], samples[f"{a}_conc_pgml"]):
            out_long.append({"ID": sid, "Analyte": a, "MFI": mfi_val, "Conc_pg/mL": cval})
    except Exception as e:
        st.warning(f"⚠️ Fit failed for {a}: {e}")

# ---------- Display ----------
st.success("✅ Fits complete. Below are your curves:")
for fig in plots:
    st.plotly_chart(fig, use_container_width=True)

# ---------- Export ----------
qc = pd.DataFrame([{"Analyte": a, "R²": fit_results[a]["r2"]} for a in fit_results])
long_df = pd.DataFrame(out_long)
wide_df = long_df.pivot_table(index="ID", columns="Analyte", values="Conc_pg/mL", aggfunc="first").reset_index()

output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    long_df.to_excel(writer, index=False, sheet_name="Long")
    wide_df.to_excel(writer, index=False, sheet_name="Wide")
    qc.to_excel(writer, index=False, sheet_name="QC")
output.seek(0)

st.download_button(
    label="⬇️ Download results (.xlsx)",
    data=output,
    file_name="LegendPlex_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
