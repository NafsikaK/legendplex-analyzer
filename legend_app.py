import streamlit as st
import pandas as pd
import numpy as np
import re, io
from lmfit import Model
import plotly.express as px

st.title("LegendPlex — STRICT R-parity 4PL")

# ---------- helpers ----------
def load_file(f):
    return pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)

def extract_marker_name(col):
    # match: ".../<Analyte> | Median..."
    m = re.search(r".*?/(.*?)\s*\|\s*Median", col)
    return m.group(1).strip() if m else col

def fourPL_R(Concentration, deltaA, log10EC50, n, Amin_fixed):
    # Concentration and MFI are already log10 in STRICT mode
    return Amin_fixed + (deltaA / (1 + np.exp(n * (Concentration - log10EC50))))

def inverse_fourPL_R(MFI, Amin_fixed, deltaA, log10EC50, n):
    # All on log10 scale
    Amax = Amin_fixed + deltaA
    return (1.0/n) * (np.log((Amax - MFI) / (MFI - Amin_fixed)) + n * log10EC50)

# ---------- upload ----------
f = st.file_uploader("Upload FlowJo (.csv or .xlsx) used in R", type=["csv", "xlsx"])
if not f:
    st.stop()

# ---------- read & R-parity preprocessing ----------
df = load_file(f)

# rename first column to ID
df.columns = df.columns.astype(str)
df.rename(columns={df.columns[0]: "ID"}, inplace=True)

# drop last two rows (R: head(df, n = nrow(df) - 2))
if df.shape[0] >= 2:
    df = df.iloc[:-2, :].copy()

# extract analyte names exactly like R did
new_cols = [extract_marker_name(c) for c in df.columns]
df.columns = new_cols

# OPTIONAL: mirror your script if it drops a trailing non-analyte column (e.g. df <- df[,-16])
# If your file has that extra column, uncomment the next 2 lines:
# if df.shape[1] >= 16:
#     df = df.drop(df.columns[15], axis=1)

# ensure numeric for analytes (from col 3 to last, like your R code)
if df.shape[1] < 3:
    st.error("Not enough columns after parsing. Check file format.")
    st.stop()

analyte_cols = df.columns[2:]  # columns 3..end
for c in analyte_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

st.write("First rows (after strict R preprocessing):")
st.dataframe(df.head())

# ---------- STRICT: standards are FIRST 12 ROWS ----------
if df.shape[0] < 12:
    st.error("File has < 12 rows; cannot take first 12 as standards like R.")
    st.stop()

std_df = df.iloc[:12, :].copy()
samp_df = df.iloc[12:, :].copy()

# ---------- log10 transform MFI (analyte cols) ----------
df_std_log = std_df.copy()
for c in analyte_cols:
    df_std_log[c] = np.log10(df_std_log[c].astype(float))

df_samp_log = samp_df.copy()
for c in analyte_cols:
    df_samp_log[c] = np.log10(df_samp_log[c].astype(float))

# ---------- ask for top CoA (ng/mL) in current analyte order ----------
st.markdown("### Top standard concentrations (ng/mL) — MUST match the CURRENT analyte column order below")
st.write(list(analyte_cols))

top_ng = {a: st.number_input(f"{a}", min_value=0.0001, value=10.0) for a in analyte_cols}
dilution_factor = st.number_input("Serial dilution factor", min_value=1.0, value=3.0, step=0.5)
run = st.button("Run 4PL (STRICT R)")

if not run:
    st.stop()

# ---------- build ladder: Top / d^(i-1), then log10 ----------
powers = np.arange(12, dtype=float)
conc_pg = {a: (top_ng[a] * 1000.0) / (dilution_factor ** powers) for a in analyte_cols}
reps = pd.DataFrame(conc_pg)
reps.insert(0, "ID", std_df["ID"].values)

# x = log10(concentration pg/mL)
X_log = {a: np.log10(reps[a].astype(float).to_numpy()) for a in analyte_cols}
Y_log = {a: df_std_log[a].to_numpy() for a in analyte_cols}

st.write("Standard ladder preview (pg/mL):")
st.dataframe(reps)

# ---------- fit per analyte (STRICT) ----------
fit_params = {}
plots = []
long_rows = []

for a in analyte_cols:
    x = X_log[a]
    y = Y_log[a]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]

    # Amin fixed = 5th percentile of log10(MFI) (STRICT)
    Amin_fixed = float(np.quantile(y, 0.05))
    Amax_start = float(np.quantile(y, 0.95))

    mid_y = (Amax_start + Amin_fixed) / 2.0
    idx_mid = int(np.argmin(np.abs(y - mid_y)))
    log10EC50_start = float(x[idx_mid]) if len(x) > 0 else float(np.median(x))
    n_start = -1.0

    model = Model(fourPL_R)
    params = model.make_params(deltaA=Amax_start - Amin_fixed,
                               log10EC50=log10EC50_start,
                               n=n_start,
                               Amin_fixed=Amin_fixed)
    params["Amin_fixed"].vary = False
    params["deltaA"].min, params["deltaA"].max = 1e-6, (Amax_start - Amin_fixed) * 5.0
    params["log10EC50"].min, params["log10EC50"].max = float(np.min(x)), float(np.max(x))
    params["n"].min, params["n"].max = -10.0, -1e-3

    result = model.fit(y, Concentration=x, params=params, method="leastsq", max_nfev=20000)
    y_fit = result.best_fit
    r2 = 1 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)

    fit_params[a] = dict(result.best_values)
    fit_params[a]["Amin_fixed"] = Amin_fixed
    fit_params[a]["R2"] = float(r2)

    # plot standards + curve
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_curve = model.eval(Concentration=x_fit, **result.best_values)

    fig = px.scatter(x=x, y=y,
                     title=f"{a} — 4PL STRICT (R²={r2:.3f})",
                     labels={"x": "log10(Conc pg/mL)", "y": "log10(MFI)"},
                     color_discrete_sequence=["red"])
    fig.add_scatter(x=x_fit, y=y_curve, mode="lines", name="Fit", line=dict(color="orange"))

    # interpolate samples (use EXACT inverse on log10 scale)
    y_samp = df_samp_log[a].to_numpy()
    mask_s = np.isfinite(y_samp)
    p = result.best_values
    try:
        x_pred = inverse_fourPL_R(y_samp[mask_s], Amin_fixed, p["deltaA"], p["log10EC50"], p["n"])
        conc_pred = 10 ** x_pred  # pg/mL
        # store into original samples frame:
        samp_df.loc[mask_s, f"{a}_conc_pgml"] = conc_pred

        # overlay samples
        fig.add_scatter(x=x_pred, y=y_samp[mask_s], mode="markers",
                        name="Samples", marker=dict(color="blue", size=6))
    except Exception as e:
        samp_df[f"{a}_conc_pgml"] = np.nan
        st.warning(f"{a}: interpolation failed: {e}")

    plots.append(fig)

# ---------- show curves ----------
for fig in plots:
    st.plotly_chart(fig, use_container_width=True)

# ---------- export like R long/wide ----------
long_records = []
for _, row in samp_df.iterrows():
    for a in analyte_cols:
        long_records.append({
            "ID": row["ID"],
            "Analyte": a,
            "MFI": row[a],
            "Concentration_pg_mL": row.get(f"{a}_conc_pgml", np.nan)
        })
long_df = pd.DataFrame(long_records)
wide_df = samp_df[["ID"] + [f"{a}_conc_pgml" for a in analyte_cols]].copy()

qc = pd.DataFrame([
    {"Analyte": a, "Amin_fixed": fit_params[a]["Amin_fixed"],
     "deltaA": fit_params[a]["deltaA"], "log10EC50": fit_params[a]["log10EC50"],
     "n": fit_params[a]["n"], "R2": fit_params[a]["R2"]}
    for a in analyte_cols
])

out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as w:
    long_df.to_excel(w, index=False, sheet_name="Long")
    wide_df.to_excel(w, index=False, sheet_name="Wide")
    qc.to_excel(w, index=False, sheet_name="QC")
out.seek(0)

st.download_button(
    "⬇️ Download .xlsx (Long/Wide/QC)",
    data=out,
    file_name="LegendPlex_STRICT_R.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
