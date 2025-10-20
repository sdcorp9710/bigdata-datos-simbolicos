"""
# ==========================
# Doctorado en Tecnologias para la Transformacion Digital
# Materia: Big Data
# Instructor: Dr. Jonás Velasco Álvarez
# Estudiante: Luis Alejandro Santana Valadez
# ==========================
Ejercicio 2.2 — SDA Demostración de estadísticos por Histograma (Tablas 2.12, 2.13, 2.14 Y = Hora de vuelo)
# ==========================
Pasos:
1) Parsear las Tablas 2.12, 2.13, 2.14 del PDF "4.1 Intro_SDA.pdf" 
2) Exportar y unir los pares de tablas a CSVs canónicos de bins
3) Calcular estadísticos por aerolínea y estadísticos muestrales (SDA)
4) Guardar un histograma promedio (PNG) y salidas CSV por tabla.

Dependencias:
  pip install PyPDF2 pandas numpy matplotlib
"""

import pandas as pd, numpy as np, math, re, os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

BASE = Path(".")

def parse_interval_str(s):
    m = re.match(r'\s*[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]\s*', str(s))
    if not m:
        raise ValueError(f"Bad interval format: {s}")
    a = float(m.group(1)); b = float(m.group(2))
    return (a,b) if a<=b else (b,a)

def table_to_bins(path, which):
    df = pd.read_csv(path)
    if which == "2.12":  # (Y1,Y2) = (AirTime, ArrDelay)
        y1_col, y2_col = "Y1_interval", "Y2_interval"
    elif which == "2.13":  # (Y1,Y3) -> tratamos como (Y1, Y2_gen)
        y1_col, y2_col = "Y1_interval", "Y3_interval"
    else:                 # 2.14: (Y2,Y3) -> (Y1_gen=Y2, Y2_gen=Y3)
        y1_col, y2_col = "Y2_interval", "Y3_interval"

    bins = []
    for _, r in df.iterrows():
        u = int(r["airline"])
        a1,b1 = parse_interval_str(r[y1_col])
        a2,b2 = parse_interval_str(r[y2_col])
        p = float(r["joint_probability"])
        bins.append({"unit":u,"y1_lower":a1,"y1_upper":b1,"y2_lower":a2,"y2_upper":b2,"prob":p})
    out = pd.DataFrame(bins)
    # normalizar p por unidad
    sums = out.groupby("unit")["prob"].sum()
    out["prob"] = out.apply(lambda r: r["prob"]/sums.loc[r["unit"]] if sums.loc[r["unit"]] else r["prob"], axis=1)
    return out

def compute_bivariate_sda(df_bins):
    df = df_bins.copy()
    df["y1_mid"] = (df["y1_lower"] + df["y1_upper"])/2.0
    df["y2_mid"] = (df["y2_lower"] + df["y2_upper"])/2.0
    df["y1_width"] = (df["y1_upper"] - df["y1_lower"])
    df["y2_width"] = (df["y2_upper"] - df["y2_lower"])
    g = df.groupby("unit")
    mu1 = g.apply(lambda x: (x["prob"]*x["y1_mid"]).sum())
    mu2 = g.apply(lambda x: (x["prob"]*x["y2_mid"]).sum())
    var1_within = g.apply(lambda x: (x["prob"]*(x["y1_width"]**2)/12.0).sum())
    var2_within = g.apply(lambda x: (x["prob"]*(x["y2_width"]**2)/12.0).sum())
    var1_between = g.apply(lambda x: (x["prob"]*((x["y1_mid"]-mu1[x.name])**2)).sum())
    var2_between = g.apply(lambda x: (x["prob"]*((x["y2_mid"]-mu2[x.name])**2)).sum())
    var1_total = var1_within + var1_between
    var2_total = var2_within + var2_between
    cov12_total = g.apply(lambda x: (x["prob"]*((x["y1_mid"]-mu1[x.name])*(x["y2_mid"]-mu2[x.name]))).sum())
    per_unit = pd.DataFrame({
        "unit": mu1.index, "mu1": mu1.values, "mu2": mu2.values,
        "var1_total": var1_total.values, "var2_total": var2_total.values,
        "cov12_total": cov12_total.values
    })
    Ybar1 = per_unit["mu1"].mean(); Ybar2 = per_unit["mu2"].mean()
    S2_1 = per_unit["var1_total"].mean() + per_unit["mu1"].var(ddof=0)
    S2_2 = per_unit["var2_total"].mean() + per_unit["mu2"].var(ddof=0)
    S12  = per_unit["cov12_total"].mean() + np.cov(per_unit["mu1"], per_unit["mu2"], bias=True)[0,1]
    summary = pd.DataFrame({"n":[per_unit.shape[0]], "Ybar1":[Ybar1], "Ybar2":[Ybar2], "S2_1":[S2_1], "S2_2":[S2_2], "S12":[S12]})
    return df, per_unit, summary

def plot_avg_joint_hist(df_bins, out_png, title):
    avg = df_bins.groupby(["y1_lower","y1_upper","y2_lower","y2_upper"]).agg(avg_prob=("prob","mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6,5))
    for _, r in avg.iterrows():
        x = r["y1_lower"]; w = r["y1_upper"]-r["y1_lower"]
        y = r["y2_lower"]; h = r["y2_upper"]-r["y2_lower"]
        ax.add_patch(plt.Rectangle((x,y), w, h, alpha=min(0.9, r["avg_prob"]*3), edgecolor="black", linewidth=0.25))
    ax.set_xlabel("Y1"); ax.set_ylabel("Y2"); ax.set_title(title)
    ax.autoscale(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_unit_means_ellipse(per_unit, out_png, title):
    means = per_unit[["mu1","mu2"]].values
    mu = means.mean(axis=0)
    cov = np.cov(means.T, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:,order]
    theta = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    width, height = 2*np.sqrt(eigvals)
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    ax.scatter(per_unit["mu1"], per_unit["mu2"], s=40)
    ax.add_patch(Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False, linewidth=2))
    ax.set_xlabel("E[Y1]"); ax.set_ylabel("E[Y2]"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def process_all():
    mapping = {
        "Table 2.12": ("tabla_2_12_completa.csv","2.12","ex2_2a"),
        "Table 2.13": ("tabla_2_13_completa.csv","2.13","ex2_2b"),
        "Table 2.14": ("tabla_2_14_completa.csv","2.14","ex2_2c"),
    }
    outputs = []
    comp_list = []
    for label, (fname, which, tag) in mapping.items():
        bins = table_to_bins(BASE/fname, which)
        bins_csv = BASE / f"{tag}_{label.replace(' ','_').lower()}_joint_bins.csv"
        per_csv  = BASE / f"{tag}_per_unit_stats.csv"
        sum_csv  = BASE / f"{tag}_summary_stats.csv"
        bins.to_csv(bins_csv, index=False)
        norm_bins, per_unit, summary = compute_bivariate_sda(bins)
        per_unit.to_csv(per_csv, index=False)
        summary.to_csv(sum_csv, index=False)
        plot_avg_joint_hist(norm_bins, str(BASE / f"{tag}_avg_joint_hist.png"), f"{label} — Average joint histogram")
        plot_unit_means_ellipse(per_unit, str(BASE / f"{tag}_unit_means_ellipse.png"), f"{label} — Unit means & ellipse")
        comp_list.append(summary.assign(table=label)[["table","n","Ybar1","Ybar2","S2_1","S2_2","S12"]])
        outputs += [bins_csv, per_csv, sum_csv, BASE / f"{tag}_avg_joint_hist.png", BASE / f"{tag}_unit_means_ellipse.png"]

    comp = pd.concat(comp_list, ignore_index=True)
    comp.to_csv(BASE/"ex2_2_comparative_summary_from_usercsv.csv", index=False)
    outputs.append(BASE/"ex2_2_comparative_summary_from_usercsv.csv")

    # trivariante
    t212 = pd.read_csv(BASE/"ex2_2a_per_unit_stats.csv")
    t213 = pd.read_csv(BASE/"ex2_2b_per_unit_stats.csv")
    t214 = pd.read_csv(BASE/"ex2_2c_per_unit_stats.csv")
    dfm = pd.DataFrame({"unit": t212["unit"]})
    dfm["Y1_from_212"] = t212["mu1"]; dfm["Y2_from_212"] = t212["mu2"]
    dfm = dfm.merge(t213[["unit","mu1","mu2"]].rename(columns={"mu1":"Y1_from_213","mu2":"Y3_from_213"}), on="unit")
    dfm = dfm.merge(t214[["unit","mu1","mu2"]].rename(columns={"mu1":"Y2_from_214","mu2":"Y3_from_214"}), on="unit")
    dfm["Y1_mu"] = dfm[["Y1_from_212","Y1_from_213"]].mean(axis=1)
    dfm["Y2_mu"] = dfm[["Y2_from_212","Y2_from_214"]].mean(axis=1)
    dfm["Y3_mu"] = dfm[["Y3_from_213","Y3_from_214"]].mean(axis=1)

    Ybar1 = dfm["Y1_mu"].mean(); Ybar2 = dfm["Y2_mu"].mean(); Ybar3 = dfm["Y3_mu"].mean()
    S2_Y1 = t212["var1_total"].mean() + dfm["Y1_mu"].var(ddof=0)
    S2_Y2 = t212["var2_total"].mean() + dfm["Y2_mu"].var(ddof=0)
    S2_Y3 = t213["var2_total"].mean() + dfm["Y3_mu"].var(ddof=0)
    Cov12 = t212["cov12_total"].mean() + np.cov(dfm["Y1_mu"], dfm["Y2_mu"], bias=True)[0,1]
    Cov13 = t213["cov12_total"].mean() + np.cov(dfm["Y1_mu"], dfm["Y3_mu"], bias=True)[0,1]
    Cov23 = t214["cov12_total"].mean() + np.cov(dfm["Y2_mu"], dfm["Y3_mu"], bias=True)[0,1]
    S_Y1, S_Y2, S_Y3 = math.sqrt(S2_Y1), math.sqrt(S2_Y2), math.sqrt(S2_Y3)
    Corr12 = Cov12/(S_Y1*S_Y2); Corr13 = Cov13/(S_Y1*S_Y3); Corr23 = Cov23/(S_Y2*S_Y3)

    tri = pd.DataFrame([
        {"metric":"Ybar1","value":Ybar1},
        {"metric":"Ybar2","value":Ybar2},
        {"metric":"Ybar3","value":Ybar3},
        {"metric":"S_Y1","value":S_Y1},
        {"metric":"S_Y2","value":S_Y2},
        {"metric":"S_Y3","value":S_Y3},
        {"metric":"Cov12","value":Cov12},
        {"metric":"Cov13","value":Cov13},
        {"metric":"Cov23","value":Cov23},
        {"metric":"Corr12","value":Corr12},
        {"metric":"Corr13","value":Corr13},
        {"metric":"Corr23","value":Corr23},
    ])
    tri.to_csv(BASE/"ex2_2_targets_calc_values_from_usercsv.csv", index=False)
    outputs.append(BASE/"ex2_2_targets_calc_values_from_usercsv.csv")

    print("\nArchivos generados:")
    for p in outputs:
        print(" -", p.resolve())

if __name__ == "__main__":
    process_all()
