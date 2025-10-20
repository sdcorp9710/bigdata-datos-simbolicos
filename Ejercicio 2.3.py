"""
# ==========================
# Doctorado en Tecnologias para la Transformacion Digital
# Materia: Big Data
# Instructor: Dr. Jonás Velasco Álvarez
# Estudiante: Luis Alejandro Santana Valadez
# ==========================
Ejercicio 2.3 — SDA Demostración de estadísticos por Histograma (Tablas 2.15, 2.16, 2.17)
Calculo de estadísticas SDA para las Tablas 2.15–2.17 (Ejercicio 2.3).
# ==========================
Pasos:
1) Parsear las Tablas 2.15, 2.16, 2.17 del PDF "4.1 Intro_SDA.pdf" 
2) Exportar y unir los pares de tablas a CSVs canónicos de bins
3) Calcular estadísticos por aerolínea y estadísticos muestrales (SDA)
4) Guardar un histograma promedio (PNG) y salidas CSV por tabla.

Dependencias:
  pip install PyPDF2 pandas numpy matplotlib
"""

import pandas as pd, numpy as np, math, re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

BASE = Path(".")

# -------- Helpers de lectura --------
INTERVAL_RX = re.compile(r'\s*[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]\s*')
def parse_interval(s):
    m = INTERVAL_RX.match(str(s))
    if not m: raise ValueError(f"Bad interval format: {s}")
    a, b = float(m.group(1)), float(m.group(2))
    return (a,b) if a<=b else (b,a)

def infer_pair(df):
    cols = {c.lower(): c for c in df.columns}
    has = lambda k: k in cols
    if has("y1_interval") and has("y2_interval"):
        return (cols["y1_interval"], cols["y2_interval"], "Y1", "Y2")
    if has("y1_interval") and has("y3_interval"):
        return (cols["y1_interval"], cols["y3_interval"], "Y1", "Y3")
    if has("y2_interval") and has("y3_interval"):
        return (cols["y2_interval"], cols["y3_interval"], "Y2", "Y3")
    # fallback por búsqueda parcial
    def find(name):
        for k,v in cols.items():
            if name in k: return v
        return None
    c1 = find("y1_interval") or find("y2_interval") or find("y3_interval")
    c2 = None
    for t in ["y2_interval","y3_interval","y1_interval"]:
        v = find(t)
        if v and v != c1: c2=v; break
    lab1 = "Y" + (c1.split("_")[0][-1] if c1 else "?")
    lab2 = "Y" + (c2.split("_")[0][-1] if c2 else "?")
    return (c1,c2,lab1,lab2)

def table_to_bins(path: Path):
    df = pd.read_csv(path)
    var1_col, var2_col, lab1, lab2 = infer_pair(df)
    unit_col = "airline" if "airline" in df.columns else "unit"
    p_col = "joint_probability" if "joint_probability" in df.columns else "prob"
    rows=[]
    for _, r in df.iterrows():
        u = int(r[unit_col])
        a1,b1 = parse_interval(r[var1_col]); a2,b2 = parse_interval(r[var2_col])
        p = float(r[p_col])
        rows.append({"unit":u,"y1_lower":a1,"y1_upper":b1,"y2_lower":a2,"y2_upper":b2,"prob":p,
                     "label1":lab1,"label2":lab2})
    out = pd.DataFrame(rows)
    sums = out.groupby("unit")["prob"].sum()
    out["prob"] = out.apply(lambda r: r["prob"]/sums.loc[r["unit"]] if sums.loc[r["unit"]] else r["prob"], axis=1)
    return out, lab1, lab2

# -------- Cálculos SDA --------
def compute_bivariate_sda(df_bins: pd.DataFrame):
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
    summary = pd.DataFrame({"n":[per_unit.shape[0]], "Ybar1":[Ybar1], "Ybar2":[Ybar2],
                            "S2_1":[S2_1], "S2_2":[S2_2], "S12":[S12]})
    return df, per_unit, summary

# -------- Gráficas --------
def plot_avg_joint_hist(df_bins, out_png, title, xlab, ylab):
    avg = df_bins.groupby(["y1_lower","y1_upper","y2_lower","y2_upper"]).agg(avg_prob=("prob","mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6,5))
    for _, r in avg.iterrows():
        x = r["y1_lower"]; w = r["y1_upper"]-r["y1_lower"]
        y = r["y2_lower"]; h = r["y2_upper"]-r["y2_lower"]
        ax.add_patch(plt.Rectangle((x,y), w, h, alpha=min(0.9, r["avg_prob"]*3), edgecolor="black", linewidth=0.25))
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.autoscale(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_unit_means_ellipse(per_unit, out_png, title, xlab, ylab):
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
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# -------- Pipeline principal --------
def main():
    tables = [
        ("Table 2.15", "ex2_3a", BASE/"tabla_2_15_completa.csv"),
        ("Table 2.16", "ex2_3b", BASE/"tabla_2_16_completa.csv"),
        ("Table 2.17", "ex2_3c", BASE/"tabla_2_17_completa.csv"),
    ]
    pair_summaries = []
    per_unit_by_pair = {}
    for label, tag, path in tables:
        bins, lab1, lab2 = table_to_bins(path)
        # guardar bins normalizados
        bins.to_csv(BASE/f"{tag}_{label.replace(' ','_').lower()}_joint_bins.csv", index=False)
        # cálculos
        _, per_unit, summary = compute_bivariate_sda(bins)
        per_unit.to_csv(BASE/f"{tag}_per_unit_stats.csv", index=False)
        summary.to_csv(BASE/f"{tag}_summary_stats.csv", index=False)
        # gráficas
        plot_avg_joint_hist(bins, str(BASE/f"{tag}_avg_joint_hist.png"),
                            f"{label} — Average joint histogram", lab1, lab2)
        plot_unit_means_ellipse(per_unit, str(BASE/f"{tag}_unit_means_ellipse.png"),
                                f"{label} — Unit means & ellipse", f"E[{lab1}]", f"E[{lab2}]")
        # almacenar
        s = summary.copy(); s["table"] = label
        pair_summaries.append(s[["table","n","Ybar1","Ybar2","S2_1","S2_2","S12"]])
        per_unit_by_pair[label] = (lab1, lab2, per_unit)

    # resumen por pares
    pair_comp = pd.concat(pair_summaries, ignore_index=True)
    pair_comp.to_csv(BASE/"ex2_3_pairwise_summary.csv", index=False)

    # resumen trivariante
    def get_pu(labA, labB):
        for k,(a,b,pu) in per_unit_by_pair.items():
            if {a,b} == {labA,labB}: return pu, a, b
        raise RuntimeError(f"No per-unit table for pair ({labA},{labB}).")

    pu_12, a12, b12 = get_pu("Y1","Y2")
    pu_13, a13, b13 = get_pu("Y1","Y3")
    pu_23, a23, b23 = get_pu("Y2","Y3")

    dfm = pd.DataFrame({"unit": pu_12["unit"]})
    dfm["Y1_from_12"] = pu_12["mu1"] if a12=="Y1" else pu_12["mu2"]
    dfm["Y2_from_12"] = pu_12["mu2"] if b12=="Y2" else pu_12["mu1"]
    dfm = dfm.merge(pu_13[["unit","mu1","mu2"]], on="unit")
    dfm["Y1_from_13"] = pu_13["mu1"] if a13=="Y1" else pu_13["mu2"]
    dfm["Y3_from_13"] = pu_13["mu2"] if b13=="Y3" else pu_13["mu1"]
    dfm = dfm.merge(pu_23[["unit","mu1","mu2"]], on="unit")
    dfm["Y2_from_23"] = pu_23["mu1"] if a23=="Y2" else pu_23["mu2"]
    dfm["Y3_from_23"] = pu_23["mu2"] if b23=="Y3" else pu_23["mu1"]

    dfm["Y1_mu"] = dfm[["Y1_from_12","Y1_from_13"]].mean(axis=1)
    dfm["Y2_mu"] = dfm[["Y2_from_12","Y2_from_23"]].mean(axis=1)
    dfm["Y3_mu"] = dfm[["Y3_from_13","Y3_from_23"]].mean(axis=1)

    Ybar1 = dfm["Y1_mu"].mean(); Ybar2 = dfm["Y2_mu"].mean(); Ybar3 = dfm["Y3_mu"].mean()
    S2_Y1 = pu_12["var1_total"].mean() + dfm["Y1_mu"].var(ddof=0)
    S2_Y2 = pu_12["var2_total"].mean() + dfm["Y2_mu"].var(ddof=0)
    S2_Y3 = pu_13["var2_total"].mean() + dfm["Y3_mu"].var(ddof=0)
    Cov12 = pu_12["cov12_total"].mean() + np.cov(dfm["Y1_mu"], dfm["Y2_mu"], bias=True)[0,1]
    Cov13 = pu_13["cov12_total"].mean() + np.cov(dfm["Y1_mu"], dfm["Y3_mu"], bias=True)[0,1]
    Cov23 = pu_23["cov12_total"].mean() + np.cov(dfm["Y2_mu"], dfm["Y3_mu"], bias=True)[0,1]
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
    tri.to_csv(BASE/"ex2_3_trivariate_summary.csv", index=False)

    print("OK. Generados CSV y PNG en:", BASE.resolve())

if __name__ == "__main__":
    main()
