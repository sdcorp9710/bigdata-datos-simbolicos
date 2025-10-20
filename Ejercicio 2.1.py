"""
# ==========================
# Doctorado en Tecnologias para la Transformacion Digital
# Materia: Big Data
# Instructor: Dr. Jonás Velasco Álvarez
# Estudiante: Luis Alejandro Santana Valadez
# ==========================
Ejercicio 2.1 — SDA Demostración de estadísticos por Histograma (Tabla 2.4: Y = Hora de vuelo)
# ==========================
Pasos:
1) Parsear la Tabla 2.4 del PDF "4.1 Intro_SDA.pdf" (página 8)
2) Exportar CSV canónico de bins
3) Calcular estadísticos por aerolínea y estadísticos muestrales (SDA)
4) Guardar un histograma promedio (PNG) y salidas CSV.

Dependencias:
  pip install PyPDF2 pandas numpy matplotlib
"""

from pathlib import Path
import re, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader


def parse_table_2_4_from_pdf(pdf_path: str, page_index_zero_based: int = 7) -> pd.DataFrame:
    """
    Parsea la Tabla 2.4 (Histogram data: flight times) de la página indicada del PDF.
    Devuelve DataFrame con columnas: airline, lower, upper, closed_right, prob
    """
    reader = PdfReader(pdf_path)
    text = reader.pages[page_index_zero_based].extract_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Extraer sólo las líneas de la tabla 2.4
    tbl_lines, capture = [], False
    for ln in lines:
        if ln.startswith("Table 2.4"):
            capture = True
            continue
        if capture:
            if re.match(r'^\d+\{', ln):
                tbl_lines.append(ln)
            if ln.startswith("Table 2.5") or ln.startswith("2.2.5"):
                break

    def parse_airline_hist(line: str):
        m = re.match(r'^(\d+)\{(.+)\}$', line)
        if not m:
            return []
        airline = int(m.group(1))
        inside = m.group(2)
        parts = [p.strip() for p in inside.split(';') if p.strip()]
        rows = []
        for p in parts:
            # Separar intervalo y probabilidad (robusto a OCR)
            if '),' in p:
                interval, prob = p.split('),', 1); interval = interval.strip() + ')'
            elif '],' in p:
                interval, prob = p.split('],', 1); interval = interval.strip() + ']'
            else:
                msp = re.split(r'\)\s*,\s*|\]\s*,\s*', p)
                if len(msp) == 2:
                    interval = msp[0] + (')' if ')' in p else ']')
                    prob = msp[1]
                else:
                    continue
            prob = float(prob.strip())
            m2 = re.match(r'^\[(\-?\d+\.?\d*),\s*(\-?\d+\.?\d*)(\)|\])$', interval)
            if not m2:
                continue
            a = float(m2.group(1)); b = float(m2.group(2)); right = m2.group(3)
            rows.append({
                "airline": airline,
                "lower": a,
                "upper": b,
                "closed_right": 1 if right == ']' else 0,
                "prob": prob
            })
        return rows

    rows = []
    for ln in tbl_lines:
        rows.extend(parse_airline_hist(ln))

    df = pd.DataFrame(rows)
    return df.sort_values(["airline","lower"]).reset_index(drop=True)


def compute_sda_stats(df_bins: pd.DataFrame):
    """
    A partir del CSV canónico (bins), calcula:
    - por aerolínea: E_u, Var_within_u, Var_between_bins_u, Var_total_u
    - global SDA: n, Ybar, S2, SD
    Fórmula SDA que reproduce el ejercicio:
        S^2 = mean(Var_total_u) + Var(E_u)    (Var con denominador n)
    """
    df = df_bins.copy()
    df["mid"] = (df["lower"] + df["upper"]) / 2.0
    df["width"] = (df["upper"] - df["lower"])

    g = df.groupby("airline")
    E_u = g.apply(lambda x: (x["prob"] * x["mid"]).sum())
    Var_within = g.apply(lambda x: (x["prob"] * (x["width"]**2) / 12.0).sum())
    Var_between_bins = g.apply(lambda x: (x["prob"] * ((x["mid"] - (x["prob"]*x["mid"]).sum())**2)).sum())
    Var_total_u = Var_within + Var_between_bins

    per_airline = pd.DataFrame({
        "airline": E_u.index,
        "E_u": E_u.values,
        "Var_within_u": Var_within.values,
        "Var_between_bins_u": Var_between_bins.values,
        "Var_total_u": Var_total_u.values
    })

    n = len(E_u)
    Ybar = E_u.mean()
    S2 = Var_total_u.mean() + E_u.var(ddof=0)  # ddof=0 para igualar el resultado del ejercicio
    SD = math.sqrt(S2)

    summary = pd.DataFrame({"n":[n], "Ybar":[Ybar], "S2":[S2], "SD":[SD]})
    return per_airline, summary


def save_avg_histogram(df_bins: pd.DataFrame, out_png: str):
    """
    Histograma promedio: probabilidad media por bin [lower, upper] a través de aerolíneas.
    """
    bins = df_bins.groupby(["lower","upper"]).apply(lambda x: x["prob"].mean()).reset_index(name="avg_prob")
    mids = (bins["lower"] + bins["upper"]) / 2.0
    widths = (bins["upper"] - bins["lower"]) * 0.9

    plt.figure()
    plt.bar(mids, bins["avg_prob"], width=widths)
    plt.xlabel("Flight time")
    plt.ylabel("Average probability across airlines")
    plt.title("Exercise 2.1 — Average histogram (Table 2.4)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    base = Path(".")
    pdf_path = base / "4.1 Intro_SDA.pdf"  # Ajusta la ruta si el PDF está en otro sitio
    out_bins = base / "ex2_1_table_2_4_flight_times_bins.csv"
    out_per = base / "ex2_1_per_airline_stats.csv"
    out_summary = base / "ex2_1_summary_stats.csv"
    out_png = base / "ex2_1_avg_histogram.png"

    df_bins = parse_table_2_4_from_pdf(str(pdf_path), page_index_zero_based=7)
    df_bins.to_csv(out_bins, index=False)

    per_airline, summary = compute_sda_stats(df_bins)
    per_airline.to_csv(out_per, index=False)
    summary.to_csv(out_summary, index=False)

    save_avg_histogram(df_bins, str(out_png))

    print("Saved:")
    print(f"- {out_bins}")
    print(f"- {out_per}")
    print(f"- {out_summary}")
    print(f"- {out_png}")


if __name__ == "__main__":
    main()
