"""
# ==========================
# Doctorado en Tecnologias para la Transformacion Digital
# Materia: Big Data
# Instructor: Dr. Jonás Velasco Álvarez
# Estudiante: Luis Alejandro Santana Valadez
# ==========================
Ejercicio 4.3 — Reproducción del ejemplo de regresión lineal (Billard & Diday).
# ==========================
Pasos:
1) Parsear la Tabla 1 del PDF "4.3 Articulo__Regresion_SDA.pdf" 
2) Exportar la tabla a CSV canónico de bins
3) Calcular estadísticos de regresión lineal para las presiones arteriales de 11 personas
4) Guardar un histograma promedio (PNG) y salidas CSV de la tabla.

Dependencias:
  pip install PyPDF2 pandas numpy matplotlib
Genera:
- sda_regression_table1_base.csv  (datos Tabla 1)
- sda_regression_results.csv      (estadísticos y coeficientes)
- sda_regression_predictions.csv  (predicción para X1 in [118,126])
- fig_midpoints_regline.png       (dispersión midpoints + recta simbólica)
- fig_low_high_reglines.png       (rectas con extremos)
- fig_interval_prediction.png     (comparación de predicciones)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(".")

# -------------------------------
# 1) Datos de la Tabla 1 (artículo)
#    Y: Pulse Rate, X1: Systolic, X2: Diastolic
#    Intervalos [a,b] por paciente u=1..11
# -------------------------------
rows = [
 # u,   Y_low, Y_high,  X1_low, X1_high,  X2_low, X2_high
 ( 1,   44,     68,      90,      100,      50,     70 ),
 ( 2,   60,     72,      90,      130,      70,     90 ),
 ( 3,   56,     90,     140,      180,      90,    100 ),
 ( 4,   70,    112,     110,      142,      80,    108 ),
 ( 5,   54,     72,      90,      100,      50,     70 ),
 ( 6,   70,    100,     130,      160,      80,    110 ),
 ( 7,   63,     75,      60,      100,     140,    150 ),  # viola X2<=X1 -> excluir
 ( 8,   72,    100,     130,      160,      76,     90 ),
 ( 9,   76,     98,     110,      190,      70,    110 ),
 (10,   86,     96,     138,      180,      90,    110 ),
 (11,   86,    100,     110,      150,      78,    100 ),
]

cols = ["u","Y_low","Y_high","X1_low","X1_high","X2_low","X2_high"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUT/"sda_regression_table1_base.csv", index=False)

# Aplicar regla: excluir u=7 (X2 <= X1)
df = df[df["u"] != 7].reset_index(drop=True)

# Midpoints y widths (SDA asume uniforme dentro de intervalo)
for v in ["Y","X1","X2"]:
    df[f"{v}_mid"] = (df[f"{v}_low"] + df[f"{v}_high"]) / 2.0
    df[f"{v}_w"]   = (df[f"{v}_high"] - df[f"{v}_low"])

n = len(df)  # debería ser 10

# -------------------------------
# 2) Funciones SDA (univar y bivar)
# -------------------------------
def sda_mean(mid):
    return float(np.mean(mid))

def sda_var(mid, width):
    # S^2 = average(within) + var(mid)   (ddof=0 entre unidades)
    within = np.mean((width**2)/12.0)
    between = float(np.var(mid, ddof=0))
    return within + between

def sda_cov(mid_x, mid_y):
    # Con 1 rectángulo por unidad, la covarianza intra-bin es 0;
    # queda la covarianza entre medias por unidad (ddof=0).
    return float(np.cov(mid_x, mid_y, bias=True)[0,1])

def sda_corr(mid_x, mid_y, sx, sy):
    return sda_cov(mid_x, mid_y) / (sx*sy)

# -------------------------------
# 3) Estadísticos para Y ~ X1 (dos versiones)
# -------------------------------

# Medias de midpoints
Ybar  = float(df["Y_mid"].mean())
X1bar = float(df["X1_mid"].mean())

# --- Componentes dentro / entre ---
within_Y_mean  = float((df["Y_w"]**2 / 12.0).mean())       # promedio de varianzas intra-intervalo
within_X1_mean = float((df["X1_w"]**2 / 12.0).mean())

var_Y_mid  = float(df["Y_mid"].var(ddof=0))                # varianza entre-unidades (midpoints)
var_X1_mid = float(df["X1_mid"].var(ddof=0))

# --- Versión TOTAL (dentro + entre) ---
S2Y_total  = within_Y_mean  + var_Y_mid
S2X1_total = within_X1_mean + var_X1_mid
SY_total   = math.sqrt(S2Y_total)
SX1_total  = math.sqrt(S2X1_total)

# --- Versión ENTRE-MIDPOINTS (solo entre) ---
SY_mid  = math.sqrt(var_Y_mid)
SX1_mid = math.sqrt(var_X1_mid)

# --- Covarianza y correlaciones ---
# Con un rectángulo por unidad, la covarianza intra-bin es 0 -> covariamos midpoints
Cov_mid = float(np.cov(df["Y_mid"], df["X1_mid"], bias=True)[0,1])

# Correlación entre-midpoints (la que usa el artículo para la recta simbólica)
Corr_mid = Cov_mid / (SY_mid * SX1_mid)

# Correlación "total" si se usan desvíos totales en el denominador (útil para describir variabilidad total)
Corr_total = Cov_mid / (SY_total * SX1_total)

# --- Recta simbólica (usar SIEMPRE la varianza de midpoints en el denominador) ---
beta1 = Cov_mid / var_X1_mid
beta0 = Ybar - beta1 * X1bar


# -------------------------------
# 4) Predicción para X1 in [118,126]
# -------------------------------
x1_a, x1_b = 118.0, 126.0
y_hat_a = beta0 + beta1*x1_a
y_hat_b = beta0 + beta1*x1_b

# -------------------------------
# 5) Regresión con extremos
# -------------------------------
# a) sólo inferiores
coef_low = np.polyfit(df["X1_low"], df["Y_low"], 1)  # slope, intercept
beta1_low, beta0_low = coef_low[0], coef_low[1]
y_low_a = beta0_low + beta1_low*x1_a
y_low_b = beta0_low + beta1_low*x1_b

# b) sólo superiores
coef_up  = np.polyfit(df["X1_high"], df["Y_high"], 1)
beta1_up, beta0_up = coef_up[0], coef_up[1]
y_up_a = beta0_up + beta1_up*x1_a
y_up_b = beta0_up + beta1_up*x1_b

# -------------------------------
# 6) Regresión múltiple Y ~ X1 + X2 (midpoints)
# -------------------------------
X = np.column_stack([np.ones(n), df["X1_mid"], df["X2_mid"]])
y = df["Y_mid"].values
# beta = (X'X)^(-1) X'y
XtX = X.T @ X
Xty = X.T @ y
beta = np.linalg.solve(XtX, Xty)
beta0_m, beta1_m, beta2_m = map(float, beta)

# -------------------------------
# 7) Resultados a CSV
# -------------------------------
res = {
    "n": n,
    "Ybar": Ybar, "X1bar": X1bar,

    # Desvíos y varianzas: TOTAL (dentro+entre) vs ENTRE-midpoints
    "SY_total": SY_total, "SX1_total": SX1_total,
    "S2Y_total": S2Y_total, "S2X1_total": S2X1_total,
    "SY_mid": SY_mid, "SX1_mid": SX1_mid,
    "S2Y_mid": var_Y_mid, "S2X1_mid": var_X1_mid,

    # Covarianza entre midpoints (única que corresponde aquí)
    "Cov_mid(Y,X1)": Cov_mid,

    # Correlaciones: entre-midpoints (paper) y 'total' (descriptiva)
    "Corr_mid(Y,X1)": Corr_mid,
    "Corr_total(Y,X1)": Corr_total,

    # Recta simbólica (paper) basada en varianza de midpoints
    "beta0_sym": beta0, "beta1_sym": beta1,

    # Predicciones simbólicas (ya calculadas más abajo en tu script)
    "pred_sym_low118": y_hat_a, "pred_sym_high126": y_hat_b,

    # Rectas con extremos (ya las tienes calculadas)
    "beta0_low": beta0_low, "beta1_low": beta1_low,
    "pred_low_low118": y_low_a, "pred_low_high126": y_low_b,
    "beta0_up": beta0_up, "beta1_up": beta1_up,
    "pred_up_low118": y_up_a, "pred_up_high126": y_up_b,

    # Regresión múltiple (ya calculada)
    "beta0_mult": beta0_m, "beta1_mult": beta1_m, "beta2_mult": beta2_m,
}

pd.DataFrame([res]).to_csv(OUT/"sda_regression_results.csv", index=False)

pred = pd.DataFrame({
 "model": ["symbolic","lower-only","upper-only"],
 "Y_at_118": [y_hat_a, y_low_a, y_up_a],
 "Y_at_126": [y_hat_b, y_low_b, y_up_b]
})
pred.to_csv(OUT/"sda_regression_predictions.csv", index=False)

# -------------------------------
# 8) Gráficas
# -------------------------------
# (i) Midpoints + recta simbólica
plt.figure(figsize=(6,5))
plt.scatter(df["X1_mid"], df["Y_mid"])
xline = np.linspace(df["X1_mid"].min()-5, df["X1_mid"].max()+5, 200)
yline = beta0 + beta1*xline
plt.plot(xline, yline)
plt.title("Midpoints (X1 vs Y) + Recta simbólica")
plt.xlabel("Systolic (mid)")
plt.ylabel("Pulse (mid)")
plt.tight_layout()
plt.savefig(OUT/"fig_midpoints_regline.png", dpi=150)
plt.close()

# (ii) Rectas con extremos inferiores y superiores
plt.figure(figsize=(6,5))
plt.scatter(df["X1_low"],  df["Y_low"])
plt.scatter(df["X1_high"], df["Y_high"])
xgrid = np.linspace(min(df["X1_low"].min(), df["X1_high"].min())-5,
                    max(df["X1_low"].max(), df["X1_high"].max())+5, 200)
plt.plot(xgrid, beta0_low + beta1_low*xgrid)
plt.plot(xgrid, beta0_up  + beta1_up*xgrid)
plt.title("Rectas con extremos (inferior y superior)")
plt.xlabel("Systolic")
plt.ylabel("Pulse")
plt.tight_layout()
plt.savefig(OUT/"fig_low_high_reglines.png", dpi=150)
plt.close()

# (iii) Predicción para X1 in [118,126]
plt.figure(figsize=(6,5))
# franja de X
plt.axvspan(x1_a, x1_b, alpha=0.15)
# líneas horizontales de predicción
plt.hlines([y_hat_a, y_hat_b], x1_a, x1_b)
plt.hlines([y_low_a, y_low_b], x1_a, x1_b)
plt.hlines([y_up_a,  y_up_b],  x1_a, x1_b)
plt.title("Predicción de Pulse para X1∈[118,126]")
plt.xlabel("Systolic")
plt.ylabel("Predicted Pulse")
plt.tight_layout()
plt.savefig(OUT/"fig_interval_prediction.png", dpi=150)
plt.close()
