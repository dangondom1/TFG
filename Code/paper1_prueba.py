import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
from pathlib import Path
import matplotlib.pyplot as plt # <-- 1. IMPORTACIÓN AÑADIDA

#Cambio de directorio al ejecutar el código
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

#Inicialización del modelo
print(os.getcwd())
M = pyo.ConcreteModel()

# --- Definición de parámetros (tu código original) ---
M.pro = pyo.RangeSet(1,5)
M.tran = pyo.Set(dimen=2, initialize =
                 [(i,j) for i in M.pro for j in M.pro if i!=j])
M.ts = pyo.RangeSet(1,4)
M.dt = pyo.Param(initialize=(15/60))

df_gE = pd.read_csv("../Data/Gen_pw.csv", sep=',').to_numpy()
df_lE = pd.read_csv("../Data/Con_pw.csv", sep=',').to_numpy()

M.gE = pyo.Param(M.pro, M.ts, initialize = lambda 
             model, i, t: df_gE[t-1,i-1]*M.dt,
             within = pyo.NonNegativeReals)
M.lE = pyo.Param(M.pro, M.ts, initialize = lambda
             model, i, t: df_lE[t-1,i-1]*M.dt,
             within = pyo.NonNegativeReals)

df_GBP = pd.read_csv("../Data/precio_compra_red.csv", sep=',').to_numpy()

M.GBP = pyo.Param(M.ts, initialize = lambda
              model, t: df_GBP[t-1],
              within = pyo.PositiveReals)
M.GSP = pyo.Param(initialize=0.07,
                  within = pyo.PositiveReals)
M.P2P = pyo.Param(M.ts, initialize = lambda
               model, t:((M.GBP[t]+M.GSP)/2),
               within = pyo.PositiveReals)
M.maxE = pyo.Param(initialize=10.0)
M.maxEbat = pyo.Param(initialize=10.0)
M.maxEch = pyo.Param(initialize=8.0)
M.maxEdch = pyo.Param(initialize=8.0)
M.batcheff = pyo.Param(initialize=0.9)
M.batdcheff = pyo.Param(initialize=0.9)

# --- Definición de variables (tu código original) ---
M.GBE = pyo.Var(M.pro, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.GSE = pyo.Var(M.pro, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.P2BE = pyo.Var(M.tran, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.P2SE = pyo.Var(M.tran, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.GB = pyo.Var(M.pro, M.ts, domain=pyo.Binary)
M.GS = pyo.Var(M.pro, M.ts, domain=pyo.Binary)
M.PB = pyo.Var(M.tran, M.ts, domain=pyo.Binary)
M.PS = pyo.Var(M.tran, M.ts, domain=pyo.Binary)
M.Ebat = pyo.Var(M.pro, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxEbat))
M.Ech = pyo.Var(M.pro, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.Edch = pyo.Var(M.pro, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.Bch = pyo.Var(M.pro, M.ts, domain=pyo.Binary)
M.Bdch = pyo.Var(M.pro, M.ts, domain=pyo.Binary)

# --- Definición de la función objetivo (tu código original) ---
M.Z = pyo.Objective(
    expr=sum(M.GBE[i,t]*M.GBP[t] for i in M.pro for t in M.ts) - 
    sum(M.GSE[i,t]*M.GSP for i in M.pro for t in M.ts),
    sense=pyo.minimize)

# --- Definición de las restricciones (tu código original) ---
def eq1_rule(model,i,t):
    return (M.gE[i,t] + M.GBE[i,t] + M.Edch[i,t] + sum(M.P2BE[i,j,t] for j in M.pro if j != i) -
            M.lE[i,t] - M.GSE[i,t] - M.Ech[i,t] - sum(M.P2SE[i,j,t] for j in M.pro if j != i) ==
            0.0)
M.eq1 = pyo.Constraint(M.pro, M.ts, rule=eq1_rule)

def eq2_rule(model,i,t):
    return (M.maxE*M.GB[i,t] - M.GBE[i,t] >=
            0.0)
M.eq2 = pyo.Constraint(M.pro, M.ts, rule=eq2_rule)

def eq3_rule(model,i,t):
    return (M.maxE*M.GS[i,t] - M.GSE[i,t] >=
            0.0)
M.eq3 = pyo.Constraint(M.pro, M.ts, rule=eq3_rule)

def eq4_rule(model,i,t):
    return (M.GB[i,t] + M.GS[i,t] - 1.0 <=
            0.0)
M.eq4 = pyo.Constraint(M.pro, M.ts, rule=eq4_rule)

def eq5_rule(model,i,j,t):
    return(M.maxE*M.PB[i,j,t] - M.P2BE[i,j,t] >=
           0.0)
M.eq5 = pyo.Constraint(M.tran, M.ts, rule=eq5_rule)

def eq6_rule(model,i,j,t):
    return(M.maxE*M.PS[i,j,t] - M.P2SE[i,j,t] >=
           0.0)
M.eq6 = pyo.Constraint(M.tran, M.ts, rule=eq6_rule)

def eq7_rule(model,i,t):
    return(M.GB[i,t] + sum(M.PS[i,j,t] for j in M.pro if j != i) - 1.0 <=
           0.0)
M.eq7 = pyo.Constraint(M.pro, M.ts, rule=eq7_rule)

def eq8_rule(model,i,t):
    return(M.GS[i,t] + sum(M.PB[i,j,t] for j in M.pro if j != i) - 1.0 <=
           0.0)
M.eq8 = pyo.Constraint(M.pro, M.ts, rule=eq8_rule)

def eq9_rule(model,t):
    return(sum(M.P2BE[i,j,t] for (i,j) in M.tran) - sum(M.P2SE[i,j,t] for (i,j) in M.tran) ==
           0.0)
M.eq9 = pyo.Constraint(M.ts, rule = eq9_rule)

def eq10_rule(model,i,t):
    return(sum(M.PB[i,j,t] for j in M.pro if j != i) + sum(M.PS[j,i,t] for j in M.pro if j != i) - 2.0 <=
           0.0)
M.eq10 = pyo.Constraint(M.pro, M.ts, rule=eq10_rule)

def eq11_rule(model,j,t):
    return(sum(M.PB[i,j,t] for i in M.pro if i != j) + sum(M.PS[j,i,t] for i in M.pro if i != j) - 2.0 <= 
           0.0)
M.eq11 = pyo.Constraint(M.pro, M.ts, rule=eq11_rule)

def eq12_rule(model,i,t):
    if t == M.ts.first():
        return(M.Ebat[i,t] == M.maxEbat*0.8)
    else:
        return(M.Ebat[i,t] - M.Ebat[i,t-1] - M.Ech[i,t]*M.batcheff + M.Edch[i,t]*1/M.batdcheff == 0.0)
M.eq12 = pyo.Constraint(M.pro, M.ts, rule=eq12_rule)

def eq13_rule(model,i,t):
    return(M.Bch[i,t] + M.Bdch[i,t] <= 1)
M.eq13 = pyo.Constraint(M.pro, M.ts, rule=eq13_rule)

def eq14_rule(model,i,t):
    return(M.Ech[i,t] - M.maxEch*M.Bch[i,t] <= 0.0)
M.eq14 = pyo.Constraint(M.pro, M.ts, rule=eq14_rule)

def eq15_rule(model,i,t):
    return(M.Edch[i,t] - M.maxEdch*M.Bdch[i,t] <= 0.0)
M.eq15 = pyo.Constraint(M.pro, M.ts, rule=eq15_rule)


#Resolvemos el modelo
solver = pyo.SolverFactory('glpk')
result = solver.solve(M)

if result.solver.status == pyo.SolverStatus.ok:
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        print('Se ha llegado a la solución óptima.')
    else:
        print('Se ha llegado a una solución factible.')
    
    # --- Creación del DataFrame de resultados (tu código original) ---
    filas = []
    for t in M.ts:
        for i in M.pro:
            compra_p2p = sum(pyo.value(M.P2BE[i,j,t]) for j in M.pro if j != i)
            venta_p2p = sum(pyo.value(M.P2SE[i,j,t]) for j in M.pro if j != i)
            balance_total = (
                M.gE[i,t] + pyo.value(M.GBE[i,t]) - pyo.value(M.GSE[i,t]) +
                compra_p2p - venta_p2p +
                pyo.value(M.Edch[i,t]) - pyo.value(M.Ech[i,t])
                )
            filas.append({
                'Ts': t,
                'Prosumer': i,
                'Pgen': M.gE[i,t],
                'P_buyed_grid': pyo.value(M.GBE[i,t]),
                'P_selled_grid': pyo.value(M.GSE[i,t]),
                'P_buyed_P2P': compra_p2p,
                'P_selled_P2P': venta_p2p,
                'P_total': balance_total,
                'P_demand': M.lE[i,t]
                })
    df_cons = pd.DataFrame(filas)
    archivo_result = "../Results/p1_nobattery.csv"
    if os.path.exists(archivo_result):
        os.remove(archivo_result)
    df_cons.to_csv(archivo_result, index=False)
    print(f"Los resultados se han guardado en {archivo_result}.")

    # -----------------------------------------------------------------------------------
    # --- INICIO DE LA MODIFICACIÓN: CÁLCULO Y GRÁFICA DE ENERGÍA TOTAL P2P ---
    # -----------------------------------------------------------------------------------

    # 1. Calcular la energía total comerciada en el mercado P2P para cada intervalo
    total_p2p_traded_per_ts = []
    for t in M.ts:
        # Sumamos toda la energía vendida (P2SE) en el mercado P2P en el tiempo t.
        # Podríamos usar P2BE y el resultado sería el mismo gracias a tu restricción eq9.
        total_in_t = sum(pyo.value(M.P2SE[i, j, t]) for i, j in M.tran)
        total_p2p_traded_per_ts.append(total_in_t)

    # Imprimir los valores para verificación
    print("\nEnergía total comerciada en el mercado P2P por intervalo de tiempo:")
    for t, energy in zip(M.ts, total_p2p_traded_per_ts):
        print(f"  Intervalo {t}: {energy:.2f} kWh")

    # 2. Generar la gráfica con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear el gráfico de LÍNEA en lugar de barras
    # Se añade un marcador 'o' para señalar claramente los puntos de datos
    ax.plot(list(M.ts), total_p2p_traded_per_ts, 
            color="#0B21E9", 
            marker='o', 
            linestyle='-', 
            linewidth=2, 
            markersize=8,
            label='Energía P2P Comerciada')

    # Añadir títulos y etiquetas para que sea claro
    ax.set_title('Energía Total Comerciada en el Mercado P2P', fontsize=16, weight='bold')
    ax.set_xlabel('Intervalo de Tiempo (ts)', fontsize=12)
    ax.set_ylabel('Energía Total Comerciada (kWh)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(list(M.ts)) # Asegura que todos los intervalos se muestren en el eje X
    ax.legend()
    ax.set_facecolor("#FFFFFFCA")

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()


else:
    print('No se ha encontrado solución.')