import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os

#Inicialización del modelo

M = pyo.ConcreteModel()

#Definición de parámetros
#pro -> Lista de prosumidores del sistema.
#tran -> Lista de transacciones posibles. Un prosumidor no puede comerciar consigo mismo.
#ts -> Lista de ventanas de tiempo.
#dt -> Duración de la ventana de tiempo.
#gE -> Energía generada por cada prosumidor en kWh.
#lE -> Energía consumida por cada prosumidor en kWh.
#GBP -> Precio de compra a la red (grid), en EUR/kWh. Grid Buy Price.
#GSP -> Precio de venta a la red (grid), en EUR/kWh. Se asume mayor al GBP. Grid Sell Price.
#P2P -> Precio de compra/venta al mercado P2P, en EUR/kWh. Se toma la media entre el GSP y el GBP.

M.pro = pyo.RangeSet(1,5)
M.tran = pyo.Set(dimen=2, initialize =
                 [(i,j) for i in M.pro for j in M.pro if i!=j])
M.ts = pyo.RangeSet(1,5)
M.dt = pyo.Param(initialize=(15/60))

df_gE = pd.read_csv("../Datos/Gen_pw.csv", sep=',').to_numpy()
df_lE = pd.read_csv("../Datos/Con_pw.csv", sep=',').to_numpy()

M.gE = pyo.Param(M.pro, M.ts, initialize = lambda 
             model, i, t: df_gE[t-1,i-1]*M.dt,
             within = pyo.NonNegativeReals)
M.lE = pyo.Param(M.pro, M.ts, initialize = lambda
             model, i, t: df_lE[t-1,i-1]*M.dt,
             within = pyo.NonNegativeReals)

df_GBP = pd.read_csv("../Datos/precio_compra_red.csv", sep=',').to_numpy()

M.GBP = pyo.Param(M.ts, initialize = lambda
              model, t: df_GBP[t-1],
              within = pyo.PositiveReals)
M.GSP = pyo.Param(initialize=0.07,
                  within = pyo.PositiveReals)
#M.P2P = pyo.Param(M.ts, initialize = lambda
#               model, t:((M.GBP[t]+M.GSP)/2),
#               within = pyo.PositiveReals)
M.maxE = pyo.Param(initialize=10.0)

#Definición de variables
#GBE -> Energía comprada a la red (grid), en kWh. Grid Buyed Energy.
#GSE -> Energía vendida a la red (grid), en kWh. Grid Selled Energy.
#P2BE -> Energía comprada en el mercado P2P, en kWh.
#P2SE -> Energía vendida en el mercado P2P, en kWh.
#GB -> Variable binaria que determina si se compra a la red (Grid).
#GB -> Variable binaria que determina si se vende a la red (Grid).
#PB -> Variable binaria que determina si se compra al mercado P2P. 
#PS -> Variable binaria que determina si se vende al mercado P2P. 

M.GBE = pyo.Var(M.pro, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.GSE = pyo.Var(M.pro, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.P2BE = pyo.Var(M.tran, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.P2SE = pyo.Var(M.tran, M.ts, domain=pyo.PositiveReals, bounds=(0.0,M.maxE))
M.GB = pyo.Var(M.pro, M.ts, domain=pyo.Binary)
M.GS = pyo.Var(M.pro, M.ts, domain=pyo.Binary)
M.PB = pyo.Var(M.tran, M.ts, domain=pyo.Binary)
M.PS = pyo.Var(M.tran, M.ts, domain=pyo.Binary)

#Definición de la función objetivo.

M.Z = pyo.Objective(
    expr=sum(M.GBE[i,t]*M.GBP[t] for i in M.pro for t in M.ts) - 
    sum(M.GSE[i,t]*M.GSP for i in M.pro for t in M.ts),
    sense=pyo.minimize)

#Definición de las restricciones
#eq1: Balance de potencia. Lo que se genera debe ser mayor igual a lo que se consume.
#eq2: Establece que solo se puede comprar de la red (Grid) cuando GB = 1.
#eq3: Establece que solo se puede vender de la red (Grid) cuando GS = 1.
#eq4: Establece que solo se puede comprar o vender de la red (Grid).
#eq5: Establece que solo se puede comprar del mercado p2p cuando PB = 1.
#eq6: Establece que solo se puede vender mercado p2p cuando PS = 1.
#eq7: Establece que no se puede comprar de la red (Grid) para vender al mecado p2p.
#eq8: Establece que no se puede comprar del mercado p2p para vender a la red (Grid).
#eq9: Lo que se compra al mercado p2p debe ser igual que lo que se vende.
#eq10-11: Se aseguran de que los prosumidores solo comercian con un prosumidor.

def eq1_rule(model,i,t):
    return (M.gE[t,i] + M.GBE[i,t] + sum(M.P2BE[i,j,t] for j in M.pro if j != i) -
            M.lE[t,i] - M.GSE[i,t] - sum(M.P2SE[i,j,t] for j in M.pro if j != i) ==
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

#Resolvemos el modelo

solver = pyo.SolverFactory('glpk')
result = solver.solve(M)

#print(result.solver.termination_condition)
if result.solver.status == pyo.SolverStatus.ok:
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        print('Se ha llegado a la solución óptima.')
    else:
        print('Se ha llegado a una solución factible.')
    # Lista para almacenar las filas
    filas = []

# Iterar sobre los períodos y prosumidores
    for t in M.ts:
        for i in M.pro:
            # Calcular compras y ventas P2P
            compra_p2p = sum(pyo.value(M.P2BE[i,j,t]) for j in M.pro if j != i)
            venta_p2p = sum(pyo.value(M.P2SE[i,j,t]) for j in M.pro if j != i)
        
            # Calcular el balance total de energía
            balance_total = (
                M.gE[t,i] + pyo.value(M.GBE[i,t]) - pyo.value(M.GSE[i,t]) +
                compra_p2p - venta_p2p
                )
        
            # Añadir la fila a la lista
            filas.append({
                'Ts': t,
                'Prosumer': i,
                'Pgen': M.gE[t,i],
                'P_buyed_grid': pyo.value(M.GBE[i,t]),
                'P_selled_grid': pyo.value(M.GSE[i,t]),
                'P_buyed_P2P': compra_p2p,
                'P_selled_P2P': venta_p2p,
                'P_total': balance_total,
                'P_demand': M.lE[t,i]
                })

    # Crear el DataFrame una sola vez
    df_cons = pd.DataFrame(filas)

    # Guardar el DataFrame en un archivo CSV
    archivo_result = "../Resultados/p1_nobattery.csv"
    if os.path.exists(archivo_result):
        os.remove(archivo_result)
    df_cons.to_csv(archivo_result, index=False)

    print(f"Los resultados se han guardado en {archivo_result}.")

else:
    print('No se ha encontrado solución.')