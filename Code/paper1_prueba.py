import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
from pathlib import Path
import matplotlib.pyplot as plt

#Cambio de directorio al ejecutar el código
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

#Inicialización del modelo
print(os.getcwd())
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
#MaxE -> Máxima energía transportada.
#MaxEbat -> Energía máxima presente en las baterías.
#MaxEch -> Energía máxima cargada por una batería.
#MaxEdch -> Energía máxima descargada por una batería.
#batcheff -> Eficiencia de carga de las baterías.
#batdcheff -> Eficiencia de descarga de las baterías.
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

#Definición de variables
#GBE -> Energía comprada a la red (grid), en kWh. Grid Buyed Energy.pyo.value(M.Ech[i,t])
#GSE -> Energía vendida a la red (grid), en kWh. Grid Selled Energy.
#P2BE -> Energía comprada en el mercado P2P, en kWh.
#P2SE -> Energía vendida en el mercado P2P, en kWh.
#GB -> Variable binaria que determina si se compra a la red (Grid).
#GS -> Variable binaria que determina si se vende a la red (Grid).
#PB -> Variable binaria que determina si se compra al mercado P2P. 
#PS -> Variable binaria que determina si se vende al mercado P2P. 
#Ebat -> Energía presente en las baterías.
#Ech -> Energía cargada en las baterías.
#Edch -> Energía descargada en las baterías.
#Bch -> Variable binaria para decidir si una batería se carga.
#Bdch -> Variable binaria para decidir si una batería se descarga.
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
        return(M.Ebat[i,t] - M.maxEbat*0.8 - M.Ech[i,t]*M.batcheff + M.Edch[i,t]*1/M.batdcheff == 0.0)
    else:
        return(M.Ebat[i,t] - M.Ebat[i,M.ts.prev(t)] - M.Ech[i,t]*M.batcheff + M.Edch[i,t]*1/M.batdcheff == 0.0)
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
solver.options['mipgap'] = 1e-5
result = solver.solve(M, tee=True)

if result.solver.status == pyo.SolverStatus.ok:
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        print('Se ha llegado a la solución óptima.')
    else:
        print('Se ha llegado a una solución factible.')
    
    # --- Creación del DataFrame de resultados ---
    filas = []
    for t in M.ts:
        for i in M.pro:
            compra_p2p = sum(pyo.value(M.P2BE[i,j,t]) for j in M.pro if j != i)
            venta_p2p = sum(pyo.value(M.P2SE[i,j,t]) for j in M.pro if j != i)
            bateria_soc_percent = (pyo.value(M.Ebat[i, t]) / M.maxEbat.value) * 100
            balance_total = (
                M.gE[i,t] + pyo.value(M.GBE[i,t]) - pyo.value(M.GSE[i,t]) +
                compra_p2p - venta_p2p +
                pyo.value(M.Edch[i,t]) - pyo.value(M.Ech[i,t])
                )
            filas.append({
                'Ts': t,
                'Prosumer': i,
                'Pgen_kWh': M.gE[i,t],
                'P_buy_grid_kWh': pyo.value(M.GBE[i,t]),
                'P_sell_grid_kWh': pyo.value(M.GSE[i,t]),
                'P_buy_P2P_kWh': compra_p2p,
                'P_sell_P2P_kWh': venta_p2p,
                'Bateria_Energia_kWh': pyo.value(M.Ebat[i, t]),
                'Bateria_SoC_%': bateria_soc_percent,
                'P_demand_kWh': M.lE[i,t],
                'Balance_kWh': balance_total
                })
    df_resultados = pd.DataFrame(filas)
    archivo_result = "../Results/resultados_completos.csv"
    if os.path.exists(archivo_result):
        os.remove(archivo_result)
    df_resultados.to_csv(archivo_result, index=False, float_format='%.2f')
    print(f"\nTodos los resultados se han guardado en {archivo_result}.")


    # -----------------------------------------------------------------------------------
    # --- INICIO DE LA MODIFICACIÓN: GRÁFICA EN SUBPLOTS PARA EL PROSUMIDOR 1 ---
    # -----------------------------------------------------------------------------------
    
    # 1. Definir el prosumidor que queremos analizar
    target_prosumer = 1

    # 2. Preparar listas para almacenar los datos del prosumidor seleccionado
    time_steps = list(M.ts)
    demand_p1 = []
    grid_buy_p1 = []
    p2p_buy_p1 = []
    grid_sell_p1 = []
    charge_p1 = []
    discharge_p1 = []

    # 3. Extraer los datos para cada intervalo de tiempo para ESE prosumidor
    for t in time_steps:
        demand_p1.append(pyo.value(M.lE[target_prosumer, t]))
        grid_buy_p1.append(pyo.value(M.GBE[target_prosumer, t]))
        grid_sell_p1.append(pyo.value(M.GSE[target_prosumer, t]))
        charge_p1.append(pyo.value(M.Ech[target_prosumer, t]))
        discharge_p1.append(pyo.value(M.Edch[target_prosumer, t]))
        
        # La compra P2P del prosumidor 1 es la suma de lo que compra a los demás
        p2p_buy_in_t = sum(pyo.value(M.P2BE[target_prosumer, j, t]) for j in M.pro if j != target_prosumer)
        p2p_buy_p1.append(p2p_buy_in_t)

    # 4. Crear la figura y la parrilla de subplots (3 filas, 2 columnas)
    # sharex=True hace que todos los subplots compartan el mismo eje X
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Evolución Energética Detallada del Prosumidor {target_prosumer}', fontsize=16, weight='bold')

    # Diccionarios para configurar cada subplot de forma limpia
    plot_configs = {
        'Demanda': {'data': demand_p1, 'ax': axs[0, 0], 'color': 'red'},
        'Compra a Red': {'data': grid_buy_p1, 'ax': axs[0, 1], 'color': 'darkorange'},
        'Compra P2P': {'data': p2p_buy_p1, 'ax': axs[1, 0], 'color': 'green'},
        'Venta a Red': {'data': grid_sell_p1, 'ax': axs[1, 1], 'color': 'skyblue'},
        'Carga Batería': {'data': charge_p1, 'ax': axs[2, 0], 'color': 'purple'},
        'Descarga Batería': {'data': discharge_p1, 'ax': axs[2, 1], 'color': 'blue'}
    }

    # 5. Dibujar cada variable en su respectivo subplot
    for title, config in plot_configs.items():
        ax = config['ax']
        ax.plot(time_steps, config['data'], color=config['color'], marker='o', linestyle='-')
        ax.set_title(title)
        ax.set_ylabel('Energía (kWh)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(time_steps)

    # 6. Añadir etiquetas solo a los subplots inferiores para no saturar la vista
    axs[2, 0].set_xlabel('Intervalo de Tiempo (ts)')
    axs[2, 1].set_xlabel('Intervalo de Tiempo (ts)')

    # Ajustar el layout para que no se solapen los títulos y etiquetas
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Dejar espacio para el suptitle
    plt.show()

    # --- FIN DE LA MODIFICACIÓN ---
    # -----------------------------------------------------------------------------------

else:
    print('No se ha encontrado solución.')