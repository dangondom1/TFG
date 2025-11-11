import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN INICIAL ---
# Nota: La ejecución de esta sección puede fallar si los archivos de datos no están presentes.
BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(BASE_DIR)

# --- CREACIÓN DEL MODELO ---
M = pyo.ConcreteModel()
 
# --- CREACIÓN DE SETS ---
N_prosumers = int(3)
N_ts = int(5)
test = 1

# --- LECTURA Y DEFINICIÓN DE PARÁMETROS ---
# Datos de ejemplo para que el modelo sea ejecutable
M.Ni = pyo.RangeSet(0, N_prosumers - 1, 1)
M.Nt = pyo.RangeSet(0, N_ts-1, 1)
dt = 15 / 60
# Se asume que los archivos de datos están en un directorio llamado 'Data' en el nivel superior.
# Si la estructura es diferente, ajusta las rutas.
try:
    df_generated_energy = pd.read_csv(f"./Data/Prueba_{test}/Gen_pw.csv", sep=',').to_numpy()
    df_load_energy = pd.read_csv(f"./Data/Prueba_{test}/Cons_pw.csv", sep=',').to_numpy()
    df_grid_buy_price = pd.read_csv("./Data/buy_price.csv", sep=',').to_numpy()
    has_bat = np.random.choice([0, 1], size=N_prosumers, p=[0.5, 0.5])
    battery_cap = np.random.choice([5, 10, 15, 20], size=N_prosumers, p=[0.25, 0.25, 0.25, 0.25])
except FileNotFoundError:
    print("Advertencia: No se encontraron los archivos de datos. Usando datos aleatorios para la ejecución.")
    df_generated_energy = np.random.rand(N_ts, N_prosumers) * 5
    df_load_energy = np.random.rand(N_ts, N_prosumers) * 4
    df_grid_buy_price = np.random.rand(N_ts, 1) * 0.3 + 0.1


# Creación de diccionarios para inicializar parámetros
generated_energy_data = {(t, i): df_generated_energy[t, i] for t in M.Nt for i in M.Ni}
load_energy_data = {(t, i): df_load_energy[t, i] for t in M.Nt for i in M.Ni}
grid_buy_price_data = {t: df_grid_buy_price[t,0] for t in M.Nt} # Ajustado para asegurar que es un escalar
grid_sell_price_data = 0.19
p2p_price_data = {t: (grid_buy_price_data[t] + grid_sell_price_data) / 2 for t in M.Nt}
max_energy_data = 10.0
max_battery_energy_data = 20.0
max_energy_charge_data = 8.0
max_energy_discharge_data = 8.0
battery_charge_efficiency_data = 0.9
battery_discharge_efficiency_data = 0.9

# Asignación de parámetros al modelo
M.P_gen = pyo.Param(M.Nt, M.Ni, initialize=generated_energy_data, within=pyo.NonNegativeReals)
M.P_load = pyo.Param(M.Nt, M.Ni, initialize=load_energy_data, within=pyo.NonNegativeReals)
M.pi_buyGrid = pyo.Param(M.Nt, initialize=grid_buy_price_data, within=pyo.Reals)
M.pi_sellGrid = pyo.Param(initialize=grid_sell_price_data, within=pyo.NonNegativeReals)

M.pi_p2p = pyo.Param(M.Nt, initialize=p2p_price_data, within=pyo.Reals)
M.delta_t = pyo.Param(initialize=dt, within=pyo.NonNegativeReals)
M.p2p_pairs = pyo.Set(initialize=M.Ni * M.Ni, filter=lambda model, i, j: i != j)
M.P_max_buyGrid = pyo.Param(initialize=8.625, within=pyo.NonNegativeReals)
M.P_max_sellGrid = pyo.Param(initialize=4.3125, within=pyo.NonNegativeReals)
M.P_max_buy_p2p = pyo.Param(initialize=8.625, within=pyo.NonNegativeReals)
M.P_max_sell_p2p = pyo.Param(initialize=8.625, within=pyo.NonNegativeReals)
M.P_max_ch = pyo.Param(initialize=max_energy_charge_data, within=pyo.NonNegativeReals)
M.P_max_dch = pyo.Param(initialize=max_energy_discharge_data, within=pyo.NonNegativeReals)
M.E_max_Bat = pyo.Param(initialize=max_battery_energy_data, within=pyo.NonNegativeReals)
M.n_ch = pyo.Param(initialize=battery_charge_efficiency_data, within=pyo.NonNegativeReals)
M.n_dch = pyo.Param(initialize=battery_discharge_efficiency_data, within=pyo.NonNegativeReals)

# --- DEFINICIÓN DE VARIABLES ---
M.P_buyGrid = pyo.Var(M.Nt, M.Ni, bounds=(0.0, M.P_max_buyGrid))
M.P_sellGrid = pyo.Var(M.Nt, M.Ni, bounds=(0.0, M.P_max_sellGrid))
M.P_buy_p2p = pyo.Var(M.Nt, M.p2p_pairs, bounds=(0.0, M.P_max_buy_p2p))
M.P_sell_p2p = pyo.Var(M.Nt, M.p2p_pairs, bounds=(0.0, M.P_max_sell_p2p))
M.E_bat = pyo.Var(M.Nt, M.Ni)
M.P_ch = pyo.Var(M.Nt, M.Ni, bounds=(0.0, M.P_max_ch))
M.P_dch = pyo.Var(M.Nt, M.Ni, bounds=(0.0, M.P_max_dch))

M.B_buyGrid = pyo.Var(M.Nt, M.Ni, within=pyo.Binary)
M.B_sellGrid = pyo.Var(M.Nt, M.Ni, within=pyo.Binary)
M.B_buy_p2p = pyo.Var(M.Nt, M.p2p_pairs, within=pyo.Binary)
M.B_sell_p2p = pyo.Var(M.Nt, M.p2p_pairs, within=pyo.Binary)
M.B_ch = pyo.Var(M.Nt, M.Ni, within=pyo.Binary)
M.B_dch = pyo.Var(M.Nt, M.Ni, within=pyo.Binary)

# --- FUNCIÓN OBJETIVO ---

M.obj = pyo.Objective(
    expr=sum(M.P_buyGrid[t, i] * M.pi_buyGrid[t] * M.delta_t for t in M.Nt for i in M.Ni) -
         sum(M.P_sellGrid[t, i] * M.pi_sellGrid * M.delta_t for t in M.Nt for i in M.Ni),
sense = pyo.minimize)

# --- RESTRICCIONES ---

# ---- RESTRICCIONES DE BALANCE ----

def energy_balance_rule(model, t, i):
    buy_p2p_sum = sum(model.P_buy_p2p[t, (i, j)] for j in model.Ni if j != i)
    sell_p2p_sum = sum(model.P_sell_p2p[t, (i, j)] for j in model.Ni if j != i)
    return (model.P_gen[t, i] + model.P_buyGrid[t, i] + buy_p2p_sum + model.P_dch[t, i] ==
            model.P_load[t, i] + model.P_sellGrid[t, i] + sell_p2p_sum + model.P_ch[t, i])

M.energy_balance = pyo.Constraint(M.Nt, M.Ni, rule=energy_balance_rule)

# ---- RESTRICCIONES DE GRID ----

def grid_buy_limit_rule(model, t, i):
    return model.P_buyGrid[t, i] <= model.B_buyGrid[t, i] * model.P_max_buyGrid
M.grid_buy_limit = pyo.Constraint(M.Nt, M.Ni, rule=grid_buy_limit_rule)

def grid_sell_limit_rule(model, t, i):
    return model.P_sellGrid[t, i] <= model.B_sellGrid[t, i] * model.P_max_sellGrid
M.grid_sell_limit = pyo.Constraint(M.Nt, M.Ni, rule=grid_sell_limit_rule)

def grid_buy_sell_rule(model, t, i):
    return model.B_buyGrid[t, i] + model.B_sellGrid[t, i] <= 1
M.grid_buy_sell = pyo.Constraint(M.Nt, M.Ni, rule=grid_buy_sell_rule)

# ---- RESTRICCIONES DE P2P ----

def p2p_buy_limit_rule(model, t, i, j):
    return model.P_buy_p2p[t, (i, j)] <= model.B_buy_p2p[t, (i, j)] * model.P_max_buy_p2p
M.p2p_buy_limit = pyo.Constraint(M.Nt, M.p2p_pairs, rule=p2p_buy_limit_rule)

def p2p_sell_limit_rule(model, t, i, j):
    return model.P_sell_p2p[t, (i, j)] <= model.B_sell_p2p[t, (i, j)] * model.P_max_sell_p2p
M.p2p_sell_limit = pyo.Constraint(M.Nt, M.p2p_pairs, rule=p2p_sell_limit_rule)

def p2p_buy_sell_rule(model, t, i):
    return model.B_buyGrid[t, i] + sum(model.B_sell_p2p[t, i, j] for j in M.Ni if j != i) <= 1
M.p2p_buy_sell = pyo.Constraint(M.Nt, M.Ni, rule=p2p_buy_sell_rule)

def p2p_sell_buy_rule(model, t, i):
    return model.B_sellGrid[t, i] + sum(model.B_buy_p2p[t, i, j] for j in M.Ni if j != i) <= 1
M.p2p_sell_buy = pyo.Constraint(M.Nt, M.Ni, rule=p2p_sell_buy_rule)

def p2p_balance_rule(model, t):
    buy_p2p_sum = sum(model.P_buy_p2p[t, (i, j)] for (i, j) in model.p2p_pairs)
    sell_p2p_sum = sum(model.P_sell_p2p[t, (i, j)] for (i, j) in model.p2p_pairs)
    return buy_p2p_sum == sell_p2p_sum
M.p2p_balance = pyo.Constraint(M.Nt, rule=p2p_balance_rule)

def p2p_equal_rule(model, t, i, j):
    return model.P_buy_p2p[t, (i, j)] == model.P_sell_p2p[t, (j, i)]
M.p2p_equal = pyo.Constraint(M.Nt, M.p2p_pairs, rule=p2p_equal_rule)

def p2p_equal_binary_rule(model, t, i, j):
    return model.B_buy_p2p[t, (i, j)] == model.B_sell_p2p[t, (j, i)]
M.p2p_equal_binary = pyo.Constraint(M.Nt, M.p2p_pairs, rule=p2p_equal_binary_rule)

def p2p_one_buy(model, t, i):
    return sum(model.B_buy_p2p[t, (i, j)] for j in model.Ni if j != i) <= 1
M.p2p_one_buy = pyo.Constraint(M.Nt, M.Ni, rule=p2p_one_buy)

def p2p_one_sell(model, t, i):
    return sum(model.B_sell_p2p[t, (i, j)] for j in model.Ni if j != i) <= 1
M.p2p_one_sell = pyo.Constraint(M.Nt, M.Ni, rule=p2p_one_sell)

def p2p_no_simultaneous_buy_sell(model, t, i):
    return sum(model.B_buy_p2p[t, (i, j)] for j in model.Ni if j != i) + sum(model.B_sell_p2p[t, (i, j)] for j in model.Ni if j != i) <= 1
M.p2p_no_simultaneous_buy_sell = pyo.Constraint(M.Nt, M.Ni, rule=p2p_no_simultaneous_buy_sell)

# ---- RESTRICCIONES DE BATERÍA ----

def battery_model_rule(model, t, i):
    if t == model.Nt.first():
        return model.E_bat[t, i] == 0.5*(battery_cap[i] * has_bat[i]) + (model.P_ch[t, i] * model.n_ch - model.P_dch[t, i] / model.n_dch)
    else:
        return model.E_bat[t, i] == model.E_bat[t - 1, i] + (model.P_ch[t, i] * model.n_ch - model.P_dch[t, i] / model.n_dch)
M.battery_model = pyo.Constraint(M.Nt, M.Ni, rule=battery_model_rule)

def battery_charge_rule(model, t, i):
    return model.P_ch[t, i] <= model.B_ch[t, i] * model.P_max_ch
M.battery_charge = pyo.Constraint(M.Nt, M.Ni, rule=battery_charge_rule)

def battery_discharge_rule(model, t, i):
    return model.P_dch[t, i] <= model.B_dch[t, i] * model.P_max_dch
M.battery_discharge = pyo.Constraint(M.Nt, M.Ni, rule=battery_discharge_rule)

def battery_charge_discharge_rule(model, t, i):
    return model.B_ch[t, i] + model.B_dch[t, i] <= 1
M.battery_charge_discharge = pyo.Constraint(M.Nt, M.Ni, rule=battery_charge_discharge_rule)

def battery_capacity_up_rule(model, t, i):
    return model.E_bat[t, i] <= 0.8 * battery_cap[i] * has_bat[i]
M.battery_up_capacity = pyo.Constraint(M.Nt, M.Ni, rule=battery_capacity_up_rule)

def battery_capacity_low_rule(model, t, i):
    return model.E_bat[t, i] >= 0.2 * battery_cap[i] * has_bat[i]
M.battery_low_capacity = pyo.Constraint(M.Nt, M.Ni, rule=battery_capacity_low_rule)

solver = pyo.SolverFactory('gurobi')
#solver.options['MIPgap'] = 0.0
#solver.options['tmlim'] = 7200 #2 horas
result = solver.solve(M, tee=True)

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Optimal solution found")
    print(f"Objective value: {pyo.value(M.obj)}")

    # --- EXPORTAR RESULTADOS A CSV ---
    results_dir = "./Results"

    # Timeseries por prosumer (grid, batería, generación, carga)
    rows = []
    for t in M.Nt:
        for i in M.Ni:
            rows.append({
                "t": int(t),
                "prosumer": int(i),
                "P_buyGrid": np.round(float(pyo.value(M.P_buyGrid[t, i])), 2),
                "P_sellGrid": np.round(float(pyo.value(M.P_sellGrid[t, i])), 2),
                "P_sellP2P": np.round(float(sum(pyo.value(M.P_sell_p2p[t, (i, j)]) for j in M.Ni if j != i)), 2),
                "P_buyP2P": np.round(float(sum(pyo.value(M.P_buy_p2p[t, (i, j)]) for j in M.Ni if j != i)), 2),
                "P_ch": np.round(float(pyo.value(M.P_ch[t, i])), 2),
                "P_dch": np.round(float(pyo.value(M.P_dch[t, i])), 2),
                "E_bat": np.round(float(pyo.value(M.E_bat[t, i])), 2),
                "P_gen": np.round(float(pyo.value(M.P_gen[t, i])), 2),
                "P_load": np.round(float(pyo.value(M.P_load[t, i])), 2),
                "Bat_cap": np.round(float(pyo.value(M.E_bat[t, i])), 2),
                "Bat_cap_max": np.round(battery_cap[i] * has_bat[i], 2)
            })
    df_all = pd.DataFrame(rows)
    df_all.to_csv("./Results/Prueba_{}/results_all.csv".format(test), index=False)

    # Desglose P2P: buyer (i) compra a seller (j) la potencia indicada
    p2p_rows = []
    tol = 1e-8
    for t in M.Nt:
        for pair in M.p2p_pairs:
            # pair suele ser (i,j)
            try:
                i, j = pair
            except Exception:
                continue
            val = float(pyo.value(M.P_buy_p2p[t, (i, j)]))
            if val > tol:
                p2p_rows.append({
                    "t": int(t),
                    "buyer": int(i),
                    "seller": int(j),
                    "power": round(val, 2)
                })
    df_p2p = pd.DataFrame(p2p_rows)
    df_p2p.to_csv("./Results/Prueba_{}/results_p2p.csv".format(test), index=False)

    print(f"Results exported to {results_dir}")
else:
    print("No optimal solution found")
    exit()

