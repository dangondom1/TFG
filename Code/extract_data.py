import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

#Cambio de directorio al ejecutar el código
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

df1 = pd.read_excel('../Data/CREST_Demand_Model_v2.3.3.xlsm', sheet_name='Results - aggregated')
prod_br = df1.iloc[3:1444,5].to_numpy()
demand_br = df1.iloc[3:1444,4].to_numpy()
demand = []
prod = []
mean_demand = 0
mean_prod = 0

for i in range(1,97):
    mean_demand = np.mean(demand_br[i*15-15:i*15-1])
    demand = np.append(demand, mean_demand)
    mean_prod = np.mean(prod_br[i*15-15:i*15-1])
    prod = np.append(prod, mean_prod)

""" 
data_demand = {'pro1': demand}
df_demand = pd.DataFrame(data_demand)
data_prod = {'pro1': prod}
df_prod = pd.DataFrame(data_prod)


df_demand.to_csv('../Data/Con_pw.csv', index=False)
df_prod.to_csv('../Data/Gen_pw.csv', index=False) """

obj = 'pro10'

df_demand = pd.read_csv('../Data/Con_pw.csv')
df_demand[obj] = demand

df_prod = pd.read_csv('../Data/Gen_pw.csv')
df_prod[obj] = prod

df_demand.to_csv('../Data/Con_pw.csv', index=False)
df_prod.to_csv('../Data/Gen_pw.csv', index=False)