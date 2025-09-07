import pandas as pd
import numpy as np
import os
from pathlib import Path

#Cambio de directorio al ejecutar el código
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

df1 = pd.read_excel('../Data/CREST_Demand_Model_v2.3.3.xlsm', sheet_name='Results - aggregated')
demand_br = df1.iloc[3:1444,4].to_numpy()
demand = []
mean = 0

for i in range(1,97):
    mean = np.mean(demand_br[i*15-15:i*15-1])
    demand = np.append(demand, mean)

df_demand = pd.read_csv('../Data/Con_pw.csv')
df_demand['pro2'] = demand

df_demand.to_csv('../Data/Con_pw.csv', index=False)