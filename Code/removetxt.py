#@author=Daniel González Domínguez
#This code removes the txt from .csv.txt files.

import pandas as pd
import numpy as np
import os
from pathlib import Path

#Cambio de directorio al ejecutar el código
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

df = pd.read_csv('../Data/export_mercados-y-precios_2025-09-10_14_19.csv', sep=";")

data_raw = df.iloc[24:48, 3].to_numpy()
data = np.repeat(data_raw, 4)

#print(data_raw)
#print(data.size)

df_result = pd.DataFrame({'price': data})
df_result.to_csv('../Data/buy_price.csv', index=False)