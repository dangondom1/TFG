import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

#Cambio de directorio al ejecutar el código
BASE_DIR = Path(__file__).resolve().parent

os.chdir(BASE_DIR)

#Obtención de datos:
df = pd.DataFrame({
    "Año": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Uds": [0.1, 0.2, 0.2, 0.3, 0.4, 0.6, 1.4, 2.3, 2.6, 3.2, 3.2]
})

#Exportamos los datos a un csv
df.to_csv('../Data/EV_Europe.csv', sep=';')

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.bar(df["Año"], df["Uds"])
ax.set_xlabel("Año")
ax.set_ylabel("Ventas (en millones de uds)")
ax.grid(True)
plt.show()