import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

pro = 0;

#Cambio de directorio al ejecutar el código
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)  
os.chdir(BASE_DIR)

df = pd.read_csv('./Results/Prueba_3/results_all.csv')
t = np.arange(0, 5, 1)

buy_grid = df.loc[df['prosumer'] == pro, 'P_buyGrid'].values
sell_grid = df.loc[df['prosumer'] == pro, 'P_sellGrid'].values
buy_p2p = df.loc[df['prosumer'] == pro, 'P_buyP2P'].values
sell_p2p = df.loc[df['prosumer'] == pro, 'P_sellP2P'].values
dch = df.loc[df['prosumer'] == pro, 'P_dch'].values
ch = df.loc[df['prosumer'] == pro, 'P_ch'].values
prod = df.loc[df['prosumer'] == pro, 'P_gen'].values
load = df.loc[df['prosumer'] == pro, 'P_load'].values

bar_width = 0.35
bar1_pos = t - bar_width/2
bar2_pos = t + bar_width/2

fig,ax = plt.subplots(figsize=(10,6))
ax.bar(bar1_pos, buy_grid, width=bar_width, label='Buy from Grid', color='blue')
ax.bar(bar1_pos, buy_p2p, width=bar_width, bottom=buy_grid ,label='Buy from p2p', color='green')
ax.bar(bar1_pos, dch, width=bar_width, bottom=buy_grid + buy_p2p, label='Discharge Battery', color='orange')
ax.bar(bar1_pos, prod, width=bar_width, bottom=buy_grid + buy_p2p + dch, label='Generation', color='yellow')

ax.bar(bar2_pos, sell_grid, width=bar_width, label='Sell to Grid', color='red')
ax.bar(bar2_pos, sell_p2p, width=bar_width, bottom=sell_grid, label='Sell to p2p', color='purple')
ax.bar(bar2_pos, ch, width=bar_width, bottom=sell_grid + sell_p2p, label='Charge Battery', color='brown')
ax.bar(bar2_pos, load, width=bar_width, bottom=sell_grid + sell_p2p + ch, label='Load', color='grey')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Power (kW)')
ax.legend()

plt.show()