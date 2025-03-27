#@author=Daniel González Domínguez
#This code removes the txt from .csv.txt files.

import os

for filename in os.listdir():
    if filename.endswith(".csv.txt"):
        newname = filename.rsplit(".txt",1)[0]
        os.rename(filename,newname)
        print("Renombrado archivo {} a {}".format(filename,newname))