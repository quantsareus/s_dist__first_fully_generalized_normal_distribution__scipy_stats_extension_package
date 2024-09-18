#! /bin/python3


import numpy as np
import pandas as pd
import os
import sys



#############################################################################################################################################################################################################################################
### Inits


# Table init and modul imports

syspath= sys.path
syspathlen= len(syspath)
# print(syspath)
path=""
envir=""
for i in range(1, syspathlen):
    path= syspath[i] +"/s_dist/table.csv"
    # print(path)
    if os.path.exists("table.csv"):
        envir="local"
        tabledf= pd.read_csv("table.csv")
    elif os.path.exists(path):
	    envir="installed"
	    tabledf= pd.read_csv(path)


global table
if envir== "installed":
    table= np.array(tabledf)
    import s_dist.s_dist as s_dist
    import s_dist.fglm as fglm
    print("Package s_dist has been successfully initialized in an", envir, "environment." )
    
elif envir== "local":
    table= np.array(tabledf)
    import s_dist
    import fglm
    print("Package s_dist has been successfully initialized in a", envir, "environment." )
    
else:
    print("Error. Package s_dist could NOT get initialized." )


