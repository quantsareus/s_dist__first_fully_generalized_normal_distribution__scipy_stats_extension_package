#! /bin/python3


"""
This is an until to create or recreate a (damaged) S dist table.

USUALLY YOU DO NOT NEED IT, BECAUSE S Dist DELIVERS A TABLE OUT-OUF-THE-BOX. 

The script runs (also depending on your machine and the installed (auto) parallel computing packages) several hours.  

To change to the new table, you have to replace table.csv in the installed path of s_dist, manually.

"""


#############################################################################################################################################################################################################################################
### Imports


import numpy as np
import pandas as pd
import sys
import os
import time


from s_dist.s_dist import *
import s_dist.fglm as fglm


#############################################################################################################################################################################################################################################
### Inits


# Num. setup
# The num. setup sets a list out of [resolution, min, max], under which all stochastic calculations will be performed. 
# Do not change this, until you have understood the concept and you have got back-checks for the results from alternative num. setups at hand    

# global numsetup
numsetup= [0.005, -250, 250]



#############################################################################################################################################################################################################################################
### Functions

#   
def table_calc(k_min=1.0, k_max=3.0, k_res=0.005, a_min=0.5, a_max=2, a_res=0.005, dtol_max=5e-2, a_half_low=False, u_res=0.01, u_min=-250, u_max=250 ):
    """
    Table calculator
    A default dist table is provided out-of-the-box. The output file is table.csv
    A more broad parameter range than the defaults may also require a more broad x-range. Otherwise you may get a dtol_max error and NaN moments will be written into table.csv, where a certain S distribution exceeds the x-interval
     
    --------------------------------------------------------------------- 
    u_min: minimum value of the independent helper variable
    u_max: maximum of the independent helper variable
    u_res: scalar resolution value of the independent helper variable
    k_min: minimum power and kurtosis value
    k_max: maximum power and kurtosis value
    k_res: resolution of k
    a_min: minimum asymmetry value
    a_max: maximum asymmetry value
    a_res: resolution of a
    """

    if os.path.exists("dsets/table.py"):
        print("Error. The file > dsets/table.py < already exists. Rename, move or delete the file before calculating a new one.")
        print("Aborted.")
        print("")
        exit()
    
    u= np.arange(u_min, u_max, u_res, dtype=np.float128)
        
    k= pd.Series(np.arange(k_min, k_max +k_res, k_res, dtype=np.float128))
    
    a= pd.Series(np.arange(a_min, a_max +a_res, a_res, dtype=np.float128))
    # 1/a values addition in case of half intervall definition of a
    
    if a_half_low== True:
        a_right= pd.Series( 1/ a[:-1] )
        a= pd.concat([ a, a_right])
    
    z= np.array([1], dtype=np.float128)
    
    c= np.array([0], dtype=np.float128)
    
    k= pd.DataFrame(({"key": np.repeat(0, k.shape[0]), "k": pd.Series(k)}))
    a= pd.DataFrame(({"key": np.repeat(0, a.shape[0]), "a": pd.Series(a)}))
    z= pd.DataFrame({"key": np.repeat(0, z.shape[0]), "z": pd.Series(z)})
    c= pd.DataFrame({"key": np.repeat(0, c.shape[0]), "c": pd.Series(c)})
    
    params= pd.merge(k, a)
    params= pd.merge(params, z)
    params= pd.merge(params, c)
    params= np.array(params, dtype=np.float128)
    # Deleting key:
    params= params[:,1:]
    
    print("Current Size of Parameter Set  ", params.shape)
    
    print("")
    print("Machine precision of params table containing k,a,z,c:")
    print(params.dtype)
    print("")
    print("parameter preview:")
    print(params)
    print("")


    ### Calculating the moments

    table= np.zeros((params.shape[0], 9), dtype= np.float128)

    # for i in range (0, 1, 1):
    for i in range (0, params.shape[0], 1):
    	print("iteration ", i, "/", params.shape[0], "  time ", time.asctime(time.localtime()) )

    	table[i, 0:4]= params[i, 0:4]
    	
    	y= s_pdf_hp(u, params[i, 0], params[i, 1], params[i, 2], params[i, 3])
    	dtol= abs(1 -f_sum(y, u_res)) 
    	table[i, 8]= dtol
    	
    	table[i, 4:8]= f_cmoments(y,  u_res, u)
    	if dtol >dtol_max:
    	    table[i, 4:8]= np.array( [np.nan, np.nan, np.nan, np.nan] )
    	    print("Warning. More than ", dtol_max, " probability got lost. Maybe the independent variable interval used for construction is too small")
    	
    ### Writing the table

    tabledf= pd.DataFrame(table, columns= ["k","a","z","c","mean","std","skew","kurt","dtol"], dtype= np.float128)

    if tabledf.to_csv("table.csv", index= False, header= True):
    	print("")
    	print("writing table.csv has failed")
    else:
    	print("")
    	print("the moments table (preview):")
    	print(tabledf)
    	print("")
    	print("has been written succesfully")
    	print("")



#############################################################################################################################################################################################################################################
### Run


# Test table 
# table_calc(k_min=1.99, k_max=2.01, a_min=0.99, a_max=1.01, a_half_low=True)


# Recreate the default table 
table_calc(a_min=0.5, a_max=1.0, a_half_low=True)



