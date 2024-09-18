#! /bin/python3


import numpy as np
# from numpy.linalg import inv
import pandas as pd
import scipy.stats as stats


from s_dist.s_dist import *



#############################################################################################################################################################################################################################################
### Test


# Finding the appropriate dtol_max for table lookups

# dtol_max= 5e-6
dtol_max= 1e-6
# dtol_max= 5e-7
# dtol_max= 1e-7
# dtol_max= 5e-8
# dtol_max= 1e-8


print("")
print("Total table size")
print(table.shape)
print("")

# Selecting problematic lines with effective k lower than 1
table_select= table[ :, [0, 1, 2, 3, 8]].copy()
# index= table_select[ :, 0] * table_select[ :, 1] < 1.0 
index= table_select[ :, 0] / table_select[ :, 1] < 1.0 
table_select= table_select[ index, :]
print("")
print(table_select.shape)
print("")
print(table_select)
print("")


# Selecting problematic lines 
index= table_select[ :, 4] < dtol_max 
table_solve= table_select[ index, :]
print("")
print(table_solve.shape)
print("")
print(table_solve)
print("")


# Control (if having to restricted)
table_control= table[ :, [0, 1, 2, 3, 8]].copy()
index= table_control[ :, 4] < dtol_max 
table_control= table_control[ index, :]

print("")
print(table_control.shape)
print("")
# print(table_control)
# print("")

"""
"""
