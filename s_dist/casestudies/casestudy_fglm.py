#! /bin/python3


import numpy as np
# from numpy.linalg import inv
import pandas as pd


from s_dist.s_dist import *
import s_dist.fglm as fglm



##############################################################################################################################################################################################################################################
#  Generating Test Data



n= 5000


# Generate Random Error
print("")
print("Creating error ...")
print("")


k= 1.33
# k= 1.52
# k= 2
# k= 2.08
# k= 2.95
# a= 1
a= 1.27
# a= 1.42
# a= 1.95
# a= 1/a
z= 0.84
# z= 1.44
# z= 2
# z= 3.14
# k= 1.16
c= 0


e_creat_params= np.array([k, a, z, c])

print("")
print("Defined error params ", e_creat_params)
print("")

e_theo_moments= s_dist.stats(k, a, z, c)
print("Corresponding perfect table moments  ", e_theo_moments)
print("")

epop= s_mkpop(n, k, a, z, c, dtol_max=1e-4)
e= sample(n, epop)

e_effect_moments= d_cmoments(e)
print("Effective error moments created  ", e_effect_moments, " (perfect random number generation problem)" )
print("")
e_effect_params= s_dist.fit(e)
print("Effective error params  ", e_effect_params)
print("")


# Generate independent Variables 

print("")
print("Creating independent variable ...")
print("")

# Define target line coefficients 

b0= -2.0
# b0= -1.5
b1= 0.2
# b1= 0.3
# b2= 0.7
b2= 0.8

b= np.array([b0, b1, b2])
print("Defined target beta ", b)
print("")
print("")


x0= np.repeat(1, n)

# Normal distributed x
# x1= random.normal(0, 1, n)
# x2= random.normal(0, 1, n)

# Normal distributed x
# xpop= s_mkpop(n, 2, 1, 1, 0, dtol_max=1e-4)
# S distributed x
# xpop= s_mkpop(n, 1.2, 1.333, 1, 0, dtol_max=1e-4)
# xpop= s_mkpop(n, 1.4, 1.333, 1, 0,  dtol_max=1e-4)
xpop= s_mkpop(n, 1.52, 1.333, 1, 0, dtol_max=1e-4)
# xpop= s_mkpop(n, 1.65, 1.333, 1, 0, dtol_max=1e-4)
# xpop= s_mkpop(n, 1.71, 1.333, 1, 0,  dtol_max=1e-4)
# xpop= s_mkpop(n, 2, 1.333, 1, 0, dtol_max=1e-4)
x1= sample(n, xpop)
x2= sample(n, xpop)

X= np.array([x0, x1, x2]).T


# Construct target variable with noise
# y= X.dot(b) +e
y= X @ b +e 



##############################################################################################################################################################################################################################################
# FGML Regression


# target prec
# b_change_min= 1e-3
b_change_min= 5e-4
# b_change_min= 1e-4
# b_change_min= 5e-5
# b_change_min= 1e-5


# fglm.fglm_fit_report(y, X, b_change_min=b_change_min)
fglm_results= fglm.fglm_fit_report(y, X, b_change_min=b_change_min)
# print(fglm_results)


# For Production 
# fglm_results= fglm.fglm_fit(y, X, b_change_min=b_change_min)
# print(fglm_results)


#############################################################################################################################################################################################################################################
### Summary



