#! /bin/python3


#############################################################################################################################################################################################################################################
### Imports


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


from s_dist.s_dist import *
import s_dist.fglm as fglm


#############################################################################################################################################################################################################################################
### Functions


# Full rank exponential function
def exp_fun(x, b0, b1, b2):
    # scipy.optimize.curve_fit does not process
    # return np.exp( b2 *(x -b1)) +b0
    return np.exp( -b2 *(x -b1)) +b0

# Reduced rank exponential function
def exp_fun_a(x, b0, b1, b2):
    return b2 *np.exp( -b1 *x) +b0
  


#############################################################################################################################################################################################################################################
### Creating data

# 100 poins ^= 1 x unit
x_res= 0.01

x= np.arange(0, 100, x_res)
# print("x  ", x.shape)
# print("")


# Noise Generation
y_noise= 0.02 *np.random.randn(x.shape[0])
scalfac_noise= x_res* np.sum(np.abs(y_noise))
scalfac_signal= scalfac_noise


# Signal Generation
# params_sig= [ 2, 1, 1, 50]
# params_sig= [ 2.23, 1.31, 2.61, 50]
params_sig= [1.95, 1.31, 2.61, 50]
# print("Signal creation parameters)
# print(params_sig)
# print("")

# y_signal= s_pdf(x, params_sig[0],  params_sig[1], params_sig[2], params_sig[3] )
# Creating a spectral line signal with an area signal to noise ratio of 1
y_signal= s_pdf(x, params_sig[0], params_sig[1], params_sig[2], params_sig[3] ) *scalfac_signal


# Base Funcion Generation

# base_exp
# fast horizontal 
# y_base= exp_fun(x, 0.5, -10, -0.1)
# schräg 
# y_base= exp_fun(x, 0.5, -10, -0.05)

# base exp_a
# schräg
y_base= exp_fun_a(x, 1.5, 0.01, 1)


y_tot= y_base +y_signal +y_noise



#############################################################################################################################################################################################################################################
### Modeling y_base estimation  


# Fitting on all bad result
# The signal massive causes negative bias in the intercept
# x_= x
# y_tot_= y_tot

# Fitting on the outer values without signal
x_= np.array( pd.concat( [ pd.Series( x[:3000]), pd.Series( x[7000:]) ] ) )  
y_tot_= np.array( pd.concat( [ pd.Series( y_tot[:3000]), pd.Series( y_tot[7000:]) ] ) )


# One-Step exp fit bad result
# base_b, base_b_cov = curve_fit(exp_fun, x_, y_tot_)

# print("Base Parameters by One-Step Exponential Fit", base_b)
# print("")


# Boostrap fit

# Each iteration cuts off the two outer x-axis points

# Best
imax= 300

cut= 1
base_b_array= np.zeros((imax, 3))
for i in range(0, imax):
    base_b_step, base_b_cov_step = curve_fit(exp_fun, x_, y_tot_)
    base_b_array[i, 0]= base_b_step[0] 
    base_b_array[i, 1]= base_b_step[1]
    base_b_array[i, 2]= base_b_step[2]
    x_= x_[cut: -cut]
    y_tot_= y_tot_[cut: -cut]

base_b= np.zeros(3)
base_b[0]= np.mean(base_b_array[:, 0])
base_b[1]= np.mean(base_b_array[:, 1])
base_b[2]= np.mean(base_b_array[:, 2])


print("")
print("")
print("base_b mean estimates by bootstrapping")
print(base_b)
print("")



#############################################################################################################################################################################################################################################
### Confidence Interval of NLS Parameters


print("")
print("It is extremely difficult to compute a confidence interval of the exponential base function parameters here, for the following reasons:")
print("- Basically, there is no pertinent probability dist for the parameters of a (full rank) exponential function fitted by some (in detail unknown) non-linear squares method")
print("- Making it worse, the fit process has only been performed on the outer x-axis values. Thus, the parameter estimates will be more volatile than usual (hat value effect)")
print("Thus till now, one cannot seriously calculate an e.g. 95% confidence interval")
print("")
print("")


print("To the help the eating almost anything S dist comes in")
print("")


base_b_0_sparams= s_fit(base_b_array[:, 0])
base_b_1_sparams= s_fit(base_b_array[:, 1])
base_b_2_sparams= s_fit(base_b_array[:, 2])
"""
print("S dist parameters of the the dist of base_b_0 (intercept)")
print(base_b_0_sparams)
print("")

print("S dist parameters of the the dist of base_b_1")
print(base_b_1_sparams)
print("")

print("S dist parameters of the the dist of base_b_2")
print(base_b_2_sparams)
print("")
"""

print("base_b_0 95% Interval")
print(s_interval(0.95, base_b_0_sparams[0], base_b_0_sparams[1], base_b_0_sparams[2], base_b_0_sparams[3]) )
print("")
print("base_b_1 95% Interval")
print(s_interval(0.95, base_b_1_sparams[0], base_b_1_sparams[1], base_b_1_sparams[2], base_b_1_sparams[3]) )
print("")
print("base_b_2 95% Interval")
print(s_interval(0.95, base_b_2_sparams[0], base_b_2_sparams[1], base_b_2_sparams[2], base_b_2_sparams[3]) )
print("")


print("")
print("The conf. interval result for base_b_1 IS plausible. As an exponential function is a one-side asymptotically flat function, it is extremely hard to fit an x-axis shift parameter (here base_b_1). Frightening, huh?")
print("")

