#! /bin/python3


import numpy as np
import scipy
import scipy.stats as stats
# import pandas as pd


from matplotlib import pyplot as plt
import matplotlib.style


from s_dist.s_dist import *


#############################################################################################################################################################################################################################################
# Inits


# help(stats)

# help(stats.norm)
# help(stats.norm.isf)



q= np.array([0.0, 1.0, 2.0])

p= np.array([-0.001, 0, 0.5, 0.95])


k=2
a=1
z=1.5
c=0


n=1000


##########################################################################################################################################################
# Tests


# stats
print("")
n_stats= stats.norm.stats(scale=z, loc=c)
print("n_stats")
print(n_stats)

s_stats= s_dist.stats(k=k, a=a, scale=z, loc=c)
print("s_stats")
print(s_stats)
print("")



# fit
print("")
print("k, a, z, c") 
print( [k, a, z, c] ) 

s_rvs= s_dist.rvs(n, k=k, a=a, scale=1, loc=0)
s_fit= s_dist.fit(s_rvs)
print("s_fit")
print(s_fit)
print("")



# median
print("")
n_median= stats.norm.median(scale=z, loc=c)
print("n_median")
print(n_median)

s_median= s_dist.median(k=k, a=a, scale=z, loc=c)
print("s_median")
print(s_median)
print("")



# mean
print("")
n_mean= stats.norm.mean(scale=z, loc=c)
print("n_mean")
print(n_mean)

s_mean= s_dist.mean(k=k, a=a, scale=z, loc=c)
print("s_mean")
print(s_mean)
print("")



# std
print("")
n_std= stats.norm.std(scale=z, loc=c)
print("n_std")
print(n_std)

s_std= s_dist.std(k=k, a=a, scale=z, loc=c)
print("s_std")
print(s_std)
print("")

"""
"""



# std geneuaso unbekannt wie stats
# var
print("")
n_var= stats.norm.var(scale=z, loc=c)
print("n_var")
print(n_var)

s_var= s_dist.var(k=k, a=a, scale=z, loc=c)
print("s_var")
print(s_var)
print("")



# interval
print("")
n_interval= stats.norm.interval(0.95, scale=z, loc=c)
print("n_interval")
print(n_interval)

s_interval= s_dist.interval(0.95, k=k, a=a, scale=z, loc=c)
print("s_interval")
print(s_interval)
print("")



