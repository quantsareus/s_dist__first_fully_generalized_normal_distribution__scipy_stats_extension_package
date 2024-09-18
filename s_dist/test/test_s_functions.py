#! /bin/python3


import numpy as np
import scipy.stats as sp_stats
# import pandas as pd


from s_dist.s_dist import *



#############################################################################################################################################################################################################################################
# Init

# k=1.5
# k=2
k=2.5

# a=1/ 1.333
# a=1
a=1.333

z=100

c=-0.5


q= np.array( [-1.2, 0, 1.645] )

p= np.array( [0.1, 0.5, 0.95] )


#############################################################################################################################################################################################################################################
# Test


# pdf
print("")
print("Test s_... foundation functions")
print("")
print("")


# cdf
print("Test cdf")
prob_1= s_cdf(1.0, k, a, z, c)
print("Cumulated probability 1")
print(prob_1)
print("")

prob_2= s_cdf(1.0, k, a, z, c, direction="left")
print("Cumulated probability 2")
print(prob_2)
print("")

prob_sum= prob_1 +prob_2
print("Sum")
print(prob_sum)
print("")
print("")


# sf
print("Test sf")
prob_1= s_sf(1.0, k, a, z, c)
print("Cumulated probability 1")
print(prob_1)
print("")

prob_2= s_sf(1.0, k, a, z, c, direction="left")
print("Cumulated probability 2")
print(prob_2)
print("")

prob_sum= prob_1 +prob_2
print("Sum")
print(prob_sum)
print("")
print("")


# interval and quantile
interval= s_interval(0.90, k, a, z, c)
print("Test interval (= quantile(direction='left') and quantile(direction='right') ")
print(interval)
print("")

prob_1= s_cdf( interval[0], k, a, z, c)
print("Cumulated probability 1")
print(prob_1)
print("")

prob_2= s_cdf( interval[1], k, a, z, c)
print("Cumulated probability 2")
print(prob_2)
print("")

prob_sum= prob_1 +prob_2
print("Sum")
print(prob_sum)
print("")
print("")





