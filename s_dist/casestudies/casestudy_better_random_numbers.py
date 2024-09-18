#! /bin/python3


import numpy as np
import pandas as pd
import scipy.stats as stats


from s_dist.s_dist import *
import s_dist.fglm as fglm


#############################################################################################################################################################################################################################################
### Test


n=5000


print("")
print("")
print("S table central moments corresponding to standard normal distribution [ mean, stdev, skewness, kurtosis ]")
print(s_dist.stats(2, 1, 1, 0))
print("")


sample_1= stats.norm.rvs(size=n)
# print(sample_1.shape)
print("Effective central moments of stats.norm.rvs sample of size ", n)
print(d_cmoments(sample_1))
print("")


sample_2= s_dist.rvs(n, 2, 1, 1, 0)
print("Effective central moments of S dist sample of size ", n)
print(d_cmoments(sample_2))
print("")


print("")
print("Conclusion: S dist produces better quality random numbers, especially in the higher central moments.")
print("Rerun the script and get suprised how unstable the stats.norm.rvs central moments are. (And all the other generators starting with uniform random numbers)")
print("")



