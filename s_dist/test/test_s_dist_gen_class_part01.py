#! /bin/python3


import numpy as np
import scipy
import scipy.stats as stats
# import pandas as pd


from matplotlib import pyplot as plt
import matplotlib.style


from s_dist.s_dist import *



###########################################################################################################################################################################
# Inits


# help(stats)
# help(stats.norm)
# help(stats.norm.isf)


q= np.array([0.0, 1.0, 2.0])
p= np.array([-0.001, 0, 0.5, 0.95])


c=0
z=1
k=2
a=1


n=1000


###########################################################################################################################################################################
# Tests


# pdf
print("")
print("q")
print(q)
print("")


n_pdf= stats.norm.pdf(q)
print("n_pdf")
print(n_pdf)
print(n_pdf.shape)

# Works
# s= s_dist.s_dist_gen()
# s_pdf= s.pdf(q, k=k, a=a, scale=1, loc=0)
s_pdf= s_dist.pdf(q, k=k, a=a, scale=1, loc=0)
print("s_dist_pdf")
print(s_pdf)
print(s_pdf.shape)
print("")


# rvs

n_rvs= stats.norm.rvs(size=n)

print("")
print("n_rvs")
print(d_cmoments(n_rvs))

s_rvs= s_dist.rvs(n, k=k, a=a, scale=1, loc=0)
print("s_rvs")
print(d_cmoments(s_rvs))
print("")



# cdf
print("")
print("q")
print(q)
print("")

n_cdf= stats.norm.cdf(q)
print("n_cdf")
print(n_cdf)

s_cdf= s_dist.cdf(q, k=k, a=a, scale=1, loc=0)
print("s_dist_cdf")
print(s_cdf)
print("")



# ppf
print("")
print("p")
print(p)
print("")

n_ppf= stats.norm.ppf(p, scale=z, loc=c)
print("n_ppf")
print(n_ppf)

s_ppf= s_dist.ppf(p, k=k, a=a, scale=z, loc=c)
print("s_ppf")
print(s_ppf)
print("")



# sf
print("")
print("q")
print(q)
print("")

n_sf= stats.norm.sf(q, scale=z, loc=c)
print("n_sf")
print(n_sf)

s_sf= s_dist.sf(q, k=k, a=a, scale=z, loc=c)
print("s_sf")
print(s_sf)
print("")



# isf
print("")
print("p")
print(p)
print("")

n_isf= stats.norm.isf(p, scale=z, loc=c)
print("n_isf")
print(n_isf)

s_isf= s_dist.isf(p, k=k, a=a, scale=z, loc=c)
print("s_isf")
print(s_isf)
print("")



# loppdf
print("")
print("q")
print(q)
print("")

n_logpdf= stats.norm.logpdf(q, scale=z, loc=c)
print("n_logpdf")
print(n_logpdf)

s_logpdf= s_dist.logpdf(q, k=k, a=a, scale=z, loc=c)
print("s_logpdf")
print(s_logpdf)
print("")



# lopcdf
print("")
print("q")
print(q)
print("")

n_logcdf= stats.norm.logcdf(q, scale=z, loc=c)
print("n_logcdf")
print(n_logcdf)

s_logcdf= s_dist.logcdf(q, k=k, a=a, scale=z, loc=c)
print("s_logcdf")
print(s_logcdf)
print("")



# lopsf
print("")
print("q")
print(q)
print("")

n_logsf= stats.norm.logsf(q, scale=z, loc=c)
print("n_logsf")
print(n_logsf)

s_logsf= s_dist.logsf(q, k=k, a=a, scale=z, loc=c)
print("s_logsf")
print(s_logsf)
print("")


