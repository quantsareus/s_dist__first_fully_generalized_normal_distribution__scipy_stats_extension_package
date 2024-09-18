#! /bin/python3


import numpy as np
# import pandas as pd
import scipy.stats as stats


from matplotlib import pyplot as plt
import matplotlib.style


from s_dist.s_dist import *
import s_dist.fglm as fglm


#############################################################################################################################################################################################################################################
# Init


# Some independent help variable
u_res= 0.005
u= np.arange(-50, 50, u_res)


f_dfn=19
f_dfd=19

n= f_dfn +1

#############################################################################################################################################################################################################################################
# Program


# Generating f distributed x

# help(stats.f)
f_dens= stats.f.pdf(u, dfn=f_dfn, dfd=f_dfd)
x= mk_a_pop(10000, f_dens, u_res, u)


# Appr. trial
s_params= s_dist.fit(x)
s_dens= s_dist.pdf(u, s_params[0], s_params[1], s_params[2], s_params[3] )


#############################################################################################################################################################################################################################################
# Reporting


r_l1= fglm.r_l_k(f_dens, s_dens, k=1)
r_l2= fglm.r_l_k(f_dens, s_dens, k=2)


print("")
print("Test of appr.-quality by an S dist")
print("")
print("Target. f dist. of degrees", f_dfn, ",", f_dfd)
print("")
print("Appr. by S dist. with params ", s_params)

print("")
print("R_L1 goodness of fit  ", r_l1)
print("R_L2 goodness of fit  ", r_l2)
print("")
print("Interpretation. Good appr.-quality.")
print("On the first look, a f dist of degrees ", f_dfn, ",", f_dfd, " (thus samples beginning with a size of", n, ") - and very likely also more easy ones above - seem to be suitable for approximation by S dist")


print("")
print("Used Method.")
print("An f-distributed random variable x, pretty unstable in the central moments, has been replaced by a stable f-distributed population x of size 10'000 using mk_a_pop().")
print("As the test does not require random number generator parts any more, the test result counts always the same in each run of the script. Thus has become much more reliable")

print("")
print("")


# Using stylesheets
# plt.style.use(['ggplot',  'fast'])
plt.style.use(['seaborn-v0_8-whitegrid','fast'])


# Creating the graph
plt.plot(u, f_dens)
plt.plot(u, s_dens)


# plt.axis([startx, endx, starty, endy])
plt.axis([-20, 20, 0, 1.2])

# Title 
plt.title("Line Plot")
# x-Label
plt.xlabel('u')
# y-Label
plt.ylabel('density')
# Legends
plt.legend(["f", "s appr."])


plt.show()




