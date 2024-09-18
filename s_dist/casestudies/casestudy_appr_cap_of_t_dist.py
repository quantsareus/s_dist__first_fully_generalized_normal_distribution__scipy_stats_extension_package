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


# An independent variable
x_res= 0.005
x= np.arange(-50, 50, x_res)


t_df= 6
n= t_df +1


#############################################################################################################################################################################################################################################
# Program


# Generating t densities

# help(stats.t)
t_dens= stats.t.pdf(x, df=t_df)


# Appr. trial
moments= f_cmoments(t_dens, x_res, x)
s_params= s_dist.params( moments[0], moments[1], moments[2], moments[3] )
s_dens= s_dist.pdf( x, s_params[0], s_params[1], s_params[2], s_params[3] )


#############################################################################################################################################################################################################################################
# Reporting


r_l1= fglm.r_l_k(t_dens, s_dens, k=1)
r_l2= fglm.r_l_k(t_dens, s_dens, k=2)


print("")
print("Test of appr.-quality by an S dist")
print("")
print("Target. t dist. of degree", t_df)
print("")
print("Appr. by S dist. with params ", s_params)

print("")
print("R_L1 goodness of fit  ", r_l1)
print("R_L2 goodness of fit  ", r_l2)
print("")
print("Interpretation. Good appr.-quality.")
print("On the first look, a t dist of degree ", t_df, " (thus samples beginning with a size of", n, ") - and very likely also more easy ones above- seem to be suitable for approximation by S dist")

print("")
print("Used method.")
print("The approximation test is neither based on an unstable t-distributed random variable x, nor on an t-distributed population x.")
print("Instead a vector of t-densities has been created, from which the central moments have been calculated directly using f_cmomentscalc().")
print("Second, these moments have looked up in the S table; the corresponding S parameters have been returned.")
print("Third, the S densities has been drawn with the found parameters.")
print("As the test does not require random number generator parts any more, the test result counts always the same in each run of the script. Thus has become much more reliable")

print("")
print("")


# Using stylesheets
# plt.style.use(['ggplot',  'fast'])
plt.style.use(['seaborn-v0_8-whitegrid','fast'])


# Creating the graph
plt.plot(x, t_dens)
plt.plot(x, s_dens)


# plt.axis([startx, endx, starty, endy])
plt.axis([-20, 20, 0, 0.5])

# Title 
plt.title("Line Plot")
# x-Label
plt.xlabel('x')
# y-Label
plt.ylabel('density')
# Legends
plt.legend(["t", "s appr."])


plt.show()




