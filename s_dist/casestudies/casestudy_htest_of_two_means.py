#! /bin/python3


#############################################################################################################################################################################################################################################
### Imports


import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


from s_dist.s_dist import *



#############################################################################################################################################################################################################################################
### Getting the data


cat2_df= pd.read_csv("../dsets/cataneo2.csv")
# print(cat2_df.head)

"""
Original data set source: ~/anaconda3/lib/python3.11/site-packages/statsmodels/treatment/tests/results/cataneo2.csv

Assumed data set variables (No data description on the internet)

bweight'= birth weight (target variable),  
'mmarried'= mother married, 
'mhisp'= mother hispano, 
'fhisp'= father hispano, 
'foreign'= foreign,
'alcohol'= alcohole consumption,
'deadkids'= dead kids,
'mage'= mother age, 
'medu'= mother education, 
'fage'= father age,
'fedu'= father education,
'nprenatal'= number of total prenatal treatments, 
'monthslb'= months pound,
'order'= order, 
'msmoke'= mother smoke, 
'mbsmoke'= mother bsmoke, 
'mrace'= mother race,
'frace'= father race, 
'prenatal'= total prenatal treatment code (= all kinds of treatments ?),
'birthmonth'= birth month, 
'lbweight'= pound weight, 
'fbaby'= first baby, 
'prenatal1'= prenatal treatment with clinical cataneo2 drug, 
'mbsmoke_'= mother bsmoke dummy, 
'mmarried_'= mother married dummy,
'fbaby_'= first baby dummy, 
'prenatal1_'= prenatal treatment with clinical cataneo2 drug dummy, 
'mage2'= mother age squared,
"""


# Analysis of tretment variables
# print(cat2_df['nprenatal'])
# print(cat2_df['prenatal'])
# print(cat2_df['prenatal1'])
# print(cat2_df['prenatal1_'])


# Treatment group
a_index= cat2_df['prenatal1']== 'Yes'
a_group= np.array( cat2_df['bweight'])[a_index]

# Control group
b_index= cat2_df['prenatal1']== 'No'
b_group= np.array( cat2_df['bweight'])[b_index]



#############################################################################################################################################################################################################################################
#
# Descriptive statistics


a_group_mean= np.mean(a_group)
b_group_mean= np.mean(b_group)
diff= b_group_mean -a_group_mean 

print("")
print("")
print("Mean birth weight of group A (treatment group)")
print(a_group_mean)
print("")
print("Mean birth weight of group B (control group)")
print(b_group_mean)
print("")

print("")
print("Question: Has group A performed _significantly_ better (as a result of cataneo2 treatment) or is the higher birth weight just a random effect?")
print("")
print("Hypothesis H0: mean_A= mean_B; Ha: mean_A> mean_B")
print("")
print("")


#############################################################################################################################################################################################################################################
#
# Classical t-test

print("Classical T test")
print("")

print("T-test results (case equal variance)")
# alternative : {'two-sided', 'less', 'greater'},
print(ttest_ind(a=a_group, b=b_group, equal_var=True, alternative='greater'))
print("")

print("T-test results (case unequal variance)")
print(ttest_ind(a=a_group, b=b_group, equal_var=False, alternative='greater'))
print("")



#############################################################################################################################################################################################################################################
#
# Bootstrap

a_n= a_group.shape[0]
b_n= b_group.shape[0]


imax= 10000
means= np.zeros([imax, 2])
for i in range(0, imax):
    a_sample= np.random.choice(a_group, size=a_n, replace=True)
    b_sample= np.random.choice(b_group, size=b_n, replace=True)
    
    means[i, 0]= np.mean(a_sample)
    means[i, 1]= np.mean(b_sample)
    

mean_diff= means[:, 1] -means[:, 0]


#############################################################################################################################################################################################################################################
#
# Bootstrap Result

print("")
print("Bootstrap Test")

params= s_dist.fit(mean_diff)
print("S parameters of mean differences")
print(params)
print("")

p_value= s_dist.sf(0, params[0], params[1], params[2], params[3])

print("P Value")
print(p_value)
print("")


#############################################################################################################################################################################################################################################
#
# Conclusions

print("")
print("Conclusions")
print("The bootstrap test of two means produces very similar test results as a classical T test of two means, when the empirical bootstrap density gets stabilized by S dist. [This would also be true for any other T test beginning from a minimum number of 7 observations (compare casestudy_appr_cap_of_t_dist.py)]")
print("")
print("")
print("The big advantage of the bootstrap test: One can generalize the bootstrap test approach onto almost any statistical metric; but one cannot generalize the classical T test statistic onto all other metrics. E.g. the straightforward difference of two medians does _not_ follow a T dist. To get specific the difference of two medians does also _not_ follow any other classical dist. Thus, a test for the difference of two medians can only be performed by the bootstrap method. Anyway, the full classical test statistic universe is a fragmented piecework and very complicated to remember. Frequently, there is even more than one pertinent test for a test case and sometimes they even do produce contrary test results. So, if you are clever, you just learn the new bootstrap S dist test approach and you can forget all the old fragmented test piecework for number counts beginning from 7 ;-) [for variance tests and dist tests for number counts beginning from 20 (compare casestudy_appr_cap_of_f_dist.py)]" )

