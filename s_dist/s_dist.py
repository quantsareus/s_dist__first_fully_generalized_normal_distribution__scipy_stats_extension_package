#! /bin/python3


"""

This is an implementation of the fully generalized S dist invented by Schlingmann, H., that was published in the Phd thesis 'Introducing a fully generalized normal distribution' in January 2024 on Github (github.com/quantsareus/introducing_a_fully_generalized_normal_distribution). The thesis proves, that the new distribution is valid; further it also explains a lot of the numerical methods applied in this stats lib. 

Summing it up short, the new S dist is the first generalized normal dist, that is flexible in both, skewness and kurtosis (power), at the same time (= fully generalized). The new giant flexibility makes it a very univeral distribution, that eats about 95% of the empirical dists (including the dists of metrics and parameters), that do occur in daily data science practice.   

Further, it is directly downward compatible to the normal distribution (without parameter conversion). The new S dist is closely related to the Gaussian; it can be described as a Gaussian (already flexible in kurosis/ power), that additionally got enhanced for flexibility in skewness and that has been normed to a total density of 1. For detailed information about S dist refer to the thesis (github.com/quantsareus/introducing_a_fully_generalized_normal_distribution/introducing_a_fully_generalized_normal_distribution_published.pdf). 

The MathML version of the S dist PDF formula is

f(x) = left langle binom {2 over{%lambda_1 + %lambda_2} e^ {-0.5({ abs{x-c}} over z)^ak}   for x -c < 0 } { 2 over {%lambda_1 + %lambda_2} e^ {-0.5({ abs{x-c}} over z)^{k over a}} for x -c > 0}  right rangle 

where

%lambda_1 = {2z %GAMMA(1 over ak)} over {ak 0.5^{1 over ak}} 

%lambda_2 = {2z %GAMMA(a over k)} over {k over a 0.5^{a over k}} 

and
    
x: independent input variable (quantile variable) 
k: power and kurtosis parameter € [1, 3]
a: asymmetry parameter € [0.5, 2]
z: deviation scale parameter
c: central location parameter


The valid parameter space of the current implementation is

k: k € [ 1, 3]
a: a € [ 0.5, 2]
z: basically unlimited (within Python number and precision limits)
c: basically unlimited (within Python number and precision limits)

where k *a >1 and k /a >1  

The current parameter space will roughly covers about 95% of the empirical dists (including the dists of metrics and parameters), that do occur in daily data science practice. This, without using any other distribution than S distribution! In future the valid parameter space and the performance will get further expanded by additional mathematical solve of the distribution.  
 


The modul contains the following:


-----------------------------------------------------------------------------------------------------------------------------
General foundation functions

f_sum()
    Performant num. area under the curve calculation
f_cmoments()
    Performant central moments for PD function values
d_cmoments()
    Central moments for data (of some outcome variable), without sample size correction (in order to get a maximum correspondence to the distribution table)
mk_a_pop()  
    Creates an arbitrary distributed population with high moment precision from a density vector corresponding to an independent variable
sample() 
    Sampling function
    
-----------------------------------------------------------------------------------------------------------------------------
S dist foundation functions. The functions follow the [k, a, z, c] (:Kassy:) parameter notation standard and the [mean, std, skewness, kurtosis] central moment notation standard 

s_pdf() 
    PDF of S dist
s_pdf_hp() 
    High precision version of the PDF
s_stats()
    Return the central moments mean, stdev, skewness, kurtosis from the dist parameters
s_params()
    Return the dist parameters k, a, z, c from the central moments
s_fit()
    Return the dist parameters k, a, z, c from generic data
s_mkpop() 
    Creates an S distributed population
s_rands()
    S distributed random numbers
s_cdf_obj() 
    Low level cumulated distribution object
s_cdf()
    Cumulative distribution function
sf()
    Survival function (also defined as ``1 - cdf``)
s_quantile()
    Quantile function (inverse of CDF)
s_interval()
    confidence interval starting in the center with borders left and right 

-----------------------------------------------------------------------------------------------------------------------------
'scipy.stats.s_dist_gen' class with the standard functions of a scipy.stats dist 

rvs()
    Random variates wrapper function
pdf()
    Probability density wrapper function
logpdf()
    Log of the probability density function
cdf()
    Cumulative distribution wrapper function
logcdf()
    Log of the cumulative distribution function
sf()
    Survival function wrapper
logsf)
    Log of the survival function
ppf()
     'Percent point function'. Compatibiliy wrapper function to s_quantile() function. The functions naming follows the here misguiding scipy.stats.dist naming standard
isf()
    Inverse survival function (inverse of ``sf``)
[Missing: moment()
    Non-central moment of the specified order
stats()
    Return the central moments mean, stdev, skewness, kurtosis from the dist parameters wrapper function
params()
    Return the dist parameters k, a, z, c from the central moments wrapper function
fit()
    Return the dist parameters k, a, z, c from generic data wrapper function
[Missing: entropy()
    Differential) entropy of the RV
[Missing: expect()
    Expected value of a function (of one argument) with respect to the distribution
median()
    Median wrapper function
mean()
    Mean wrapper function
var()
    Variance wrapper function
std()
    Standard deviation wrapper function
interval()
    Confidence interval with equal areas around the median wrapper function


Of cause there are also the extra '_method' versions of each function for the standard dist

"""


#############################################################################################################################################################################################################################################
### Imports


import numpy as np
import pandas as pd
import scipy.stats
# import sympy as sym
from sympy.functions.special.gamma_functions import gamma


from s_dist import *



#############################################################################################################################################################################################################################################
### Inits


# Num. setup
# The num. setup defines the resolution and the lower and upper limits, on which all the stochastic calculations will be performed.   
# It is a python list out of [resolution, min, max]. Do not change this, until you have understood the concept and you have got back-checks for the results from alternative num. setups at hand   

global numsetup
numsetup= [0.005, -100, 100]



#############################################################################################################################################################################################################################################
### General foundation functions; function names beginning with 's_' follow the (., k, a, z, c, ...) function parameter notation standard

#
def f_sum(y, x_res):
    """
    Performant num. area under the curve calculation
    
    Method: Summing rectangles of height y and width x_res
    
    y: values vector of output variable 1 (curve 1)
    x_res: scalar resolution value of input variable
    =====================================================================
    Returns sum scalar
    
    """
    
    area= x_res* np.sum(y)   
    return area


#
def f_cmoments(f, x_res, x):
    """
    Performant central moments for PD function values
    
    Method: Summing rectangles of height central moment generating function and width x_res
    
    f: vector of density function values
    x_res: scalar resolution value of input variable
    x: values vector of the input variable
    =====================================================================
    Returns np.array([mean, standard_dev, skewness, kurtosis])
    
    """
    
    mean= np.float128( x_res* np.sum( f *x) )
    var= np.float128( x_res* np.sum( f *(x-mean)**2) )
    stdev= np.float128( np.sqrt(abs( var)) )
    skew= np.float128( x_res* np.sum( f *((x-mean)/stdev)**3 ) )
    kurt= np.float128( x_res* np.sum( f *((x-mean)/stdev)**4 ) )
    
    cmoments= np.array( [mean, stdev, skew, kurt], dtype=np.float128)
    return cmoments


#
def d_cmoments(x):
    """
    Central moments for data (of some outcome variable), without sample size correction (in order to get a maximum correspondence to the distribution table)
    
    Method: The well known stats formulas  
    
    x: values vector of an outcome variable
    =====================================================================
    Returns np.array([mean, standard_dev, skewness, kurtosis])
    
    """
    
    n= x.shape[0]
    mean= 1/n *np.sum(x)
    var= 1/n *np.sum( (x -mean)**2)
    stdev= np.sqrt(abs(var))
    skew= 1/n* np.sum( ((x- mean)/stdev)**3)
    kurt= 1/n* np.sum( ((x- mean)/stdev)**4)

    cmoments= np.array( [mean, stdev, skew, kurt] )
    return cmoments


#
def mk_a_pop(n, d, u_res, u, dtol_max=1e-6):
    """
    Creates an arbitrary distributed population from a density vector corresponding to an independent variable
    
    n: number of observations (size)
    d: density values vector containing at least (1 -dtol_max) density
    u_res: scalar resolution value of independent helper variable
    u: values vector of independent helper variable
    ---------------------------------------------------------------------
	dtol_max: maximum tolerated deviation from density sum 1
    =====================================================================
    Returns np.array([])
    
    """
    
    l= u.shape[0]
    
    # Density sum up to 1 test
    dcontrsum= u_res *np.sum(d) 
    dtol= abs(1 -dcontrsum)
    
    if dtol < dtol_max:
    	dcum= np.cumsum(d) *u_res *n
    	dresid= 0.0
    	dint= np.zeros(l, dtype= np.int64)
    	for i in range(1, l):
    		if round(dcum[i] -dcum[i-1] +dresid) >= 0.5:
    			dint[i]= round(dcum[i] -dcum[i-1] +dresid)
    			dresid= (dcum[i] -dcum[i-1] +dresid) - round(dcum[i] -dcum[i-1] +dresid)
    		else:
    			dresid= dresid+ dcum[i] -dcum[i-1]
    		# print(d[i], dcum[i], dint[i], dresid)
    	pop= u.repeat(dint)
    	return pop

    else:
    	print("")
    	print("Warning from mk_a_pop()")
    	print("The density values vector is not as reliable as desired in this parameter setup")
    	print("More than ", dtol_max, " probability got lost. Maybe the independent variable interval used for construction is too small")
    	print("")
    	return pop


#
def sample(n, pop, replace=False):
    """
    Sampling function
    
    n: desired number of observations (size)
    pop: value vector of population values
    replace=false: Already drawn items cannot be drawn again
    """
    
    x= np.random.choice(pop, size=n, replace=replace)
    
    return x



#############################################################################################################################################################################################################################################
### S dist specific foundation functions following the (., k, a, z, c, ...) function parameter notation standard and the [mean, std, skewness, kurtosis] central moment notation standard 


#
def s_pdf(x, k, a, z, c):
    """
    S dist PDF 
    
    x: independent input variable
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    =====================================================================
    Returns np.array([function_values])
    
    """
    
    norm1= (2 *z *gamma(1/(k*a))) /(k*a *0.5**(1/(k*a))) 
    norm2= (2 *z *gamma(1/(k/a))) /(k/a *0.5**(1/(k/a)))
    norm= (norm1 +norm2) /2 
    f1= 1/norm *np.exp(-0.5 *(np.abs(x-c)/z)**(k*a) )
    f2= 1/norm *np.exp(-0.5 *(np.abs(x-c)/z)**(k/a) )
    f= x *0
    f[x-c <=0]= f1[x-c <=0]
    f[x-c >0]= f2[x-c >0]
    
    return f
    
    
#
def s_pdf_hp(x, k, a, z, c):
    """
    S dist high precision PDF
    
    x: independent input variable
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    =====================================================================
    Returns np.array([function_values])
    
    """
    
    norm1= np.float128( (2 *z *gamma(1/(k*a))) /(k*a *0.5**(1/(k*a))) )
    norm2= np.float128( (2 *z *gamma(1/(k/a))) /(k/a *0.5**(1/(k/a))) )
    norm= np.float128( (norm1 +norm2) /2 ) 
    f1= np.float128( 1/norm *np.exp(-0.5 *(np.abs(x-c)/z)**(k*a) ) )
    f2= np.float128( 1/norm *np.exp(-0.5 *(np.abs(x-c)/z)**(k/a) ) )
    f= np.float128( x *0 )
    f[x-c <=0]= f1[x-c <=0]
    f[x-c >0]= f2[x-c >0]
    
    return f


#
def s_mkpop(n, k, a, z, c, dtol_max=1e-6, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
    """
    Creates an S distributed population using mk_a_pop
    
    n: desired number of observations (size)
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    ---------------------------------------------------------------------
    u_min: minimum value of independent helper variable
    u_maximum: maximum value of independent helper variable
    u_res: scalar resolution value of independent helper variable
    dtol_max: maximum tolerated deviation from density sum 1
    
    """

    u_min= (u_min *z) +c
    u_max= (u_max *z) +c
    u_res= u_res *z
    u= np.arange(u_min, u_max, u_res)
    
    d= s_pdf(u, k, a, z, c)
        
    x_pop= mk_a_pop(n, d=d, u_res=u_res, u=u, dtol_max=dtol_max)
    
    return x_pop


#
def s_rands(n, k, a, z, c, popsizefac=1, dtol_max=1e-6, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
    """
    Random Number Generator
    
    n: desired number of observations (size)
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    ---------------------------------------------------------------------
    popsizefac: size factor multiple of the in background created population   
    dtol_max: maximum tolerated deviation from density sum 1
	"""

    pop= s_mkpop(n *popsizefac, k, a, z, c, dtol_max=dtol_max, u_res=u_res, u_min=u_min, u_max=u_max )
    x= sample(n, pop)
    
    return x


#
def s_stats(k, a, z, c, stddist=False, dtol_max=1e6):
    """     
    Central moments of S dist (table lookup from parameters)  
    
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    ---------------------------------------------------------------------
    dtol_max: maximum tolerated deviation from density sum 1    
    =====================================================================
    Returns np.array([mean, standard_dev, skewness, kurtosis])
    
    """
    
    diff= (np.abs(table[:, 0] -k)) +(np.abs(table[:, 1] -a)) 
    table_match= table[ np.argmin(diff), :].copy()
    # print(table_match)
    
    # Standard dist switch
    if stddist==False:
        table_match[5]= table_match[5] *z
        table_match[4]= table_match[4] *z
        table_match[4]= table_match[4] +c
        
    if table_match[8] <dtol_max:
        return table_match[4:8]
    else:
        print("")
        print("Warning from s_stats()")
        print("The table values in this parameter setup are not as reliable as desired")
        print("When computing this table entry line, ", table_match[8], " density has been lost. This usually occurs when exceeding the valid parameter range k *a <1 or k /a <1 or moments corresponding to such parameter set")
        print("")
        return table_match[4:8]


#
def s_params(mean, std, skew, kurt, stddist=False, dtol_max=1e6):
    """
    Parameters of S dist (table lookup from central moments)  
    
    mean: mean
    std: standard deviation
    skew: skewness
    kurt: kurtosis
    ---------------------------------------------------------------------
    dtol_max: maximum tolerated deviation from density sum 1
    =====================================================================
    Returns np.array([k, a, z, c])
    
    """
    
    diff= (np.abs(table[:, 6] -skew)) +(np.abs(table[:, 7] -kurt)) 
    table_match= table[ np.argmin(diff), :].copy()
    # print(table_match)
    
    # Calculating z!=1 using z-transform
    z= std/ table_match[5]
    
    # Standard dist switch
    if stddist==False:
        table_match[2]= z    
        table_match[5]= table_match[5] *z
        table_match[4]= table_match[4] *z
        c= mean -table_match[4]
        table_match[3]= c
    
    if table_match[8] <dtol_max:
        return table_match[0:4]
    else:
        print("")
        print("Warning from s_params()")
        print("The table values in this parameter setup are not as reliable as desired")
        print("When computing this table entry line, ", table_match[8], " density has been lost. This usually occurs when exceeding the valid parameter range k *a <1 or k /a <1 or moments corresponding to such parameter set")
        print("")
        return table_match[0:4]


#
def s_fit(x, stddist=False, dtol_max=1e6):
    """
    Fit a static S distribution to data of outcome values and return the parameters 

    x: values vector of an outcome variable
    ---------------------------------------------------------------------
    dtol_max: maximum tolerated deviation from density sum 1
    =====================================================================
    Returns np.array([k, a, z, c])
    
    """
    
    cmoments= d_cmoments(x)
    mean=cmoments[0]
    std=cmoments[1]
    skew=cmoments[2]
    kurt=cmoments[3]
    
    diff= (np.abs(table[:, 6] -skew)) +(np.abs(table[:, 7] -kurt)) 
    table_match= table[ np.argmin(diff), :].copy()
    
    # Calculating z!=1 using z-transform
    z= std/ table_match[5]
    
    # Standard dist switch
    if stddist==False:
        table_match[2]= z    
        table_match[5]= table_match[5] *z
        table_match[4]= table_match[4] *z
        c= mean -table_match[4]
        table_match[3]= c
    
    if table_match[8] <dtol_max:
        return table_match[0:4]
    else:
        print("")
        print("Warning from fit()")
        print("The table values in this parameter setup are not as reliable as desired")
        print("When computing this table entry line, ", table_match[8], " density has been lost. This usually occurs when exceeding the valid parameter range k *a <1 or k /a <1 or moments corresponding to such parameter set")
        print("")
        return table_match[0:4]


#
def s_cdf_obj(k, a, z, c, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
    """
    Low level numerical cumulated distribution function
    
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    direction: direction of cumulation
    =====================================================================
    Returns np.array([ [cumulated_density, x_values] ])
    
    """
    
    u_min= (u_min *z) +c
    u_max= (u_max *z) +c
    u_res= u_res *z
    u= np.arange(u_min, u_max, u_res)
    n= u.shape[0]
    
    if direction== "right":
        y= s_pdf(u, k, a, z, c)
        dsum= u_res* np.cumsum(y)
    
    else:
        y= s_pdf(u, k, 1/a, z, c)
        dsum= u_res* np.cumsum(y)    
        revindex= range(n-1, -1, -1)
        u= u[revindex]
    
    cd_obj= np.array([dsum, u])
    
    return cd_obj


#
def s_cdf(x, k, a, z, c, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
    """
    Cumulative distribution function
    
    x: quantile value(s) of variable
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    direction: direction of cumulation
    =====================================================================
    Returns scalar p or np.array([ p])
    
    """
    
    if np.ndim(x) ==0:
        n=1
        x_= np.array([x])
        p= np.array([np.nan])
        # print("if")
        # print(x_.shape)
        # print(p.shape)
    else:
        n= x.shape[0]
        x_= x
        p= np.ones((n)) *np.nan
        # print("else")
        # print(x_.shape)
        # print(p.shape)
    
    cd_obj= s_cdf_obj(k, a, z, c, direction=direction, u_res=u_res, u_min=u_min, u_max=u_max)    
        
    for i in range(0, n):
        diff= np.abs(cd_obj[1, :] -x_[i] )
        p[i]= cd_obj[0, np.argmin( diff) ]
    
    if n ==1:
        return p[0]
    else:
        return p


#
def s_sf(x, k, a, z, c, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
    """
    Survival function; (= 1 - CDF)
    
    x: quantile value(s) of variable
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    direction: direction of cumulation
    =====================================================================
    Returns scalar p or np.array([ p])
    
    """
    
    return 1- s_cdf(x, k=k, a=a, z=z, c=c, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])


#
def s_quantile(p, k, a, z, c, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
    """
    Quantile function. The inverse of the CDF
    
    p: probability value € [0, 1]
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    =====================================================================
    Returns scalar q or np.array([ q])
    
    """
            
    if np.ndim(p) ==0:
        n=1
        p_= np.array([p])
        q= np.array([np.nan])
    else:
        n= p.shape[0]
        p_= p
        q= np.ones(n) *np.nan
        
    cd_obj= s_cdf_obj(k, a, z, c, direction=direction, u_res=u_res, u_min=u_min, u_max=u_max)    
        
    for i in range(0, n):
        
        if p_[i] <0:
            q[i]= np.nan
        elif p_[i] ==0:
            q[i]= -np.inf          
        elif p_[i] ==1:
            q[i]= np.inf
        elif p_[i] >1:
            q[i]= np.nan    
        else:
            diff= np.abs(cd_obj[0, :] -p_[i] )
            q[i]= cd_obj[1, np.argmin( diff) ]
    
    if n ==1:
        return q[0]
    else:
        return q


#
def s_interval(confidence, k, a, z, c, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
    """
    Two-sided confidence interval calculation with equal areas around the median        
    confidence: two-sided probability value € [0, 1
    k: power and kurtosis parameter € [1, 3]
    a: asymmetry parameter € [0.5, 2]
    z: deviation scale parameter
    c: central location parameter
    ====================================================================
    Returns (left_q, right_)

    """
    
    p= 1- ((1- confidence) /2)
        
    q_left= s_quantile(p, k, a, z, c, direction="left", u_res=u_res, u_min=u_min, u_max=u_max)
    
    q_right= s_quantile(p, k, a, z, c, direction="right", u_res=u_res, u_min=u_min, u_max=u_max)
    
    return (q_left, q_right)


#
class s_dist_gen(scipy.stats.rv_continuous):
    r""" S dist class

    %(before_notes)s

    Notes
    -----

    %(after_notes)s

    %(example)s

    """

    #
    def _shape_info(self):
        k = _ShapeInfo("k", False, (1.0, 3.0), (False, False))
        a = _ShapeInfo("a", False, (0.5, 2.0), (False, False))
        return [k, a]
    
    
    #
    def rvs(self, n, k, a, scale=1, loc=0, popsizefac= 1, direction="right", dtol_max=1e-6, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
        """
        random variates wrapper function
        
        n: number of observations (size)
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        popsizefac: size factor multiple of the in background created population   
        dtol_max: maximum tolerated deviation from density sum 1
        
        """
        
        return s_rands(n, k=k, a=a, z=scale, c=loc, popsizefac= popsizefac, dtol_max=dtol_max, u_res=u_res, u_min=u_min, u_max=u_max)
    
    
    #
    def _rvs(self, n, k, a, popsizefac= 1, dtol_max=1e-6, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
        """
        
        """
        
        return rvs(n, k=k, a=a, scale=1, loc=0, popsizefac= popsizefac, dtol_max=dtol_max, u_res=u_res, u_min=u_min, u_max=u_max)
            
    
    #
    def pdf(self, x, k, a, scale=1, loc=0):
        """
        PDF wrapper function
        
        x: independent input variable
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns np.array([function_values])
        
        """
    
        return s_pdf(x, z=scale, c=loc, k=k, a=a)

    
    #
    def _pdf(self, x, k, a):
        """
        
        """
        
        return pdf(x, k=k, a=a, scale=1, loc=0)
    
    
    #
    def logpdf(self, x, k, a, scale=1, loc=0):
        """
        Logarithmic PDF function
        
        x: quantile value(s) of variable
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar p or np.array([ p])
        
        """
        
        return np.log( s_pdf(x, z=scale, c=loc, k=k, a=a) ) 
    
    
    #
    def _logpdf(self, x, k, a):
        """
        
        """
        
        return logpdf(x, k=k, a=a, scale=1, loc=0) 
    
    
    #
    def cdf(self, x, k, a, scale=1, loc=0, direction="right"):
        """
        CDF wrapper function
        
        x: independent input variable
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns np.array([function_values])
        
        """
    
        return s_cdf(x, k=k, a=a, z=scale, c=loc, direction=direction)

    
    #
    def _cdf(self, x, k, a,):
    # def _cdf(self, x, k, a, direction="right"):
        """
        
        """
        
        return cdf(x, k=k, a=a, scale=1, loc=0, direction="right")
        # return cdf(x, k=k, a=a, scale=1, loc=0, direction=direction)
    
    
    #
    def ppf(self, p, k, a, scale=1, loc=0, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
        """
        Percent point function wrapper 
        The functions naming follows the here misguiding scipy.stats.dist naming standard. As the CDF is already the 'percent point function' when using straight output orientated function naming, qualified data scientists do call it the quantile() or the percentile() function. A percentile is a special >quantile value<, that (just) corresponds to a certain percent probability. If you call a function f(x) the 'x-function', how do you call g(x), then? And the parameter q is usually used for quantile values :-D
        
        p: probability value € [0, 1]
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar q or np.array([ q])
        
        """
        
        return s_quantile(p, k=k, a=a, z=scale, c=loc, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] )
    
    
    #
    def _ppf(self, p, k, a, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
    #def _ppf(self, p, k, a, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
        """
        
        """
        
        return ppf(p, k=k, a=a, scale=1, loc=0, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] )
        # return ppf(p, k=k, a=a, scale=1, loc=0, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] )
    
    
    #
    def sf(self, x, k, a, scale=1, loc=0, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        Survival function; (= 1 - CDF)     
        
        x: quantile value(s) of variable
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar s or np.array([ s])
        
        """ 
        
        return s_sf(x, k=k, a=a, z=scale, c=loc, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
    
    
    #
    def _sf(self, x, k, a, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
    # def _sf(self, x, k, a, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        
        """ 
        
        return sf(x, k=k, a=a, scale=1, loc=0, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
        # return sf(x, k=k, a=a, scale=1, loc=0, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
    
    
    #
    def isf(self, p, k, a, scale=1, loc=0, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        Inverse survival function
        
        p: probability value € [0, 1]
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar q or np.array([ q])
        
        """
        
        p_= 1 -p
        
        return s_quantile(p_, k=k, a=a, z=scale, c=loc, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
    
    
    #
    def _isf(self, p, k, a, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
    # def _isf(self, p, k, a, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        
        """
        
        return isf(p, k=k, a=a, scale=1, loc=0, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
        # return isf(p, k=k, a=a, scale=1, loc=0, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
    
    
    #
    def logcdf(self, x, k, a, scale=1, loc=0, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        Logarithmic CDF function
        
        x: quantile value(s) of variable
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar log_p or np.array([ log_p])
        
        """
        
        return np.log( s_cdf(x, k=k, a=a, z=scale, c=loc, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]) )
    
    
    #
    def _logcdf(self, x, k, a, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
    # def _logcdf(self, x, k, a, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        
        """
        
        return logcdf(x, k=k, a=a, scale=1, loc=0, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
        # return logcdf(x, k=k, a=a, scale=1, loc=0, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
    
    
    #
    def logsf(self, x, k, a, scale=1, loc=0, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        Logarithmic survival function
        
        x: quantile value(s) of variable
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar log_s or np.array([ log_s])
        
        """
           
        return np.log( s_sf(x, k=k, a=a, z=scale, c=loc, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]) )
    
    
    #
    def _logsf(self, x, k, a, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
    # def _logsf(self, x, k, a, direction="right", u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2]):
        """
        
        """
        
        return logsf(x, k=k, a=a, scale=1, loc=0, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
        # return logsf(x, k=k, a=a, scale=1, loc=0, direction=direction, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2])
    
    
    #
    def params(self, mean, std, skew, kurt, stddist=False, dtol_max=1e6):
        """
        Parameters of S dist (table lookup from central moments)  
        
        mean: mean
        std: standard deviation
        skew: skewness
        kurt: kurtosis
        ---------------------------------------------------------------------
        dtol_max: maximum tolerated deviation from density sum 1
        =====================================================================
        Returns np.array([k, a, z, c])
        
        """
    
        return s_params(mean, std, skew, kurt, stddist=False, dtol_max=1e6)
    
    
    #
    def _params(self, mean, std, skew, kurt, dtol_max=1e6):
        """
        Parameters of S dist (table lookup from central moments)  
        
        mean: mean
        std: standard deviation
        skew: skewness
        kurt: kurtosis
        ---------------------------------------------------------------------
        dtol_max: maximum tolerated deviation from density sum 1
        =====================================================================
        Returns np.array([k, a, z, c])
        
        """
    
        return s_params(mean, std, skew, kurt, stddist=True, dtol_max=1e6)
    
    
    #
    def stats(self, k, a, scale=1, loc=0, dtol_max=1e6):
        """     
        Statistics of central moments wrapper function 
        
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        dtol_max: maximum tolerated deviation from density sum 1    
        =====================================================================
        Returns np.array([mean, standard_dev, skewness, kurtosis])
        
        """
    
        return s_stats(k=k, a=a, z=scale, c=loc, stddist=False, dtol_max=dtol_max)
    
    
    #
    def _stats(self, k, a, dtol_max=1e6):
        """     
        
        """
    
        return s_stats(k=k, a=a, z=1, c=0, stddist=True, dtol_max=dtol_max)
    
    
    #
    def fit(self, x, dtol_max=1e6):
        """
        Fit a static S distribution to data of outcome values and return the parameters; wrapper function 
    
        x: values vector of an outcome variable
        ---------------------------------------------------------------------
        dtol_max: maximum tolerated deviation from density sum 1
        =====================================================================
        Returns np.array([k, a, z, c])
        
        """
        
        return s_fit(x, stddist=False, dtol_max=dtol_max)
    
    
    #
    def _fit(self, x, dtol_max=1e6):
        """
        
        """
        
        return s_fit(x, stddist=True, dtol_max=dtol_max)
    
    
    #
    def median(self, k, a, scale=1, loc=0):
        """
        Median wrapper function 
        
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar median
        
        """
        
        return s_quantile(0.5, k=k, a=a, z=scale, c=loc)
    
    
    #
    def _median(self, k, a):
        """
        
        """
        
        return median(k=k, a=a, scale=1, loc=0)
    
    
    #
    def mean(self, k, a, scale=1, loc=0):
        """
        Self speaking wrapper function for the mean
        
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar mean
        
        """
        
        moments= s_stats(k=k, a=a, z=scale, c=loc)
        return moments[0]
    	
    	
    #
    def _mean(self, k, a):
        """
        
        """
        
        return mean(k=k, a=a, scale=1, loc=0)
    
    
    #
    def std(self, k, a, scale=1, loc=0):
        """
        Self speaking wrapper function for the standard deviation
        
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar standard deviation
        
        """
        
        moments= s_stats(k=k, a=a, z=scale, c=loc)
        return moments[1]
    
    
    #
    def _std(self, k, a):
        """
        
        """
        
        return std(k=k, a=a, scale=1, loc=0)
    
    
    #
    def var(self, k, a, scale=1, loc=0):
        """
        Self speaking wrapper function for the variance
        
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns scalar variance
        
        """
        
        moments= s_stats(k=k, a=a, z=scale, c=loc)
        return moments[1] **2
    
    
    #
    def _var(self, k, a):
        """
        
        """
        
        return var(k=k, a=a, scale=1, loc=0)
    
    
    #
    def interval(self, confidence, k, a, scale=1, loc=0, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
        """
        Two-sided confidence interval wrapper function   
        
        confidence: two-sided probability value € [0, 1]
        k: power and kurtosis parameter € [1, 3]
        a: asymmetry parameter € [0.5, 2]
        ---------------------------------------------------------------------
        scale: deviation scale parameter
        loc: central location parameter
        =====================================================================
        Returns (left_q, right_q)
        
        """
        
        q_left, q_right= s_interval(confidence, k=k, a=a, z=scale, c=loc, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] )
        
        return (q_left, q_right)
    
    
    #
    def _interval(self, confidence, k, a, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] ):
        """
        
        """
        
        return intervall(confidence, k=k, a=a, scale=1, loc=0, u_res=numsetup[0], u_min=numsetup[1], u_max=numsetup[2] )



# Dummy instance to assign a second handy access name to the class
s_dist= s_dist_gen(name='s_dist')



#############################################################################################################################################################################################################################################
###


### - An implementation of "_methods" with direction="right"/"left" in s_dist_gen class is blocked by the parent class rv_continuous (only shape parameters and parameter of _rvs() are allowed?)


## Update License.txt?


