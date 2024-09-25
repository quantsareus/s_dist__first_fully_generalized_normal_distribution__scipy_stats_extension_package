#! /bin/python3


"""

The modul extends the package s_dist for a new fully generalized linear model (FGLM). 


The package s_dist is an implementation of the fully generalized S dist invented by Schlingmann, H., that was published in the Phd thesis 'Introducing a fully generalized normal distribution' in January 2024 on Github (github.com/quantsareus/introducing_a_fully_generalized_normal_distribution). 

Summing it up short, the new S dist is the first generalized normal dist, that is flexible in both, skewness and kurtosis (power), at the same time (= fully generalized). The new giant flexibility makes it a very univeral distribution, that eats about 95% of the empirical dists (including the dists of metrics and parameters), that do occur in daily data science practice.   

Further, it is directly downward compatible to the normal distribution (without parameter conversion). The new S dist is closely related to the Gaussian; it can be described as a Gaussian (already flexible in kurosis/ power), that  additionally got enhanced for flexibility in skewness and that has been normed to a total density of 1. For detailed information about S dist refer to the thesis (github.com/quantsareus/introducing_a_fully_generalized_normal_distribution/introducing_a_fully_generalized_normal_distribution_published.pdf).


The FGLM solver is realized by a plain vanilla iteratively reweighted least squares algorithm. As usual, iteration 0 gets done by an OLS estimation. The solver logic of every further iteration step is the following:
1. Analyze the S dist parameters of the residuals.
2. Take the residuals of the previous iteration step and tense them with their S dist parameters in a P-norm like manner (weight tension).
3. Perform a weighted least squares regression.
4. Go to step no. 1 until the target precision of the beta parameters is reached.


The plain vanilla iteratively reweighted least squares algorithm finds better solutions than the OLS regression, when the effective power is between 1 and 3. Thus, k*a and k/a have to remain within the value range [1.0, 3.0].


The modul also implements two new goodness of fit metrics (GoF metrics). Area-not-fit (an area between the curves based goodness of not fit metric) and R_Lk (fully generalized R square). Because the traditional R square GoF metric has exactly the same loss function as OLS regression, it evaluates the final goodness of fit not objective, but always in strong favor for the OLS regression solution. Thus, the new GoF metrics do provide additional GoF measurement.  

"""


import numpy as np
from numpy.linalg import inv
import pandas as pd
# import scipy as sc
# import sympy as sym
# from sympy.functions.special.gamma_functions import gamma
# import sklearn as skl


from s_dist.s_dist import *
import s_dist.fglm as fglm


#############################################################################################################################################################################################################################################
### Inits



#############################################################################################################################################################################################################################################
### Functions


#
def area_not_fit(y_t, y_e, verbose=False):
    """
    Area between the curves based goodness of (not) fit metric
    
    y_t: value vector of the target variable
    y_e: value vector of the estimator
    =====================================================================
    Returns scalar GoF metric
    
    """

    area_dev= np.sum(np.abs(y_t -y_e))
    area_targ= np.sum(np.abs(y_t))
    
    if verbose== True:
        print("Area of deviations from target ", area_dev)
        print("Area of target different from 0  ", area_targ)
    
    return area_dev /area_targ


#
def r_l_k(y_t, y_e, k=1):
    """
    R coefficient of determination with variable loss function power of e.g. 1, 1.717, 2, 2.511, ...; aka fully generalized R-square
    
    y_t: value vector of the target variable
    y_e: value vector of the estimator
    k: power of the loss function
    =====================================================================
    Returns scalar GoF metric
    
    """
    
    y_t_mean= np.mean(y_t)
    fit= 1 -( np.sum( np.abs(y_e -y_t)**k ) / np.sum( np.abs(y_t -y_t_mean)**k ) )
    return fit


#
def fglm_fit_report(y, X, iter_max=25, b_change_min=1e-3):
    """
    A fully generalized linear model to run a multivariate regression on a target variable, that contains an assumed linear relationship to the independent variable matrix X and an assumed S distributed (maybe skew and kurtosed) random error
    
    y: value vector of the target variable
    X: matrix of independent variable values containing an ones vector in the first columns
    ---------------------------------------------------------------------
    iter_max: maximum number of iterations 
    b_change_min: cumulative target precision of beta (exclusive intercept value)
    =====================================================================
    Returns a list of results objects [y_, b_, b_interc_corr_, e_params_, e_, r_l1_, r_l2_, r_lk_]

    """

    # Iteration No.0 OLS-Regression 
    
    # OLS regression weights
    w= np.repeat(1, y.shape[0])
    w= np.abs(w)/ sum(np.abs(w))
    W= np.diag(w)

    b_= inv(X.T @ W @ X) @ (X.T @ W @ y)

    b_change= 1e0

    y_= X @ b_
    e_= y - y_

    # Value conservation
    y_0_= y_.copy()
    e_0= e_.copy()
    b_0= b_.copy()
    
    ################################################################
    ### Reporting
    print("Iteration No.", "0 ", "OLS-Regression Init" )
    print("")


    for i in range(1, iter_max +1):
	    
	    if b_change < b_change_min:
	    	print (" -- target precision reached -- ")
	    	print ("beta change to last  ", b_change)
	    	print ("")
	    	break
	    
	    b_last = b_.copy()
	    
	    ################################################################
	    ### parameter search
	    
	    e_moments= d_cmoments(e_)
	    e_mean= e_moments[0]
	    e_std= e_moments[1]
	    e_skew= e_moments[2]
	    e_kurt= e_moments[3]
	    
	    e_params_= s_fit(e_)
	    k_= e_params_[0]
	    a_= e_params_[1]
	    z_= e_params_[2]
	    c_= e_params_[3]
	    
             # Forward of the table moments
	    e_moments_= s_stats(c_, z_, k_, a_, stddist=True)
	    e_mean_= e_moments_[0]
	    e_std_= e_moments_[1]
	    e_skew_= e_moments_[2]
	    e_kurt_= e_moments_[3]
            
            ################################################################
	    ### Tense Weights
	    
	    # Correction of skewness mean shift 
	    w= e_ -c_
	    
            # Appropriate weight tense for k_*a_ >1 respectively k_/a_ >1
	    w[w <0]= (np.abs(w[w <0]) ) **(0.25* (k_*a_ -2))
	    w[w >0]= (np.abs(w[w >0]) ) **(0.25* (k_/a_ -2))
	    
            # Renorming weights to sum up to 1 again
	    w= np.abs(w)/ sum(np.abs(w))
	    
	    W= np.diag(w)
	    
	    ################################################################
	    
	    # Iteratively reweighted least squares
	    b_= inv(X.T @ W @ X) @ (X.T @ W @ y)
	    
	    # Skewness mean shift correction
	    b_interc_corr_= b_[0] +c_
	    
	    # Precision reached criterion
	    b_change= sum(np.abs(b_[1:] -b_last[1:])) /sum(np.abs(b_[1:]))
	    
	    # Estimate Update
	    y_= X @ b_
            
	    e_= y - y_
	    
	    r_l1_= r_l_k(y, y_, k=1)
	    r_l2_= r_l_k(y, y_, k=2)
	    r_lk_= r_l_k(y, y_, k=k_)
	    # area_nf= area_not_fit(y, y_)    
	    
	    ################################################################
	    ### Reporting
	    print("Iteration No.", i)
	    print("")
	    print("error moments", "  mean:", e_mean, "  std:", e_std, "  skew:", e_skew, "  kurt:", e_kurt)	 
	    print("table moments", "  mean_:", e_mean_, "  std_:", e_std_, "  skew_:", e_skew_, "  kurt_:", e_kurt_)	 
	    print("table parameters", "  k_:", k_, "  a_:", a_, "  z_:", z_, "  c_:", c_, )
	    print("beta ", b_, "  corr.intercept", b_interc_corr_, "  beta change to last:", b_change) 
	    print("GoF", "  R_L1:", r_l1_, "  R_L2:", r_l2_, "  R_Lk:", r_lk_)
	    print("")
    
    
    r_l1_0= r_l_k(y, y_0_, k=1)
    r_l2_0= r_l_k(y, y_0_, k=2)
    r_lk_0= r_l_k(y, y_0_, k=k_)
    # area_nf_0= area_not_fit(y, y_0_)
    
    e_0_moments= d_cmoments(e_0)
    e_0_mean= e_0_moments[0]
    e_0_std= e_0_moments[1]
    e_0_skew= e_0_moments[2]
    e_0_kurt= e_0_moments[3]
    
    e_0_params= s_fit(e_0)
    k_0= e_0_params[0]
    a_0= e_0_params[1]
    z_0= e_0_params[2]
    c_0= e_0_params[3]
    
    e_0_moments_= s_stats(c_0, z_0, k_0, a_0, stddist=True)
    e_0_mean_= e_0_moments_[0]
    e_0_std_= e_0_moments_[1]
    e_0_skew_= e_0_moments_[2]
    e_0_kurt_= e_0_moments_[3]
    
    b_0_interc_corr_= b_0[0] +c_0
    
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("")
    print("  In comparison to")
    print("")

    print("Iteration No.", "0 ", "OLS Regression" )
    print("")
    print("error moments", "  mean:", e_0_mean, "  std:", e_0_std, "  skew:", e_0_skew, "  kurt:", e_0_kurt)
    print("table moments", "  mean_:", e_0_mean_, " std_:", e_0_std_, "  skew_:", e_0_skew_, "  kurt_:", e_0_kurt_)
    print("table parameters", "  k_:", k_0, "  a_:", a_0, "  z_:", z_0, "  c_:", c_0 )
    print("beta ", b_0, "  corr.intercept", b_0_interc_corr_) 
    print("GoF", "  R_L1:", r_l1_0, "  R_L2:", r_l2_0 ,"  R_Lk:", r_lk_0)
    print("")
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")

    # Returns a list of results objects
    return [y_, b_, b_interc_corr_, e_params_, e_, r_l1_, r_l2_, r_lk_]


#
def fglm_fit(y, X, iter_max=25, b_change_min=1e-3):
    """
    The core version (reporting free version) of the fully generalized linear model to run a multivariate regression on a target variable, that contains an assumed linear relationship to the independent variable matrix X and an assumed S distributed (maybe skew and kurtosed) random error
    
    y: value vector of the target variable
    X: matrix of independent variable values containing an ones vector in the first columns
    ---------------------------------------------------------------------
    iter_max: maximum number of iterations 
    b_change_min: cumulative target precision of beta (exclusive intercept value)
    =====================================================================
    Returns a list of results objects [y_, b_, b_interc_corr_, e_params_, e_]
    
    """
    
    # Iteration No.0 OLS-Regression 
    
    # OLS regression weightss
    
    w= np.repeat(1, y.shape[0])
    w= np.abs(w)/ sum(np.abs(w))
    W= np.diag(w)
    
    b_= inv(X.T @ W @ X) @ (X.T @ W @ y)

    b_change= 1e0

    y_= X @ b_
    e_= y - y_

    
    ################################################################
    ### Reporting
    print("Iteration No.", "0 ", "OLS-Regression Init" )
    # print("")
    

    for i in range(1, iter_max +1):
	    
	    if b_change < b_change_min:
	    	print (" -- target precision reached -- ")
	    	print ("beta change to last  ", b_change)
	    	print ("")
	    	break
	    
	    b_last = b_.copy()
	    
	    ################################################################
	    ### parameter search
	    
	    e_moments= d_cmoments(e_)
	    e_mean= e_moments[0]
	    e_std= e_moments[1]
	    e_skew= e_moments[2]
	    e_kurt= e_moments[3]
    	    
	    e_params_= s_fit(e_)
	    k_= e_params_[0]
	    a_= e_params_[1]
	    z_= e_params_[2]
	    c_= e_params_[3]
	    
            # Forward of the table moments
	    e_moments_= s_stats(c_, z_, k_, a_, stddist=True)
	    e_mean_= e_moments_[0]
	    e_std_= e_moments_[1]
	    e_skew_= e_moments_[2]
	    e_kurt_= e_moments_[3]

            ################################################################
	    ### Tense Weights
	    
	    # Correction of skewness mean shift
	    w= e_ +c_
	    
            # Appropriate weight tense for k_*a_ >1 respectively k_/a_ >1
	    w[w <0]= (np.abs(w[w <0]) ) **(0.25* (k_*a_ -2))
	    w[w >0]= (np.abs(w[w >0]) ) **(0.25* (k_/a_ -2))
	    
            # Renorming weights to sum up to 1 again
	    w= np.abs(w)/ sum(np.abs(w))
	    
	    W= np.diag(w)
	    
	    ################################################################
	    
	    # Iteratively reweighted least squares
	    b_= inv(X.T @ W @ X) @ (X.T @ W @ y)
	    
	    # Skewness mean shift correction
	    b_interc_corr_= b_[0] +c_
	    
	    # Precision reached criterion
	    b_change= sum(np.abs(b_[1:] -b_last[1:])) /sum(np.abs(b_[1:]))
	    
	    # Estimate Update
	    y_= X @ b_

	    e_= y - y_
	    
	    
	    ################################################################
	    ### Reporting
	    print("Iteration No.", i)
	    # print("")
	    
            
    # Return a list of result objects
    return [y_, b_, b_interc_corr_, e_params_, e_]





#############################################################################################################################################################################################################################################
### 






