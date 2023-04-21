'''
Code to perform data analysis on the PBC data set in Davies, Coolen and Galla (2023) for delayed kernel model B.
Written by A. Davies (2023).
The 10 groups of PBC data (obtained from the R code PBC_JM_LM.R) are read in.
The 10 groups are used to perform 10-fold cross validation.
At each iteration, one group is taken as the test data and the other nine groups comprise the training data.
The delayed kernel model B is fitted to the training data using maximum likelihood.
Minimisation of the negative log likelihood is performed using Powell's method via the function scipy.optimize()

First we perform the fixed base time analysis with base time t = 3 years:
The prediction time u is varied in steps of 0.2 years from 3 to 8 years
Prediction error calculated using the fitted model and test data for each combination of t and u is stored at each iteration.

Then we perform the fixed prediction window analysis for three windows:
w1=1 year, w2=2 years, w3=3 years
At w1 we use base times t=0-9 years in steps of 0.2 years
At w2 we use base times t=0-8 years in steps of 0.2 years
At w3 we use base times t=0-7 years in steps of 0.2 years
Again, prediction error for each scenario is stored for each iteration.
An average of PE over the 10 iterations is performed subsequently.

The model fitted here specifies a fixed assocation parameter for individuals with s=0 (n=1)
For the decaying parameter model, the relevant changes to the code are commented out and labelled 's=0 Decaying Association Model'
The changes correpsond to one line in three functions: mu_sum1, mus_sum2 and EXP_B 
The model fitted here treats the transplant event as a censoring event.
For the model that treats the two events as a composite event we simply read in the files Censor_comp.txt instead of Censor.txt for each group.
These files are commented out and labelled 'Composite Event Model'. 
Description of delayed kernel model B can be found in the manuscriupt 'Delayed kernels for longitudinal survival analysis and dynamic predictions' Davies, Coolen and Galla (2023). 
Detailed derivations of the functions evaluated by this code can be found in the corresponding Supplementary Material.
'''
import numpy as np
import os
import math
from numpy import exp, arange, sqrt, pi, log
import scipy.optimize as optimize
import time
import multiprocessing as mp
from multiprocessing import Pool

#############################
#Maximum likelihood functions
#############################

'''
To perform maximum likelihood inference of delayed kernel model B we minimise the negative log likelihood for the training data.
The function func() outputs the negative log likelihood for model B calling to functions mu_sum1(), j_sum() and mu_sum2().
The equation that correpsonds to this function is Eq (22) in Davies, Coolen and Galla (2023).
We make use of Eq (S16) in the correpsonding supplementary material which gives the result of the integration in Eq (22) when using the LOCF interpolation procedure for Model B.
'''
def mu_sum1(V, T, Z, Z_fix, X, i, j, p, n):
    mu_sum = 0.0
    #work out the sum over n[] up to j-1 for use in Z and T indicators
    sum_nj = 0
    for k in range(0,j):
        sum_nj = sum_nj + n[k]

    s_j = T[sum_nj + n[j] - 1] #final observation time 
    min_XS = min(s_j, X[i])

    #sum over p longitudinal covariates
    for mu in range(0, p):
        tau = p + mu
        if n[j]==1: 
            a = 0
            mu_sum = mu_sum + V[mu]*Z[sum_nj*p + a*p + mu]                      #s=0 Fixed Association Model
            #mu_sum = mu_sum + V[mu]*Z[sum_nj*p + a*p + mu]*exp(-X[i]/V[tau])   #s=0 Decaying Association Model
        else: 
            #sum over n[j]-1 longitudinal observations (LOCF)
            for a in range(0, n[j]-1): 
                if X[i] <= T[sum_nj + a]: 
                    mu_sum = mu_sum
                else:
                    min_XT = min(X[i], T[sum_nj + a + 1])
                    #first term
                    mu_sum = mu_sum + V[mu]*Z[sum_nj*p + a*p + mu]*(min_XT - T[sum_nj + a])/min_XS #CHANGE
                    #second term
                    #term 2a:
                    diffa = X[i] - T[sum_nj + a] 
                    diffa1 = X[i] - min_XT
                    term2a = exp(-diffa1/V[tau]) - exp(-diffa/V[tau])
                    #term 2b:
                    diffs = X[i] - min_XS
                    term2b = (min_XT-T[sum_nj + a])*(exp(-diffs/V[tau])-exp(-X[i]/V[tau]))/min_XS #CHANGE
                    #result:
                    mu_sum = mu_sum + V[mu]*Z[sum_nj*p + a*p + mu]*(term2a - term2b)
    #sum over q fixed covariates
    for nu in range(0, q):
        mu_sum = mu_sum + V[2*p+nu]*Z_fix[j*q+nu]
    return mu_sum

def j_sum(V, T, Z, Z_fix, X, i, N_train, p, n):
    j_sum = 0.0
    #sum over j (individuals in the training data)
    for j in range(0, N_train):
        #work out the sum over n[] up to j-1 for use in Z and T indicators
        sum_nj = 0
        for k in range(0,j):
            sum_nj = sum_nj + n[k]
            
        s_j = T[sum_nj + n[j] - 1]  #the final observation time

        if X[j] - X[i] < 0.0 or X[i] < 0.0:
            j_sum = j_sum
        else:
            e = exp(mu_sum1(V, T, Z, Z_fix, X, i, j, p, n))
            j_sum = j_sum + e
    return j_sum


def mu_sum2(V, T, Z, Z_fix, X, i, p, n):
    mu_sum = 0.0
    #work out the sum over n[] up to i-1 for use in Z and T indicators
    sum_ni = 0
    for k in range(0,i):
        sum_ni = sum_ni + n[k]

    s_i = T[sum_ni + n[i] - 1] #the final observation time 
    min_XS = min(s_i,X[i]) #should always be s_i

    #sum over p longitudinal covariates
    for mu in range(0, p):
        tau = p + mu
        if n[i]==1:
            a = 0
            mu_sum = mu_sum + V[mu]*Z[sum_ni*p + a*p + mu]                      #s=0 Fixed Association Model
            #mu_sum = mu_sum + V[mu]*Z[sum_ni*p + a*p + mu]*exp(-X[i]/V[tau])   #s=0 Decaying Association Model
        else: 
            #sum over n[i] longitudinal observations
            for a in range(0, n[i]-1):
                if X[i] <= T[sum_ni + a]: #this shouldn't ever be the case 
                    print("oh no mu sum2")
                    mu_sum = mu_sum
                else:
                    min_XT = min(X[i], T[sum_ni + a + 1])   #should always be T 
                    #first term
                    mu_sum = mu_sum + V[mu]*Z[sum_ni*p + a*p + mu]*(min_XT - T[sum_ni + a])/min_XS 
                    #second term
                    #term 2a:
                    diffa = X[i] - T[sum_ni + a] 
                    diffa1 = X[i] - min_XT
                    term2a = exp(-diffa1/V[tau]) - exp(-diffa/V[tau])
                    #term 2b:
                    diffs = X[i] - min_XS
                    term2b = (min_XT-T[sum_ni + a])*(exp(-diffs/V[tau])-exp(-X[i]/V[tau]))/min_XS 
                    #Result:
                    mu_sum = mu_sum + V[mu]*Z[sum_ni*p + a*p + mu]*(term2a - term2b)
    #sum over q fixed covariates
    for nu in range(0, q):
        mu_sum = mu_sum + V[2*p+nu]*Z_fix[i*q+nu]
    return mu_sum

def func(V, T, Z, Z_fix, X, N_train, p, n, Censor):
    function = 0.0
    #penalise for negative tau (NB: for PBC data V[3], V[4] and V[5] represent tau)
    if V[3] < 0 or V[4] < 0 or V[5] < 0: 
        function = function + 10e15
    else:
        #sum over i (individuals in training data)
        for i in range(0, N_train):
            if Censor[i] == 0: 
                function = function 
            else:
                function = function + log(j_sum(V, T, Z, Z_fix, X, i, N_train, p, n)) - mu_sum2(V, T, Z, Z_fix, X, i, p, n)
    return function

#######################
# Prediction error functions
#######################
'''
Functions to calculate the prediction error for delayed kernel model B fitted to training data evaluated on test data.
Function PE_SQ() returns the prediction error for given base time, prediction time, and training and test data vectors.
PE_SQ() calls to the function S_t() which calculates survival probability.
Function S_t() calls to function EXP_B().
Within function PE_SQ() 'new data' vectors are created from the test data vectors restricting observations to be <= t (base time).
The Eq for prediction error is given in Eq (26) of Davies, Coolen and Galla (2023). 
The Eq for survival probability (of the DK model) for a given base time and prediction time is given in Eq (27). 
For the integrals we again we make use of Eq (S14) in the correpsonding Supplementary Material.
'''
def EXP_B(ti, kappa, tau, gamma, Z_t, T_t, Zf_t, n_t, j):
    summ = 0.0

    #work out the sum over n[] up to j-1 for use in Z and T indicators
    sum_nj = 0
    for k in range(0,j):
        sum_nj = sum_nj + n_t[k]
    sum_nj = int(sum_nj)
    s_j = T_t[sum_nj + int(n_t[j]) - 1] #the final observation time
    min_XS = min(s_j,ti)

    #sum over p longitudinal covariates
    for mu in range(0,p):
        if int(n_t[j])==1:
            a = 0
            summ = summ + kappa[mu]*Z_t[sum_nj*p + a*p + mu]                    #s=0 Fixed Association Model
            #summ = summ + kappa[mu]*Z_t[sum_nj*p + a*p + mu]*exp(-ti/tau[mu])  #s=0 Decaying Association Model
        else: 
            #sum over n[j] longitudinal observations
            for a in range(0,int(n_t[j])-1): 
                if ti <= T_t[sum_nj + a]: 
                    summ = summ
                else:
                    min_XT = min(ti, T_t[sum_nj + a + 1]) 
                    #first term
                    summ = summ + kappa[mu]*Z_t[sum_nj*p + a*p + mu]*(min_XT - T_t[sum_nj + a])/min_XS 
                    #second term
                    #term 2a:
                    diffa = ti - T_t[sum_nj + a] 
                    diffa1 = ti - min_XT
                    term2a = exp(-diffa1/tau[mu]) - exp(-diffa/tau[mu])
                    #term 2b:
                    diffs = ti - min_XS; 
                    term2b = (min_XT-T_t[sum_nj + a])*(exp(-diffs/tau[mu])-exp(-ti/tau[mu]))/min_XS 
                    summ = summ + kappa[mu]*Z_t[sum_nj*p + a*p + mu]*(term2a - term2b)
    #sum over q fixed covariates
    for nu in range(0,q):
        summ = summ + gamma[nu]*Zf_t[j*q+nu]
    e = exp(summ) 
    return e


def S_t(kappa, tau, gamma, predTime, baseTime, X, Z_fix, Z_fix_test, Z, Z_new, T, T_new, n, n_new, Censor, j, N_train):
    #j refers to the indiviudal in the test data
    #i and k label individuals in the training data

    #sum over individuals (event times) in the training data
    sum_i = 0.0
    for i in range(0,N_train):
        if Censor[i] == 0 or X[i] < baseTime or X[i] > predTime:
            sum_i = sum_i
        else:
            #base hazard - sum over individuals in the training data:
            denom = 0.0
            for k in range(0,N_train):
                sum_nk = 0
                for l in range(0,k):
                    sum_nk = sum_nk + n[l]
                s_k = T[sum_nk + n[k] - 1] #the final observation time 
                if X[i] < 0.0 or X[i] > X[k]:
                    denom = denom
                else:
                    denom = denom + EXP_B(X[i], kappa, tau, gamma, Z, T, Z_fix, n, k)
            #for individual j (test) we only have data up to baseTime hence 'new' vectors (from test data)
            numer = EXP_B(X[i], kappa, tau, gamma, Z_new, T_new, Z_fix_test, n_new, j)
            sum_i = sum_i + numer/denom
    S = exp(-sum_i)
    return S


def PE_SQ(predTime, baseTime, kappa, tau, gamma, X, X_test, Censor, Censor_test, Z, Z_test, Z_fix, Z_fix_test, T, T_test, n, n_test, N_train, N_test):
    #create Z_new, n_new, t_obs_new and T_new from baseTime and TEST vectors
    #keep only observations <= baseTime

    #No. of observations
    n_new = np.zeros(N_test)
    sum_new = 0
    for j in range(0,N_test): 
        sum_nj = 0
        for i in range(0,j):
            sum_nj = sum_nj + n_test[i]

        count_n = 0
        for a in range(0,n_test[j]):
            if t_obs_test[sum_nj + a] <= baseTime:
                count_n = count_n + 1
            else:
                count_n = count_n
        n_new[j] = count_n  #new n defined 
        sum_new = sum_new + n_new[j]

    sum_new = int(sum_new)
    Z_new = [0.0]*p*sum_new         #longitudinal covariates (create vector of zeroes of size p*sum_new)
    T_new = [0.0]*sum_new           #observation times (create vector of zeroes of size sum_new)

    for j in range(0,N_test): 
        new_nj = 0
        sum_nj = 0
        for i in range(0,j):
            new_nj = new_nj + n_new[i]
            sum_nj = sum_nj + n_test[i]
        new_nj = int(new_nj)
        sum_nj = int(sum_nj)     
        #T new************************************
        for a in range(0,int(n_new[j])):
            T_new[new_nj+a] = T_test[sum_nj+a]
        #Z_new****************************************
        for mu in range(0,p):
            for a in range(0,int(n_new[j])):
                Z_new[p*new_nj + p*a + mu] = Z_test[p*sum_nj + p*a + mu]


    r = 0.0     #no. of subjects in test data at risk at baseTime
    summ = 0.0
    #sum over individuals in test data still at risk at baseTime (Xj>=baseTime)
    for j in range(0,N_test):
        if X_test[j] < baseTime:
            summ = summ 
        else:
            r = r + 1.0
            #Prob of j's survival to predTime given covariates up to baseTime (new vectors) and survival to baseTime
            SprobTau = S_t(kappa, tau, gamma, predTime, baseTime, X, Z_fix, Z_fix_test, Z, Z_new, T, T_new, n, n_new, Censor, j, N_train)
            #term 1: still alive at predTime
            if X_test[j] >= predTime:
                summ = summ + pow((1.0-SprobTau), 2.0)
            #term 2: experienced an event by predTime (and after baseTime)
            elif Censor_test[j]==1:
                summ = summ + pow(SprobTau, 2.0)
            #term 3: censored by predTime (and after baseTime)
            else:
                #Prob of j's survival to predTime given covariates up to baseTime (new vectors) and survival to X_test[j]
                SprobTj = S_t(kappa, tau, gamma, predTime, X_test[j], X, Z_fix, Z_fix_test, Z, Z_new, T, T_new, n, n_new, Censor, j, N_train)
                summ = summ + pow((1.0-SprobTau),2.0)*SprobTj + pow(SprobTau,2.0)*(1.0-SprobTj)
    M_Y_sq = summ/r
    return M_Y_sq




######################
# Function to read in data
######################

#Function to read in longitudinal covariates for group g
#Z vector is organised as follows (with notation Z^i_mu(ta))
#Z^0_0(t0), Z^0_1(t0), ..., Z^0_p(t0), Z^0_0(t1), Z^0_1(t1), ...,  Z^0_p(t1),...,
#Z^0_0(tn0), Z^0_1(tn0), ..., Z^0_p(tn0),...Z^1_0(t0), Z^1_1(t0), ...,  Z^1_p(t0),...
def read_Zlong(fn_read, g, N_t, n):
    fnZ1 = fn_read + str(g) + "/Zlong_bili.txt"
    Z_bili = np.loadtxt(fnZ1, dtype=float)

    fnZ2 = fn_read + str(g) + "/Zlong_alb.txt"
    Z_alb = np.loadtxt(fnZ2, dtype=float)

    fnZ3 = fn_read + str(g) + "/Zlong_proth.txt"
    Z_proth = np.loadtxt(fnZ3, dtype=float)

    #create full Z vector (length = p * sum_n)
    sum_n = 0
    for j in range(0,N_t):
        sum_n = sum_n + n[j]

    Z = np.zeros(p*sum_n)
    for i in range(0, N_t):
        sum_ni=0
        for k in range(0,i):
            sum_ni = sum_ni + n[k]
        for a in range(0,n[i]):
            Z[sum_ni*p+a*p+0] = log(Z_bili[sum_ni+a])
            Z[sum_ni*p+a*p+1] = log(Z_alb[sum_ni+a])
            Z[sum_ni*p+a*p+2] = log(Z_proth[sum_ni+a])

    return Z

#Read in data for group g
def readfiles(fn_read, g):
    fnX = fn_read + str(g) +"/Events.txt" #event times
    X = np.loadtxt(fnX, dtype=float)

    fnN = fn_read + str(g)+ "/n_obs.txt" #number of observations
    n = np.loadtxt(fnN, dtype=int)

    N_g = len(n)
    
    #use function to read in longitudinal covariates and define Z vector
    Z = read_Zlong(fn_read, g, N_g, n) 
    
    fnZF = fn_read + str(g)+ "/Zfix_age.txt" #fixed covariate (age)
    Z_fix = np.loadtxt(fnZF, dtype=float)
    
    fnT = fn_read + str(g)+ "/t_obs.txt" #observation times
    T = np.loadtxt(fnT, dtype=float)

    fnC = fn_read + str(g) + "/Censor.txt" #censoring indicator (transplant = censor event model)
    #fnC = fn_read + str(g) + "/Censor_comp.txt" #censoring indicator (composite event model)
    Censor = np.loadtxt(fnC, dtype=int)    

    return X, n, Z, Z_fix, T, Censor, N_g 

#function to create training data from remaining 9 groups
def create_train(vec, g):
    train = vec[:g]+vec[g+1:]
    train = [l.tolist() for l in train]
    train = [item for sublist in train for item in sublist]
    return train

######################
# Function to perform main tasks
#####################
def main_func(g):
    #depends on group g = test group
    #Create test data = group g
    X_test = [l.tolist() for l in X_vec[g]]
    n_test = [l.tolist() for l in n_vec[g]]
    Z_test = [l.tolist() for l in Z_vec[g]]
    Z_fix_test = [l.tolist() for l in Zf_vec[g]]
    T_test = [l.tolist() for l in T_vec[g]]
    Censor_test = [l.tolist() for l in C_vec[g]]
    N_test = Ng_vec[g]
    #Create train data = all except group g
    X_train = create_train(X_vec, g)
    n_train = create_train(n_vec, g)
    Z_train = create_train(Z_vec, g)
    Z_fix_train = create_train(Zf_vec, g)
    T_train = create_train(T_vec, g)
    Censor_train = create_train(C_vec, g)
    N_train = sum(Ng_vec) - N_test
    
    #MINMISE
    print("minimizing " + str(g))
    #Initialise parameter estimates 
	#order is kappa(bili), kappa(alb), kappa(proth), tau(bili), tau(alb), tau(proth), gamma(age)
    initial_guess = [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0]
    result = optimize.minimize(func, initial_guess, args = (T_train, Z_train, Z_fix_train, X_train, N_train, p, n_train, Censor_train), method = 'Powell', options = {'xtol':3e-8, 'ftol':3e-8, 'return_all':True, 'disp':True}) 
    print("finished minimizing " + str(g))
    
    ##maximum likelihood estimates of model parameters
    kappa = []
    for i in range(0,p):
        kappa.append(result.x[i])
    tau = []
    for i in range(0,p):
        tau.append(result.x[p + i])
    gamma = []
    for i in range(0,q):
        gamma.append(result.x[2 * p + i])
      
    #Save results to file
    fileMin = fn_write + "G" + str(g+1) + "/ModelB_estimates.txt"
    with open(fileMin, "w") as file:
        file.write("g = " + str(g) + "\n result of min: " + str(result.x) + "\n successful minimisation? " + str(result.success) + "\n function value = " + str(result.fun) + "\n \n")
  
    ######################
    #Fix t
    ######################
    #for fixed t=3 years, iterate over prediction time u = 3-8 years in steps of 0.2 years
  	#calculate PE for Model A for each combination of (t,u) using newdata = test data
    print("\n Fixt begin, g = "+str(g))#-------------------------------------------------------------------------
    PE = []
    baseTime = 3.0
    for l in range(0,26):
        predTime = baseTime + l*0.2
        predErr = PE_SQ(predTime, baseTime, kappa, tau, gamma, X_train, X_test, Censor_train, Censor_test, Z_train, Z_test, Z_fix_train, Z_fix_test, T_train, T_test, n_train, n_test, N_train, N_test)
        PE.append(predErr)

    filePE = fn_write + "G" + str(g+1) + "/ModelB_fixt.txt" 
    with open(filePE, "w") as file:
        file.write('\n'.join(str(item) for item in PE))

    print("fix t done, g = "+str(g))

    ######################
    #Prediction window
    ######################
    #For fixed window w1=1 year work out PE for base time t=0-9 years (steps of 0.2)
    print("\n Window 1 begin, g = "+str(g))#-------------------------------------------------------------------------
    PE1 = []
    w1 = 1.0
    for l in range(0,46):
        baseTime = l*0.2 
        predTime = baseTime + w1
        predErr = PE_SQ(predTime, baseTime, kappa, tau, gamma, X_train, X_test, Censor_train, Censor_test, Z_train, Z_test, Z_fix_train, Z_fix_test, T_train, T_test, n_train, n_test, N_train, N_test)
        PE1.append(predErr)

    filePE1 = fn_write + "G" + str(g+1) + "/ModelB_w1.txt" 
    with open(filePE1, "w") as file:
        file.write('\n'.join(str(item) for item in PE1))

    print("w1 done, g = "+str(g))

    #For fixed window w2=2 years work out PE for base time t=0-8 years (steps of 0.2)
    print("\n Window 2 begin, g = "+str(g))#-------------------------------------------------------------------------
    PE2 = []
    w2 = 2.0
    for l in range(0,41):
        baseTime = l*0.2 #initialise
        predTime = baseTime + w2
        predErr = PE_SQ(predTime, baseTime, kappa, tau, gamma, X_train, X_test, Censor_train, Censor_test, Z_train, Z_test, Z_fix_train, Z_fix_test, T_train, T_test, n_train, n_test, N_train, N_test)
        PE2.append(predErr)

    filePE2 = fn_write + "G" + str(g+1) + "/ModelB_w2.txt" 
    with open(filePE2, "w") as file:
        file.write('\n'.join(str(item) for item in PE2))

    print("w2 done, g = "+str(g))

    #For fixed window w3=3 years work out PE for base time t=0-7 years (steps of 0.2)
    print("\n Window 3 begin, g = "+str(g))#-------------------------------------------------------------------------
    PE3 = []
    w3 = 3.0
    for l in range(0,36):
        baseTime = l*0.2
        predTime = baseTime + w3
        predErr = PE_SQ(predTime, baseTime, kappa, tau, gamma, X_train, X_test, Censor_train, Censor_test, Z_train, Z_test, Z_fix_train, Z_fix_test, T_train, T_test, n_train, n_test, N_train, N_test)
        PE3.append(predErr)

    filePE3 = fn_write + "G" + str(g+1) + "/ModelB_w3.txt"
    with open(filePE3, "w") as file:
        file.write('\n'.join(str(item) for item in PE3))

    print("w3 done, g = "+str(g))

    return 0

########################
# Define global variables
########################
st = time.time()

#parameters
p = 3 #number of longitudinal covariates
q = 1 #number of fixed covariates
V = 10 #number of cross validations

#Data:
fn_read = "~...PBC_DATA/G"
X_vec = []
n_vec = []
Z_vec = []
Zf_vec = []
T_vec = []
C_vec = []
Ng_vec = []

for i in range(1,V+1):
    res = readfiles(fn_read,i)
    X_vec.append(res[0])
    n_vec.append(res[1])
    Z_vec.append(res[2])
    Zf_vec.append(res[3])
    T_vec.append(res[4])
    C_vec.append(res[5])
    Ng_vec.append(res[6])

fn_write = "~.../PBC_Results/"


###################
# Run main_func parallel
###################

if __name__ == '__main__':
    __spec__ = None

    pool = Pool(mp.cpu_count())
    res = pool.map(main_func,range(0,V))
    
    pool.close()
    pool.join()
    
    et2 = time.time()
    elapsed_time = et2 - st
    print('Time since start:', elapsed_time, 'seconds')

