'''
Code to perform data analysis using delayed kernel model A for the simulation study in Davies, Coolen and Galla (2023)
The same code can be used for scenario 1 and 2 (simply change the data files that are read in)
Written by A. Davies (2023).

The code reads in the simulated test and training data
The delayed kernel model A is fitted to the training data using maximum likelihood.
Minimisation of the negative log likelihood is performed using Powell's method via the function scipy.optimize()

Then prediction error is evaluated for one prediction window and five base times using the test data

The model fitted here specifies a fixed assocation parameter for individuals with s=0 (n=1)
For the decaying parameter model, the relevant changes to the code are commented out and labelled 's=0 Decaying Association Model'
The changes correpsond to one line in three functions: mu_sum1, mus_sum2 and EXP_A 

Description of delayed kernel model A can be found in the manuscriupt 'Delayed kernels for longitudinal survival analysis and dynamic predictions' Davies, Coolen and Galla (2023). 
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
To perform maximum likelihood inference of delayed kernel model A we minimise the negative log likelihood for the training data.
The function func() outputs the negative log likelihood for model A calling to functions mu_sum1(), j_sum() and mu_sum2().
The equation that correpsonds to this function is Eq (22) in Davies, Coolen and Galla (2023).
We make use of Eq (S14) in the correpsonding supplementary material which gives the result of the integration in Eq (22) when using the LOCF interpolation procedure for Model A.
'''

def musum1(V, T, Z, Z_fix, X, i, j, p, n):
    mu_sum = 0.0
    #work out sum over n[] up to j-1 for use in Z and T indicators
    sum_nj = 0
    for k in range(0,j):
        sum_nj = sum_nj + n[k]
   
    s_j = T[sum_nj + n[j] - 1] #final observation time 
    
    #minimum of sj and ti 
    min_XS = min(s_j, X[i])
    
    #sum over p longitudinal covariates
    for mu in range(0, p):
        tau = p + mu
        if n[j]==1: 
            a = 0
            mu_sum = mu_sum + V[mu] * Z[sum_nj*p + a * p + mu]                  #s=0 Fixed Association Model
            #mu_sum = mu_sum + V[mu]*Z[sum_nj*p + a*p + mu]*exp(-X[i]/V[tau]) 	#s=0 Decaying Association Model
        else: 
            for a in range(0, n[j]-1): #sum over n[j]-1 longitudinal observations (LOCF)
                if V[tau] == 0: 
                    mu_sum = mu_sum
                elif X[i] <= T[sum_nj + a]: 
                    mu_sum = mu_sum
                else:
                    min_XT = min(X[i], T[sum_nj + a + 1])
                    numer = exp(-(min_XS-min_XT)/V[tau])-exp(-(min_XS-T[sum_nj+a])/V[tau])
                    denom = 1.0 - exp(-min_XS / V[tau])
                    mu_sum = mu_sum + (V[mu] * Z[sum_nj*p + a * p + mu] * numer) / denom
    #sum over q fixed covariates
    for nu in range(0, q):
        mu_sum = mu_sum + V[2 * p + nu] * Z_fix[j*q + nu]
    return mu_sum

def j_sum(V, T, Z, Z_fix, X, i, N_train, p, n):
    j_sum = 0.0
    #sum over j (individuals in the training data)
    for j in range(0, N_train):
        #work out the sum over n[] up to j-1 for use in Z and T indicators
        sum_nj = 0
        for k in range(0,j):
            sum_nj = sum_nj + n[k]
            
        s_j = T[sum_nj + n[j] - 1] #the final observation time (a=n[j]-1)

        if X[j] - X[i] < 0.0 or X[i] < 0.0: 
            j_sum = j_sum		
        else:
            e = exp(musum1(V, T, Z, Z_fix, X, i, j, p, n))
            j_sum = j_sum + e
    return j_sum
    
def musum2(V, T, Z, Z_fix, X, i, p, n):
    mu_sum = 0.0
    #work out sum over n[] up to i-1 for use in Z and T indicators
    sum_ni = 0
    for k in range(0,i):
        sum_ni = sum_ni + n[k]
    
    s_i = T[sum_ni + n[i] - 1] #final observation time
    
    #minimum of sj and ti 
    min_XS = min(s_i, X[i])
    
    #sum over p longitudinal covariates
    for mu in range(0, p):
        tau = p + mu
        if n[i]==1:
            a = 0
            mu_sum = mu_sum + V[mu] * Z[sum_ni*p + a * p + mu]                  #s=0 Fixed Association Model
            #mu_sum = mu_sum + V[mu]*Z[sum_ni*p + a*p + mu]*exp(-X[i]/V[tau])	#s=0 Decaying Association Model
        else: #sum over n[i] longitudinal observations
            for a in range(0, n[i]-1): 
                if V[tau] == 0: 
                    mu_sum = mu_sum
                elif X[i] <= T[sum_ni + a]:
                    mu_sum = mu_sum
                else:
                    min_XT = min(X[i], T[sum_ni + a + 1])
                    numer = exp(-(min_XS - min_XT) / V[tau]) - exp(-(min_XS - T[sum_ni + a]) / V[tau])#CHANGE
                    denom = 1.0 - exp(-min_XS / V[tau])
                    mu_sum = mu_sum + (V[mu] * Z[sum_ni*p + a * p + mu] * numer) / denom
    #sum over q fixed covariates
    for nu in range(0, q):
        mu_sum = mu_sum + V[2 * p + nu] * Z_fix[i*q + nu]
    return mu_sum
    
def func(V, T, Z, Z_fix, X, N_train, p, n, Censor):
    func_val = 0.0
    if V[1] < 0: #penalise for negative tau (NB: for simulated data V[1] represents tau)
        func_val = func_val + 10e15
    else:
        for i in range(0, N_train):
            if Censor[i] == 0: 
                func_val = func_val
            else:
                func_val = func_val + log(j_sum(V, T, Z, Z_fix, X, i, N_train, p, n)) - musum2(V, T, Z, Z_fix, X, i, p, n)
    return func_val
    
#######################
# Prediction error functions
#######################
'''
Functions to calculate the prediction error for delayed kernel model A fitted to training data evaluated on test data.
Function PE_SQ() returns the prediction error for given base time, prediction time, and training and test data vectors.
PE_SQ() calls to the function S_t() which calculates survival probability.
Function S_t() calls to function EXP_A().
Within function PE_SQ() 'new data' vectors are created from the test data vectors restricting observations to be <= t (base time).
The Eq for prediction error is given in Eq (26) of Davies, Coolen and Galla (2023). 
The Eq for survival probability (of the DK model) for a given base time and prediction time is given in Eq (27). 
For the integrals we again we make use of Eq (S14) in the correpsonding Supplementary Material.
'''

def EXP_A(ti, kappa, tau, gamma, Z_t, T_t, Zf_t, n_t, j):
    summ = 0.0
    
    #work out the sum over n[] up to j-1 for use in Z and T indicators
    sum_nj = 0
    for k in range(0,j):
        sum_nj = sum_nj + n_t[k]
    sum_nj = int(sum_nj)
    s_j = T_t[sum_nj + int(n_t[j]) - 1] #the final observation time
    min_XS = min(s_j, ti)

    #sum over p longitudinal covariates
    for mu in range(0,p):
        #sum over n[j]-1 longitudinal observations		
        if int(n_t[j])==1: 
            a = 0
            summ = summ + kappa[mu] * Z_t[sum_nj*p + a * p + mu]				#s=0 Fixed Association Model
            #summ = summ + kappa[mu]*Z_t[sum_nj*p + a*p + mu]*exp(-ti/tau[mu])	#s=0 Decaying Association Model
        else:
            for a in range(0,int(n_t[j])-1): 
                if tau[mu] == 0.0: 
                    summ = summ
                elif ti <= T_t[sum_nj + a]:
                    summ = summ
                else:
                    min_XT = min(ti, T_t[sum_nj + a + 1])
                    numer = exp(-(min_XS - min_XT) / tau[mu]) - exp(-(min_XS - T_t[sum_nj + a]) / tau[mu])
                    denom = 1.0 - exp(-min_XS / tau[mu])
                    summ = summ + (kappa[mu] * Z_t[sum_nj*p + a * p + mu] * numer) / denom
    #sum over q fixed covariates
    for nu in range(0,q):
        summ = summ + gamma[nu] * Zf_t[j*q + nu]
    return exp(summ)

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
					denom = denom + EXP_A(X[i], kappa, tau, gamma, Z, T, Z_fix, n, k)
			#for individual j (test) we only have data up to baseTime hence 'new' vectors (from test data)
			numer = EXP_A(X[i], kappa, tau, gamma, Z_new, T_new, Z_fix_test, n_new, j)
			sum_i = sum_i + numer / denom
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
		n_new[j] = count_n #new n defined 
		sum_new = sum_new + n_new[j]

	sum_new = int(sum_new)
	Z_new = [0.0]*p*sum_new	#longitudinal covariates 
	T_new = [0.0]*sum_new	#observation times 

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
			T_new[new_nj + a] = T_test[sum_nj + a]
		#Z_new****************************************
		for mu in range(0,p):
			for a in range(0,int(n_new[j])):
				Z_new[p*new_nj + p * a + mu] = Z_test[p*sum_nj + p * a + mu]

	r = 0.0 #no. of subjects in test data at risk at baseTime
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
				summ = summ + pow((1.0 - SprobTau), 2.0)
			#term 2: experienced an event by predTime (and after baseTime)
			elif Censor_test[j] == 1:
				summ = summ + pow(SprobTau, 2.0)
			#term 3: censored by predTime (and after baseTime)
			else:
				#Prob of j's survival to predTime given covariates up to baseTime (new vectors) and survival to X_test[j]
				SprobTj = S_t(kappa, tau, gamma, predTime, X_test[j], X, Z_fix, Z_fix_test, Z, Z_new, T, T_new, n, n_new, Censor, j, N_train)
				summ = summ + pow((1.0 - SprobTau), 2.0)*SprobTj + pow(SprobTau, 2.0)*(1.0 - SprobTj)
	M_Y_sq = summ / r
	return M_Y_sq



######################
# Function to read in data
######################

#Read in data ('label' refers to either test or training data)
def readfiles(fn_read, label):
    fnX = fn_read + label +"/Events.txt" #event times
    X = np.loadtxt(fnX, dtype=float)

    fnN = fn_read + label + "/n_obs.txt" #number of observations
    n = np.loadtxt(fnN, dtype=int)

    fnZL = fn_read + label + "/Zlong.txt" #longitudinal covariates
    Z = np.loadtxt(fnZL, dtype=float)

    fnZF = fn_read + label + "/Zfix_group.txt" #fixed covariates
    Z_fix = np.loadtxt(fnZF, dtype=float)

    fnT = fn_read + label + "/t_obs.txt" #observation times
    T = np.loadtxt(fnT, dtype=float)

    fnC = fn_read + label + "/Censor.txt" #censoring indicator
    Censor = np.loadtxt(fnC, dtype=int)

    N_g = len(n)
    return X, n, Z, Z_fix, T, Censor, N_g

######################
# Function to perform main tasks
#####################
def main_func(g):
    
    #MINMISE
    print("minimizing " )
    init1 = np.random.uniform(-100,100)
    init2 = np.random.uniform(0,100)
    init3 = np.random.uniform(-100,100)
    initial_guess = [init1, init2, init3]
    print("initialisation")
    print(initial_guess)
    result = optimize.minimize(func, initial_guess, args = (T_train, Z_train, Z_fix_train, X_train, N_train, p, n_train, Censor_train), method = 'Powell', options = {'xtol':3e-8, 'ftol':3e-8, 'return_all':True, 'disp':True}) 
    print("finished minimizing ")
    
    #maximum likelihood estimates of model parameters
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
    fileMin = fn_write + "/ModelA_SimulationEstimates.txt"
    with open(fileMin, "w") as file:
        file.write(" initialisation = " + str(init1) + ", " + str(init2) + ", " + str(init3) + "\n result of min: " + str(result.x) + "\n successful minimisation? " + str(result.success) + "\n function value = " + str(result.fun) + "\n \n")
        file.write(" kappa = " + str(kappa[0]) + "\n tau: " + str(tau[0]) + "\n gamma: " + str(gamma[0]) + "\n \n")
  
        
    print("\n PE begin")#-------------------------------------------------------------------------
    PE2 = []
    w2 = 2.0
    for l in range(0,5):
        baseTime = 1.5 + l*2.0
        print(baseTime)
        predTime = baseTime + w2
        predErr = PE_SQ(predTime, baseTime, kappa, tau, gamma, X_train, X_test, Censor_train, Censor_test, Z_train, Z_test, Z_fix_train, Z_fix_test, T_train, T_test, n_train, n_test, N_train, N_test)
        PE2.append(predErr)

    filePE2 = fn_write + "/PE_ModelA.txt" 
    with open(filePE2, "w") as file:
        file.write('\n'.join(str(item) for item in PE2))

    print("w2 done")
    
    return 0

########################
# Define global variables
########################
st = time.time()

#parameters
p = 1 #number of longitudinal covariates
q = 1 #number of fixed covariates


#Data:
fn_read = "~...SIMULATION\\SCEN1\\" #Change to \\SCEN2\\ for scenario 2

train = readfiles(fn_read,"TRAIN")
X_train = train[0]
n_train = train[1]
Z_train = train[2]
Z_fix_train = train[3]
T_train = train[4]
Censor_train = train[5]
N_train = train[6]

test = readfiles(fn_read,"TEST")
X_test = test[0]
n_test = test[1]
Z_test = test[2]
Z_fix_test = test[3]
T_test = test[4]
Censor_test = test[5]
N_test = test[6]

fn_write = "~...SIMULATION\\SCEN1\\ModelA\\"


main_func(0)

et2 = time.time()
elapsed_time = et2 - st
print('Time since start:', elapsed_time, 'seconds')
