# R Codes to perform Data Analysis of Joint Models and Landmarking Models in Davies, Coolen and Galla (2023)

This folder contains three R codes:
1. AIDS_JM_LM.R 
2. Liver_JM_LM.R
3. PBC_JM_LM.R

Each code reads in the original data set (aids, prothro and pbc2 respectively) and splits the data into 10 equally sized groups in order to perform 10-fold cross validation. The data for each group is saved so that it can be read into the delayed kernel model codes. 

At each iteration one group is assigned as the test data and the remaining groups comprise the training data. 

Models are fitted to the training data while prediction error is calculated for the test data for each combination of base time, t, and prediction time, u.

The results for the fixed base time and fixed prediction window analyses are saved at each iteration. 
An average over the iterations (along with the retarded kernel results) was performed subsequently using the Python code Plot_Average.py (in Plotting folder).  

Details of the models fitted and the analysis performed is given in the manuscript 'Delayed kernels for longitudinal survival analysis and dynamic prediction', Davies, Coolen and Galla (2023). 

## Edited_JMbayes_functions
The folder Edited_JMbayes_functions contains the edited versions of the function prederrJM (from the JMbayes package) for coxph objects, and the function tvBrier (from the JMbayes2 package). 

The original codes were copied from prederrJM.coxph.R at https://github.com/drizopoulos/JMbayes/tree/master/R and accuracy_measure.R (tvBrier()) at https://github.com/drizopoulos/JMbayes2/tree/master/R.

Edits were made so the calculation of prediction error exactly matches the prediction error equation (Eq. (26) in Davies, Galla and Coolen (2023) or, equivelently, the equation for prediction error on pg. 34 of Rizopoulos, D. (2016).  The R package JMbayes for fitting joint models for longitudinal andtime-to-event data using MCMC. Journal of Statistical Software 72(7), 1–46). 

All edits are described in the comments of each code and labelled in line. 
Further details can be found in the Supplementary Material for Davies, Galla and Coolen (2023).
