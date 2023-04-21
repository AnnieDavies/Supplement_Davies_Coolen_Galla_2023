# Supplement_Davies_Coolen_Galla_2023
Supplementary material and codes used to perform a simulation study and data analysis of the three data sets (AIDS, Liver, PBC) in 'Delayed kernels for longitudinal survival analysis and dynamic prediction' Davies, Coolen and Galla (2023).

## Data Analysis

The folder 'Data Analysis' contains codes to perform the data analysis of the three real world data sets.

&nbsp;&nbsp;&nbsp;&nbsp;-> The folder 'DK' contains Python codes for the delayed kernel models

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> The folder 'ModelA' contains the model A codes for the three datasets

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> The folder 'ModelB' contains the model B codes for the three datasets

&nbsp;&nbsp;&nbsp;&nbsp;-> The folder 'JM_LM' contains the R codes for the joint models (JM) and landmarking models (LM)

&nbsp;&nbsp;&nbsp;&nbsp;-> The folder 'Plotting' contains a Python code to average over the cross validation iterations for all models and plot the results

## Simulation

The folder 'Simulation' contains codes to perform the simualtion study.

&nbsp;&nbsp;&nbsp;&nbsp;-> The folder 'DK' contains Python codes to read in the simulated data and perform the delayed kernel models

&nbsp;&nbsp;&nbsp;&nbsp;-> The folder 'JM_LM' contains R codes to simulate data from a JM (scenarios 1 and 2) and to perform the joint models and landmarking models. 
  
## Edited PE functions

The folder 'Edited PE functions' contains the edited versions of prediction error functions from the JMbayes packages:

&nbsp;&nbsp;&nbsp;&nbsp;-> PE.AD.JM2.R edited version of the tvBrier() function in the JMbayes2 package. Original code copied from https://github.com/drizopoulos/JMbayes2/tree/master/R/accuracy_measure.R.

&nbsp;&nbsp;&nbsp;&nbsp;-> PE.AD.coxph.R edited version of prederrJM() for coxph objects in the JMbayes package. Original code copied from https://github.com/drizopoulos/JMbayes/tree/master/R/prederrJM.coxph.R

Edits were made so the calculation of prediction error exactly matches the prediction error equation (Eq. (26) in Davies, Galla and Coolen (2023) or, equivelently, the equation for prediction error on pg. 34 of Rizopoulos, D. (2016).  The R package JMbayes for fitting joint models for longitudinal andtime-to-event data using MCMC. Journal of Statistical Software 72(7), 1–46). 

All edits are described in the comments of each code and labelled in line. 
Further details can be found in the Supplementary Material for Davies, Galla and Coolen (2023).
