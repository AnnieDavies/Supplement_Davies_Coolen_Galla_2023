# Supplement_Davies_Coolen_Galla_2023
Supplementary material and codes used to perform a simulation study and data analysis of the three data sets (AIDS, Liver, PBC) in 'Delayed kernels for longitudinal survival analysis and dynamic prediction' Davies, Coolen and Galla (2023).

## Data Analysis

The folder 'Data Analysis' contains codes to perform the data analysis of the three real world data sets.
  -> The folder 'DK' contains Python codes for the delayed kernel models
      -> The folder 'ModelA' contains the model A codes for the three datasets
      -> The folder 'ModelB' contains the model B codes for the three datasets
  -> The folder 'JM_LM' contains the R codes for the joint models (JM) and landmarking models (LM)
  -> The folder 'Plotting' contains a Python code to average over the cross validation iterations for all models and plot the results

## Simulation

The folder 'Simulation' contains codes to perform the simualtion study.
  -> The folder 'DK' contains Python codes to read in the simulated data and perform the delayed kernel models
  -> The folder 'JM_LM' contains R codes to simulate data from a JM (scenarios 1 and 2) and to perform the joint models and landmarking models. 
  
