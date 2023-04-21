# Data Analysis

## DK

The folder DK contains Python codes to perform Data Analysis of Delayed Kernel models A and B in Davies, Coolen and Galla (2023)

### Model A

The folder ModelA contains three Python codes
1. AIDS_DK_ModelA.py
2. LIVER_DK_ModelA.py
3. PBC_DK_ModelA.py

Each code reads in 10 data sets created by the R code in the folder JM_LM corresponding to equal splits of the relevant data set (AIDS, Liver, PBC). At each iteration, one data set group is treated as the test data and the other groups make up the training data. The code performs the fixed base time and fixed prediction window analysis for delayed kernel Model A for these 10 sets of test/training data.

Models are fitted to the training data while prediction error is calculated for the test data for each combination of base time, t, and prediction time, u.

The results for the fixed base time and fixed prediction window analyses are saved at each iteration. An average over the iterations (along with the retarded kernel results) was performed subsequently using the Python code Plot_Average.py (in Plotting folder).

Details of the models fitted and the analysis performed is given in the manuscript 'Delayed kernels for longitudinal survival analysis and dynamic prediction', Davies, Coolen and Galla (2023).

### Model B 

The folder ModelB contains three Python codes
1. AIDS_RK_ModelB.py
2. LIVER_RK_ModelB.py
3. PBC_RK_ModelB.py

Each code reads in 10 data sets created by the R code in the folder JM_LM corresponding to equal splits of the relevant data set (AIDS, Liver, PBC). At each iteration, one data set group is treated as the test data and the other groups make up the training data. The code performs the fixed base time and fixed prediction window analysis for delayed kernel Model B for these 10 sets of test/training data.

Models are fitted to the training data while prediction error is calculated for the test data for each combination of base time, t, and prediction time, u.

The results for the fixed base time and fixed prediction window analyses are saved at each iteration. An average over the iterations (along with the retarded kernel results) was performed subsequently using the Python code Plot_Average.py (in Plotting folder).

Details of the models fitted and the analysis performed is given in the manuscript 'Delayed kernels for longitudinal survival analysis and dynamic prediction', Davies, Coolen and Galla (2023).

## JM_LM

The folder JM_LM contains three R codes:
1. AIDS_JM_LM.R 
2. Liver_JM_LM.R
3. PBC_JM_LM.R

Each code reads in the original data set (aids, prothro and pbc2 respectively) and splits the data into 10 equally sized groups in order to perform 10-fold cross validation. The data for each group is saved so that it can be read into the delayed kernel model codes. 

At each iteration one group is assigned as the test data and the remaining groups comprise the training data. 

Models are fitted to the training data while prediction error is calculated for the test data for each combination of base time, t, and prediction time, u.

The results for the fixed base time and fixed prediction window analyses are saved at each iteration. 
An average over the iterations (along with the retarded kernel results) was performed subsequently using the Python code Plot_Average.py (in Plotting folder).  

Details of the models fitted and the analysis performed is given in the manuscript 'Delayed kernels for longitudinal survival analysis and dynamic prediction', Davies, Coolen and Galla (2023). 
