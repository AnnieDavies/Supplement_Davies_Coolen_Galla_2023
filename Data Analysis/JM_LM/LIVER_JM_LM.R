# Code to perform data analysis on Liver data set in Davies, Coolen and Galla (2023).
# Written by A. Davies (2021),
# aided by code by D. Rizopoulos: https://github.com/drizopoulos/jm_and_lm
# Data set is split into 10 groups which are saved to be read in to Python codes for the 
# delayed kernel models.
# The 10 groups are used to perform 10-fold cross validation
# In turn, each group is taken as the test data while the other 9 comprise the training data
# A joint model is fitted to the training data set.
# At each landmark (base) time a landmark model is fitted to the training data.
# First we perform the fixed base time analysis with base time t = 3 years:
# The prediction time u is varied in steps of 0.2 years from 3 to 10 years
# Prediction error for each combination of t and u is stored at each iteration.
# Then we perform the fixed prediction window analysis for three windows:
# w1=1 year, w2=2 years, w3=3 years
# At w1 we use base times t=0->9 years in steps of 0.2 years
# At w2 we use base times t=0->8 years in steps of 0.2 years
# At w3 we use base times t=0->7 years in steps of 0.2 years
# Again, prediction error for each scenario is stored for each iteration.
# An average of PE over the 20 iterations is performed subsequently

library("JMbayes") #for prederrJM for LM models
library("splines")
library("xtable")
library("lattice")
library("JMbayes2") #for JM models


#########################
# Read in (and save) data and source codes
#########################
# internal JMbayes function to create landmark data sets
dataLM <- JMbayes:::dataLM 
# version of dataLM that creates a data set keeping event times >= landmark time for use
# in PE.AD.coxph only (i.e. to create the data set over which we sum in the PE eq):
source("dataLM.AD.R")
# edited versions of prederrJM (from JMbayes) so that it exactly matches the PE eq
# for coxph objects:
source("PE.AD.coxph.R")
#Edited version of tvBrier (from JMbayes2) so that it exactly matches the PE eq
# for JMBayes2 objects:
source("PE.AD.JM2.R")

#load original data 
#Liver data with all observations of covariates
data(prothro, package="JMbayes2")
#Liver data with only the baseline observations of covariates
data(prothros, package="JMbayes2")

#for joint models (baseline time indicator)
prothro$t0 <- as.numeric(prothro$time == 0)

#encode drug (pred = 1, placebo=0) for saving data 
prothro$treat_code <- as.numeric(prothro$treat == "prednisone")
prothros$treat_code <- as.numeric(prothros$treat == "prednisone")

# the original Liver data ids do not go from 1-488
# to split the data we need this to be the case, therefore:
# label prothro & prothros with id 1->488
prothro$id <- match(prothro$id, unique(prothro$id))
prothros$id <- match(prothros$id, unique(prothros$id))

#base folder in which to save test and training data in each loop
fn.base <- "~...LIVER_DATA\\G"

n <- 488 #number of subjects in orig data
V <- 10 #number of ways to split data

# base time for fixed base time analysis
timeLM <- 3.0

# prediction windows for fixed window analysis
w1.proth <- 1.0
w2.proth <- 2.0
w3.proth <- 3.0

set.seed(123)

#Functions to create data frames for each group---------------------------------
groupData <- function(g){
  G_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE)
  G_data <- prothro[prothro$id %in% G_id, ]
  G_data
}

groupData.id <- function(g){
  G_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE)
  G_data.id <- prothros[prothros$id %in% G_id, ]
  G_data.id
}

#Functions to create data frames for the training data:-------------------------
trainData <- function(g){
  #g is group for test data
  test_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE) 
  train_data <- prothro[!prothro$id %in% test_id, ]
  train_data
}

trainData.id <- function(g){
  #g is group for test data
  test_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE) 
  train_data.id <- prothros[!prothros$id %in% test_id, ]
  train_data.id
}

#function to save data for each group in files----------------------------------
write_file <- function(df_list, df_list.id, g){
  df <- df_list[[g]]
  df.id <- df_list.id[[g]]
  
  #number of observations per individual
  n <- table(df$id)  
  #fixed variables - use df.id
  events <-c(df.id[['Time']])
  censor <-c(df.id[['death']])
  Zf.drug <-c(df.id[['treat_code']])
  #longitudinal variables - use train data
  t.obs <- c(df[['time']])
  Zl.proth <-c(df[['pro']])
  
  #filenames
  fn.n <- paste(fn.base, g, "\\n_obs.txt", sep='')
  fn.events <- paste(fn.base, g, "\\Events.txt", sep='')
  fn.censor <- paste(fn.base, g, "\\Censor.txt", sep='')
  fn.Zf.drug <- paste(fn.base, g, "\\Zfix_drug.txt", sep='')
  fn.t.obs <- paste(fn.base, g, "\\t_obs.txt", sep='')
  fn.Zl.proth <- paste(fn.base, g, "\\Zlong_proth.txt", sep='')
  
  write(formatC(n, digits=15, format="fg", flag = "-"), fn.n, ncolumns=1)
  write(formatC(events, digits=15, format="fg", flag = "-"), fn.events, ncolumns=1)
  write(formatC(censor, digits=15, format="fg", flag = "-"), fn.censor, ncolumns=1)
  write(formatC(Zf.drug, digits=15, format="fg", flag = "-"), fn.Zf.drug, ncolumns=1)
  write(formatC(t.obs, digits=15, format="fg", flag = "-"), fn.t.obs, ncolumns=1)
  write(formatC(Zl.proth, digits=15, format="fg", flag = "-"), fn.Zl.proth, ncolumns=1)
  
}


#split data into 10 groups------------------------------------------------------
splits <- split(seq_len(n), sample(rep(seq_len(V), length.out = n)))

list_df <- list(groupData(1), groupData(2), groupData(3), groupData(4), groupData(5),
                groupData(6), groupData(7), groupData(8), groupData(9), groupData(10))

list_df.id <- list(groupData.id(1), groupData.id(2), groupData.id(3), groupData.id(4), groupData.id(5),
                groupData.id(6), groupData.id(7), groupData.id(8), groupData.id(9), groupData.id(10))


#write group data to files 
for(i in 1:10){
  write_file(list_df, list_df.id, i)
}

############################
# Perform cross validation analysis
############################

#Loop over the 10 groups
for(g in 1:V){
  print(paste("test group = ", g))
  
  #create test and training data
  #test data is list_df(i)
  test_data <- list_df[[g]]
  test_data.id <- list_df.id[[g]]
  #training is everything else
  train_data <- trainData(g)
  train_data.id <- trainData.id(g)
  
  ################
  # Joint Model
  ################
  
  print("JM model begin")
  print(Sys.time())
  
  #fit Joint model using train_data
  
  #longitudinal model
  long.train <- lme(pro ~ treat * (ns(time, 3) + t0),
                    random = list(id = pdDiag(form = ~ ns(time, 3))),
                    data = train_data)
  
  print("longitudinal done")
  #survival model
  Surv.train <- coxph(Surv(Time, death) ~ treat, data = train_data.id,
                      x = TRUE)
  print("survival done")
  #joint model
  Joint.train <- jm(Surv.train, long.train, time_var = "time") 
  #default chain=3, burn in = 500, iter = 3500 

  print("joint done")
  print(Sys.time())
  
  ###########################
  # Fix t
  ###########################
  
  print("begin fix t")
  print(Sys.time())
  ################
  # Landmarking
  ################
  
  # fit landmark model to training data at the fixed base time (t=3)
  
  print("LM model begin")
  
  #Create a landmark data set at the landmark time (timeLM) using training data
  train.LM <- dataLM(train_data, timeLM, respVar = "pro", timeVar = "time", 
                     evTimeVar = "Time", summary = "value")
  #Fit a standard Cox model to the landmark data set
  Cox.train.LM <- coxph(Surv(Time, death) ~ treat + pro, data = train.LM)
  
  print("done")
  
  
  ################
  # Pred Error
  ################
  
  # for fixed t=3 years, iterate over prediction time u = 3->10 years in steps of 0.2
  # calculate PE for LM and JM for each combination of (t,u) using newdata = test data
  
  print("Fix t loop begin")
  print(Sys.time())
  
  vec.pe.LM <-c()
  vec.pe.JM <-c()
  for(i in 0:35){
    #print(i)
    #print(Sys.time())
    timeHZ <- timeLM + (i*0.2) #prediction or 'horizon' time u
    
    PE.LM<-PE.AD.coxph(Cox.train.LM, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ,
                  idVar = "id", timeVar = "time", respVar = "pro", evTimeVar = "Time",
                  lossFun = "square", summary = "value")
    PE.JM <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ)
    
    vec.pe.LM<-c(vec.pe.LM,PE.LM["prederr"])
    vec.pe.JM<-c(vec.pe.JM,PE.JM["Brier"])
  }
  #save vector of prediction errors to files
  fn.fixt.LM <- paste(fn.base, g, "\\LM_fixt.txt", sep='')
  fn.fixt.JM <- paste(fn.base, g, "\\JM_fixt.txt", sep='')
  write(formatC(unlist(vec.pe.LM), digits=15, format="fg", flag = "-"), fn.fixt.LM, ncolumns=1)
  write(formatC(unlist(vec.pe.JM), digits=15, format="fg", flag = "-"), fn.fixt.JM, ncolumns=1)
 
  print("fix t loop done")
  print(Sys.time())
  
  ###########################
  # Pred Window
  ###########################
  
  ################
  # Window 1
  ################
  
  # For fixed window w1=1 year work out PE for base time t=0-9 years (steps of 0.2 years)
  # For each base time t we must create a new LM data set and fit a corresponding
  # LM model (using the training data)
  
  print("window 1 begin")
  
  vecPE.proth.LM1 <-c()
  vecPE.proth.JM1 <-c()
  for(i in 0:45){
    #print(i)
    t.proth <- i*0.2
    
    #create LM data set at LM time = t.proth using training data
    proth.LM1 <- dataLM(train_data, t.proth, respVar = "pro", timeVar = "time", 
                       evTimeVar = "Time", summary = "value")
    #Fit a standard Cox model to the landmark data set 
    Cox.proth.LM1 <- coxph(Surv(Time, death) ~ treat + pro, data = proth.LM1)
    
    #PE using newdata = test_data
    #LM
    PE.LM1 <- PE.AD.coxph(Cox.proth.LM1, newdata = test_data, Tstart = t.proth, Thoriz = t.proth+w1.proth, 
                       idVar = "id", timeVar = "time", respVar = "pro", evTimeVar = "Time",
                       lossFun = "square", summary = "value")
    #JM
    PE.JM1 <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = t.proth, Thoriz = t.proth+w1.proth)
    
    vecPE.proth.LM1<-c(vecPE.proth.LM1,PE.LM1["prederr"])
    vecPE.proth.JM1<-c(vecPE.proth.JM1,PE.JM1["Brier"])
  }
  #save vector of prediction errors to files
  fn.pw1.LM <- paste(fn.base, g, "\\LM_w1.txt", sep='')
  fn.pw1.JM <- paste(fn.base, g, "\\JM_w1.txt", sep='')
  write(formatC(unlist(vecPE.proth.LM1), digits=15, format="fg", flag = "-"), fn.pw1.LM, ncolumns=1)
  write(formatC(unlist(vecPE.proth.JM1), digits=15, format="fg", flag = "-"), fn.pw1.JM, ncolumns=1)
  print("done")
  print(Sys.time())
  
  ################
  # Window 2
  ################
  
  # For fixed window w2=2 years work out PE for base time t=0-8 years (steps of 0.2 years)
  # For each base time t we must create a new LM data set and fit a corresponding
  # LM model (using the training data)
  
  print("window 2 begin")
  print(Sys.time())
  vecPE.proth.LM2 <-c()
  vecPE.proth.JM2 <-c()
  for(i in 0:40){
    #print(i)
    t.proth <- i*0.2
    
    #create LM data set at LM time = t.proth using train data
    proth.LM2 <- dataLM(train_data, t.proth, respVar = "pro", timeVar = "time", 
                        evTimeVar = "Time", summary = "value")
    #Fit a standard Cox model to the landmark data set 
    Cox.proth.LM2 <- coxph(Surv(Time, death) ~ treat + pro, data = proth.LM2)
    
    #calculate PE using newdata = test data
    #LM
    PE.LM2 <- PE.AD.coxph(Cox.proth.LM2, newdata = test_data, Tstart = t.proth, Thoriz = t.proth+w2.proth, 
                        idVar = "id", timeVar = "time", respVar = "pro", evTimeVar = "Time",
                        lossFun = "square", summary = "value")
    #JM
    PE.JM2 <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = t.proth, Thoriz = t.proth+w2.proth)
    
    vecPE.proth.LM2<-c(vecPE.proth.LM2,PE.LM2["prederr"])
    vecPE.proth.JM2<-c(vecPE.proth.JM2,PE.JM2["Brier"])
  }
  #save vector of prediction errors to files
  fn.pw2.LM <- paste(fn.base, g, "\\LM_w2.txt", sep='')
  fn.pw2.JM <- paste(fn.base, g, "\\JM_w2.txt", sep='')
  write(formatC(unlist(vecPE.proth.LM2), digits=15, format="fg", flag = "-"), fn.pw2.LM, ncolumns=1)
  write(formatC(unlist(vecPE.proth.JM2), digits=15, format="fg", flag = "-"), fn.pw2.JM, ncolumns=1)
  print("done")
  print(Sys.time())
  
  ################
  # Window 3
  ################
  
  # For fixed window w3=3 years work out PE for base time t=0-7 years (steps of 0.2 years)
  # For each base time t we must create a new LM data set and fit a corresponding
  # LM model (using the training data)
  
  print("window 3 begin")
  print(Sys.time())
  vecPE.proth.LM3 <-c()
  vecPE.proth.JM3 <-c()
  for(i in 0:35){
    #print(i)
    t.proth <- i*0.2
    
    #create LM data set  at LM time = t.proth using train data
    proth.LM3 <- dataLM(train_data, t.proth, respVar = "pro", timeVar = "time", 
                        evTimeVar = "Time", summary = "value")
    #Fit a standard Cox model to the landmark data set 
    Cox.proth.LM3 <- coxph(Surv(Time, death) ~ treat + pro, data = proth.LM3)
    
    #calculate PE using newdata = test data
    #LM
    PE.LM3 <- PE.AD.coxph(Cox.proth.LM3, newdata = test_data, Tstart = t.proth, Thoriz = t.proth+w3.proth, 
                        idVar = "id", timeVar = "time", respVar = "pro", evTimeVar = "Time",
                        lossFun = "square", summary = "value")
    #JM
    PE.JM3 <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = t.proth, Thoriz = t.proth+w3.proth)
    
    vecPE.proth.LM3<-c(vecPE.proth.LM3,PE.LM3["prederr"])
    vecPE.proth.JM3<-c(vecPE.proth.JM3,PE.JM3["Brier"])
  }
  #save vector of prediction errors to files
  fn.pw3.LM <- paste(fn.base, g, "\\LM_w3.txt", sep='')
  fn.pw3.JM <- paste(fn.base, g, "\\JM_w3.txt", sep='')
  write(formatC(unlist(vecPE.proth.LM3), digits=15, format="fg", flag = "-"), fn.pw3.LM, ncolumns=1)
  write(formatC(unlist(vecPE.proth.JM3), digits=15, format="fg", flag = "-"), fn.pw3.JM, ncolumns=1)
  
  print("done")
  print(Sys.time())
}  

