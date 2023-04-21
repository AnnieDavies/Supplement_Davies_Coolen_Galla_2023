# Code to perform data analysis on AIDS data set in Davies, Coolen and Galla (2023)
# for joint models and landmarking models.
# Written by A. Davies (2021),
# aided by code by D. Rizopoulos: https://github.com/drizopoulos/jm_and_lm
# Data set is split into 10 groups which are saved to be read in to Python codes for the 
# delayed kernel models.
# The 10 groups are used to perform 10-fold cross validation
# In turn, each group is taken as the test data while the other 9 comprise the training data
# A joint model is fitted to the training data set.
# At each landmark (base) time a landmark model is fitted to the training data.
# First we perform the fixed base time analysis with base time t = 6 months:
# The prediction time u is varied in steps of 0.2 months from 6 to 18 months
# Prediction error calculated using the fitted model and test data 
# for each combination of t and u is stored at each iteration.
# Then we perform the fixed prediction window analysis for three windows:
# w1=6 months, w2=9 months, w3=12 months
# At w1 we use base times t=0,2,6,12 months
# At w2 & w3 we use base times t=0,2,6 months
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
#AIDS data with all observations of covariates
data(aids, package="JMbayes2")
#AIDS data with only the baseline observations of covariates
data(aids.id, package="JMbayes2")

#encode fixed covariates for saving data 
#drug (ddI=1 ddC=0)
aids$drug_code <- as.numeric(aids$drug == "ddI")
aids.id$drug_code <- as.numeric(aids.id$drug == "ddI")

#gender (male=1 female=0)
aids$gender_code <- as.numeric(aids$gender == "male")
aids.id$gender_code <- as.numeric(aids.id$gender == "male")

#PrevOI (AIDS=1 noAIDS=0)
aids$prevOI_code <- as.numeric(aids$prevOI == "AIDS")
aids.id$prevOI_code <- as.numeric(aids.id$prevOI == "AIDS")

#Stratum (AZT failure=1 intolerance=0)
aids$AZT_code <- as.numeric(aids$AZT == "failure")
aids.id$AZT_code <- as.numeric(aids.id$AZT == "failure")


#base folder in which to save test and training data in each loop
fn.base <- "~...AIDS_DATA\\G"

n <- 467 #number of subjects in orig data
V <- 10 #number of ways to split data

# base time for fixed base time analysis
timeLM.aids <- 6.0

# prediction windows for fixed window analysis
w1.aids <- 6.0
w2.aids <- 9.0
w3.aids <- 12.0

set.seed(312)


#Functions to create data frames for each group---------------------------------
groupData <- function(g){
  G_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE)
  G_data <- aids[aids$patient %in% G_id, ]
  G_data
}

groupData.id <- function(g){
  G_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE)
  G_data.id <- aids.id[aids.id$patient %in% G_id, ]
  G_data.id
}

#Functions to create data frames for the training data:-------------------------
trainData <- function(g){
  #g is group for test data
  test_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE) 
  train_data <- aids[!aids$patient %in% test_id, ]
  train_data
}

trainData.id <- function(g){
  #g is group for test data
  test_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE) 
  train_data.id <- aids.id[!aids.id$patient %in% test_id, ]
  train_data.id
}

#function to save data for each group in files----------------------------------
write_file <- function(df_list, df_list.id, g){
  df <- df_list[[g]]
  df.id <- df_list.id[[g]]
  
  #number of observations per individual
  n <- table(df$patient)  
  #fixed variables - use df.id
  events <-c(df.id[['Time']])
  censor <-c(df.id[['death']])
  Zf.drug <-c(df.id[['drug_code']])
  Zf.gender <-c(df.id[['gender_code']])
  Zf.prevOI <-c(df.id[['prevOI_code']])
  Zf.AZT <-c(df.id[['AZT_code']])
  #longitudinal variables
  t.obs <- c(df[['obstime']])
  Zl.CD4 <-c(df[['CD4']])
  
  #filenames
  fn.n <- paste(fn.base, g, "\\n_obs.txt", sep='')
  fn.events <- paste(fn.base, g, "\\Events.txt", sep='')
  fn.censor <- paste(fn.base, g, "\\Censor.txt", sep='')
  fn.Zf.drug <- paste(fn.base, g, "\\Zfix_drug.txt", sep='')
  fn.Zf.gender <- paste(fn.base, g, "\\Zfix_gender.txt", sep='')
  fn.Zf.prevOI <- paste(fn.base, g, "\\Zfix_prevOI.txt", sep='')
  fn.Zf.AZT <- paste(fn.base, g, "\\Zfix_AZT.txt", sep='')
  fn.t.obs <- paste(fn.base, g, "\\t_obs.txt", sep='')
  fn.Zl.CD4 <- paste(fn.base, g, "\\Zlong_CD4.txt", sep='')
  
  write(formatC(n, digits=15, format="fg", flag = "-"), fn.n, ncolumns=1)
  write(formatC(events, digits=15, format="fg", flag = "-"), fn.events, ncolumns=1)
  write(formatC(censor, digits=15, format="fg", flag = "-"), fn.censor, ncolumns=1)
  write(formatC(Zf.drug, digits=15, format="fg", flag = "-"), fn.Zf.drug, ncolumns=1)
  write(formatC(Zf.gender, digits=15, format="fg", flag = "-"), fn.Zf.gender, ncolumns=1)
  write(formatC(Zf.prevOI, digits=15, format="fg", flag = "-"), fn.Zf.prevOI, ncolumns=1)
  write(formatC(Zf.AZT, digits=15, format="fg", flag = "-"), fn.Zf.AZT, ncolumns=1)
  write(formatC(t.obs, digits=15, format="fg", flag = "-"), fn.t.obs, ncolumns=1)
  write(formatC(Zl.CD4, digits=15, format="fg", flag = "-"), fn.Zl.CD4, ncolumns=1)
  
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
  long.train <- lme(CD4 ~ obstime + obstime:drug,
                    random = ~ obstime | patient, data = train_data)
  print("longitudinal done")
  #survival model
  Surv.train <- coxph(Surv(Time, death) ~ drug + prevOI + AZT + gender, 
                      data = train_data.id, x = TRUE)
  print("survival done")
  #joint model
  Joint.train <- jm(Surv.train, long.train, time_var = "obstime") 
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
  
  # fit landmark model to training data at the fixed base time (t=6)
  
  print("LM model begin")
  
  #Create a landmark data set at the landmark time (timeLM.aids) using training data
  train.LM <- dataLM(train_data, timeLM.aids, respVar = "CD4", timeVar = "obstime", 
                     evTimeVar = "Time", idVar = "patient", summary = "value")
  
  #Fit a standard Cox model to the landmark data set 
  Cox.train.LM <- coxph(Surv(Time, death) ~ drug + prevOI + AZT + gender + CD4, data = train.LM)
  
  print("done")
  print(Sys.time())
  
  ################
  # Pred Error
  ################
  
  # for fixed t=6 months, iterate over prediction time u = 6->18 months in steps of 0.2
  # calculate PE for LM and JM for each combination of (t,u) using newdata = test data
  
  print("Fix t loop begin")
  print(Sys.time())
  
  vec.pe.LM <-c()
  vec.pe.JM <-c()
  for(i in 0:60){
    timeHZ <- timeLM.aids+(i*0.2) #prediction or 'horizon' time u
    
    PE.LM<-PE.AD.coxph(Cox.train.LM, newdata = test_data, Tstart = timeLM.aids, Thoriz = timeHZ,
                       idVar = "patient", timeVar = "obstime", respVar = "CD4", evTimeVar = "Time",
                       lossFun = "square", summary = "value")
    PE.JM <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = timeLM.aids, Thoriz = timeHZ)
    
    vec.pe.LM <-c(vec.pe.LM, PE.LM["prederr"])
    vec.pe.JM <-c(vec.pe.JM, PE.JM["Brier"])
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
  
  # For fixed window w1=6 months work out PE for base time t=0,2,6,12 months
  # For each base time t we must create a new LM data set and fit a corresponding
  # LM model (using the training data)
  
  print("window 1 begin")
  print(Sys.time())  
  vec.pe.LM1 <-c()
  vec.pe.JM1 <-c()
  
  #t = 0, 2, 6, 12
  for(i in 0:3){
    if(i<2){
      t.aids <- i*2
    } else if(i==2){
      t.aids <- 6
    }
    else if(i==3){
      t.aids <- 12
    }
    
    #create LM data set at LM time = t.aids using training data
    aids.LM1 <- dataLM(train_data, t.aids, respVar = "CD4", timeVar = "obstime", 
                       evTimeVar = "Time", idVar = "patient", summary = "value")
    #Fit a standard Cox model to the landmark data set
    Cox.aids.LM1 <- coxph(Surv(Time, death) ~ drug + prevOI + AZT + gender + CD4, 
                          data = aids.LM1)
    
    #PE using test_data
    #LM
    PE.LM1<-PE.AD.coxph(Cox.aids.LM1, newdata = test_data, Tstart = t.aids, Thoriz = t.aids+w1.aids,
                        idVar = "patient", timeVar = "obstime", respVar = "CD4", evTimeVar = "Time",
                        lossFun = "square", summary = "value")
    #JM
    PE.JM1 <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = t.aids, Thoriz = t.aids+w1.aids)
    
    vec.pe.LM1 <-c(vec.pe.LM1, PE.LM1["prederr"])
    vec.pe.JM1 <-c(vec.pe.JM1, PE.JM1["Brier"])
  }
  #save vector of prediction errors to files
  fn.pw1.LM <- paste(fn.base, g, "\\LM_w1.txt", sep='')
  fn.pw1.JM <- paste(fn.base, g, "\\JM_w1.txt", sep='')
  write(formatC(unlist(vec.pe.LM1), digits=15, format="fg", flag = "-"), fn.pw1.LM, ncolumns=1)
  write(formatC(unlist(vec.pe.JM1), digits=15, format="fg", flag = "-"), fn.pw1.JM, ncolumns=1)
  print("done")
  print(Sys.time())
  
  ################
  # Window 2
  ################
  
  # For fixed window w2=9 months work out PE for base time t=0,2,6 months
  # For each base time t we must create a new LM data set and fit a corresponding
  # LM model (using the training data)
  
  print("window 2 begin")
  print(Sys.time())  
  vec.pe.LM2 <-c()
  vec.pe.JM2 <-c()
  #t = 0, 2, 6
  for(i in 0:2){
    if(i<2){
      t.aids <- i*2
    } else if(i==2){
      t.aids <- 6
    }
    #print(t.aids)
    
    #create LM data set at LM time = t.aids using training data
    aids.LM2 <- dataLM(train_data, t.aids, respVar = "CD4", timeVar = "obstime", 
                       evTimeVar = "Time", idVar = "patient", summary = "value")
    #Fit a standard Cox model to the landmark data set
    Cox.aids.LM2 <- coxph(Surv(Time, death) ~ drug + prevOI + AZT + gender + CD4, 
                          data = aids.LM2)
    
    #PE using test_data
    #LM
    PE.LM2<-PE.AD.coxph(Cox.aids.LM2, newdata = test_data, Tstart = t.aids, Thoriz = t.aids+w2.aids,
                        idVar = "patient", timeVar = "obstime", respVar = "CD4", evTimeVar = "Time",
                        lossFun = "square", summary = "value")
    #JM
    PE.JM2 <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = t.aids, Thoriz = t.aids+w2.aids)
    
    vec.pe.LM2 <-c(vec.pe.LM2, PE.LM2["prederr"])
    vec.pe.JM2 <-c(vec.pe.JM2, PE.JM2["Brier"])
  }
  #save vector of prediction errors to files
  fn.pw2.LM <- paste(fn.base, g, "\\LM_w2.txt", sep='')
  fn.pw2.JM <- paste(fn.base, g, "\\JM_w2.txt", sep='')
  write(formatC(unlist(vec.pe.LM2), digits=15, format="fg", flag = "-"), fn.pw2.LM, ncolumns=1)
  write(formatC(unlist(vec.pe.JM2), digits=15, format="fg", flag = "-"), fn.pw2.JM, ncolumns=1)
  print("done")
  print(Sys.time())
  
  ################
  # Window 3
  ################
  
  # For fixed window w3=12 months work out PE for base time t=0,2,6 months
  # For each base time t we must create a new LM data set and fit a corresponding
  # LM model (using the training data)
  
  print("window 3 begin")
  print(Sys.time())  
  vec.pe.LM3 <-c()
  vec.pe.JM3 <-c()
  #t = 0, 2, 6
  for(i in 0:2){
    if(i<2){
      t.aids <- i*2
    } else if(i==2){
      t.aids <- 6
    }
    #print(t.aids)
    
    #create LM data set at LM time = t.aids using training data
    aids.LM3 <- dataLM(train_data, t.aids, respVar = "CD4", timeVar = "obstime", 
                       evTimeVar = "Time", idVar = "patient", summary = "value")
    #Fit a standard Cox model to the landmark data set
    Cox.aids.LM3 <- coxph(Surv(Time, death) ~ drug + prevOI + AZT + gender + CD4, 
                          data = aids.LM3)
    
    #PE using test_data
    #LM
    PE.LM3<-PE.AD.coxph(Cox.aids.LM3, newdata = test_data, Tstart = t.aids, Thoriz = t.aids+w3.aids,
                        idVar = "patient", timeVar = "obstime", respVar = "CD4", evTimeVar = "Time",
                        lossFun = "square", summary = "value")
    #JM
    PE.JM3 <- PE.AD.JM2(Joint.train, newdata = test_data, Tstart = t.aids, Thoriz = t.aids+w3.aids)
    
    vec.pe.LM3 <-c(vec.pe.LM3, PE.LM3["prederr"])
    vec.pe.JM3 <-c(vec.pe.JM3, PE.JM3["Brier"])
  }
  #save vector of prediction errors to files
  fn.pw3.LM <- paste(fn.base, g, "\\LM_w3.txt", sep='')
  fn.pw3.JM <- paste(fn.base, g, "\\JM_w3.txt", sep='')
  write(formatC(unlist(vec.pe.LM3), digits=15, format="fg", flag = "-"), fn.pw3.LM, ncolumns=1)
  write(formatC(unlist(vec.pe.JM3), digits=15, format="fg", flag = "-"), fn.pw3.JM, ncolumns=1)
  print("done")
  print(Sys.time())
}  


