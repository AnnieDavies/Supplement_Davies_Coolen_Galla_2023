# Code to perform data analysis on PBC data set in Davies, Coolen and Galla (2023).
# Written by A. Davies (2023),
# aided by code by D. Rizopoulos: https://github.com/drizopoulos/jm_and_lm
# Data set is split into 10 groups which are saved to be read in to Python codes for the 
# delayed kernel models.
# The 10 groups are used to perform 10-fold cross validation
# In turn, each group is taken as the test data while the other 9 comprise the training data
# Analysis is performed for data that treats the transplant event as a 
# censoring event (labelled 'cen') and for data that treats the two events (death
# and transplant) as a composite event (labelled 'comp').
# For each version (cen and comp) we fit two joint models (one with a linear
# longitudinal model and one using cubic splines).
# Joint models are fitted to the training data set.
# At each landmark (base) time landmark models are fitted to the training data.
# First we perform the fixed base time analysis with base time t = 3 years:
# The prediction time u is varied in steps of 0.2 years from 3 to 8 years
# Prediction error for each combination of t and u is stored at each iteration.
# Then we perform the fixed prediction window analysis for three windows:
# w1=1 year, w2=2 years, w3=3 years
# At w1 we use base times t=0->9 years in steps of 0.2 years
# At w2 we use base times t=0->8 years in steps of 0.2 years
# At w3 we use base times t=0->7 years in steps of 0.2 years
# Again, prediction error for each scenario is stored for each iteration.
# An average of PE over the 10 iterations is performed subsequently

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
#PBC data with all observations of covariates
data(pbc2, package="JMbayes2")
#PBC data with only the baseline observations of covariates
data(pbc2.id, package="JMbayes2")

#In pbc2 and pbc2.id, status2 indicates the event "death" treating the event
#'transplant' as a censoring event i.e.
pbc2.id$status2 <- as.numeric(pbc2.id$status == "dead")
pbc2$status2 <- as.numeric(pbc2$status == "dead")

#create the indicator status3 for the composite event (death or transplant)
pbc2.id$status3 <- as.numeric(pbc2.id$status != "alive")
pbc2$status3 <- as.numeric(pbc2$status != "alive")

#base folder in which to save test and training data in each loop
fn.base <- "~...PBC_DATA\\G"


n <- 312 #number of subjects in orig data
V <- 10 #number of ways to split data

# base time for fixed base time analysis
timeLM <- 3.0

# prediction windows for fixed window analysis
w1.pbc <- 1.0
w2.pbc <- 2.0
w3.pbc <- 3.0

set.seed(231)

#Functions to create data frames for each group--------------------------------- 
groupData <- function(g){
  G_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE)
  G_data <- pbc2[pbc2$id %in% G_id, ]
  G_data
}

groupData.id <- function(g){
  G_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE)
  G_data.id <- pbc2.id[pbc2.id$id %in% G_id, ]
  G_data.id
}

#Functions to create data frames for the training data:-------------------------
trainData <- function(g){
  #g is group for test data
  test_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE) 
  train_data <- pbc2[!pbc2$id %in% test_id, ]
  train_data
}

trainData.id <- function(g){
  #g is group for test data
  test_id <- unlist(splits[g], recursive = TRUE, use.names = TRUE) 
  train_data.id <- pbc2.id[!pbc2.id$id %in% test_id, ]
  train_data.id
}

#function to save data for each group in files----------------------------------
write_file <- function(df_list, df_list.id, g){
  df <- df_list[[g]]
  df.id <- df_list.id[[g]]
  
  #number of observations per individual
  n <- table(df$id)  
  #fixed variables - use df.id
  events <-c(df.id[['years']])
  censor <-c(df.id[['status2']])
  censor.comp <-c(df.id[['status3']])
  Zf.age <-c(df.id[['age']])
  #longitudinal variables - use train data
  t.obs <- c(df[['year']])
  Zl.bili <-c(df[['serBilir']])
  Zl.alb <-c(df[['albumin']])
  Zl.proth <-c(df[['prothrombin']])
  
  #filenames
  fn.n <- paste(fn.base, g, "\\n_obs.txt", sep='')
  fn.events <- paste(fn.base, g, "\\Events.txt", sep='')
  fn.censor <- paste(fn.base, g, "\\Censor.txt", sep='')
  fn.censor.comp <- paste(fn.base, g, "\\Censor_comp.txt", sep='')
  fn.Zf.age <- paste(fn.base, g, "\\Zfix_age.txt", sep='')
  fn.t.obs <- paste(fn.base, g, "\\t_obs.txt", sep='')
  fn.Zl.bili <- paste(fn.base, g, "\\Zlong_bili.txt", sep='')
  fn.Zl.alb <- paste(fn.base, g, "\\Zlong_alb.txt", sep='')
  fn.Zl.proth <- paste(fn.base, g, "\\Zlong_proth.txt", sep='')
  
  write(formatC(n, digits=15, format="fg", flag = "-"), fn.n, ncolumns=1)
  write(formatC(events, digits=15, format="fg", flag = "-"), fn.events, ncolumns=1)
  write(formatC(censor, digits=15, format="fg", flag = "-"), fn.censor, ncolumns=1)
  write(formatC(censor.comp, digits=15, format="fg", flag = "-"), fn.censor.comp, ncolumns=1)
  write(formatC(Zf.age, digits=15, format="fg", flag = "-"), fn.Zf.age, ncolumns=1)
  write(formatC(t.obs, digits=15, format="fg", flag = "-"), fn.t.obs, ncolumns=1)
  write(formatC(Zl.bili, digits=15, format="fg", flag = "-"), fn.Zl.bili, ncolumns=1)
  write(formatC(Zl.alb, digits=15, format="fg", flag = "-"), fn.Zl.alb, ncolumns=1)
  write(formatC(Zl.proth, digits=15, format="fg", flag = "-"), fn.Zl.proth, ncolumns=1)
  
}


#split data into 10 groups------------------------------------------------------
splits <- split(seq_len(n), sample(rep(seq_len(V), length.out = n)))

list_df <- list(groupData(1), groupData(2), groupData(3), groupData(4), groupData(5),
                groupData(6), groupData(7), groupData(8), groupData(9), groupData(10))

list_df.id <- list(groupData.id(1), groupData.id(2), groupData.id(3), groupData.id(4), groupData.id(5),
                   groupData.id(6), groupData.id(7), groupData.id(8), groupData.id(9), groupData.id(10))

View(list_df[[1]])

#write group data to files
for(i in 1:10){
  write_file(list_df, list_df.id, i)
}

############################
# Perform cross validation analysis
############################

ctrl <- lmeControl(opt='optim', maxIter=500, msMaxIter = 500, msMaxEval = 500) #for LME

#Loop over the 10 groups 
for(g in 1:V){
  print(paste("test group = ", g))
  
  #create test and training data------------------------------------------------
  #test data is list_df(i)
  test_data <- list_df[[g]]
  test_data.id <- list_df.id[[g]]
  #training is everything else
  train_data <- trainData(g)
  train_data.id <- trainData.id(g)
  
  ################
  # Joint Model
  ################
  
  #fit Joint models using train_data
  
  # Longitudinal models --------------------------------------------------------
  # spline longitudinal model
  print("spline longitudinal model begin")
  long.spline1 <- lme(log(serBilir) ~ ns(year,2,B=c(0,14.4)), random = ~ ns(year,2,B=c(0,14.4)) | id,
                      data = train_data, control = ctrl)
  long.spline2 <- lme(log(albumin) ~ ns(year,2,B=c(0,14.4)), random = ~ ns(year,2,B=c(0,14.4)) | id,
                      data = train_data, control = ctrl)
  long.spline3 <- lme(log(prothrombin) ~ ns(year,2,B=c(0,14.4)), random = ~ ns(year,2,B=c(0,14.4)) | id,
                      data = train_data, control = ctrl)
  print("done")
  print(Sys.time())
  
  # linear longitudinal model
  print("linear longitudinal model begin")
  long.linear1 <- lme(log(serBilir) ~ year, random = ~ year | id, data = train_data, control = ctrl)
  long.linear2 <- lme(log(albumin) ~ year, random = ~ year | id, data = train_data, control = ctrl)
  long.linear3 <- lme(log(prothrombin) ~ year, random = ~ year | id, data = train_data, control = ctrl)
  print("done")
  print(Sys.time())
  
  # Survival models ------------------------------------------------------------
  print("survival models")
  
  # censor event survival model
  surv.cen <- coxph(Surv(years, status2) ~ age, data = train_data.id, model = TRUE)
  
  # composite event survival model
  surv.comp <- coxph(Surv(years, status3) ~ age, data = train_data.id, model = TRUE)
  
  print("done")
  print(Sys.time())
  
  # Joint models ---------------------------------------------------------------
  # JM: Spline Censor event
  print("JM: Spline Censor event begin")
  JM.spline.cen <- jm(surv.cen, list(long.spline1, long.spline2, long.spline3), time_var = "year",
                  n_iter = 12000L, n_burnin = 2000L, n_thin = 5L)
  print("done")
  print(Sys.time())
  
  # JM: Linear Censor event
  print("JM: Linear Censor event begin")
  JM.linear.cen <- jm(surv.cen, list(long.linear1, long.linear2, long.linear3), time_var = "year",
                      n_iter = 12000L, n_burnin = 2000L, n_thin = 5L)
  print("done")
  print(Sys.time())
  
  # JM: Spline Comp event
  print("JM: Spline Comp event begin")
  JM.spline.comp <- jm(surv.comp, list(long.spline1, long.spline2, long.spline3), time_var = "year",
                      n_iter = 12000L, n_burnin = 2000L, n_thin = 5L)
  print("done")
  print(Sys.time())
  
  # JM: Linear Comp event
  print("JM: Linear Comp event begin")
  JM.linear.comp <- jm(surv.comp, list(long.linear1, long.linear2, long.linear3), time_var = "year",
                      n_iter = 12000L, n_burnin = 2000L, n_thin = 5L)
  print("done")
  print(Sys.time())
  
  ###########################
  # Fix t
  ###########################
  
  print("begin fix t")
  
  ################
  # Landmarking 
  ################
  
  # fit landmark models to training data at the fixed base time (t=3)
  
  # Create a landmark data set at the landmark time (timeLM) using training data
  # We fit three longitudinal covariates (serBilir, albumin and prothrombin)
  # because we use summary = "value" we need only specify one longitudinal covariate
  # (arbitrarily) as 'respVar' (here we use serBilir)
  # NB: this would not hold for other arguments for summary
  pbc.LM <- dataLM(train_data, timeLM, respVar = "serBilir", timeVar = "year", 
                   idVar = "id", evTimeVar = "years", summary = "value")
  
  #we use the same LM data set for both the censor event and composite event models
  
  ################
  # Landmarking CENSOR EVENT
  ################
  
  print("LM censor model begin")
  
  #Fit a standard Cox model to the landmark data set (for the censor event)
  Cox.LM.cen <- coxph(Surv(years, status2) ~ age + log(serBilir) + log(albumin) + 
                        log(prothrombin), data = pbc.LM)
  
  print("done")
  
  ################
  # Landmarking COMP EVENT
  ################
  
  print("LM comp model begin")
  
  #Fit a standard Cox model to the landmark data set (for the composite event)
  Cox.LM.comp <- coxph(Surv(years, status3) ~ age + log(serBilir) + log(albumin) + 
                         log(prothrombin), data = pbc.LM)
  
  print("done")
  print(Sys.time())
  
  
  ################
  # Pred Error
  ################
  
  # for fixed t=3 years, iterate over prediction time u = 3->8 years in steps of 0.2
  # calculate PE for LMs and JMs for each combination of (t,u) using newdata = test data
  
  print("Fix t loop begin")
  
  vec.pe.LM.cen <-c()
  vec.pe.JM.spline.cen <-c()
  vec.pe.JM.linear.cen <-c()
  
  vec.pe.LM.comp <-c()
  vec.pe.JM.spline.comp <-c()
  vec.pe.JM.linear.comp <-c()
  
  for(i in 0:25){
    timeHZ <- timeLM + (i*0.2) #prediction or 'horizon' time u
    
    #censor event
    PE.LM.cen <- PE.AD.coxph(Cox.LM.cen, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ, 
                             idVar = "id", timeVar = "year", respVar = "serBilir",
                             evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.cen <- PE.AD.JM2(JM.spline.cen, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ)
    PE.JM.linear.cen <- PE.AD.JM2(JM.linear.cen, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ)

    #comp event
    PE.LM.comp <- PE.AD.coxph(Cox.LM.comp, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ, 
                              idVar = "id", timeVar = "year", respVar = "serBilir",
                              evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.comp <- PE.AD.JM2(JM.spline.comp, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ)
    PE.JM.linear.comp <- PE.AD.JM2(JM.linear.comp, newdata = test_data, Tstart = timeLM, Thoriz = timeHZ)
    
    #censor event
    vec.pe.LM.cen <-c(vec.pe.LM.cen, PE.LM.cen["prederr"])
    vec.pe.JM.spline.cen <-c(vec.pe.JM.spline.cen, PE.JM.spline.cen["Brier"])
    vec.pe.JM.linear.cen <-c(vec.pe.JM.linear.cen, PE.JM.linear.cen["Brier"])
    #comp event
    vec.pe.LM.comp <-c(vec.pe.LM.comp, PE.LM.comp["prederr"])
    vec.pe.JM.spline.comp <-c(vec.pe.JM.spline.comp, PE.JM.spline.comp["Brier"])
    vec.pe.JM.linear.comp <-c(vec.pe.JM.linear.comp, PE.JM.linear.comp["Brier"])
    
  }
  # save vector of prediction errors to files 
  # censor event
  fn.fixt.LM.cen <- paste(fn.base, g, "\\LM_cen_fixt.txt", sep='')
  fn.fixt.JM.spline.cen <- paste(fn.base, g, "\\JM_spline_cen_fixt.txt", sep='')
  fn.fixt.JM.linear.cen <- paste(fn.base, g, "\\JM_linear_cen_fixt.txt", sep='')
  write(formatC(unlist(vec.pe.LM.cen), digits=15, format="fg", flag = "-"), fn.fixt.LM.cen, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.cen), digits=15, format="fg", flag = "-"), fn.fixt.JM.spline.cen, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.cen), digits=15, format="fg", flag = "-"), fn.fixt.JM.linear.cen, ncolumns=1)
  
  # comp event
  fn.fixt.LM.comp <- paste(fn.base, g, "\\LM_comp_fixt.txt", sep='')
  fn.fixt.JM.spline.comp <- paste(fn.base, g, "\\JM_spline_comp_fixt.txt", sep='')
  fn.fixt.JM.linear.comp <- paste(fn.base, g, "\\JM_linear_comp_fixt.txt", sep='')
  write(formatC(unlist(vec.pe.LM.comp), digits=15, format="fg", flag = "-"), fn.fixt.LM.comp, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.comp), digits=15, format="fg", flag = "-"), fn.fixt.JM.spline.comp, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.comp), digits=15, format="fg", flag = "-"), fn.fixt.JM.linear.comp, ncolumns=1)
  
  
  print("fix t loop done")
  print(Sys.time())
  
  
  ###########################
  # Pred Window
  ###########################
  
  ################
  # Window 1
  ################
  
  # For fixed window w1=1 year work out PE for base time t=0-9 years (steps of 0.2 years)
  # For each base time t we must create a new LM data set and fit corresponding
  # LM models (using the training data)
  
  print("window 1 begin")
  
  vec.pe.LM.cen1 <-c()
  vec.pe.JM.spline.cen1 <-c()
  vec.pe.JM.linear.cen1 <-c()
  
  vec.pe.LM.comp1 <-c()
  vec.pe.JM.spline.comp1 <-c()
  vec.pe.JM.linear.comp1 <-c()
  
  for(i in 0:45){
    t.pbc <- i*0.2
    
    #create LM data set from training data at time t.pbc
    pbc.LM1 <- dataLM(train_data, t.pbc, respVar = "serBilir", timeVar = "year", 
                      idVar = "id", evTimeVar = "years", summary = "value")
    
    #censor event
    #Fit a standard Cox model to the landmark data set (for the censor event)
    Cox.LM.cen1 <- coxph(Surv(years, status2) ~ age + log(serBilir) + log(albumin) + 
                           log(prothrombin), data = pbc.LM1)
    PE.LM.cen1 <- PE.AD.coxph(Cox.LM.cen1, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w1.pbc, 
                              idVar = "id", timeVar = "year", respVar = "serBilir",
                              evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.cen1 <- PE.AD.JM2(JM.spline.cen, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w1.pbc)
    PE.JM.linear.cen1 <- PE.AD.JM2(JM.linear.cen, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w1.pbc)
    
    #comp event
    #Fit a standard Cox model to the landmark data set (for the composite event)
    Cox.LM.comp1 <- coxph(Surv(years, status3) ~ age + log(serBilir) + log(albumin) + 
                            log(prothrombin), data = pbc.LM1)
    PE.LM.comp1 <- PE.AD.coxph(Cox.LM.comp1, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w1.pbc, 
                               idVar = "id", timeVar = "year", respVar = "serBilir",
                               evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.comp1 <- PE.AD.JM2(JM.spline.comp, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w1.pbc)
    PE.JM.linear.comp1 <- PE.AD.JM2(JM.linear.comp, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w1.pbc)
    
    #censor event
    vec.pe.LM.cen1 <-c(vec.pe.LM.cen1, PE.LM.cen1["prederr"])
    vec.pe.JM.spline.cen1 <-c(vec.pe.JM.spline.cen1, PE.JM.spline.cen1["Brier"])
    vec.pe.JM.linear.cen1 <-c(vec.pe.JM.linear.cen1, PE.JM.linear.cen1["Brier"])
    #comp event
    vec.pe.LM.comp1 <-c(vec.pe.LM.comp1, PE.LM.comp1["prederr"])
    vec.pe.JM.spline.comp1 <-c(vec.pe.JM.spline.comp1, PE.JM.spline.comp1["Brier"])
    vec.pe.JM.linear.comp1 <-c(vec.pe.JM.linear.comp1, PE.JM.linear.comp1["Brier"])
    
  }
  # save vector of prediction errors to files
  # censor event
  fn.w1.LM.cen1 <- paste(fn.base, g, "\\LM_cen_w1.txt", sep='')
  fn.w1.JM.spline.cen1 <- paste(fn.base, g, "\\JM_spline_cen_w1.txt", sep='')
  fn.w1.JM.linear.cen1 <- paste(fn.base, g, "\\JM_linear_cen_w1.txt", sep='')
  write(formatC(unlist(vec.pe.LM.cen1), digits=15, format="fg", flag = "-"), fn.w1.LM.cen1, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.cen1), digits=15, format="fg", flag = "-"), fn.w1.JM.spline.cen1, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.cen1), digits=15, format="fg", flag = "-"), fn.w1.JM.linear.cen1, ncolumns=1)
  
  # comp event
  fn.w1.LM.comp1 <- paste(fn.base, g, "\\LM_comp_w1.txt", sep='')
  fn.w1.JM.spline.comp1 <- paste(fn.base, g, "\\JM_spline_comp_w1.txt", sep='')
  fn.w1.JM.linear.comp1 <- paste(fn.base, g, "\\JM_linear_comp_w1.txt", sep='')
  write(formatC(unlist(vec.pe.LM.comp1), digits=15, format="fg", flag = "-"), fn.w1.LM.comp1, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.comp1), digits=15, format="fg", flag = "-"), fn.w1.JM.spline.comp1, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.comp1), digits=15, format="fg", flag = "-"), fn.w1.JM.linear.comp1, ncolumns=1)
  
  
  print("w1 loop done")
  print(Sys.time())
  
  ################
  # Window 2
  ################
  
  # For fixed window w2=2 years work out PE for base time t=0-8 years (steps of 0.2 years)
  # For each base time t we must create a new LM data set and fit corresponding
  # LM models (using the training data)
  
  
  print("window 2 begin")
  
  vec.pe.LM.cen2 <-c()
  vec.pe.JM.spline.cen2 <-c()
  vec.pe.JM.linear.cen2 <-c()
  
  vec.pe.LM.comp2 <-c()
  vec.pe.JM.spline.comp2 <-c()
  vec.pe.JM.linear.comp2 <-c()
  
  for(i in 0:40){
    t.pbc <- i*0.2
    
    #create LM data set from training data at time t.pbc
    pbc.LM2 <- dataLM(train_data, t.pbc, respVar = "serBilir", timeVar = "year", 
                      idVar = "id", evTimeVar = "years", summary = "value")
    
    #censor event
    #Fit a standard Cox model to the landmark data set (for the censor event)
    Cox.LM.cen2 <- coxph(Surv(years, status2) ~ age + log(serBilir) + log(albumin) + 
                           log(prothrombin), data = pbc.LM2)
    PE.LM.cen2 <- PE.AD.coxph(Cox.LM.cen2, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w2.pbc, 
                              idVar = "id", timeVar = "year", respVar = "serBilir",
                              evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.cen2 <- PE.AD.JM2(JM.spline.cen, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w2.pbc)
    PE.JM.linear.cen2 <- PE.AD.JM2(JM.linear.cen, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w2.pbc)
    
    #comp event
    #Fit a standard Cox model to the landmark data set (for the composite event)
    Cox.LM.comp2 <- coxph(Surv(years, status3) ~ age + log(serBilir) + log(albumin) + 
                            log(prothrombin), data = pbc.LM2)
    PE.LM.comp2 <- PE.AD.coxph(Cox.LM.comp2, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w2.pbc, 
                               idVar = "id", timeVar = "year", respVar = "serBilir",
                               evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.comp2 <- PE.AD.JM2(JM.spline.comp, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w2.pbc)
    PE.JM.linear.comp2 <- PE.AD.JM2(JM.linear.comp, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w2.pbc)
    
    #censor event
    vec.pe.LM.cen2 <-c(vec.pe.LM.cen2, PE.LM.cen2["prederr"])
    vec.pe.JM.spline.cen2 <-c(vec.pe.JM.spline.cen2, PE.JM.spline.cen2["Brier"])
    vec.pe.JM.linear.cen2 <-c(vec.pe.JM.linear.cen2, PE.JM.linear.cen2["Brier"])
    #comp event
    vec.pe.LM.comp2 <-c(vec.pe.LM.comp2, PE.LM.comp2["prederr"])
    vec.pe.JM.spline.comp2 <-c(vec.pe.JM.spline.comp2, PE.JM.spline.comp2["Brier"])
    vec.pe.JM.linear.comp2 <-c(vec.pe.JM.linear.comp2, PE.JM.linear.comp2["Brier"])
    
  }
  # save vector of prediction errors to files
  # censor event
  fn.w2.LM.cen2 <- paste(fn.base, g, "\\LM_cen_w2.txt", sep='')
  fn.w2.JM.spline.cen2 <- paste(fn.base, g, "\\JM_spline_cen_w2.txt", sep='')
  fn.w2.JM.linear.cen2 <- paste(fn.base, g, "\\JM_linear_cen_w2.txt", sep='')
  write(formatC(unlist(vec.pe.LM.cen2), digits=15, format="fg", flag = "-"), fn.w2.LM.cen2, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.cen2), digits=15, format="fg", flag = "-"), fn.w2.JM.spline.cen2, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.cen2), digits=15, format="fg", flag = "-"), fn.w2.JM.linear.cen2, ncolumns=1)
  
  # comp event
  fn.w2.LM.comp2 <- paste(fn.base, g, "\\LM_comp_w2.txt", sep='')
  fn.w2.JM.spline.comp2 <- paste(fn.base, g, "\\JM_spline_comp_w2.txt", sep='')
  fn.w2.JM.linear.comp2 <- paste(fn.base, g, "\\JM_linear_comp_w2.txt", sep='')
  write(formatC(unlist(vec.pe.LM.comp2), digits=15, format="fg", flag = "-"), fn.w2.LM.comp2, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.comp2), digits=15, format="fg", flag = "-"), fn.w2.JM.spline.comp2, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.comp2), digits=15, format="fg", flag = "-"), fn.w2.JM.linear.comp2, ncolumns=1)
  
  
  print("w2 loop done")
  print(Sys.time())
  
  ################
  # Window 3
  ################
  
  # For fixed window w3=3 years work out PE for base time t=0-7 years (steps of 0.2 years)
  # For each base time t we must create a new LM data set and fit corresponding
  # LM models (using the training data)
  
  print("window 3 begin")
  
  vec.pe.LM.cen3 <-c()
  vec.pe.JM.spline.cen3 <-c()
  vec.pe.JM.linear.cen3 <-c()
  
  vec.pe.LM.comp3 <-c()
  vec.pe.JM.spline.comp3 <-c()
  vec.pe.JM.linear.comp3 <-c()
  
  for(i in 0:35){
    t.pbc <- i*0.2
    
    #create LM data set from training data at time t.pbc
    pbc.LM3 <- dataLM(train_data, t.pbc, respVar = "serBilir", timeVar = "year", 
                      idVar = "id", evTimeVar = "years", summary = "value")
    
    #censor event
    #Fit a standard Cox model to the landmark data set (for the censor event)
    Cox.LM.cen3 <- coxph(Surv(years, status2) ~ age + log(serBilir) + log(albumin) + 
                           log(prothrombin), data = pbc.LM3)
    PE.LM.cen3 <- PE.AD.coxph(Cox.LM.cen3, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w3.pbc, 
                              idVar = "id", timeVar = "year", respVar = "serBilir",
                              evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.cen3 <- PE.AD.JM2(JM.spline.cen, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w3.pbc)
    PE.JM.linear.cen3 <- PE.AD.JM2(JM.linear.cen, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w3.pbc)
    
    #comp event
    #Fit a standard Cox model to the landmark data set (for the composite event)
    Cox.LM.comp3 <- coxph(Surv(years, status3) ~ age + log(serBilir) + log(albumin) + 
                            log(prothrombin), data = pbc.LM3)
    PE.LM.comp3 <- PE.AD.coxph(Cox.LM.comp3, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w3.pbc, 
                               idVar = "id", timeVar = "year", respVar = "serBilir",
                               evTimeVar = "years", lossFun = "square", summary = "value")
    PE.JM.spline.comp3 <- PE.AD.JM2(JM.spline.comp, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w3.pbc)
    PE.JM.linear.comp3 <- PE.AD.JM2(JM.linear.comp, newdata = test_data, Tstart = t.pbc, Thoriz = t.pbc + w3.pbc)

    #censor event
    vec.pe.LM.cen3 <-c(vec.pe.LM.cen3, PE.LM.cen3["prederr"])
    vec.pe.JM.spline.cen3 <-c(vec.pe.JM.spline.cen3, PE.JM.spline.cen3["Brier"])
    vec.pe.JM.linear.cen3 <-c(vec.pe.JM.linear.cen3, PE.JM.linear.cen3["Brier"])
    #comp event
    vec.pe.LM.comp3 <-c(vec.pe.LM.comp3, PE.LM.comp3["prederr"])
    vec.pe.JM.spline.comp3 <-c(vec.pe.JM.spline.comp3, PE.JM.spline.comp3["Brier"])
    vec.pe.JM.linear.comp3 <-c(vec.pe.JM.linear.comp3, PE.JM.linear.comp3["Brier"])
    
  }
  # save vector of prediction errors to files
  # censor event
  fn.w3.LM.cen3 <- paste(fn.base, g, "\\LM_cen_w3.txt", sep='')
  fn.w3.JM.spline.cen3 <- paste(fn.base, g, "\\JM_spline_cen_w3.txt", sep='')
  fn.w3.JM.linear.cen3 <- paste(fn.base, g, "\\JM_linear_cen_w3.txt", sep='')
  write(formatC(unlist(vec.pe.LM.cen3), digits=15, format="fg", flag = "-"), fn.w3.LM.cen3, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.cen3), digits=15, format="fg", flag = "-"), fn.w3.JM.spline.cen3, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.cen3), digits=15, format="fg", flag = "-"), fn.w3.JM.linear.cen3, ncolumns=1)
  
  # comp event
  fn.w3.LM.comp3 <- paste(fn.base, g, "\\LM_comp_w3.txt", sep='')
  fn.w3.JM.spline.comp3 <- paste(fn.base, g, "\\JM_spline_comp_w3.txt", sep='')
  fn.w3.JM.linear.comp3 <- paste(fn.base, g, "\\JM_linear_comp_w3.txt", sep='')
  write(formatC(unlist(vec.pe.LM.comp3), digits=15, format="fg", flag = "-"), fn.w3.LM.comp3, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.spline.comp3), digits=15, format="fg", flag = "-"), fn.w3.JM.spline.comp3, ncolumns=1)
  write(formatC(unlist(vec.pe.JM.linear.comp3), digits=15, format="fg", flag = "-"), fn.w3.JM.linear.comp3, ncolumns=1)
  
  
  print("w3 loop done")
  print(Sys.time())
  
  
}