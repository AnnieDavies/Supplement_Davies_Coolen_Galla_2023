# Code to perform the simulation study for scenario 2 (JM and LM) in Davies, Coolen and Galla (2023).
# Written by A. Davies (2023).
# Based on code by D. Rizopoulos: https://github.com/drizopoulos/jm_and_lm
# Simulation methodology is described in full in the Supplementary Material of 
# Davies, Coolen and Galla (2023).
# Data is simulated from a joint model with a linear random slopes and random intercept
# longitudinal model and a survival model with a cumulative association.
# Data is split in half into a training and test data set.
# The two simulated data sets are saved in text files (to be read into Python DK codes).
# The longitudinal trajectories and histogram of event times are plotted.
# 2 joint models are fitted to the training data: both with the correct longitudinal model,
# one with an instantaneous association in the survival model (incorrect),
# and one with a cumulative association in the survival model (correct).
# Parameter estimates from the two models are recorded.
# Prediction error is evaluated for one prediction window and 5 base times for
# the two joint models and a standard landmarking model.


library("JMbayes")
library(ggplot2) # For plotting trajectories
library(lcsm)
library("xtable")
library("MASS")
library("splines")

# internal JMbayes function to create landmark data sets
dataLM <- JMbayes:::dataLM

################################################################################
# SET PARAMETERS

print("begin")
print(Sys.time())

n <- 1200 # number of subjects
K <- 15  # number of planned repeated measurements per subject, per outcome
t.max <- 15 # maximum follow-up time

# parameters for the linear mixed effects model---------------------------------
betas <- c("Intercept" = 2.0, "Time" = 1.25)
sigma.y <- 0.5 # measurement error standard deviation

D <- matrix(c(0.55, 0.2, 0.15, 0.45), 2, 2)
D <- (D + t(D)) / 2

# parameters for the survival model---------------------------------------------
gammas <- c("(Intercept)" = -7.0, "Group" = 0.25) # coefficients for baseline covariates
alpha <- 0.25 # association parameter 
phi <- 0.95 # shape for the Weibull baseline hazard

meanCens0 <- 10 # mean of the uniform censoring distribution for group 0
meanCens1 <- 14 # mean of the uniform censoring distribution for group 1


################################################################################
# Define model vectors and matrices

# generate observation time points for longitudinal measurements
times <- c(replicate(n, c(0, sort(runif(K-1, 0, t.max))))) 

# group indicator, i.e., '0' placebo, '1' active treatment
group <- rep(0:1, each = n/2) 

DF <- data.frame(year = times, group = factor(rep(group, each = K)))

# design matrices for the longitudinal measurement model
X <- model.matrix(~ year, data = DF)
Z <- model.matrix(~ year, data = DF)

# design matrix for the survival model
W <- cbind("(Intercept)" = 1, "Group" = group)

################################################################################
# SIMULATE DATA

#simulate random effects (covariance matrix = D)--------------------------------
b <- mvrnorm(n, rep(0, nrow(D)), D) 

print("simulated b")
print(Sys.time())

# simulate longitudinal responses-----------------------------------------------
id <- rep(1:n, each = K) 
eta.y <- as.vector(X %*% betas + rowSums(Z * b[id, ])) #fixed effects of linear model
y <- rnorm(n * K, eta.y, sigma.y) #simulate n*K values from a normal dist with mean = eta.y and variance = sigma.y

print("simulated y")
print(Sys.time())

# simulate event times----------------------------------------------------------
eta.t <- as.vector(W %*% gammas) # gamma0 + gamma1*groupi

# function to solve for event time: log(u) + int_0^t h(s) ds = 0 
invS <- function (t, u, i) {
  # hazard function:
  h <- function (s) {
    # integrate mi(t') [fixed effects] from 0 to t where mi(t')_FE = beta0 + beta2*t' 
    # result: beta0*t + beta2*t^2/2 
    # Now enter the coefficients of beta into XX (t=s):
    XX <- cbind(s, (s^2)/2)
    # integrate mi(t') [random effects] from 0 to t where mi(t')_RE = bi0 + bi1*t'
    # same as fixed effects - enter coefficients into ZZ:
    ZZ <- cbind(s, (s^2)/2)
    f1 <- as.vector(XX %*% betas + rowSums(ZZ * b[rep(i, nrow(ZZ)), ])) #mi(t)
    exp(log(phi) + (phi - 1) * log(s) + eta.t[i] + f1 * alpha) #phi = sigma_t in the paper
  }
  integrate(h, lower = 0, upper = t)$value + log(u)
}

#simulate event probabilities for all n subjects (unif dist between 0 and 1)
u <- runif(n) 

trueTimes <- numeric(n) 

#Solve invS for each simulated survival probability
for (i in 1:n) {
  Up <- 50 #upper limit of search
  tries <- 5 #number of times to increase limit and try to find the root
  
  #uniroot(f, interval[low, up], ) searches the interval [low, upper] for a root (f=0)
  Root <- try(uniroot(invS, interval = c(1e-05, Up), u = u[i], i = i)$root, TRUE)
  #result of uniroot: $root = location of root, $f.root = value of function at root
  
  #If no root is found, increase interval and try again
  while(inherits(Root, "try-error") && tries > 0) {
    tries <- tries - 1
    Up <- Up + 200
    Root <- try(uniroot(invS, interval = c(1e-05, Up), u = u[i], i = i)$root, TRUE)
  }
  trueTimes[i] <- if (!inherits(Root, "try-error")) Root else NA
}

print("simulated event times")
print(Sys.time())

#Remove any indiviudals where a root was not found ----------------------------- 
na.ind <- !is.na(trueTimes) #list of length n, =TRUE if not NA, = FALSE if NA
trueTimes <- trueTimes[na.ind] 
W <- W[na.ind, , drop = FALSE] 
group <- group[na.ind]
long.na.ind <- rep(na.ind, each = K) #extend the na.ind vector for vectors with K elements per individual
y <- y[long.na.ind] 
X <- X[long.na.ind, , drop = FALSE] 
Z <- Z[long.na.ind, , drop = FALSE] 
DF <- DF[long.na.ind, ] 
n <- length(trueTimes) #redefine number of individuals

# simulate censoring times -----------------------------------------------------
# and calculate the observed event times = min(true event times, censoring times)
Ctimes <- numeric(n)
Ctimes[group == 0] <- runif(sum(group == 0), 0, 2 * meanCens0)
Ctimes[group == 1] <- runif(sum(group == 1), 0, 2 * meanCens1)
Time <- pmin(trueTimes, Ctimes)
event <- as.numeric(trueTimes <= Ctimes) # event indicator

print("simulated censoring times")
print(Sys.time())

################################################################################
#CREATE DATAFRAME OF SIMULATED DATA

# drop the longitudinal measurements that were taken after the observed event time:
ind <- times[long.na.ind] <= rep(Time, each = K) #=TRUE if the observation time < event time and FALSE otherwise
y <- y[ind] 
X <- X[ind, , drop = FALSE] 
Z <- Z[ind, , drop = FALSE] 
id <- id[long.na.ind][ind] 
id <- match(id, unique(id)) #rename the ID labels 

dat <- DF[ind, ] #new data frame with only the observations to keep 
dat$id <- id 
dat$y <- y #longitudinal measurements
dat$Time <- Time[id] #event times 
dat$event <- event[id] #event indicator
names(dat) <- c("time", "group", "id", "y", "Time", "event")

#dat = completed data frame for simulated data

################################################################################
#DEFINE TEST AND TRAINING DATA

#split simulated data into a training and test data set
set <- sample(unique(id), n/2) 
train_data <- dat[!dat$id %in% set, ]
train_data$id <- match(train_data$id, unique(train_data$id)) 
test_data <- dat[dat$id %in% set, ]
test_data$id <- match(test_data$id, unique(test_data$id))

#create train.id and test.id for survival model:
train_data.id <- train_data[!duplicated(train_data$id), ]
test_data.id <- test_data[!duplicated(test_data$id), ]

print("split data")
print(Sys.time())

################################################################################
#SAVE SIMULATED DATA (to read into Python codes)

#extract values from test and train data--------------------------------------
#TRAIN:
n.train <- table(train_data$id)
events.train <-c(train_data.id[['Time']])
censor.train <-c(train_data.id[['event']])
Zf.group.train <-c(train_data.id[['group']])
t.obs.train <- c(train_data[['time']])
Zl.train <-c(train_data[['y']])

#TEST:
n.test <- table(test_data$id)
events.test <-c(test_data.id[['Time']])
censor.test <-c(test_data.id[['event']])
Zf.group.test <-c(test_data.id[['group']])
t.obs.test <- c(test_data[['time']])
Zl.test <-c(test_data[['y']]) 


#create filenames and write to file-------------------------------------------
fn.base <- "~...SIMULATION\\SCEN2"

#TRAIN
fn.n.train <- paste(fn.base, "\\TRAIN\\n_obs.txt", sep='')
fn.events.train <- paste(fn.base, "\\TRAIN\\Events.txt", sep='')
fn.censor.train <- paste(fn.base, "\\TRAIN\\Censor.txt", sep='')
fn.Zf.group.train <- paste(fn.base, "\\TRAIN\\Zfix_group.txt", sep='')
fn.t.obs.train <- paste(fn.base, "\\TRAIN\\t_obs.txt", sep='')
fn.Zl.train <- paste(fn.base, "\\TRAIN\\Zlong.txt", sep='')

write(formatC(n.train, digits=15, format="fg", flag = "-"), fn.n.train, ncolumns=1)
write(formatC(events.train, digits=15, format="fg", flag = "-"), fn.events.train, ncolumns=1)
write(formatC(censor.train, digits=15, format="fg", flag = "-"), fn.censor.train, ncolumns=1)
write(formatC(Zf.group.train, digits=15, format="fg", flag = "-"), fn.Zf.group.train, ncolumns=1)
write(formatC(t.obs.train, digits=15, format="fg", flag = "-"), fn.t.obs.train, ncolumns=1)
write(formatC(Zl.train, digits=15, format="fg", flag = "-"), fn.Zl.train, ncolumns=1)

#TEST
fn.n.test <- paste(fn.base, "\\TEST\\n_obs.txt", sep='')
fn.events.test <- paste(fn.base, "\\TEST\\Events.txt", sep='')
fn.censor.test <- paste(fn.base, "\\TEST\\Censor.txt", sep='')
fn.Zf.group.test <- paste(fn.base, "\\TEST\\Zfix_group.txt", sep='')
fn.t.obs.test <- paste(fn.base, "\\TEST\\t_obs.txt", sep='')
fn.Zl.test <- paste(fn.base, "\\TEST\\Zlong.txt", sep='')

write(formatC(n.test, digits=15, format="fg", flag = "-"), fn.n.test, ncolumns=1)
write(formatC(events.test, digits=15, format="fg", flag = "-"), fn.events.test, ncolumns=1)
write(formatC(censor.test, digits=15, format="fg", flag = "-"), fn.censor.test, ncolumns=1)
write(formatC(Zf.group.test, digits=15, format="fg", flag = "-"), fn.Zf.group.test, ncolumns=1)
write(formatC(t.obs.test, digits=15, format="fg", flag = "-"), fn.t.obs.test, ncolumns=1)
write(formatC(Zl.test, digits=15, format="fg", flag = "-"), fn.Zl.test, ncolumns=1)

print("training and test data created and saved")
print(Sys.time())


#delete all unused objects
rm(y, X, Z, id, n, na.ind, long.na.ind, ind, Ctimes, Time, event, W,
   betas, sigma.y, gammas, alpha, eta.t, eta.y, phi, t.max,
   trueTimes, u, Root, invS, D, b, K, set,
   times, group, i, tries, Up, Bkn, kn, DF, meanCens0, meanCens1)

################################################################################
#PLOT longitudinal trajectories and histogram of event times

myplot <- ggplot(data = train_data, aes(x = time, y = y)) + 
  geom_point(size = 2, aes(color = group)) + #colour points by group
  geom_path(aes(group = id)) + #spaghetti plot
  stat_smooth(method = "lm", formula = y ~ x, aes(group = group, colour = group)) + #line of best fit by group
  scale_color_discrete(labels=c('x=0', 'x=1')) +
  ylab("Longitudinal measurement, y(t)") +
  xlab("Time, t (years)") +
  theme_bw()

myplot + theme(text = element_text(size = 20)) 

EventTimesTrain <- train_data.id$Time
hist(EventTimesTrain, main="", xlab="Event times (years)", cex.axis = 1.5, cex.lab = 1.5)

print("plots done")
print(Sys.time())

################################################################################
#FIT JOINT MODELS

#longitudinal model
longFit_lin <- lme(y ~ time, data = train_data, random = ~ time | id)
summary(longFit_lin)

# survival model with instantaneous association
survFit <- coxph(Surv(Time, event) ~ group, data = train_data.id,
                  x = TRUE)
summary(survFit)

# JM with instantaneous association
jointFit1 <- jointModelBayes(longFit_lin, survFit, timeVar = "time", n.iter = 100000L)
summary(jointFit1)


# define cumulative association:
iForm <- list(fixed = ~ 0 + time + I(time^2/2),
              indFixed = 1:2, random = ~ 0 + time + I(time^2/2), 
              indRandom = 1:2)

# JM with cumulative association
jointFit2 <- update(jointFit1, param = "td-extra", extraForm = iForm)
summary(jointFit2)


################################################################################
#PREDICTION ERROR (JM and LM)

vec.pe.JM1 <-c()
vec.pe.JM2 <-c()
vec.pe.LM <-c()
win <- 2.0

for(i in 0:4){
  
  timeBase <- 1.5+(i*2.0) 
  print(timeBase)
  print(Sys.time())
  
  ######################
  # LM model
  ######################
  
  #create LM data set at LM time = timeBase using training data
  train.LM <- dataLM(train_data, timeBase, respVar = "y", timeVar = "time", 
                     evTimeVar = "Time", idVar = "id", summary = "value")
  
  #Fit a standard Cox model to the landmark data set
  Cox.train.LM <- coxph(Surv(Time, event) ~ group + y, data = train.LM)
  
  print("LM model done:")

  
  #####################
  #PE using test_data
  #####################
  
  #LM
  PE.LM <-prederrJM(Cox.train.LM, newdata = test_data, Tstart = timeBase, Thoriz = timeBase + win,
                    idVar = "id", timeVar = "time", respVar = "y", evTimeVar = "Time",
                    lossFun = "square", summary = "value")
  print("PE LM done")
  print(Sys.time())
  #JM1:
  PE.JM1<-prederrJM(jointFit1, newdata = test_data, Tstart = timeBase, Thoriz = timeBase + win,
                   idVar = "id", timeVar = "time", respVar = "y", evTimeVar = "Time",
                   lossFun = "square", summary = "value", simulate = TRUE)
  print("PE JM1 done")
  print(Sys.time())
  #JM1:
  PE.JM2<-prederrJM(jointFit2, newdata = test_data, Tstart = timeBase, Thoriz = timeBase + win,
                   idVar = "id", timeVar = "time", respVar = "y", evTimeVar = "Time",
                   lossFun = "square", summary = "value", simulate = TRUE)
  print("PE JM2 done")
  print(Sys.time())

  vec.pe.JM1 <-c(vec.pe.JM1, PE.JM1["prederr"])
  vec.pe.JM2 <-c(vec.pe.JM2, PE.JM2["prederr"])
  vec.pe.LM <-c(vec.pe.LM, PE.LM["prederr"])
}

print("PE joint model 1 (incorrect)")
print(vec.pe.JM1)
print("PE joint model 2 (correct)")
print(vec.pe.JM2)
print("PE LM model")
print(vec.pe.LM)

print("**********DONE")
print(Sys.time())






