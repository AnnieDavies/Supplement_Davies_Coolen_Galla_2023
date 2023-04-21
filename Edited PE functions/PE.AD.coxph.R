# Edited version of prederrJM (from JMbayes package) for coxph objects
# Code copied from https://github.com/drizopoulos/JMbayes/tree/master/R/prederrJM.coxph.R
# Code edited by A Davies (2021)
# Edits are labelled by 'AD edit' throughout
# where practical I have shown the original code in a comment labelled #orig
# Edits are made to ensure calculation of prediction error (PE) exactly matches PE Eq 
# (see Eq (26) in Davies, Coolen and Galla (2023), or, equivalently the prediction 
# error equation (no number label) on pg 34 in Rizopoulos, D. (2016). The R package 
# JMbayes for fitting joint models for longitudinal and time-to-event data using MCMC.
# Journal of Statistical Software72(7), 1-46)
# NB: Ti = event time, t= base time, u=prediction time, delta_i = event indicator
# List of edits:
# (1) sum over i: change from Ti > t to Ti >= t 
# (2) 1st term of sum: change from I(Ti>u) to I(Ti>=u)
# (3) 2nd term of sum: change from delta_i*I(Ti<=u) to delta_i*I(Ti<u)
# (4) Allow calculation of PE if there are no events in the interval [t,u] (the 
#     second term in the sum is zero)
# (5) Use argument na.rm = TRUE in sum over i (so that we do not get NA if there
#     are no events or no censor events in [t,u], this treats the second or last 
#     term as zero respectively)
#
# NB: These edits were made and used for models where summary = "value", 
# interval = "FALSE", and censor type is right censoring (i.e. is_counting = FALSE)
# The validity of the function for other summary arguments, interval = "TRUE" or
# is_counting=TRUE has not been verified. 

PE.AD.coxph <- function (object, newdata, Tstart, Thoriz, lossFun = c("absolute", "square"), 
                             interval = FALSE, idVar = "id", timeVar = "time", respVar = "y", 
                             evTimeVar = "Time", summary = c("value", "slope", "area"), 
                             tranfFun = function (x) x, ...) {
  if (!inherits(object, "coxph"))
    stop("Use only with 'coxph' objects.\n")
  if (!is.data.frame(newdata) || nrow(newdata) == 0)
    stop("'newdata' must be a data.frame with more than one rows.\n")
  if (is.null(newdata[[idVar]]))
    stop("'idVar' not in 'newdata'.\n")
  lossFun <- if (is.function(lossFun)) {
    lf <- lossFun
    match.fun(lossFun)
  } else {
    lf <- match.arg(lossFun)
    if (lf == "absolute") function (x) abs(x) else function (x) x*x
  }
  summary <- match.arg(summary)
  if (summary %in% c("slope", "area"))
    newdata$area <- newdata$slope <- 0
  id <- newdata[[idVar]]
  id <- match(id, unique(id))
  TermsT <- object$terms
  SurvT <- model.response(model.frame(TermsT, newdata)) 
  is_counting <- attr(SurvT, "type") == "counting"
  Time <- if (is_counting) {
    ave(SurvT[, 2], id, FUN = function (x) tail(x, 1))
  } else {
    Time <- SurvT[, 1]
  }
  #AD edit (1) -----------------------------------------------------------------
  ## newdata2 should have event times Ti >= t 
  
  ## dataLM keeps only individuals with event times > t
  #newdata2 <- dataLM(newdata, Tstart, idVar, respVar, timeVar, evTimeVar, summary, 
  #                   tranfFun) #orig
  
  ## dataLM.AD is the same as dataLM but with Time>=Tstart instead of Time>Tstart
  newdata2 <- dataLM.AD(newdata, Tstart, idVar, respVar, timeVar, evTimeVar, summary, 
                     tranfFun) #edit
  #-----------------------------------------------------------------------------
  SurvT <- model.response(model.frame(TermsT, newdata2)) 
  if (is_counting) {
    id2 <- newdata2[[idVar]]
    f <- factor(id2, levels = unique(id2))
    Time <- ave(SurvT[, 2], f, FUN = function (x) tail(x, 1))
    delta <- ave(SurvT[, 3], f, FUN = function (x) tail(x, 1))
  } else {
    Time <- SurvT[, 1]
    delta <- SurvT[, 2]
  }
  #AD edit----------------------------------------------------------------------
  ## censored individuals: Ti < u and delta = 0
  indCens <- Time < Thoriz & delta == 0 #orig
  nr <- nrow(newdata2) #orig
  ## Edit (2): alive individuals: Ti >= u 
  #aliveThoriz.id <- newdata2[Time > Thoriz, ] #orig
  aliveThoriz.id <- newdata2[Time >= Thoriz, ] #edit
  ## Edit (3): dead individuals: Ti < u and delta = 1
  #deadThoriz.id <- newdata2[Time <= Thoriz & delta == 1, ] #orig
  deadThoriz.id <- newdata2[Time < Thoriz & delta == 1, ] #edit
  #-----------------------------------------------------------------------------
  
  #AD edit (4)------------------------------------------------------------------
  # prederr <- if (length(unique(Time)) > 1 && nrow(aliveThoriz.id) > 1 &&
  #                nrow(deadThoriz.id) > 1) { #orig
  
  ## edit so that we don't get NA if no-one dies or is censored in the interval t->u
  
  ## work out sum if there are people alive after t
  prederr <- if (length(unique(Time)) > 1) {
    ## work out term 1 (alive) if there are individuals alive after u
    if(nrow(aliveThoriz.id) >= 1){
      Surv.aliveThoriz <- c(summary(survfit(object, newdata = aliveThoriz.id), times = Thoriz)$surv) #orig
    } else{
      ## no-one alive after u
      Surv.aliveThoriz <- NA
    }
    
    ## work out term 2 (event) if there are individuals who died between t and u
    if(nrow(deadThoriz.id) >= 1){
      Surv.deadThoriz <- c(summary(survfit(object, newdata = deadThoriz.id), times = Thoriz)$surv) #orig
    } else{
      ## no-one died between t and u
      Surv.deadThoriz <- NA
    }
    
    ## work out term 3 (censor) if there are individuals who were censored between t and u
    if (sum(indCens) >= 1) {
  #-----------------------------------------------------------------------------
      censThoriz.id <- newdata2[indCens, ]
      Surv.censThoriz <- c(summary(survfit(object, newdata = censThoriz.id), times = Thoriz)$surv)
      if (is_counting) {
        tt <- model.response(model.frame(TermsT, censThoriz.id))[, 2]
      } else {
        tt <- model.response(model.frame(TermsT, censThoriz.id))[, 1]
      }
      nn <- length(tt)
      weights <- numeric(nn)
      for (i in seq_len(nn)) {
        weights[i] <- c(summary(survfit(object, newdata = censThoriz.id[i, ]), times = Thoriz)$surv) /
          c(summary(survfit(object, newdata = censThoriz.id[i, ]), times = tt[i])$surv)
      }
      
    } else {
      Surv.censThoriz <- weights <- NA
    }
    if (!interval) {
      #AD edit (5)--------------------------------------------------------------
      ## add na.rm = TRUE to sum over i
      
      # ORIG:
      # (1/nr) * sum(lossFun(1 - Surv.aliveThoriz), lossFun(0 - Surv.deadThoriz),
      #              weights * lossFun(1 - Surv.censThoriz) + (1 - weights) * lossFun(0 - Surv.censThoriz))
      
      #EDIT:
      (1/nr) * sum(lossFun(1 - Surv.aliveThoriz), lossFun(0 - Surv.deadThoriz),
                   weights * lossFun(1 - Surv.censThoriz) + (1 - weights) * lossFun(0 - Surv.censThoriz), 
                   na.rm = TRUE)
      #-------------------------------------------------------------------------
    } else {
      TimeCens <- model.response(model.frame(TermsT, newdata))[, 1]
      deltaCens <- 1 - model.response(model.frame(TermsT, newdata))[, 2]
      KMcens <- survfit(Surv(TimeCens, deltaCens) ~ 1)
      times <- TimeCens[TimeCens > Tstart & TimeCens <= Thoriz & !deltaCens]
      times <- sort(unique(times))
      k <- as.numeric(table(times))
      w <- summary(KMcens, times = Tstart)$surv / summary(KMcens, times = times)$surv
      prederr.times <- sapply(times, 
                              function (t) prederrJM(object, newdata, Tstart, t,
                                                     interval = FALSE, idVar = idVar, timeVar = timeVar,
                                                     respVar = respVar, evTimeVar = evTimeVar, 
                                                     summary = summary, tranfFun = tranfFun)$prederr)
      num <- sum(prederr.times * w * k, na.rm = TRUE)
      den <- sum(w * k, na.rm = TRUE)
      num / den
    }
  } else {
    
    nr <- NA
    NA
  }
  out <- list(prederr = prederr, nr = nr, Tstart = Tstart, Thoriz = Thoriz, interval = interval,
              classObject = class(object), nameObject = deparse(substitute(object)), lossFun = lf)
  class(out) <- "prederrJM"
  out
}