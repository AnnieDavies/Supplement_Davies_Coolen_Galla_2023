# Edited version of tvBrier (from JMbayes2 package) 
# Code copied from https://github.com/drizopoulos/JMbayes2/blob/master/R/accuracy_measures.R
# Code edited by A Davies (2023)
# Edits are labelled by 'AD edit' throughout
# where practical I have shown the original code in a comment labelled #orig
# Edits are made to ensure calculation of prediction error (PE) exactly matches PE Eq 
# (see Eq (26) in Davies, Coolen and Galla (2023), or, equivalently the prediction 
# error equation (no number label) on pg 34 in Rizopoulos, D. (2016). The R package 
# JMbayes for fitting joint models for longitudinal and time-to-event data using MCMC.
# Journal of Statistical Software72(7), 1-46)
# 
# List of edits (Ti = event time, t= base time, u=prediction time)
# (1) sum over i: change from Ti > t to Ti >= t 
# (2) 1st term of sum: change from I(Ti>u) to I(Ti>=u)
# (3) Allow calculation of PE if there are no events in the interval [t,u] (the 
#     second term in the sum is zero)

PE.AD.JM2 <- function (object, newdata, Tstart, Thoriz = NULL, Dt = NULL, ...) {
  if (!inherits(object, "jm"))
    stop("Use only with 'jm' objects.\n")
  if (!is.data.frame(newdata) || nrow(newdata) == 0)
    stop("'newdata' must be a data.frame with more than one rows.\n")
  if (is.null(Thoriz) && is.null(Dt))
    stop("either 'Thoriz' or 'Dt' must be non null.\n")
  #if (!is.null(Thoriz) && Thoriz <= Tstart)                                    #orig
    #stop("'Thoriz' must be larger than 'Tstart'.")                             #orig
  if (is.null(Thoriz))
    Thoriz <- Tstart + Dt
  type_censoring <- object$model_info$type_censoring
  if (object$model_info$CR_MS)
    stop("'tvBrier()' currently only works for right censored data.")
  #Tstart <- Tstart + 1e-06                                                     #orig
  #Thoriz <- Thoriz + 1e-06                                                     #orig
  id_var <- object$model_info$var_names$idVar
  time_var <- object$model_info$var_names$time_var
  Time_var <- object$model_info$var_names$Time_var
  event_var <- object$model_info$var_names$event_var
  if (is.null(newdata[[id_var]]))
    stop("cannot find the '", id_var, "' variable in newdata.", sep = "")
  if (is.null(newdata[[time_var]]))
    stop("cannot find the '", time_var, "' variable in newdata.", sep = "")
  if (any(sapply(Time_var, function (nmn) is.null(newdata[[nmn]]))))
    stop("cannot find the '", paste(Time_var, collapse = ", "),
         "' variable(s) in newdata.", sep = "")
  if (is.null(newdata[[event_var]]))
    stop("cannot find the '", event_var, "' variable in newdata.", sep = "")
  #newdata <- newdata[newdata[[Time_var]] > Tstart, ]                           #orig
  newdata <- newdata[newdata[[Time_var]] >= Tstart, ]                           #AD edit (1)
  newdata <- newdata[newdata[[time_var]] <= Tstart, ]
  #if (!nrow(newdata))                                                                    #orig
  #  stop("there are no data on subjects who had an observed event time after Tstart ",   #orig
  #       "and longitudinal measurements before Tstart.")                                 #orig
  newdata[[id_var]] <- newdata[[id_var]][, drop = TRUE]
  test1 <- newdata[[Time_var]] < Thoriz & newdata[[event_var]] == 1
  if (!is.null(Thoriz) && Thoriz <= Tstart){
    out <- list(Brier = NA, nr = NA, Tstart = Tstart, Thoriz = Thoriz,
                classObject = class(object),
                nameObject = deparse(substitute(object)))
    print("Tstart=Thoriz")
  }
  else if(!nrow(newdata)){
    out <- list(Brier = NA, nr = NA, Tstart = Tstart, Thoriz = Thoriz,
                classObject = class(object),
                nameObject = deparse(substitute(object)))
    print(paste("there are no data on subjects who had an observed event time after Tstart ",
                "and longitudinal measurements before Tstart. (Tstart = ", as.character(Tstart), ")"))
  }
  #else if(!any(test1)){                                                                          #orig
  #  out <- list(Brier = NA, nr = NA, Tstart = Tstart, Thoriz = Thoriz,                           #orig
  #              classObject = class(object),                                                     #orig
  #              nameObject = deparse(substitute(object)))                                        #orig
  #  print(paste("no events in the interval [Tstart, Thoriz) = [", Tstart, ", ", Thoriz, ")."))   #orig
  #}                                                                                              #orig
  else{
    if (!any(test1))
      #stop("it seems that there are no events in the interval [Tstart, Thoriz).")                #orig
      print(paste("no events in the interval [Tstart, Thoriz) = [", Tstart, ", ", Thoriz, ")."))  #AD edit (3)
    
    newdata2 <- newdata
    newdata2[[Time_var]] <- Tstart
    newdata2[[event_var]] <- 0
    preds <- predict(object, newdata = newdata2, process = "event",
                     times = Thoriz, ...)
    pi_u_t <- preds$pred
    names(pi_u_t) <- preds$id
    # cumulative risk at Thoriz
    pi_u_t <- pi_u_t[preds$times > Tstart]
    
    id <- newdata[[id_var]]
    Time <- newdata[[Time_var]]
    event <- newdata[[event_var]]
    f <- factor(id, levels = unique(id))
    Time <- tapply(Time, f, tail, 1L)
    event <- tapply(event, f, tail, 1L)
    names(Time) <- names(event) <- as.character(unique(id))
    pi_u_t <- pi_u_t[names(Time)]
    
    # subjects who had the event before Thoriz
    ind1 <- Time < Thoriz & event == 1
    # subjects who had the event after Thoriz
    #ind2 <- Time > Thoriz                                                      #orig
    ind2 <- Time >= Thoriz                                                      #AD edit (2)
    # subjects who were censored in the interval (Tstart, Thoriz)
    ind3 <- Time < Thoriz & event == 0
    if (any(ind3)) {
      nams <- names(ind3[ind3])
      preds2 <- predict(object, newdata = newdata[id %in% nams, ],
                        process = "event", times = Thoriz, ...)
      weights <- preds2$pred
      f <- factor(preds2$id, levels = unique(preds2$id))
      names(weights) <- f
      weights <- tapply(weights, f, tail, 1)
    }
    loss <- function (x) x * x
    events <- sum(loss(1 - pi_u_t[ind1]), na.rm = TRUE)
    no_events <- sum(loss(pi_u_t[ind2]), na.rm = TRUE)
    censored <- if (any(ind3)) {
      sum(weights * loss(1 - pi_u_t[ind3]) +
            (1 - weights) * loss(pi_u_t[ind3]), na.rm = TRUE)
    } else 0.0
    nr <- length(Time)
    Brier <- (events + no_events + censored) / nr
    out <- list(Brier = Brier, nr = nr, Tstart = Tstart, Thoriz = Thoriz,
                classObject = class(object),
                nameObject = deparse(substitute(object)))
  }
  class(out) <- "tvBrier"
  out
}