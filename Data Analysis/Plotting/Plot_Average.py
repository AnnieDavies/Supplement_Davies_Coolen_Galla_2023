'''
Code to read in prediction error vector files for all models (joint, landmarking and delayed kernel) for each 
cross validation iteration, average over the iterations and plot the results
At each iteration, one group is treated as the test data and the other groups comprise the training data
Same code can be used for all data sets - AIDS, Liver, PBC - folder and model names should be changed accordingly
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
from numpy import exp, arange, sqrt, pi
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

V = 10 #number of cross validation

#Function to read in a text file
def readfile(fn_base, g, model, analysis):
    fn = fn_base + str(g) + "\\" + model + "_" + analysis + ".txt"
    vec = []
    with open(fn, "r") as file:
        for line in file:
            vec.append(line)
    for i in range(0,len(vec)):
        try:
            vec[i] = float(vec[i])
        except:
            #JM outputs NA when t=u (Tstart=Thoriz)		
            #we can treat this as 0.
            vec[i] = 0.0
    return vec
    
#Function to average PE over the cross validation iterations
def avfiles(fn_base, model, analysis, V):
    #set averages = results of first iteration
    av_vals = readfile(fn_base, 1, model, analysis)
    #then add on the results from subsequent iterations:
    for i in range (2,V+1):
        loop_vals = readfile(fn_base, i, model, analysis)
        for j in range(0,len(loop_vals)):
            av_vals[j] = av_vals[j] + loop_vals[j]
    #Then average (divide by number of iterations)
    for j in range(0,len(loop_vals)):
            av_vals[j] = av_vals[j]/V   
    return av_vals

#Define folder when JM and LM files are stored
fn_base = "~...\\DATA_RESULTS\\JM_LM\\G"

JMfixt = avfiles(fn_base, "JM", "fixt", V)
JMw1 = avfiles(fn_base, "JM", "w1", V)
JMw2 = avfiles(fn_base, "JM", "w2", V)
JMw3 = avfiles(fn_base, "JM", "w3", V)

LMfixt = avfiles(fn_base, "LM", "fixt", V)
LMw1 = avfiles(fn_base, "LM", "w1", V)
LMw2 = avfiles(fn_base, "LM", "w2", V)
LMw3 = avfiles(fn_base, "LM", "w3", V)

#Define folder where DK results are stored
fn_baseDK = "~...DATA_RESULTS\\DK\\G"

Afixt = avfiles(fn_baseDK, "ModelA", "fixt", V)
Aw1 = avfiles(fn_baseDK, "ModelA", "w1", V)
Aw2 = avfiles(fn_baseDK, "ModelA", "w2", V)
Aw3 = avfiles(fn_baseDK, "ModelA", "w3", V)


Bfixt = avfiles(fn_baseDK, "ModelB", "fixt", V)
Bw1 = avfiles(fn_baseDK, "ModelB", "w1", V)
Bw2 = avfiles(fn_baseDK, "ModelB", "w2", V)
Bw3 = avfiles(fn_baseDK, "ModelB", "w3", V)



timeLM = 3.0
#Define vector of prediction times for fix t analysis (here given for Liver data)
PT = []
for i in range(0,36):
    PT.append(timeLM + (i*0.2))

#Define vectors of base times for the three prediction window analyses (here given for Liver data)
PT1 = []
for i in range(0,46):
    PT1.append(i*0.2)

PT2 = []
for i in range(0,41):
    PT2.append(i*0.2)

PT3 = []
for i in range(0,36):
    PT3.append(i*0.2)

#Plot fix t analysis ############################################################################
fig1, ax1 = plt.subplots(figsize = (8,6)) 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.tick_params(labelsize=20)

ERR_A = ax1.plot(PT,Afixt,color='darkorange', linestyle='-', linewidth=1.25, label = "Model A")
ERR_B = ax1.plot(PT,Bfixt,color='red', linestyle='-', linewidth=1.25, label = "Model B")
ERR_JM = ax1.plot(PT,JMfixt,color='purple', linestyle='--', linewidth=1.25, label = "Joint Model")
ERR_LM, = ax1.plot(PT,LMfixt,color='royalblue', linewidth=1.25, label = "Landmarking")
ERR_LM.set_dashes([2, 2, 10, 2])

plt.xlim(3, 10)

ax1.legend(loc = 'best', fontsize = 18)

plt.show()

plt.ylabel(r"$\widehat{\mathrm{PE}}(u|t)$, $t=3$ years", fontsize = 24)
plt.xlabel(r"Prediction time, $u$, in years", fontsize = 24)

fig1.tight_layout()
fig1.show()

#Plot window 1 analysis ############################################################################
fig2, ax2 = plt.subplots(figsize = (8,6)) 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.tick_params(labelsize=22)

ERR_A1 = ax2.plot(PT1,Aw1,color='darkorange', linestyle='-', linewidth=1.25, label = "Model A")
ERR_B1 = ax2.plot(PT1,Bw1,color='red', linestyle='-', linewidth=1.25, label = "Model B")
ERR_JM1 = ax2.plot(PT1,JMw1,color='purple', linestyle='--', linewidth=1.25, label = "Joint Model")
ERR_LM1, = ax2.plot(PT1,LMw1,color='royalblue', linewidth=1.25, label = "Landmarking")
ERR_LM1.set_dashes([2, 2, 10, 2])

plt.xlim(0, 9)

ax2.legend(loc = 'best', fontsize = 18)

plt.show()

plt.ylabel(r"$\widehat{\mathrm{PE}}(t+w_1|t)$, $w_1=1$ year", fontsize = 24)
plt.xlabel(r"Base time, $t$, in years", fontsize = 24)

fig2.tight_layout()
fig2.show()

#Plot window 2 analysis ############################################################################
fig3, ax3 = plt.subplots(figsize = (8,6)) 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.tick_params(labelsize=22)

ERR_A2 = ax3.plot(PT2,Aw2,color='darkorange', linestyle='-', linewidth=1.25, label = "Model A")
ERR_B2 = ax3.plot(PT2,Bw2,color='red', linestyle='-', linewidth=1.25, label = "Model B")
ERR_JM2 = ax3.plot(PT2,JMw2,color='purple', linestyle='--', linewidth=1.25, label = "Joint Model")
ERR_LM2, = ax3.plot(PT2,LMw2,color='royalblue', linewidth=1.25, label = "Landmarking")
ERR_LM2.set_dashes([2, 2, 10, 2])

ax3.legend(loc = 'best', fontsize = 18)

plt.xlim(0, 8)

plt.show()

plt.ylabel(r"$\widehat{\mathrm{PE}}(t+w_2|t)$, $w_2=2$ years", fontsize = 24)
plt.xlabel(r"Base time, $t$, in years", fontsize = 24)

fig3.tight_layout()
fig3.show()

#Plot window 3 analysis ############################################################################
fig4, ax4 = plt.subplots(figsize = (8,6))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.tick_params(labelsize=22)

ERR_A3 = ax4.plot(PT3,Aw3,color='darkorange', linestyle='-', linewidth=1.25, label = "Model A")
ERR_B3 = ax4.plot(PT3,Bw3,color='red', linestyle='-', linewidth=1.25, label = "Model B")
ERR_JM3 = ax4.plot(PT3,JMw3,color='purple', linestyle='--', linewidth=1.25, label = "Joint Model")
ERR_LM3, = ax4.plot(PT3,LMw3,color='royalblue', linewidth=1.25, label = "Landmarking")
ERR_LM3.set_dashes([2, 2, 10, 2])

ax4.legend(loc = 'best', fontsize = 18)

plt.xlim(0, 7)
	
plt.show()

plt.ylabel(r"$\widehat{\mathrm{PE}}(t+w_3|t)$, $w_3=3$ years", fontsize = 24)
plt.xlabel(r"Base time, $t$, in years", fontsize = 24)

fig4.tight_layout()
fig4.show()