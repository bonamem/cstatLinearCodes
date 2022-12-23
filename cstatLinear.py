# ========================================================
# python code that provides the best-fit 
# maximum-likelihood solutions for linear models
# using the Cash statistic
# Author: M. Bonamente, 2019, 2020 
# ========================================================
# It also needs to be complemented by a README with
# test cases to reproduce all examples in the paper
# and instructions on how to use functions
# ========================================================
exec(open('imports.py').read())

import math,sys,numbers
import numpy as np
import matplotlib.ticker as tick
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit,root_scalar
from scipy.optimize import fsolve
from scipy.stats import poisson
from scipy.stats import uniform
# functions below are on cstatFunctions.py =========
from cstatFunctions import  flinear,cminLinear,fa,faprime,bestFitLnum,bestFitL,bestFitA,ga,isModelAcceptable
from cstatFunctions import flinearPivotA,flinearPivotB,flinearConst,cstat,rangex,sanityCheck,findGap
from cstatFunctions import Rprime,bestFitLnumGap,flinearPivotAGap,flinearPivotBGap,flinearConstGap, linear, chi2
from cstatLinearFunction import bestFitLinear, bestFitLinearExtended
# ==================================================
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)

from cstatFunctionsNew import deltaa,Ga,deltalambda,deltabeta,covALambda,linearPivotedA
from cstatFunctionsNew import confidenceBand,covMatrixErrProp,covMatrixCT,OLSStats,confidenceBandOLS
from scipy.stats import linregress

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['figure.figsize'] = 5,5
plt.rcParams.update({'figure.autolayout': True})
print(sys.argv)

# ======= READ DATA FROM FILE ===========================
if(len(sys.argv)==4): # there is data input provided
	print(' === Reading data from file ====')
	xbinFile=sys.argv[1]
	yFile=sys.argv[2]
	paramFile=sys.argv[3]
	xbin=np.genfromtxt(xbinFile,delimiter=" ")
	y=np.genfromtxt(yFile,delimiter=" ")
	param=np.genfromtxt(paramFile,delimiter=' ',dtype=None, encoding=None)
	param=param.tolist()
	print(param)
	xA=param[0]
	xB=param[1]
	# Deltax is either a number or an file name
	Deltax=param[2]
	# If it is a number, Deltax are uniform
	if(isinstance(Deltax,numbers.Number))==1:
		Deltax=Deltax*np.ones(len(xbin)) # it's an array now
	# if it is a filename, read the file into the array
	if(isinstance(Deltax,str))==1:
		Deltax=np.genfromtxt(Deltax,delimiter=" ",dtype=float,encoding=None)
	print(Deltax)

if(len(sys.argv)!=4):
	print('Wrong number of arguments provided')
	print('Usage : python3 cstatLinear.py xfile yfile parameterfile')
	quit()

# ===========================================================
# package the data 
data=(xbin,y,xA,Deltax)
M=int(sum(y)) # Number of counts in data set
N=len(xbin)   # Number of bins
Ry=np.max(y)-np.min(y) # Range of y variable
R=rangex(data)
# Find how many non-zero bins we have ========================
ind=np.where(y>=1.0)    # find the non-zero indices 
gaindex=ind[0]
n=len(gaindex) # number of bins with non-zero counts
print('xbin',xbin)
print('y',y)
print('xA, xB',xA,xB)
print('Deltax',Deltax)
print(' ==== Summary of data =======')
print('Data has M=',M,'counts, N=',N,'bins, and n=',n,'unique non-zero bins')
# perform sanity checks on the data
sanityCheck(data)
# Determine if the data have a gap
hasGap,xa,xb,xG,RG,SGSum=findGap(data)
if hasGap>=1:
	Rp=Rprime(R,RG,xA,xG)
	print('R=',R,'RG=',RG,'xA=',xA,'xG=',xG,'Rprime=',Rp)
	# If the data have a gap, R -> Rprime
	R=Rp
	# For the last step, lambda(a), need a different R
	# So this R will not work for that last step
	dataGap=(xa,xb,xG,RG)
# ============================================================
# At this point we have data, hasGap, datagap
# which are needed by bestFitLinear(data,hasGap,datagap)
# or by bestFitLinearExtended(data,hasGap,datagap)a
# hasConverged[i,k],a[i,k],Lambda[i,k],cstat[i,k]=bestFitLinear(data,hasGap,data)
# cstat[i,k],modelType[i,k]=bestFitLinearExtended(data,hasGap,data)
# ============================================================


# Declare arrays 
cmin=np.zeros(n-1)
cminplot=np.zeros((n-1,N))
cminplot[:]=np.nan
cminplotPivotA=np.zeros(N)
cminplotPivotB=np.zeros(N)
cminplotConst=np.zeros(N)
ymodel=np.zeros((n-1,N))
ymodelerr=np.zeros((n-1,N))
fmodel=np.zeros((n-1,N)) # for the density function
ymodelPivotA=np.zeros(N)
ymodelerrPivotA=np.zeros(N)
fmodelPivotA=np.zeros(N)
ymodelPivotB=np.zeros(N)
ymodelerrPivotB=np.zeros(N)
fmodelPivotB=np.zeros(N)
ymodelConst=np.zeros(N)
ymodelerrConst=np.zeros(N)
fmodelConst=np.zeros(N)
ymodelAcceptable=np.zeros(n-1,dtype=int)

# These calls to the function bestFitLinear and bestFitLinearExtended 
# work, but they are not used. They can be uncommented by users.
# They can be used by the user for other purposes too.
# =======================================================================
#hasConverged,aBestFit,LambdaBestFit,cstat=bestFitLinear(data,hasGap,dataGap)
#print('convergence ',hasConverged,'a=',aBestFit,'lambda=',LambdaBestFit,'cmin=',cstat)
#cstat,modelType=bestFitLinearExtended(data,hasGap,dataGap)
#print('cmin=',cstat,'using model ',modelType)
# ========================================================================
#quit()

# ----

# ========================================================
# Now fit the data following the algorithm ===============
# described in Bonamente 2020 ============================
# ========================================================
#
# ========= FIT THE DATA (steps 1-7) =====================
# NOTE: compared to the function bestFitLinear, here all
# the solutions of F(a)=0 are found, including the ones
# that are unacceptable. 
if (M==1):
	# step (1) +++++++++++++++++++++++++++++++++++++++
	print('No solution available with linear model')
	print('1-R/2(xi-xA)=%5.5f'%(1-R/(2.*(xbin[ind]-xA))))
if (M>=2):
	# step (2): Find points of singularity of g(a) +++
	gaSing=np.zeros(n)
	gaZero=np.zeros(n-1) 	# n-1 zeros of g(a)
	for i in range(n):
		gaSing[i]=-1/(xbin[gaindex[i]]-xA)
		print("%d-th g(a) singularity at %3.3f (for x=%3.3f)"%(i,gaSing[i],xbin[gaindex[i]]))
	# step (3): Find zeros of g(a) between points of
	# singularity ++++++++++++++++++++++++++++++++++++
		Eps=1.e-6 # this is an Epsilon value
		if i > 0: 
			resg=root_scalar(ga,bracket=[gaSing[i-1]+Eps,gaSing[i]-Eps],args=data,method="brentq")
			gaZero[i-1]=resg.root	
			print("%d-th g(a) root found at %f (between %f and %f)"%(i,gaZero[i-1],gaSing[i-1],gaSing[i]))
	# step (4): Find asymptotic value of F(a) ++++++++ 
	haLim=0.0
	for i in range(N):
		haLim+=-y[i]/(xbin[i]-xA)
	haLim=haLim/M
	fAsympt=1+R/2.*haLim	# asymptotic limit of F(a)
	print('Asymptotic value of F(a): %3.3f'%fAsympt)

	# step (5): Find n-2 roots of F(a) between zeros of g(a)
	# Remember: these are always unacceptable
	FaZero=np.zeros(n-1)	# n-1 solutions of F(a)=0
	lambdaZero=np.zeros(n-1)# corresponding value of lambda
	for i in range(n-2): # n-2 roots between zeros of g(a)
		resF=root_scalar(fa,bracket=[gaZero[i]+Eps,gaZero[i+1]-Eps],args=data,method="brentq")
		FaZero[i]=resF.root  # best-fit "a" parameter
		print("%d-th F(a) root found at %f (between %f and %f), lambda=%3.4f"%(i,FaZero[i],gaZero[i],gaZero[i+1],lambdaZero[i]))
		if(hasGap==0):
			lambdaZero[i]=bestFitLnum(FaZero[i],data)
		if(hasGap>=1):
			lambdaZero[i]=bestFitLnumGap(FaZero[i],data,dataGap)	
	# step (6): Find the remaining zero of F(a) +++++++
	# This zero is to left/right of first/last zero of g(a). 
	# Note: This is only zero for M=2 
	INF=10	# this is a large number that takes the place of infinity
	if fAsympt<0: # zero is to the right
		resF=root_scalar(fa,bracket=[gaZero[n-2]+Eps,+INF],args=data,method="brentq")
		FaZero[n-2]=resF.root
		print("%d-th F(a) root found at %f (between %f and %f)"%(n-2,FaZero[n-2],gaZero[n-2],+INF))
	if fAsympt>0: # zero is to the left
		resF=root_scalar(fa,bracket=[-INF,gaZero[0]-Eps],args=data,method="brentq")
		FaZero[n-2]=resF.root
		print("%d-th F(a) root found at %f (between %f and %f)"%(n-2,FaZero[n-2],-INF,gaZero[0]))
	# after calculating the correct value of a, calculate Lambda(a)
	if(hasGap==0):
		lambdaZero[n-2]=bestFitLnum(FaZero[n-2],data)
	if(hasGap>=1):
		lambdaZero[n-2]=bestFitLnumGap(FaZero[n-2],data,dataGap)
	print("Lambda=%3.4f"%lambdaZero[n-2])
# step (7): Check that the solution is acceptable, i.e., model is non-negative.
# Also calculate C statistic +++++++++++++++++++++++++++++++++++++++++++++++++
for i in range(n-1):
	# It is redundant to check for the first n-2 models, but do it anyways
	ymodelAcceptable[i]=isModelAcceptable(FaZero[i],data)
	if (ymodelAcceptable[i]==1):
		cmin[i],cminplot[i]=cminLinear(lambdaZero[i],FaZero[i],data)
		print('%d-th solution acceptable: Cmin=%3.3f'%(i,cmin[i]))
	if (ymodelAcceptable[i]==0):
		print('%d-th solution not acceptable'%i)

#======================= PLOTTING ==========================
# 1. Plot the function F(a) and all its zeros ==============
# ==========================================================
# Parameters to control behavior of the plot
plotRangeUnacceptable=1	# shaded area showing range of unacceptable solutions
# ==========================================================
Scale=3.5  			# sets scale of y axis
# NOTE: range of plots should be in a parameter file
# make sure to include all values for the plot
aMin=1.2*FaZero[0]
aMax=1.2*FaZero[n-2]
a=np.arange(aMin,aMax,0.0001) 	# range of x axis
#a=np.arange(-.5,0.0,0.0001)
plota=np.zeros(len(a))
plotga=np.zeros(len(a))
for i in range(len(a)):
	plota[i]=fa(a[i],*data)
	plotga[i]=ga(a[i],*data)

fig,ax=plt.subplots(figsize=(8,6))
plt.xlim(aMin,aMax)
plt.plot(a,plota,color='green',linewidth=2,label='Function $F(a)$')
if (M>=2):
	#plt.xlim(min(a),max(a))
	ym=fAsympt-Scale*abs(fAsympt)
	yM=fAsympt+Scale*abs(fAsympt)
	plt.ylim(ym,yM)
	plt.scatter(FaZero,np.zeros(n-1),color='green',marker='o',label='Solution $F(a)=0$')
	plt.hlines(fAsympt,aMin,aMax,linestyle='--',linewidth=2,color='green',label='Asymptotic value')

plt.hlines(0,min(a),max(a),linestyle='--',color='black')
# line below to have the range of unacceptable solutions for F(a)=0
if plotRangeUnacceptable==1:
	plt.fill([-2/Deltax[0],-2/Deltax[0],-1/(R-Deltax[N-1]/2),-1/(R-Deltax[N-1]/2)],[-1000,1000,1000,-1000],facecolor="none",alpha=0.7,hatch='//',edgecolor='black',label='Unacceptable solutions')
plt.xlabel('Parameter $a$')
plt.ylabel('Function $F(a)$')
plt.legend(loc=4,prop={'size': 10})
plt.savefig('Fa.pdf')
#plt.show()
# === End of function F(a)= ===============================

# ==================================================
# 2. Plot the function  g(a) and all its zeros =====
# ==================================================
# reset the x and y limits of plot
fig,ax=plt.subplots(figsize=(8,6))
aMin=1.2*min(gaSing)
aMax=0.5*max(gaSing)

a=np.arange(aMin,aMax,0.0001) # range of x axis
plota=np.zeros(len(a))
plotga=np.zeros(len(a))
for i in range(len(a)):
	plota[i]=fa(a[i],*data)
	plotga[i]=ga(a[i],*data)

plt.ylim(-100,100)
#plt.xlim(min(a),max(a))
plt.xlim(aMin,aMax)
plt.plot(a,plotga,color='black',linewidth=2,label='Function g(a)')
plt.scatter(gaSing,[0]*len(gaSing),marker='x',color='green',label='Singularity of g(a)')
plt.scatter(gaZero,np.zeros(n-1),marker='o',color="blue",label='Solution $g(a)=0$')
plt.hlines(0,min(a),max(a),linestyle='--',color='black')
plt.legend(loc=2,prop={'size': 10})
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Parameter $a$')
plt.ylabel('Function $g(a)$')
#plt.show()
plt.savefig('ga.pdf')
# === End of function g(a) ==============================


# =======================================================
# 3. Plot the data and best-fit model(s) ================
# Make a PDF plot for the book
fig,ax=plt.subplots(figsize=(8,6))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0

# Parameters to control the behavior of the plots
rescale=0		# rescales the models for visibility
plotUnacceptable=0	# plot the negative (unacceptable) models too
plotExtended=0		# plot also the extended models (pivoted and constant)
plotallticks=0  	# write all ticks (for data with few bins)
plotStepModel=1		# plots the step-wise Poisson mean model
# =======================================================

# Calculate the pivotA, pivotB and Constant model ======================
if hasGap==0:
	bestFitLambdaA,ymodelPivotA,fmodelPivotA=flinearPivotA(data) # this is the model pivoted at point xA
	bestFitLambdaB,ymodelPivotB,fmodelPivotB=flinearPivotB(data) # this is the model pivoted at point xB
	bestFitLambdaC,ymodelConst,fmodelConst=flinearConst(data)
if hasGap>=1:
	bestFitLambdaA,ymodelPivotA,fmodelPivotA=flinearPivotAGap(data,dataGap) # this is the model pivoted at point xA
	bestFitLambdaB,ymodelPivotB,fmodelPivotB=flinearPivotBGap(data,dataGap) # this is the model pivoted at point xB
	bestFitLambdaC,ymodelConst,fmodelConst=flinearConstGap(data,dataGap)	# this is the constant model

cminplotPivotA=cstat(ymodelPivotA,data)
cminplotPivotB=cstat(ymodelPivotB,data)
cminplotConst=cstat(ymodelConst,data)
# package the best-fit model values in a list
bestFitParameterName=[['a','lambda'],'lambdaA','lambdaB','lambdaC']
bestFitParameter=[["{0:.5f}".format(FaZero[n-2]),"{0:.5f}".format(lambdaZero[n-2])],"{0:.5f}".format(bestFitLambdaA),"{0:.5f}".format(bestFitLambdaB),"{0:.5f}".format(bestFitLambdaC)]
for j in range(N):
	if rescale==1:
		ymodelPivotA[j]*=10
		ymodelPivotB[j]*=10
		ymodelConst[j]*=10
		fmodelPivotA[j]*=10
		fmodelPivotB[j]*=10
		fmodelConst[j]*=10
	# now calculate y errors for plots ==============================
	if(j>0):
		ymodelerrPivotA[j-1]=ymodelPivotA[j]-ymodelPivotA[j-1] 
		ymodelerrPivotB[j-1]=ymodelPivotB[j]-ymodelPivotB[j-1]
		ymodelerrConst[j-1]=ymodelConst[j]-ymodelConst[j-1]
	if(hasGap>=1): # no ymodelerr where there is a gap
		# check at the end of each bin if there is an xA
		# which indicates the beginning of a gap
		if np.any(xbin[j-1]+Deltax[j-1]/2==xa):
			ymodelerrPivotA[j-1]=0
			ymodelerrPivotB[j-1]=0	
			ymodelerrConst[j-1]=0
	# ===============================================================
for i in range(n-1): # n-1 models, one for each (a, lambda) solution
	for j in range(N):
		ymodel[i][j]=flinear(xbin[j],xA,lambdaZero[i],FaZero[i])*Deltax[j]
		# let's also make a ymodelerr for plots =================
		if(j>0):
			ymodelerr[i][j-1]=ymodel[i][j]-ymodel[i][j-1] # difference between consecutive values
		if(hasGap>=1): # no ymodelerr where there is a gap
			if np.any(xbin[j-1]+Deltax[j-1]/2==xa):
				ymodelerr[i][j-1]=0	
				print("Here: no y error at x=",xbin[j-1]+Deltax[j-1]/2)
				print("xa=",xa)
		# =======================================================
		fmodel[i][j]=flinear(xbin[j],xA,lambdaZero[i],FaZero[i])
		if rescale==1:
			ymodel[i][j]*=10.
			fmodel[i][j]*=10.
	if(ymodelAcceptable[i]==1):
		# also plot the error bars for y(x_i) model
		plt.plot(xbin,fmodel[i],color='black',linewidth=2,linestyle='-',label='$C$ stat Linear Model')
		if plotStepModel==1:
			plt.errorbar(xbin,ymodel[i],xerr=Deltax/2,fmt='none',color='black',linewidth=2)	
			plt.errorbar(xbin+Deltax/2,ymodel[i]+ymodelerr[i]/2,yerr=ymodelerr[i]/2,fmt='none',color='black',linewidth=2)
	if(ymodelAcceptable[i]==0)and(i==0)and(plotUnacceptable==1):
		plt.plot(xbin,fmodel[i],color='blue',linestyle='--',linewidth=2,label='Unacceptable Model (%d)'%i)
	if(ymodelAcceptable[i]==0)and(i==n-2)and(plotUnacceptable==1):
		plt.plot(xbin,fmodel[i],color='black',linestyle='--',linewidth=2,label='Unacceptable Model (%d)'%i)
	if(ymodelAcceptable[i]==0)and(i>0)and(plotUnacceptable==1):
		plt.plot(xbin,fmodel[i],color='red',linewidth=2,linestyle='-.')
	if(i==n-2)and(plotExtended==1): # plot other models only at the last iteration, when linear model *may* be acceptable
		plt.plot(xbin,fmodelPivotA,color='blue',linestyle='-.',linewidth=2,label='Pivoted Model A')
		plt.plot(xbin,fmodelPivotB,color='green',linestyle='-.',linewidth=2,label='Pivoted Model B')
		plt.plot(xbin,fmodelConst,color='red',linestyle='-.',linewidth=2,label='Constant Model')
		if plotStepModel==1:
			plt.errorbar(xbin,ymodelPivotA,xerr=Deltax/2,fmt='none',color='blue',linewidth=2)
			plt.errorbar(xbin+Deltax/2,ymodelPivotA+ymodelerrPivotA/2,yerr=ymodelerrPivotA/2,fmt='none',color='blue',linewidth=2)
			plt.errorbar(xbin,ymodelPivotB,xerr=Deltax/2,fmt='none',color='green',linewidth=2)
			plt.errorbar(xbin+Deltax/2,ymodelPivotB+ymodelerrPivotB/2,yerr=ymodelerrPivotB/2,fmt='none',color='green',linewidth=2)
			plt.errorbar(xbin,ymodelConst,xerr=Deltax/2,fmt='none',color='red',linewidth=2)
			plt.errorbar(xbin+Deltax/2,ymodelConst+ymodelerrConst/2,yerr=ymodelerrConst/2,fmt='none',color='red',linewidth=2)

		
print(fmodelPivotA,fmodelPivotB)
print(xbin,ymodel[n-2],'xA=',xA,'lambda=',lambdaZero[n-2],'a=',FaZero[n-2])
plt.scatter(xbin,y,color='black',marker='o',label='Data')
#plt.axhline(0.0,color='red')
#plt.yticks([0,1],("0","1"))
plt.ylim(-0.10,max(max(fmodel[n-2]),max(y))+0.2)
plt.xlim(xbin[0]-Deltax[0]/2,xbin[N-1]+Deltax[N-1]/2)
# use line below if there are only a few data points
if plotallticks==1:
	plt.xticks(xbin,fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
#plt.hlines(0,0,100,linestyle='--')
ax.yaxis.set_major_locator(MultipleLocator(2.0))
ax.yaxis.set_minor_locator(MultipleLocator(1.0))
#ax.yaxis.set_major_locator(MultipleLocator(1.0))
#ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1.0))
plt.grid(which='both')
#plt.legend(loc=1,prop={'size': 16})
# === End of data and best-fit model =======================

# ========== SUMMARY OF FIT ==========================
# These are the C statistics
# The only model that can be acceptable is the (n-1)-th
Cstat=sum(cminplot[n-2])   # "original" model
CstatA=sum(cminplotPivotA) # "pivotA" model
CstatB=sum(cminplotPivotB) # "pivotB" model
CstatC=sum(cminplotConst)  # constant model
# get the critical value of Cstat=Cmin, assuming no free parameters-----
yxi=flinear(xbin,xA,lambdaZero[i],FaZero[i])
print('E',E)
CCrit=ccritApprox(yxi,1.3)
print('*** Critical value of C (for Cmin=%3.4f) at 0.9 CL: %3.4f)'%(Cstat,CCrit))
# -----------------------------------------------------------------------
name=['original','pivotA','pivotB','constant']
cstat=[Cstat,CstatA,CstatB,CstatC]
indexSort=np.argsort(cstat)

print('======= Summary of fit ===================')
# step (9): determine which model is most accurate
print('Cstats of all linear models sorted in ascending order (best- to worst-fitting):')
for i in range(4):
	print(i,'Model "',name[indexSort[i]],'", with parameter(s)',bestFitParameterName[indexSort[i]],'=',bestFitParameter[indexSort[i]],', and Cstat=',cstat[indexSort[i]],end = '')
	if name[indexSort[i]]=='original':
		if(ymodelAcceptable[n-2]==1):
			print(' (Model is non--negative and acceptable)')
		if(ymodelAcceptable[n-2]==0):
			print(' (Model is not acceptable)')
	else:
		print(' ')
print('==========================================')
# === End of Summary ================================	

# ==============================================================
# Testing the routines for variances and covariance
# 1 - deltaa
aHat=FaZero[n-2]
lambdaHat=lambdaZero[n-2]
betaHat=aHat*lambdaHat
# These Delta C=1 below are wrong - to be deleted
'''
Deltaa=deltaa(lambdaHat,aHat,data)
print('Estimated sigmaa=%3.4e'%Deltaa)
CStatNew,_=cminLinear(lambdaHat,aHat+Deltaa,data)
print('Cstat for this value of ahat+deltaa',CStatNew)
# 2- deltaLambda
DeltaLambda=deltalambda(lambdaHat,aHat,data)
print('Estimated sigmaLambda=%3.4e'%DeltaLambda)
CStatNew,_=cminLinear(lambdaHat+DeltaLambda,aHat,data)
print('Cstat for this value of LambdaHat+deltaLambda',CStatNew)
# 3 - deltabeta for the overall slope parameter
#Deltabeta=deltabeta(lambdaZero[n-2],FaZero[n-2],data)
# 4 - Covariance and correlation coefficient between a and Lambda
Cov=covALambda(aHat,lambdaHat,data)
print('Estimated covariance between a and lambda: %3.4f'%Cov)
# --------------------------------------------------
# Also estimate the overall slope beta and its error
Deltabeta=deltabeta(lambdaHat,aHat,data)
print('Estimated beta=%3.2f, sigmabeta=%3.2f'%(betaHat,Deltabeta))
# lambda is kept constant at lambdahat, but aHat is transformed to:
#newa=aHat+(1+aHat)*Deltabeta/(aHat*lambdaHat)
betaHat=aHat*lambdaHat
newa=(betaHat+Deltabeta)/lambdaHat
CStatNew,_=cminLinear(lambdaHat,newa,data)
print('Cstat for this value of betaHat+deltaBeta',CStatNew)
'''
# Test also the covariance matrix from the direct method
CovMat=covMatrixErrProp(lambdaHat, aHat,data)
print('Covariance Matrix from error propagation',CovMat)
print('This means: lambda=%3.4f+-%3.4f, a=%3.4f+-%3.4f, cov=%3.4f'%
        (lambdaHat,CovMat[0][0]**0.5,aHat,CovMat[1][1]**0.5,CovMat[0][1]))
# test the covariance matrix from Cameron-Trivedi
CovMatCT=covMatrixCT(lambdaHat, aHat,data)
print('Covariance Matrix from direct method (Cameron-Trivedi) ',CovMatCT)
print('This means: lambda=%3.4f+-%3.4f, a=%3.4f+-%3.4f, cov=%3.4f'%
    (lambdaHat,CovMatCT[0][0]**0.5,aHat,CovMatCT[1][1]**0.5,CovMatCT[0][1]))
DeltaaCT=CovMatCT[1][1]**0.5
DeltaLambdaCT=CovMatCT[0][0]**0.5
covCT=CovMatCT[0][1]
# add the error in a*lambda from error prop method
slopeCTErr=aHat*lambdaHat*( (DeltaaCT/aHat)**2+(DeltaLambdaCT/lambdaHat)**2 + 2*covCT/(aHat*lambdaHat))**0.5
print('Overall slope a*lambda=%3.4f+-%3.4f'%(aHat*lambdaHat,slopeCTErr))
# ================================================================
# add a panel with best-fit parameters and covariances, for proposal
# Choose: (a) for full linear model
plotChoice=0 
if plotChoice==0:
    textstr = '\n'.join((
    r'$\mathrm{Fit\, Parameters}$',
    r'$a=%.2f \pm %.2f$' % (aHat,DeltaaCT ),
    r'$\lambda=%.2f \pm %.2f$' % (lambdaHat,DeltaLambdaCT),
    #r'$(\beta=%.3f \pm %.3f)$' % (betaHat,Deltabeta ),
    r'$C_{\mathrm{min}}=%.2f$' % (Cstat )))
# or (b) for pivotA model, for which we need the error
if plotChoice==1:
    aHatA=0 
    DeltaLambdaA=deltalambda(bestFitLambdaA,aHatA,data)
    textstr = '\n'.join((
    r'$\mathrm{Fit\, Parameters}$',
    r'$\lambda=%.2f \pm %.2f$' % (bestFitLambdaA,DeltaLambdaA ),
    r'$C_{\mathrm{min}}=%.2f$' % (CstatA )))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.05, 0.90, textstr, transform=ax.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)

# ============================================================
# Adding Gaussian least squares fit, just for comparison
# This is for the NSF proposal
# OLS, make sure that the bin-size is properly accounted
yLS=y/Deltax # now bin-sizes with size 2 will be reduced
print(xbin,Deltax,y)
results= linregress(xbin,yLS)
slope=results.slope
intercept=results.intercept
r_value=results.rvalue
p_value=results.pvalue
std_err=results.stderr
std_err_int=results.intercept_stderr
sigma2Hat=(1/(len(y)-2))*sum((y-(intercept+slope*xbin))**2)
# decide whether to also do chi2 fit
dochi2=0
if dochi2==1:
    print('Also doing chi^2 statistic. Warning: if data have 0 counts it will fail')
    # just like for LS, modify y and error to account for bin-size
    yChi=yLS # same as for LS, this is a density
    yErrChi=y**0.5/Deltax # this is taking yChi=y/Deltax as the density 
    resultsWeights=curve_fit(linear,xbin,yChi,sigma=yErrChi,absolute_sigma=True)
    print(resultsWeights)
    slopeW=resultsWeights[0][1]
    interceptW=resultsWeights[0][0]
    varSlope=resultsWeights[1][1,1]
    slopeWErr=varSlope**0.5
    varIntercept=resultsWeights[1][0,0]
    interceptWErr=varIntercept**0.5
    covW=resultsWeights[1][1,0]

print("Using ordinary least-squares")
print("slope: %f+-%f    intercept: %f+-%f" % (slope, std_err,intercept,std_err_int))
print('Correlation r=%3.2f, p=%3.6f'%(r_value,p_value))
print('Problem 14.3')
print("r=%3.4f, R-squared: %f" % (r_value,r_value**2))
print("Estimated data variance sigmaHat=%3.2f"%sigma2Hat)
print("============")

# Also check my calculations of the errors and t statistic
aLocal,bLocal,Vara,Varb,Covab,tStat,tCrit,pVal=OLSStats(intercept,slope,xbin,yLS)
print("My OLS calculations: a=%3.5f+-%3.5f; b=%3.5f+-%3.5f, Covab=%3.4f,t=%3.2f,tCrit=%3.2f, pVal=%3.6f"%
        (aLocal,Vara**0.5,bLocal,Varb**0.5,Covab,tStat,tCrit,pVal))
print("============")
ymodel=np.zeros(2)
xMin=xbin[0]-Deltax[0]/2
xMax=xbin[N-1]+Deltax[N-1]/2
xPlot=np.asarray([xMin,xMax])
ymodel=intercept+xPlot*slope
plt.plot(xPlot,ymodel,color='blue',linewidth=2,label='Least-squares fit')

if dochi2==1:
    ymodelW=interceptW+xPlot*slopeW
    ymodelWErr=(varIntercept+varSlope*xPlot**2+2*covW*xPlot)**0.5
    print('Using weighted chi2 fit:')
    print("slope: %f+-%f    intercept: %f+-%f, covW=%3.4f" % (slopeW, slopeWErr,interceptW,interceptWErr,covW))
    ymodW=interceptW+slopeW*xbin
    chiSq=chi2(yChi,ymodW,yErrChi)
    print('for chi2:',y,ymodW,y**0.5)
    print('chi^2: %3.3f'%chiSq)
    plt.plot(xPlot,ymodelW,color='red',linewidth=2,label='$\chi^2$ fit ($\chi^2_{\mathrm{min}}=%3.2f$)'%chiSq)
    plt.errorbar(xbin,yChi,yerr=yErrChi,color='red',linestyle=' ',capsize=3)
    plt.fill_between(xPlot,ymodelW-ymodelWErr,ymodelW+ymodelWErr,color='none',hatch='-',edgecolor='red',zorder=0)

# also the least-squares regression to one-parameter pegged model
slopePivoted, slopeVariance=curve_fit(linearPivotedA,xbin,y)
ymodelPivoted=np.zeros(2),
ymodelPivoted=xPlot*slopePivoted
#plt.plot(xPlot,ymodelPivoted)
if plotChoice==1:
    plt.plot(xPlot,ymodelPivoted,color='red',linewidth=2,linestyle='dashed',label='Least-squares fit (pivoted)')

# also the C-stat pivoted model all the way to the edges
ymodelPivotedA=np.zeros(2)
slopePivotedA=float(bestFitParameter[1])

if plotChoice==1:
    print('pivoted A slope:',slopePivotedA) # this is the slope of the pivotedA model
    ymodelPivotedA=xPlot*slopePivotedA
    plt.plot(xPlot,ymodelPivotedA,color='black',linewidth=2,linestyle='solid',label='Linear model ($C$ stat.)')


# re-plot the Cstat linear model, all the way to the edges
doBand=1
if plotChoice==0:
    print('best-fit Cstat parameters',float(bestFitParameter[0][0]),float(bestFitParameter[0][1]))
    ymodelC=np.zeros(2)
    ymodelC=flinear(xPlot,xA,lambdaHat,aHat)#float(bestFitParameter[0][1]),float(bestFitParameter[0][0]))
    plt.plot(xPlot,ymodelC,linewidth=2,color='black')
    # OPTIONALLY ADD THE CONFIDENCE BAND
    if doBand==1:
        nBand=21
        xBand=np.linspace(xMin,xMax,nBand)
        yBand=np.zeros(nBand)
        yModelBand=flinear(xBand,xA,float(bestFitParameter[0][1]),float(bestFitParameter[0][0]))
        yBandErr=confidenceBand(xBand,lambdaHat,aHat,data)
        #plt.plot(xBand,yModelBand+yBandErr)
        print('yBandErr',yBandErr)
        plt.fill_between(xBand,yModelBand+yBandErr,yModelBand-yBandErr,color='none',hatch="///",edgecolor='black',zorder=0,alpha=0.6)
        #plt.plot(xBand,yModelBand,linestyle='dashed')
        # Also add a band for the OLS
        yModelBandOLS=np.zeros(nBand)
        yBandErrOLS=confidenceBandOLS(xBand,aLocal,bLocal,Vara,Varb,Covab)
        yModelBandOLS=aLocal+bLocal*xBand
        plt.fill_between(xBand,yModelBandOLS+yBandErrOLS,yModelBandOLS-yBandErrOLS,color='none',hatch="\\\\",edgecolor='blue',zorder=0,alpha=0.6)

plt.plot(xPlot,(0,0),linewidth=2,linestyle='dotted',color='black')
plt.legend(loc=(0.37,0.68),prop={'size': 16})
#plt.ylim(-2,36)
plt.ylim(-1,13.5)
#plt.ylim(-0.2,5.2)
plt.savefig('bestFitModels.pdf')
# ==========================================================
# 4. Plot the cstat
fig,ax=plt.subplots(figsize=(8,6))
# step (8): calculate the C statistics and determine
# which model (original/pivotA/pivotB/constant)
# provides the lowest Cstat ++++++++++++++++++++++++


plt.scatter(xbin,cminplot[n-2],label='Model')
if (plotExtended==1):
	plt.scatter(xbin,cminplotPivotA,color='blue',label='Pivoted Model A')
	plt.scatter(xbin,cminplotPivotB,color='green',label='Pivoted Model B')
	plt.scatter(xbin,cminplotConst,color='orange',label='Constant Model B')

plt.xlim(xbin[0]-Deltax[0]/2,xbin[N-1]+Deltax[N-1]/2)
if plotallticks==1:
	plt.xticks(xbin,fontsize=8)
plt.xlabel('Bin')
plt.ylabel('Contribution to $C$ statistic')
plt.savefig('CContribPlot.pdf')
# === End of plot of cstat ===========================

