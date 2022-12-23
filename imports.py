import math
import numpy as np
import matplotlib.ticker as tick
import scipy.stats as stats
#from kaastra import Ce
from matplotlib import pyplot as plt
from scipy.stats import expon, binom, chi2, norm, poisson, lognorm, f, t, rdist, beta, kstest, ks_2samp, ksone, kstwo, kstwobign, uniform, linregress, pearsonr
from scipy.optimize import curve_fit, fsolve
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter,LogLocator
import matplotlib.ticker as ticker
from random import choices
from scipy.optimize import minimize 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib import rcParams

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 2.0

# xmin and xmax fill the CDF to the min and max values with 0 and 1
def sampleCDF(x,xmin,xmax):
	xOrdered=np.sort(x)
	xUnique,countsUnique=np.unique(xOrdered,return_counts=True)
	nUnique=len(xUnique) # there will be nUnique-1 steps
	tot=np.cumsum(countsUnique)
	totCDF=tot/tot[-1] # normalization
	# now pad the arrays with the min and max values for a nice plot

	xUniquePad=np.zeros(nUnique+2)
	totCDFPad=np.zeros(nUnique+2)
	xUniquePad[0]=xmin
	xUniquePad[1:nUnique+1]=xUnique[0:nUnique]
	xUniquePad[nUnique+1]=xmax
	totCDFPad[0]=0
	totCDFPad[1:nUnique+1]=totCDF[0:nUnique]
	totCDFPad[nUnique+1]=1
	return xUniquePad,totCDFPad	

def CContribwTwo(Ni,mu):
    if (Ni==0):
        return 2*mu
    if (Ni>0):
        return 2*(mu - Ni +Ni*np.log(Ni/mu)) # factor of 2 included here

def linear(x,a,b):
    return a+b*x

def constant(x,a):
	return a+0*x

# constant model with an additional term for the first bin
def constantPlusOne(x,a,b):
	Sizex=len(x)
	model=np.zeros(Sizex)
	# first bin, where the line would be
	model[0] = a+b
	# all other bins
	for j in range(Sizex-1):
		model[j+1]=a
	return model

def chisquared(y,yerr,ymod):
    N=len(y)
    result=0
    for i in range(N):
        result+=((y[i]-ymod[i])/yerr[i])**2
    return result

# This is the bivariate chi2 for linear model
def chisquaredBivLin(x, *args):
    a,b=x
    x,y,xerr,yerr=args
    #print('inside chisquaredBinLin',x,y,xerr,yerr)
    N=len(y)
    #print('Using %d datapoints'%N)
    result=0
    for i in range(N):
        contrib=(y[i]-a-b*x[i])**2/(yerr[i]**2 +(b**2)*xerr[i]**2)
        #print('x=%3.2f+-%3.2f, y=%3.2f+-%3.2f,a=%3.2f, b=%3.2f, %3.3f'%
        #        (x[i],xerr[i],y[i],yerr[i],a,b,contrib))
        result=result+contrib
    return result

def modelVariance(y,ymodel,dof):
    N=len(y)
    result=0
    for i in range(N):
        result+=(y[i]-ymodel[i])**2
    result=result/(N-dof)
    return result

def intrinsicScatter(y,ymodel,yerr,dof):
    result=0
    N=len(y)
    for i in range(N):
            result+=(y[i]-ymodel[i])**2/(N-dof)-yerr[i]**2/N
    return result**0.5

def Stirling(n):
    return (2*np.pi*n)**0.5*(n/np.e)**n

# Implements a batch variance with an=bn=sqrt(n)
# This BM variance needs to be divided by n to have variance of mean
def VarBM(x):
	n=len(x) #hopefully is the square of a number
	an=int(n**0.5)
	#an=20 #(test)
	bn=int(n/an)
	print('an=%d, bn=%d'%(an,bn))
	# In Flegal+20, Ykbar is the mean of the deviations
	ykMean=np.zeros(bn)
	ynMean=np.mean(x) # mean of whole array
	result=0
	for k in range(an):
		# mean of y over the k-th batch
		ykMean[k]=np.mean(x[k*bn:(k+1)*bn])	
	# Now do the sample variance of the batch means
	s2BM=np.var(ykMean,ddof=1)
	# Multiply by bn 
	result=bn*s2BM
	return result

# Ci contribution to Cstat
def CContribwTwo(Ni,mu):
    if (Ni==0):
        return 2*mu
    if (Ni>0):
        return 2*(mu - Ni +Ni*np.log(Ni/mu)) # factor of 2 included here

def cstat(y,ymod):
    N=len(y)
    result=np.zeros(N)
    for i in range(N):
        result[i]=CContribwTwo(y[i],ymod[i]) # these are the Ci's
    return(sum(result))

# Functions that approximate the mean and variance of Ci 
# in broad and narrow ranges, to use as useful approximations

def funcECmin(x, a,b,c,d,e,beta,f,alpha):
    return (a+b*x+c*(x-d)**2.0)*np.exp(-alpha*x)+e*np.exp(-beta*x)+f
def funcVarC(x, a,b,c,d,e,f,g,h,alpha,beta,i):
    return  (a+b*x**2+c*(x-d)**2.0)*np.exp(-alpha*x) +  (e+f*x+g*(x-h)**2.0)*np.exp(-beta*x) +i

# Parameter values as reported in Bonamente20
parNamesE=['a','b','c','d','e','beta','f','alpha']
parNamesVar=['a','b','c','d','e','f','g','h','alpha','beta','i']

# These are the best-fit values for E[Ci] on full and narrow range,
# followed by Var[Ci] for full and narrow range
A=[0.065672, -0.56709, -2.4637, -3.1971]
B=[-6.9461,-2.7336,1.5109,1.5118]
C=[-8.0124,-2.3603,-1.5109,-1.5118]
D=[0.40165,0.52816,0.60509,0.79384]
E=[0.261037,0.33133,1.4761,1.9294]
F=[1.00512,1.0174,18.358,6.1740]
G=[0,0,0.87316e-3,22.360e-3]
H=[0,0,-0.08592,-7.2981]
I=[0,0,2.02343,2.08378]
alphaB=[5.5178,3.9375,0.62652,0.750315] # note change of name B- Bonamente
betaB=[0.34817,0.48446,7.8187,4.49654]

colors=['black','red','gray','orange']
# i=0: E[Ci] full range, i=1: E[Ci] narrow range
# i=2: Var(Ci) full range, i=3=Var(Ci) narrow range
descriptor=['E[$C_i$] (range $\mu$=0.01-100)','E[$C_i$] (range $\mu$=0.1-10)','Var($C_i$) (range $\mu$=0.01-100)','Var($C_i$) (range $\mu$=0.1-10)']

# Add a function to estimate the critical value, assuming C crit = E[C] + q Var(C)**0.5
# It assume a model y(x_i) for the parent means, its length is the number of points
# qProb is the q value that corresponds to probability, 
# q = 0.5, 1.3, 2.3 for, respectively a probability p = 0.68, 0.90, 0.99
def ccritApprox(yxi,qProb):
    N=len(yxi)
    EC=0
    VarC=0
    result=0
    # Somehow, E seems to be a reserved keyword here ... 
    for i in range(N):
        j=0 # full range for mean
        print(E[j])#,betaB[j],F[j],alphaB[j])
        ECi=funcECmin(yxi[i],A[j],B[j],C[j],D[j],E[j],betaB[j],F[j],alphaB[j])
        EC+=ECi
        j=2 # full range for variance
        VarCi=funcVarC(yxi[i],A[j],B[j],C[j],D[j],E[j],F[j],G[j],H[j],alphaB[j],betaB[j],I[j])
        VarC+=VarCi
    result=EC+qProb*VarC**0.5
    return result
