# ========================================================
# python code that provides the best-fit 
# maximum-linelihood solutions for linear models
# using the Cash statistic
# bestFitLinear(data,hasGap,dataGap) and
# bestFitLinearExtended(data,hasGap,dataGap)
#
# Author: M. Bonamente, 2019, 2020
# This is intended to be a function to find best-fit
# values of the Scargle+13 linear model, if it exists.
# This was taken from cstatLinear.py and adapted as a function.
# It has the option to skip the inner, un-acceptable zeros of
# the function F(a)=0, to save computational time.
# ========================================================
import numpy as np
from scipy.optimize import root_scalar
from cstatFunctions import cminLinear,isModelAcceptable,ga,fa,rangex,Rprime,bestFitLnum,bestFitLnumGap
from cstatFunctions import cstat,flinearPivotA,flinearPivotB,flinearConst
from cstatFunctions import flinearPivotAGap,flinearPivotBGap,flinearConstGap
#
# ========= Function to FIT THE DATA (steps 1-7) ==========
def bestFitLinear(data,hasGap,dataGap):
	# -----optionally skip inner zeros of F(a)---------
	nMin=0 # this is the default, do all zeros
	doInner=0 # this is a flag to skip inner zeros
	# -----------------------------
	xbin,y,xA,Deltax=data # unpack the data
	N=len(xbin)
	R=rangex(data)
	M=sum(y)	
	# Find how many non-zero bins we have =============
	ind=np.where(y>=1.0)    # find the non-zero indices 
	gaindex=ind[0]
	n=len(gaindex) # number of bins with non-zero counts
	if doInner==0: # skip n-2 zeros of F(a)
		nMin=n-2
	if n<=1:
		print("*** No solution available for n<=1")
		return 0,np.NaN,np.NaN,np.NaN
# Define the relevant arrays ==============================
	gaSing=np.zeros(n)
	gaZero=np.zeros(n-1)
	FaZero=np.zeros(n-1)
	lambdaZero=np.zeros(n-1)
	ymodelAcceptable=np.zeros(n-1,dtype=int)
	cmin=np.zeros(n-1)
	cminplot=np.zeros((n-1,N))
# step (1) ===============================================
	if hasGap>0:
		xa,xb,xG,RG=dataGap	
		Rp=Rprime(R,RG,xA,xG)
		R=Rp # overwrites R with R'
	if (M<=1): # case when M=2 but both in same bin, for example
		print("*** No solution available for M=1")
		return 0,np.NaN,np.NaN,np.NaN
# if we are here, M>=2 ====================================

# step (2): Find n points of singularity of g(a) ==========
	for i in range(n):
		gaSing[i]=-1/(xbin[gaindex[i]]-xA)
		#print("%d-th g(a) singularity at %3.3f (for x=%3.3f)"%(i,gaSing[i],xbin[gaindex[i]]))
# step (3): Find n-1 zeros of g(a) between points of singularity
# Only need the first and last zero of g(a), since ==========
# F(a)=0 solutions between zeros of g(a) are unacceptable ===
		E=1.e-10 # this is an Epsilon value
		# if doInner=0, only seeks first and last zero of g(a)
		if ((i>0)and(doInner==1)) or((i==1)or(i==n-1)and(doInner==0)): 
			resg=root_scalar(ga,bracket=[gaSing[i-1]+E,gaSing[i]-E],args=data,method="brentq")
			gaZero[i-1]=resg.root	
			#print("%d-th g(a) root found at %f (between %f and %f)"%(i,gaZero[i-1],gaSing[i-1],gaSing[i]))
# step (4): Find asymptotic value of F(a) ==================
	haLim=0.0
	for i in range(N):
		haLim+=-y[i]/(xbin[i]-xA)
	haLim=haLim/M
	fAsympt=1+R/2.*haLim
	print('Asymptotic value of F(a): %3.e'%fAsympt)
# step (5): Find n-2 roots of F(a) between zeros of g(a) ===
# ---- this can be skipped --------------------------------
	for i in range(nMin,n-2): # n-2 roots between zeros of g(a)
		resF=root_scalar(fa,bracket=[gaZero[i]+E,gaZero[i+1]-E],args=data,method="brentq")
		FaZero[i]=resF.root  # best-fit "a" parameter
		#print("%d-th F(a) root found at %f (between %f and %f), lambda=%3.4f"%(i,FaZero[i],gaZero[i],gaZero[i+1],lambdaZero[i]))
		if(hasGap==0):
			lambdaZero[i]=bestFitLnum(FaZero[i],data)
		if(hasGap>=1):
			lambdaZero[i]=bestFitLnumGap(FaZero[i],data,dataGap)	
#-------- -------------------------------------------------
# step (6): Find the external zero of F(a) ================
	INF=1e+6
# IMPORTANT NOTE =================================================================
# *** If fAsympt is sufficiently close to 0, the solution may be near infinity and 
# difficult to obtain numerically. Set an arbitrary EPS value 
# ================================================================================
	if fAsympt<0-E: # zero is to the right
		print('Seeking solution between gaZero[n-2]+E=%e and +INF=%e'%(gaZero[n-2]+E,+INF))
		print('f(a)=%e, f(b)=%e'%(fa(gaZero[n-2]+E,*data),fa(+INF,*data)))
		resF=root_scalar(fa,bracket=[gaZero[n-2]+E,+INF],args=data,method="brentq")
		FaZero[n-2]=resF.root
		print("%d-th F(a) root found at %e (between %e and %e)"%(n-2,FaZero[n-2],gaZero[n-2],+INF))
	if fAsympt>0+E: # zero is to the left
		print('Seeking solution between -INF=%e and gaZero[0]-E=%e'%(-INF,gaZero[0]-E))
		print('f(a)=%e, f(b)=%e'%(fa(-INF,*data),fa(gaZero[0]-E,*data)))
		resF=root_scalar(fa,bracket=[-INF,gaZero[0]-E],args=data,method="brentq")
		FaZero[n-2]=resF.root
		print("%d-th F(a) root found at %e (between %e and %e)"%(n-2,FaZero[n-2],-INF,gaZero[0]))
# IMPORTANT NOTE (cont'd) =========================================================
# Need to also take care of the case of -E < fAsympt < E 
	if (fAsympt>=-E)and(fAsympt<=E):
		FaZero[n-2]=INF # set to infinity, and lambda will be 0, thus zeroing out the model
# =================================================================================	
	# after calculating the correct value of a, calculate Lambda(a)
	if(hasGap==0):
		lambdaZero[n-2]=bestFitLnum(FaZero[n-2],data)
	if(hasGap>=1):
		lambdaZero[n-2]=bestFitLnumGap(FaZero[n-2],data,dataGap)
	print("Lambda=%3.4f (n-1=%d, nMin+1=%d)"%(lambdaZero[n-2],n-1,nMin+1))
# step (7): Check that the solution is acceptable =========
# Also calculate C statistic ==============================
	for i in range(nMin,n-1):
	# It is redundant to check for the first n-2 models, but can do it anyways
		ymodelAcceptable[i]=isModelAcceptable(FaZero[i],data)
		print('ymodelAcceptable[i]=',ymodelAcceptable[i])
		if (ymodelAcceptable[i]==1):
			cmin[i],cminplot[i]=cminLinear(lambdaZero[i],FaZero[i],data)
			print('%d-th solution acceptable: Cmin=%3.3f'%(i,cmin[i]))
		if (ymodelAcceptable[i]==0):
			print('%d-th solution not acceptable'%i)	
			cmin[i]=np.NaN
# ==============================================================
	return ymodelAcceptable[n-2],FaZero[n-2],lambdaZero[n-2],cmin[n-2]
# ==============================================================

# ==============================================================
# Also a similar function for the extended model ===============
def bestFitLinearExtended(data,hasGap,dataGap):
# Calculate the pivotA, pivotB and Constant model ======================
	xbin,y,xA,Deltax=data # unpack the data
	N=len(xbin)
	#R=rangex(data)
	#M=sum(y)

	cminplotPivotA=np.zeros(N)
	cminplotPivotB=np.zeros(N)
	cminplotConst=np.zeros(N)
	ymodelPivotA=np.zeros(N)
	ymodelPivotA=np.zeros(N)
	fmodelPivotA=np.zeros(N)
	ymodelPivotB=np.zeros(N)
	fmodelPivotB=np.zeros(N)
	ymodelConst=np.zeros(N)
	fmodelConst=np.zeros(N)
	if hasGap==0:
		bestFitLambdaA,ymodelPivotA,fmodelPivotA=flinearPivotA(data) # this is the model pivoted at point xA
		bestFitLambdaB,ymodelPivotB,fmodelPivotB=flinearPivotB(data) # this is the model pivoted at point xB
		bestFitLambdaC,ymodelConst,fmodelConst=flinearConst(data)
	if hasGap>=1:
		bestFitLambdaA,ymodelPivotA,fmodelPivotA=flinearPivotAGap(data,dataGap) 
		bestFitLambdaB,ymodelPivotB,fmodelPivotB=flinearPivotBGap(data,dataGap) 
		bestFitLambdaC,ymodelConst,fmodelConst=flinearConstGap(data,dataGap)

	cminplotPivotA=cstat(ymodelPivotA,data)
	cminplotPivotB=cstat(ymodelPivotB,data)
	cminplotConst=cstat(ymodelConst,data)
	CstatA=sum(cminplotPivotA) # "pivotA" model
	CstatB=sum(cminplotPivotB) # "pivotB" model
	CstatC=sum(cminplotConst)  # constant model
	# sort the Cstats in ascending order
	cstatExtended=[CstatA,CstatB,CstatC]
	name=[1,2,3] # 1: pivotA; 2: pivotB; 3: constant (0: standard)
	indexSort=np.argsort(cstatExtended)

	return cstatExtended[indexSort[0]],name[indexSort[0]]
