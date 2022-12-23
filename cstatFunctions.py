# ========================================================
# python functions that provides the best-fit 
# maximum-linelihood solutions for linear models
# using the Cash statistic
# Author: M. Bonamente, 2019, 2020
# ========================================================

import numpy as np
from scipy.stats import poisson
from scipy.optimize import fsolve, root_scalar

# ====================================================================
def sanityCheck(data):
	x,y,xA,Deltax=data # unpack the data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	#1. check that all arrays are of proper size
	Ny=len(y)
	NDeltax=len(Deltax)
	if (N!=Ny) or (N!=NDeltax):
		print('Error: Dimensions of x, y and Deltax not matching')
		print('N=',N,'Ny=',Ny,'NDeltax=',NDeltax)
		quit()
	#2. check that there is no overlap of data points
	L=0
	for i in range(N):
		L+=Deltax[i] # length of i=th bin
		# check that there is no overlap between adjacent bins
		if(i>0) and (x[i]-Deltax[i]/2.)<(x[i-1]+Deltax[i-1]/2.):
			print('Bin',i, 'and',i-1,'overlap',x[i]-Deltax[i]/2,x[i-1]+Deltax[i-1]/2)
			quit()
	if (L>R):
		print('Error: range R not consistent with Deltax')
		print('R=',R,'sum of bin size lengths=',L)			
		quit()

def findGap(data):
	# Modified to handle many gaps
	x,y,xA,Deltax=data # unpack the data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	gapFound=0
	xa=-1*np.ones(N) # up to N gaps possible
	xb=-1*np.ones(N)
	xG=-1*np.ones(N)
	j=0
	gapFound=0
	RG=np.zeros(N)
	SG=np.zeros(N)
	for i in range(N):
		if(i>0) and (x[i]-Deltax[i]/2.)>(x[i-1]+Deltax[i-1]/2.):
		# gap starts at x[i-1]+Deltax[i-1]/2.
		# and ends at x[i]-Deltax[i]/2.
			gapFound+=1
			xa[j]=x[i-1]+Deltax[i-1]/2.
			xb[j]=x[i]-Deltax[i]/2.
			RG[j]=(xb[j]-xa[j])
			xG[j]=0.5*(xb[j]+xa[j])
			SG[j]=RG[j]*(xG[j]-xA)
			print('===>>> Gap in data detected between %3.3f and %3.3f, RG=%3.3f'%(xa[j],xb[j],RG[j]))
			j+=1
	SGSum=np.sum(SG)
	print(gapFound,'gaps found')
	return gapFound,xa,xb,xG,RG,SGSum

def Rprime(R,RG,xA,xG):
	g=len(RG) # this is how many gaps we have
	SG=0
	RGap=0
	for i in range(g):
		SG+=RG[i]*(xG[i]-xA)	
		RGap+=RG[i] # total gap length
# THIS WAS THE CASE FOR ONE GAP ONLY
#	Rp=(R**2-2*RG*(xG-xA))/(R-RG)
	Rp=(R**2-2*SG)/(R-RGap)
	return Rp

def rangex(data):
	x,y,xA,Deltax=data # unpack the data
	N=len(x)
	#print('%d datapoints for rangex'%N)
	R=x[N-1]-x[0]+Deltax[0]/2.+Deltax[N-1]/2.
	return R

def fAsymptotic(data):
	x,y,xA,Deltax=data # unpack the data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	haLim=0.0
	for i in range(N):
		haLim+=-y[i]/(x[i]-xA)
	haLim=haLim/M
	fAsympt=1+R/2.*haLim
	return fAsympt

# ==================================================================
# This function needs to be re-written based on cstatLinear.py
# IT iS OBSOLETE NOW
def findZeros(data):
	x,y,xA,Deltax=data # unpack the data
        # derive the parameters needed for the formula
	N=len(x)
	R=rangex(data)
	M=sum(y)
	# 0. first some work on the data, find how many unique non-zero points we have
	ind=np.where(y>=1.0)    # find the non-zero indices 
	gaindex=ind[0]
	n=len(gaindex)          # number of unique bins with non-zero counts
	print("Number of unique non-zero bins:%d"%n)
	# if n=1 or 0, there are no solutions =================================
	if (n<=1):
		print('Exiting ...')
		return -1,-1
	# =====================================================================
        # 1. Find point of singularity and zeros of g(a) ======================
	gaSing=np.zeros(n)
	gaZero=np.zeros(n-1)
	for i in range(n):
		gaSing[i]=-1/(x[gaindex[i]]-xA)
		#print("%d-th g(a) singularity at %3.3f (for x=%3.3f)"%(i,gaSing[i],x[gaindex[i]]))
        # (1.5 find zeros of g(a) between singularities) ======================
		E=1.e-10 # this is an Epsilon value
		if i > 0:
			resg=root_scalar(ga,bracket=[gaSing[i-1]+E,gaSing[i]-E],args=data,method="brentq")
			gaZero[i-1]=resg.root
			#print("%d-th g(a) root found at %f (between %f and %f)"%(i,gaZero[i-1],gaSing[i-1],gaSing[i]))
        # 2. Find asymptotic value of F(a) ====================================
#	haLim=0.0
#	for i in range(N):
#		haLim+=-y[i]/(x[i]-xA)
#	haLim=haLim/M
#	fAsympt=1+R/2.*haLim
	fAsympt=fAsymptotic(data)
	print('Asymptotic value of F(a): %3.3e'%fAsympt)
	# This is unnecessary, since there roots are unacceptable ==============
        # 3. Find n-2 roots of F(a) between zeros of g(a)
        # =======================================================================
	FaZero=np.zeros(n-1)
	lambdaZero=np.zeros(n-1)
#	THIS IS COMMENTED OUT BECAUSE THOSE ZEROS ARE UNACCEPTABLE =============
#	for i in range(n-2): # n-2 roots between zeros of g(a)
#		# TRY making smaller EPS value here
#		resF=root_scalar(fa,bracket=[gaZero[i]+E/M,gaZero[i+1]-E/M],args=data,method="brentq")
#		FaZero[i]=resF.root  # best-fit "a" parameter
#		lambdaZero[i]=bestFitLnum(FaZero[i],data) # matching "lambda" parameter
#		print("Unacceptable %d-th F(a) root found at %f (between %f and %f), lambda=%3.4f"%(i,FaZero[i],gaZero[i],gaZero[i+1],lambdaZero[i]))
#	END OF COMMENT ==========================================================
        # 3.5 now find the remaining zero, ======================================
        # either to left or right of first/last zero of g(a). 
        # NOTE: This is only zero for M=2 =======================================
	INF=np.inf
	# Calculate following values at the extreme of the range ================
	a1=-2./Deltax
	a2=-1./(R-Deltax/2.)
	FLeft=fa(a1,*data)
	FRight=fa(a2,*data)
	print("a1=%3.3f, a2=%3.3f,ga(a1)=%3.3f,ga(a2)=%3.3f,Fa(a1)=%3.2f, Fa(a2)=%3.2f"%(a1,a2,ga(a1,*data),ga(a2,*data),FLeft,FRight))
	FaZero[n-2]=-999 	# reset it
	hasConverged=0
	#print(-2./Deltax, -1./(R-Deltax/2.))
	print("%d-th possible outer solution with fAsympt=%3.3f, FLeft=%3.3f and FRight=%3.3f"%(n-2,fAsympt,FLeft,FRight))
	# Ensure that there is a solution by checking F(-2/Deltax) or F(-1/(R-Deltax/2) 
	if ((fAsympt<0)and(FRight>0)) : # zero is guaranteed to be to the right
		resF=root_scalar(fa,bracket=[gaZero[n-2]+E,+INF],args=data,method="brentq")
		FaZero[n-2]=resF.root
		lambdaZero[n-2]=bestFitLnum(FaZero[n-2],data)
		print("Acceptable %d-th F(a) root found at %f (between %f and %f), lambda=%3.4f"%(n-2,FaZero[n-2],gaZero[n-2],+INF,lambdaZero[n-2]))
		hasConverged=1 # if code is here, there is an acceptable solution
	if ((fAsympt>0)and(FLeft<0)): # zero is to guaranteed to be to the left
		resF=root_scalar(fa,bracket=[-INF,gaZero[0]-E],args=data,method="brentq")
		FaZero[n-2]=resF.root
		lambdaZero[n-2]=bestFitLnum(FaZero[n-2],data)
		print("Acceptable %d-th F(a) root found at %f (between %f and %f), lambda=%3.4f"%(n-2,FaZero[n-2],-INF,gaZero[0],lambdaZero[n-2]))	
		hasConverged=1 # if code is here, there is an acceptable solution
# Finally, return the value of the only acceptable solution
	return hasConverged,FaZero[n-2]
# ===========================================================================

def cstat(ymodel,data):
	x,y,xA,Deltax=data # unpack the data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	cplot=np.zeros(N)
	for i in range(N):
		if(y[i]>0):
			cplot[i]=2.*(ymodel[i]-y[i]-y[i]*np.log(ymodel[i])+y[i]*np.log(y[i]))
		if(y[i]==0):
			cplot[i]=2.*ymodel[i]
	return cplot

def cminLinear(l,a,data):
	result=0.
	x,y,xA,Deltax=data # unpack the data
        # derive the parameters needed for the formula
	N=len(x)
	#R=rangex(data) I don't think that this is needed
	M=sum(y)
	cplot=np.zeros(N) # the contribution to Cmin of each point
	#print('a=%3.2f, lambda=%3.2f, R=%3.2f, xa=%3.2f, Deltax=%3.2f, '%(a,l,R,xA,Deltax))
	ymodel=np.zeros(N)
        # --------------------------------------------
	for i in range(N):
		# Remember the Deltax to convert f(x) to ymodel
		ymodel[i]=flinear(x[i],xA,l,a)*Deltax[i]
		if(y[i]>0):
			cplot[i]=2.*(ymodel[i]-y[i]-y[i]*np.log(ymodel[i])+y[i]*np.log(y[i]))
			result+=cplot[i]
		if(y[i]==0):
			cplot[i]=2*ymodel[i]
			result+=cplot[i]
		#print('%d cplot[i]=%3.3f, y[i]=%d x[i]=%3.2f ymodel[i]=%3.3f'%(i,cplot[i],y[i],x[i],ymodel[i]))
	return result,cplot

# good old linear model
def linear(x,a,b):
    return a+b*x

def chi2(y,ymod,yerr):
    result=sum(((ymod-y)/yerr)**2)
    return result

def flinear(x,xA,l,a):
# this is the density function f(x), counts/unit length of the bin
        return l*(1.+a*(x-xA))

# ===================================================
# For the extended model with Pivot points at A and B

def flinearPivotA(data):
# this is the model with just one fixed parameter, with analytical solution 
	x,y,xA,Deltax=data # unpack the data
        # derive the parameters needed for the formula
	N=len(x)
	R=rangex(data)
	M=sum(y)
	bestFitL=2*M/R**2
	
# this is the model for y data, contains already Deltax!
	return bestFitL,bestFitL*(x-xA)*Deltax,bestFitL*(x-xA)

def flinearPivotAGap(data,dataGap):
# this is the model with just one fixed parameter, with analytical solution 
	x,y,xA,Deltax=data # unpack the data
	xa,xb,xG,RG=dataGap
        # derive the parameters needed for the formula
	N=len(x)
	R=rangex(data)
	M=sum(y)
	g=len(RG)
	SG=0
	for i in range(g):
		SG+=RG[i]*(xG[i]-xA)
		#RGap+=RG[i] # total gap length
# this was for one gap
#	bestFitL=2*M/(R**2-2*RG*(xG-xA))
	bestFitL=2*M/(R**2-2*SG)
# this is the model for y data, contains already Deltax!
	return bestFitL,bestFitL*(x-xA)*Deltax,bestFitL*(x-xA)

def flinearPivotB(data):
# this is the model with just one fixed parameter, with analytical solution 
	x,y,xA,Deltax=data # unpack the data
        # derive the parameters needed for the formula
	N=len(x)
	R=rangex(data)
	M=sum(y)
	bestFitL=2*M/R
# this is the model for y data, contains already Deltax!
	return bestFitL, bestFitL*(1.-(x-xA)/R)*Deltax,bestFitL*(1.-(x-xA)/R)

def flinearPivotBGap(data,dataGap):
	x,y,xA,Deltax=data # unpack the data
	xa,xb,xG,RG=dataGap
	N=len(x)
	R=rangex(data)
	xB=xA+R
	M=sum(y)
	g=len(RG)
	SB=0
	for i in range(g):
		SB+=(RG[i]/R)*(xB-xG[i])
                #RGap+=RG[i] # total gap length
# this was for just one gap
#	bestFitL=2*M/(R+2*(RG/R)*(xG-xB))
	bestFitL=2*M/(R-2*SB)
# this is the model for y data, contains already Deltax!
	return bestFitL, bestFitL*(1.-(x-xA)/R)*Deltax,bestFitL*(1.-(x-xA)/R)

def flinearConst(data):
# this is the model with just one fixed parameter, the costant
# as shown in the other paper, const = sample mean
	x,y,xA,Deltax=data # unpack the data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	bestFitL=M/R
	return bestFitL,bestFitL*np.ones(N)*Deltax,bestFitL*np.ones(N)

def flinearConstGap(data,dataGap):
# this is the model with just one fixed parameter, the costant
# as shown in the other paper, const = sample mean
	x,y,xA,Deltax=data # unpack the data
	xa,xb,xG,RG=dataGap
	N=len(x)
	R=rangex(data)
        # when there is a gap, R must account for the gap
	M=sum(y)
	RGap=sum(RG) # sum of all gap lengths
	bestFitL=M/(R-RGap)
	return bestFitL,bestFitL*np.ones(N)*Deltax,bestFitL*np.ones(N)
# =======================================================
# g(a) is the function whose zeros cause a singularity in f(a)
def ga(a, *data):
	x,y,xA,Deltax=data #unpack the data
        # derive the parameters needed for the formula
	N=len(x)
	R=rangex(data)
	M=sum(y)
        # --------------------------------------------
	result=0
	for i in range(len(x)):
		if y[i]==0:
			localga=0
		if y[i]>0:
			localga=y[i]*(x[i]-xA)/(1.+a*(x[i]-xA))
		result+=localga
		#print('%d, localga=%3.3f, y[i]=%3.3f, x[i]=%3.3f, xA=%3.3f,a=%3.3f'%(i,localga,y[i],x[i],xA,a))	
	return result
	
def fa(a, *data):
	x,y,xA,Deltax=data #unpack the data
        # derive the parameters needed for the formula
	N=len(x)
	R=rangex(data)
	M=sum(y)
        # --------------------------------------------
	result=0
	galocal=0
	galocal=ga(a,*data)	
	result=1.+a*R/2.-M*R/(2.*galocal)
	# if the dataset has all 0's, lambda=0, which makes a indeterminate. 
	# for M=0 the equation should return a=-2/R, which is fine, since the
	# best-fit model will be y=0 anyways.
	return result

# also its derivative, which is used for finding the zero of the function
def faprime(a,*data):
	x,y,xA,Deltax=data #unpack the data
        # derive the parameters needed for the formula
	N=len(x)
	R=rangex(data)
	M=sum(y)
        # --------------------------------------------
	result=0
	localSum1=0
	localSum2=0
	for i in range(len(x)):
		localSum1+=y[i]*(x[i]-xA)/(1.+a*(x[i]-xA))
		localSum2+=y[i]*(x[i]-xA)**2*(1.+a*(x[i]-xA))**(-2)
        # result is f(a) such that f(a)=0
	result=R/2.-(M*R/2.0)*localSum1**(-2)*localSum2
        # if the dataset has all 0's, lambda=0, which makes a indeterminate. 
        # for M=0 the equation should return a=-2/R, which is fine, since the
        # best-fit model will be y=0 anyways.
	return result

def bestFitLnum(a, data):
	x,y,xA,Deltax=data #unpack the data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	return M/(R*(1.+a*R/2.))

# If there is a gap, there is a different function lambda(a)
def bestFitLnumGap(a,data,dataGap):
	x,y,xA,Deltax=data #unpack the data
	xa,xb,xG,RG=dataGap
	R=rangex(data)
	M=sum(y)
	g=len(RG) # this is how many gaps we have
	RGap=0
	SG=0
	for i in range(g):
		SG+=RG[i]*(xG[i]-xA)
		RGap+=RG[i] # total gap length
# this was for one gap	
#	return M/(R*(1.+a*R/2.) - RG*(1+a*(xG-xA)))
	return M/(R*(1.+a*R/2.) -(RGap+a*SG))
# ===================================================
# ==== FUNCTIONS BELOW ARE UNUSED ===================
# These are the approximations for the small-a case
def bestFitL(data):
# R is range, x2 is final point of range, x is array of N measurements (binned)
	x,y,xA,Deltax=data
	N=len(x)
	R=rangex(data)
	result=0
	for i in range(len(x)):
                # use mid point of bin as x coordinate for the bin
		xbinlocal=x[i]-xA
		result+=y[i]*xbinlocal
	result=result*2./R**2
	return result

def bestFitA(data):
# l is the best-fit value of lambda, using prior function
	x,y,xA,Deltax=data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	l=bestFitL(data)
	return (2./R)*(M/(l*R)-1.)
# =======================================================

def isModelAcceptable(a,data):
	result=0
	x,y,xA,Deltax=data
	N=len(x)
	R=rangex(data)
	M=sum(y)
	if ((a<-2/Deltax[0]) or (a>-1/(R-Deltax[N-1]/2))):
		result=1
	return result
# ==================================================
# These functions come from the earlier paper
def funcECmin(x, a,b,c,d,e,f,alpha,beta):
        return (a+b*x+c*(x-d)**2.0)*np.exp(-alpha*x)+e*np.exp(-beta*x)+f

# function that calculates E[Cmin] for a linear model from n simulations

def funcVarC(x, a,b,c,d,e,f,g,h,i,alpha,beta):
        return  (a+b*x**2+c*(x-d)**2.0)*np.exp(-alpha*x) +  (e+f*x+g*(x-h)**2.0)*np.exp(-beta*x) +i


# =========================================================
# This function does all the work of simulating and finding 
# best-fit value for the linear model  ====================
def  ECminSimLinear(AParent, LambdaParent, dataSim):
	xbin,xA,Deltax,n=dataSim # unpack the data for the simulations
	# n is the number of simulated datasets
	N=len(xbin)
	mu=np.zeros(N)

	for i in range(N):
		mu[i]=flinear(xbin[i],xA,LambdaParent,AParent)*Deltax

	cminSim=np.zeros(n)
	cminSim.fill(np.nan)
	y=np.zeros(N) # simulated dataset
	aMLnum=np.zeros(n)
	aMLnum.fill(np.nan)
	lMLnum=np.zeros(n)
	lMLnum.fill(np.nan)
	cminplot=np.zeros(N)
	counterNan=0
	for j in range(n): # iterate over the n simulations
		for i in range(N):
			y[i]=poisson.rvs(mu[i],size=1) # generate Poisson sample
		data=(xbin,y,xA,Deltax)
		M=sum(y) # to check if there are all 0's
		# solve numerically for best-fit "a" parameter	
		# seems like a dataset with all 0's can't be handled by fsolve
		# the "Aparent" parameter is the best-guess for the solution
		# when there are all 0's, this should be -2/R. Need to work on this []
#		if(M==0):
#			aMLnum[j]=0
#			lMLnum[j]=0
#			cminSim[j]=0
		EPS=0.0
#	===========================================================
		if(M>=0): # this is the proper case
			# this is the function we need to write====
			hasAConverged,res=findZeros(data)
			# ==========================================
			#res=root_scalar(fa,x0=AParent+EPS,fprime=faprime,args=data,method='newton')
			# res=fsolve(fa,AParent,full_output=1,xtol=1.e-8,args=data)
			#hasAConverged=res[2] # this is for fsolve
			#hasAConverged=res.converged # this is for root_scalar
			if (hasAConverged!=1): # not converged: discard this dataset
				counterNan+=1
				#print(res.flag, M)
			if (hasAConverged==1): # converged
				#aMLnum[j]=res[0] # accept the solution of fsolve
				#aMLnum[j]=res.root # accept solution of root_scalar
				aMLnum[j]=res
				lMLnum[j]=bestFitLnum(aMLnum[j],data) # and calculate corresponding Lambda
# (c) calculate cmin for this simulated dataset ---------------
# it is possible that, for low S/N data, y(xi) is negative,
# making it such that Cmin=NaN. Try by resetting them
				cminSim[j],cminplot=cminLinear(lMLnum[j],aMLnum[j],data)
				#print('-')
				#print(j, y,cminSim[j])
				if(np.isnan(cminSim[j])): # there are NaN in Cmin
					aMLnum[j]=float('NaN')
					lMLnum[j]=float('NaN')
					counterNan+=1
#	==========================================================
	EcminSim=np.nanmean(cminSim)
# Standard deviation - should it be devided by sqrt(n) for error of mean?
	EcminStdev=np.nanstd(cminSim)#/n**0.5
# Calculate % of sets that have not converged or have negative best--fit model
	return cminSim,aMLnum,lMLnum,EcminSim,EcminStdev,counterNan
		
