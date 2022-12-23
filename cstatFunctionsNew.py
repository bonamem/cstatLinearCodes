# new functions defined for the MIPS proposal
from cstatFunctions import rangex, ga
import numpy as np
from cstatFunctions import findGap,Rprime
from scipy.stats import t
def Ga(a, *data):
    x,y,xA,Deltax=data #unpack the data
    # derive the parameters needed for the formula
    N=len(x)
    R=rangex(data)
    M=sum(y)
        # --------------------------------------------
    result=0
    for i in range(len(x)):
        if y[i]==0:
            localGa=0
        if y[i]>0:
            localGa=y[i]*( (x[i]-xA)/(1.+a*(x[i]-xA)) )**2
        result+=localGa
    return result

def g2a(a, *data):
    x,y,xA,Deltax=data #unpack the data
    # derive the parameters needed for the formula
    N=len(x)
    R=rangex(data)
    M=sum(y)
    result=0
    for i in range(len(x)):
        if y[i]==0:
            localGa=0
        if y[i]>0:
            localGa=y[i]*(x[i]-xA)/(1.+a*(x[i]-xA))**2
        result+=localGa
    return result

def deltaa(lambdaHat, aHat,data):
    x,y,xA,Deltax=data
    R=rangex(data) 
    print('x=',x)
    Sum=sum(x-xA) # NOTE: this is also R, the range
    Galocal=Ga(aHat,*data)
    print('xA=%3.2f,R=%3.4e, Sum=%3.2e, lambdaHat=%3.2f,Ga(aHat)=%3.2e'%(xA,R,Sum,lambdaHat,Galocal))
    result=1/Galocal**0.5 # this is the second-order approximation
    return result

def deltalambda(lambdaHat, aHat,data):
    # Notice that aHat is not actually used
    x,y,xA,Deltax=data
    R=rangex(data)
    M=sum(y)
    result=lambdaHat/M**0.5
    return result

# now define beta=a*lambda and find deltabeta that gives \Delta C=1
# this is useful because this is the slope of the linear relationship
def deltabeta(lambdaHat, aHat,data):
    Galocal=Ga(aHat,*data)
    # line belwo applies when varying both beta and lambda in DeltaC=1 - seems wrong
    #result=lambdaHat*aHat/(1+aHat)/Galocal**0.5
    # line below appears the correct method
    result=lambdaHat/Galocal**0.5
    return result

def covALambda(lambdaHat, aHat,data):
    Galocal=Ga(aHat,*data)
    x,y,xA,Deltax=data
    M=sum(y)
    #method a:
    #result = (aHat*lambdaHat/2)*(1/Galocal*(1/(aHat+1)**2 - 1/aHat**2) - 1/M)
    # method (b):
    result= - (aHat*lambdaHat)/2*(1/M)
    return result

def linearPivotedA(x,b):
    # assumes an intercept of a=0
    result = b*x
    return result

def confidenceBandOLS(xPlot,aHat,bHat,Vara,Varb,Covab):
    nPlots=len(xPlot) # number of points for this confidence band
    varTerm1=np.zeros(nPlots)
    varTerm2=np.zeros(nPlots)
    covTerm=np.zeros(nPlots)
    varDeltaMethod=np.zeros(nPlots)

    for i in range(nPlots):
        varTerm1[i]=Vara*1.0
        varTerm2[i]=Varb*xPlot[i]**2
        covTerm[i]=2*Covab*xPlot[i]
        varDeltaMethod[i]=varTerm1[i]+varTerm2[i]+covTerm[i]
    print('OLS conf. band variance',varDeltaMethod)
    return varDeltaMethod**0.5

# New function for the confidence band plot
# It returns the +- error at a given fixed value of xPlot
def confidenceBand(xPlot,lambdaHat,aHat,data):
    x,y,xA,Deltax=data # x itself is not used
    Galocal=Ga(aHat,*data)
    M=sum(y)
    print('Detected %d counts'%M)
    print('G(aHat)=%3.5f'%Galocal)
    nPlots=len(xPlot) # number of points for this confidence band
    #result=(lambdaHat**2/M)+(lambdaHat**2*(xPlot-xA)**2/Galocal)-lambdaHat**2*aHat*(xPlot-xA)/M**0.5
    # do error propagation directly on the Scargle product
    varTerm1=np.zeros(nPlots)
    varTerm2=np.zeros(nPlots)
    covTerm=np.zeros(nPlots)
    varDeltaMethod=np.zeros(nPlots)
    yHat=np.zeros(nPlots)
    # method:   0: Delta C=1; 
    #           1: Cov. matrix (error prop); 
    #           2: Cov. matrix (Likelihood).
    method=2
    if method==0:
    # This is the methods based on Delta C=1 quantities
        for i in range(nPlots):
            yHat[i]=lambdaHat*(1+aHat*(xPlot[i]-xA)) # best-fit model
            varTerm1[i]=1/M # only variances
            varTerm2[i]=(xPlot[i]-xA)**2/(Galocal*(1+aHat*(xPlot[i]-xA))**2)
            covTerm[i]=-(aHat*lambdaHat/M)*(xPlot[i]-xA)/yHat[i] # covariance term
            print('%d: x=%3.2f,terms: %3.6f %3.6f %3.6f'%(i,xPlot[i],varTerm1[i],varTerm2[i],covTerm[i]))

    # ----------------------------------------------------------
        result=varTerm1+varTerm2+covTerm  # add covariance
        result=result**0.5*yHat # square root and multiply by yHat
        print(xA,xPlot,result)
    
    # This method is based on the covariance matrix
    if method==1: # covariance matrix from error propagation
        CovMat=covMatrixErrProp(lambdaHat, aHat,data)
    if method==2: # Cameron-Trivedi covariance matrix
        CovMat=covMatrixCT(lambdaHat, aHat,data)
    if (method==1) or (method==2):
    #CovMat=covMatrixCT(lambdaHat, aHat,data)
        sigma2a=CovMat[1][1]
        sigma2lambda=CovMat[0][0]
        covalambda=CovMat[0][1]
        for i in range(nPlots):
        #yHat[i]=lambdaHat*(1+aHat*(xPlot[i]-xA)) # best-fit model
            varTerm1[i]=sigma2lambda*(1+aHat*(xPlot[i]-xA))**2
            varTerm2[i]=sigma2a*(lambdaHat*(xPlot[i]-xA))**2
            covTerm[i]=2*lambdaHat*covalambda*(1+aHat*(xPlot[i]-xA))*(xPlot[i]-xA)
            #print('check: lambdaHat^2*(x-xA)^2*sigma^2a=%3.4f, 2 lambdaHat*sigma^2alambda (x-xA)=%3.4f, sigma^2lambda=%3.4f: %3.4f'%(varTerm2[i],covTerm[i],varTerm1[i],varTerm1[i]+varTerm2[i]+covTerm[i]))
        result=varTerm1+varTerm2+covTerm  # add covariance
        result=result**0.5 # square root 
    

    return result

# covariance matrix from error propagation formulas
def covMatrixErrProp(lambdaHat, aHat,data):
    x,y,xA,Deltax=data
    # use proper formula in presence of gap
    hasGap,xa,xb,xG,RG,SGSum=findGap(data)
    RGSum=np.sum(RG) # this is the sum R_G
    print('SGSum=%3.2f, RGSum=%3.2f'%(SGSum,RGSum))
    R=rangex(data)
    # hasGap=0 if there are no gaps
    if hasGap>=1:
        Rm=Rprime(R,RG,xA,xG)
    if hasGap==0:
        Rm=R
    print('%d gaps detected'%hasGap)
    gaHat=ga(aHat, *data)
    GaHat=Ga(aHat,*data)
    g2aHat=g2a(aHat,*data)
    M=sum(y)
    N=len(x)
    dady=np.zeros(N)
    dlambdady=np.zeros(N)
# These are the derivatives needed for covariance matrix
# The derivatives are d g(a)/d y_i where y_i are the datapoints
# and x_i are the x coordinates of the datapoints
# eq. (5) and (8) of latest derivative
    # eq (5), full differential daHat/dyi
    dady=(1-(2/Rm)*(x-xA))/(1+aHat*(x-xA))/(g2aHat-(2/Rm)*GaHat)
    # eq (5') instead, only partial derivative 
    #dady=(1-2*(x-xA)/R)/(1+aHat*(x-xA))/gaHat
    #dlambdady=-M/(R*(1+aHat*R/2)**2)*(R/2)*dady
    # from eq. (8)
    #dlambdady=(2/R**2)*( (x-xA)/(1+aHat*(x-xA)) - GaHat*dady)
    # replace with new derivatives, eq. (10)
    dlambdady=(R-RGSum +((R**2)/2-SGSum)*(aHat-M*dady))/(R-RGSum + aHat*((R**2)/2-SGSum))**2
    result=np.zeros((2,2))
    # Now we implement the sum over all datapoints
    # sigma^2_i=y_i is assumed - or should it be from the model?
    # doesn't look like it makes much of a difference
    sigma2i=y
    # use the best-fit model instead
    #yHat=lambdaHat*(1+aHat*(x-xA))*Deltax
    #sigma2i=yHat
    result[1][1]=sum(sigma2i*dady**2)
    result[0][0]=sum(sigma2i*dlambdady**2)
    result[0][1]=sum(sigma2i*dady*dlambdady)
    result[1][0]=result[0][1]
    return result

# This version uses the formulas in the book by Cameron and Trivedi
# It needs to be made to work for gaps in data
def covMatrixCT(lambdaHat, aHat,data):
    x,y,xA,Deltax=data
    hasGap,xa,xb,xG,RG,SGSum=findGap(data)

    N=len(x)
    # Not sure about the use of the SIGN in log-likelihood
    Sum1=sum(1+aHat*(x-xA))
    Sum2=sum(x-xA)
    Sum3=sum((x-xA)**2/(1+aHat*(x-xA))*Deltax)
    result=np.zeros((2,2))
    factor=1 # 1/N 
    result[0][0]=Sum1/lambdaHat
    result[1][1]=lambdaHat*Sum3
    result[0][1]=Sum2
    result[1][0]=result[0][1]

    invResult=np.linalg.inv(factor*result)
    print('check: A inv(A)=',np.dot(factor*result,invResult))
    return invResult

# Now investigate the errors and Wald t-test for OLS
def OLSStats(a,b,x,y):
    # a,b are the best-fit OLS estimators
    # do not worry about gaps here
    N=len(x)
    print('OLS: Using N=%d observations'%N)
    # define the sums to be used
    Sumx=sum(x)
    SumxSq=sum(x**2)
    Sumxy=sum(x*y)
    Sumy=sum(y) # This is also known as M
    Delta=N*SumxSq-Sumx**2
    # Check my own calculations of OLS estimates
    aLocal=(SumxSq*Sumy-Sumx*Sumxy)/Delta
    bLocal=(N*Sumxy-Sumx*Sumy)/Delta
    # ===============================================
    # variance of the two OLS estimators
    # Need to understand how to handle (a+bx) term ...
    # --------------------------------------
    # METHOD 1 Use the estimated hat(y) (these are close ....) Eq 3a-rc
    Vary=np.zeros(N) 
    Vary=a+b*x
    Vara=sum( (SumxSq-x*Sumx)**2*Vary)/Delta**2
    Varb=sum( (N*x-Sumx)**2*Vary)/Delta**2
    Covab=((N*SumxSq+Sumx**2)*sum(x*Vary) - SumxSq*Sumx*sum(Vary) -N*Sumx*sum(x**2*Vary))/Delta**2
    # ---------------------------------------
    # METHOD 2 try measured y 
    #Vara=sum( (SumxSq-x*Sumx)**2*y)/Delta**2
    #Varb=sum( (N*x-Sumx)**2*y)/Delta**2
    # ---------------------------------------
    # METHOD 3 as surmized from r statistic, from linregress
    SumxxbarSq=sum((x-np.mean(x))**2)
    SumyybarSq=sum((y-np.mean(y))**2)
    SumxySq=sum((x-np.mean(x))*(y-np.mean(y)))
    rSq=SumxySq**2/(SumxxbarSq*SumyybarSq)
    #Varb=(1-rSq)*SumyybarSq/SumxxbarSq/(N-2)
    #Vara=Varb*(SumxxbarSq/N+np.mean(x)**2)
    # ===============================================
    # now onto the t-statistic
    tStat=bLocal/Varb**0.5
    tCrit=t.ppf(0.9,N-2) # one--sided critical value - should it be two--sided??
    pVal=2.*t.sf(tStat,N-2) # yes, two--sided p--value
    return aLocal,bLocal,Vara,Varb,Covab,tStat,tCrit,pVal
