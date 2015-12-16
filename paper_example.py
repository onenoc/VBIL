import numpy as np
from scipy import special

def fisher_info(params):
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    I[0][0]=special.polygamma(1,alpha)-special.polygamma(1,alpha+beta)
    I[0][1]=special.polygamma(1,alpha+beta)
    I[1][0]=special.polygamma(1,alpha+beta)
    I[1][1]=special.polygamma(1,beta)-special.polygamma(1,alpha+beta)
    return I

def inv_fisher(params):
    #print "params"
    #print params
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    a=special.polygamma(1,alpha)-special.polygamma(1,alpha+beta)
    #print "inv"
    #print a
    b=special.polygamma(1,alpha+beta)
    #print b
    c=special.polygamma(1,alpha+beta)
    #print c
    d=special.polygamma(1,beta)-special.polygamma(1,alpha+beta)
    #print d
    const = 1./(a*d-b*c)
    #print const
    I[0][0]=d
    I[0][1]=-b
    I[1][0]=-c
    I[1][1]=a
    I=const*I
    return I

def H(params,n,k):
    alpha = params[0]
    beta = params[1]
    H=np.zeros(2)
    H[0]=k*special.polygamma(1,alpha)-n*special.polygamma(1,alpha+beta)
    H[1]=(n-k)*special.polygamma(1,beta)-n*special.polygamma(1,alpha+beta)
    return H

def iterate(logparams,i):
    params = np.exp(logparams)
    a = 1./(50000+i)
    H_val = H(params,200,57)
    fisher = fisher_info(params)
    logparams = logparams-a*(np.dot(fisher,params)-H_val)*params
    return logparams

def iterate_nat_grad(logparams,i):
    params = np.exp(logparams)
    a = 1./(3000000000+3000000000*i)
    H_val = H(params,200,57)
    logparams = logparams+a*(params-np.dot(inv_fisher(params),H_val))*params
    return logparams
    

if __name__=='__main__':
    logparams = np.array([np.random.randint(0,10),np.random.randint(0,10)])
    for i in range(1,500000):
       logparams = iterate(logparams,i)
       if i%1000==0:
           params = np.exp(logparams)
           print params
           alpha = params[0]
           beta = params[1]
           print "mean is %f" % (alpha/(alpha+beta))
    '''
    print "params"
    print params 
    alpha = params[0]
    beta = params[1]
    print "mean is %f" % (alpha/(alpha+beta))
    '''
