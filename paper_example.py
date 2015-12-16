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
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    a=special.polygamma(1,alpha)-special.polygamma(1,alpha+beta)
    b=special.polygamma(1,alpha+beta)
    c=special.polygamma(1,alpha+beta)
    d=special.polygamma(1,beta)-special.polygamma(1,alpha+beta)
    const = 1./(a*d-b*c)
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

def iterate(params,i):
    a = 1./(1+i)
    H_val = H(params,200,57)
    fisher = fisher_info(params)
    params = params-a*(np.dot(fisher,params)-H_val)
    return params

def iterate_nat_grad(params,i):
    a = 1./(5+i)
    H_val = H(params,200,57)
    params = (1-a)*params+a*np.dot(inv_fisher(params),H_val)
    return params
    

if __name__=='__main__':
    params = np.array([400.,400.])
    for i in range(3000000):
       params = iterate_nat_grad(params,i)
       if i%1==0:
           print params
           alpha = params[0]
           beta = params[1]
           print "mean is %f" % (alpha/(alpha+beta))
    
