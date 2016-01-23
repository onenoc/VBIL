import numpy as np
from scipy import special
from matplotlib import pyplot as plt
import seaborn as sns

def fisher_info(params):
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    I[0][0]=special.polygamma(1,alpha)-special.polygamma(1,alpha+beta)
    I[0][1]=-special.polygamma(1,alpha+beta)
    I[1][0]=-special.polygamma(1,alpha+beta)
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
    b=-special.polygamma(1,alpha+beta)
    #print b
    c=-special.polygamma(1,alpha+beta)
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
'''
def iterate(logparams,i):
    params = np.exp(logparams)
    a = 1./(5+i)
    H_val = H(params,200,57)
    fisher = fisher_info(params)
    #logparams = logparams-a*(np.dot(fisher,params)-H_val)*params
    params = params-a*(np.dot(fisher,params)-H_val)#*params
    logparams = np.log(params)
    return logparams
'''
def iterate(params,i):
    a = 1./(5+i)
    H_val = H(params,200,57)
    fisher = fisher_info(params)
    #params = params-a*(np.dot(fisher,params)-H_val)
    return params

def iterate_nat_grad(params,i):
    a = 1./(5+i)
    H_val = H(params,200,57)
    #params = (1.-a)*params+a*np.dot(inv_fisher(params),H_val)
    params = params-a*(params-np.dot(inv_fisher(params),H_val))
    print "nat grad"
    print params-np.dot(inv_fisher(params),H_val)
    #params = params+a*(params-np.dot(inv_fisher(params),H_val))
    return params

def nat_grad(params):
    H_val = H(params,200,57)
    return params-np.dot(inv_fisher(params),H_val)
    
if __name__=='__main__':
    #logparams = np.array([np.random.randint(0,1),np.random.randint(0,1)])
    params = np.array([100,100])
    n = 200
    k = 57
    true_alpha = k+1.
    true_beta = n-k+1.
    print params
    alpha = 0
    beta = 0
    for i in range(1,1000):
       params = iterate_nat_grad(params,i)
       if i%1==0:
           print "param estimates"
           print params
           alpha = params[0]
           beta = params[1]
           print "true params"
           print true_alpha, true_beta
           print "mean is %f" % (alpha/(alpha+beta))
           print "true mean is %f" % (true_alpha/(true_alpha+true_beta))
    '''
    print "params"
    print params 
    alpha = params[0]
    beta = params[1]
    print "mean is %f" % (alpha/(alpha+beta))
    '''
    true_beta_samples = np.random.beta(k+1,n-k+1,100000)
    recognition_beta_samples = np.random.beta(alpha,beta,100000)
    sns.distplot(true_beta_samples)
    sns.distplot(recognition_beta_samples)
    plt.show()
