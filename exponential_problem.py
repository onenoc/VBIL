import numpy as np
from scipy import special
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import math

def H_i(samples,params,data,i):
    n=len(data)
    H_i = 0
    S = len(samples)
    c = c_i(params,n,data,S,i)
    #c = 0
    for theta in samples:
        inner = (h_s(theta,n,data)-c)*gradient_log_recognition(params,theta,i)
        H_i += inner
    H_i = H_i/S
    return H_i

def c_i(params,n,data,S,i):
    first = np.zeros(S)
    second = np.zeros(S)
    samples = sample_theta(params,S)
    for s in range(S):
        theta = samples[s]
        first[s] = h_s(theta,n,data)*gradient_log_recognition(params,theta,i)
        second[s] = gradient_log_recognition(params,theta,i)
    return np.cov(first,second)[0][1]/np.cov(first,second)[1][1]
    
def sample_theta(params,S):
    return np.random.gamma(params[0],1/params[1],size=S)

def fisher_info(params):
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    I[0][0] = special.polygamma(1,alpha)
    I[0][1] = -1/beta
    I[1][0] = -1/beta
    I[1][1] = alpha/(beta**2)
    return I

def inv_fisher(params):
    return np.linalg.inv(fisher_info(params))

#CHECKED
def gradient_log_recognition(params,theta,i):
    '''
    @summary: calculate the gradient of the log of your
    recognition density.  For now we assume gamma recognition model
    @param theta: the value
    @param params: the parameters of your recognition model.
    '''
    alpha = params[0]
    beta = params[1]
    delta=[]
    delta.append(np.log(theta)+np.log(beta)-special.psi(alpha))
    delta.append(alpha/beta-theta)
    return delta[i]

def prior_density(theta):
    return stats.gamma.pdf(theta,1,scale=1/1)

def log_likelihood(theta,n,data):
    return n*np.log(theta)-theta*np.sum(data)

def h_s(theta,n,data):
    h_s = np.log(prior_density(theta))+log_likelihood(theta,n,data)
    return h_s

def data_Sy(theta,n):
    return np.random.exponential(1/theta,n)

def iterate_nat_grad(params,data,i):
    a = 1./(5+i)
    samples = sample_theta(params,1)
    H_val = np.array([H_i(samples,params,data,0),H_i(samples,params,data,1)])
    #print H_val
    #print inv_fisher(params)
    params = params-a*(params-np.dot(inv_fisher(params),H_val))
    return params

if __name__=='__main__':
    data = data_Sy(0.1,500)
    params = np.array([100.,100.])
    for i in range(5000):
        params = iterate_nat_grad(params,data,i)
        if i%100==0:
            alpha = params[0]
            beta = params[1]
            print "estimated params"
            print params
            print "estimated mean"
            print alpha/beta
            print "true params"
            print 1+500, 1+np.sum(data)
            print "true mean"
            print (1+500)/(1+np.sum(data))
    true_gamma_samples = np.random.gamma(501,1/(1+np.mean(data)*500),100000)
    recognition_gamma_samples = np.random.gamma(params[0],1/params[1],100000)
    sns.distplot(true_gamma_samples)
    sns.distplot(recognition_gamma_samples)
    plt.show()
