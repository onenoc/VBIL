import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import math

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
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    a=special.polygamma(1,alpha)-special.polygamma(1,alpha+beta)
    b=-special.polygamma(1,alpha+beta)
    c=-special.polygamma(1,alpha+beta)
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

def iterate_nat_grad(params,i,n,k):
    a = 1./(5+i)
    samples = sample_theta(params,1000)
    H_val = np.array([H_i(samples,params,n,k,0),H_i(samples,params,n,k,1)])
    H_true = H(params,n,k)
    params = params-a*(params-np.dot(inv_fisher(params),H_val))
    return params

def H_i(samples,params,n,k,i):
    H_i = 0
    S = len(samples)
    c = c_i(params,n,k,i,S)
    inner = (h_s(samples,n,k)-c)*gradient_log_recognition(params,samples,i)
    H_i = np.mean(inner)
    return H_i

def c_i(params,n,k,i,S):
    first = np.zeros(S)
    second = np.zeros(S)
    samples = sample_theta(params,S)
    first = h_s(samples,n,k)*gradient_log_recognition(params,samples,i)
    second = gradient_log_recognition(params,samples,i)
    return np.cov(first,second)[0][1]/np.cov(first,second)[1][1]


def sample_theta(params,S):
    '''
    @param params: list of parameters for recognition model, gamma
    @param S: number of samples
    '''
    #print params[0],params[1]
    return np.random.beta(params[0],params[1],size=S)

def h_s(theta,n,k):
    h_s = np.log(prior_density(theta))+log_likelihood(theta,n,k)
    #print h_s
    return h_s

def log_likelihood(theta, n, k):
    return np.log(stats.binom.pmf(k,n,theta))
    
def gradient_log_recognition(params,theta,i):
    alpha = params[0]
    beta = params[1]
    if i==0:
        return np.log(theta)-special.polygamma(0,alpha)+special.polygamma(0,alpha+beta)
    if i==1:
        return np.log(1-theta)-special.polygamma(0,beta)+special.polygamma(0,alpha+beta)

def prior_density(theta):
    return stats.beta.pdf(theta,1,1)

def numerical_gradient_log_recognition(params,theta,i):
    alpha = params[0]
    beta = params[1]
    h = 0.000001
    if i==0:
        f1 = stats.beta.logpdf(theta,alpha,beta)
        f2 = stats.beta.logpdf(theta,alpha+h,beta)
    else:
        f1 = stats.beta.logpdf(theta,alpha,beta)
        f2 = stats.beta.logpdf(theta,alpha,beta+h)
    return (f2-f1)/h

if __name__=='__main__':
    params = np.array([30.,30.])
    n = 10
    k = 1
    true_alpha = k+1.
    true_beta = n-k+1.
    alpha = 0
    beta = 0
    for i in range(1,10000):
       params = iterate_nat_grad(params,i,n,k)
       if i%100==0:
           print "param estimates"
           print params
           alpha = params[0]
           beta = params[1]
           print "true params"
           print true_alpha, true_beta
           print "mean is %f" % (alpha/(alpha+beta))
           print "true mean is %f" % (true_alpha/(true_alpha+true_beta))
    true_beta_samples = np.random.beta(k+1,n-k+1,100000)
    recognition_beta_samples = np.random.beta(alpha,beta,100000)
    sns.distplot(true_beta_samples)
    sns.distplot(recognition_beta_samples)
    plt.show()

