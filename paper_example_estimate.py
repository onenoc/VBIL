import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy import stats
from scipy.stats import beta
from scipy import misc
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
    samples = sample_theta(params,100)
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

def abc_log_likelihood(samples,n,k):
    N=100
    S = len(samples)
    log_kernels = np.zeros(N)
    ll = np.zeros(S)
    for s in range(S):
        theta = samples[s]
        x = simulator(theta,n,N)
        log_kernels = log_abc_kernel(x,k)
        ll[s] = misc.logsumexp(log_kernels)
        ll[s] = np.log(1./N)+ll[s]
    return ll

def simulator(theta,n,N):
    return np.random.binomial(n,theta,size=N)

def log_abc_kernel(x,k):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    #e=np.std(x)/np.sqrt(len(data))
    e = 0.1
    Sx = x
    Sy = k
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

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
    k = 4
    true_alpha = k+1.
    true_beta = n-k+1.
    e_alpha = 0
    e_beta = 0
    for i in range(1,1000):
       params = iterate_nat_grad(params,i,n,k)
       if i%10==0:
           print "param estimates"
           print params
           e_alpha = params[0]
           e_beta = params[1]
           print "true params"
           print true_alpha, true_beta
           print "mean is %f" % (e_alpha/(e_alpha+e_beta))
           print "true mean is %f" % (true_alpha/(true_alpha+true_beta))
    x = np.linspace(0,1,100)
    fig,ax=plt.subplots(1,1)
    ax.plot(x, beta.pdf(x,true_alpha,true_beta),'r-', lw=5, alpha=0.6, label='true pdf',color='blue')
    ax.plot(x, beta.pdf(x,params[0],params[1]),'r-', lw=5, alpha=0.6, label='VBIL pdf',color='green')
    plt.show()


