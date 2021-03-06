import numpy as np
from scipy import special
from scipy import stats
from scipy import misc
from scipy.stats import beta
from scipy.stats import lognorm
from scipy.stats import gamma
from matplotlib import pyplot as plt
import math

all_gradients = []


def H_i(samples,params,data,i):
    n=len(data)
    H_i = 0
    S = len(samples)
    c = c_i(params,n,data,S,i)
    #c = 0
    inner = (h_s(samples,n,data)-c)*gradient_log_recognition(params,samples,i)
    H_i = np.mean(inner)
    return H_i

def c_i(params,n,data,S,i):
    first = np.zeros(S)
    second = np.zeros(S)
    samples = sample_theta(params,S)
    first = h_s(samples,n,data)*gradient_log_recognition(params,samples,i)
    second = gradient_log_recognition(params,samples,i)
    return np.cov(first,second)[0][1]/np.cov(first,second)[1][1]
    
def sample_theta(params,S):
    print params[0], params[1]
    return np.random.gamma(np.exp(params[0]),1/np.exp(params[1]),size=S)

def fisher_info(params):
    alpha = np.exp(params[0])
    beta = np.exp(params[1])
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
    alpha = np.exp(params[0])
    beta = np.exp(params[1])
    delta=[]
    delta.append(np.log(theta)+np.log(beta)-special.psi(alpha))
    delta.append(alpha/beta-theta)
    return delta[i]

def prior_density(theta):
    return stats.gamma.pdf(theta,1,scale=1/1)

def log_likelihood(theta,n,data):
    return n*np.log(theta)-theta*np.sum(data)

def abc_log_likelihood(samples,n,data):
    N=500
    M = len(data)
    S = len(samples)
    log_kernels = np.zeros(N)
    ll = np.zeros(S)
    for s in range(S):
        theta = samples[s]
        x = simulator(theta,N,M).reshape(len(data),N)
        log_kernels = log_abc_kernel(x,data)
        ll[s] = misc.logsumexp(log_kernels)
        ll[s] = np.log(1./N)+ll[s]
    return ll

def log_abc_kernel(x,data):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    #e=np.std(x)/np.sqrt(len(data))
    e = 0.8
    Sx = np.mean(x,0)
    Sy = np.mean(data)
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))
    #return stats.norm.pdf(Sy, loc=Sx, scale=e)
    #return np.log(1/(e*np.sqrt(2*np.pi)))-(Sy-Sx)**2/(2*e**2)

#CORRECT
def simulator(theta,N,M):
    '''
    @given a parameter theta, simulate
    CORRECT
    '''
    w = np.random.exponential(1./theta,M*N)
    return w

def h_s(theta,n,data):
    h_s = np.log(prior_density(theta))+abc_log_likelihood(theta,n,data)
    #print "log-like"
    #print log_likelihood(theta,n,data)
    #print abc_log_likelihood(theta,n,data)
    return h_s

def data_Sy(theta,n):
    #return np.random.exponential(1/theta,n)
    return 10.0*np.ones(n)

def iterate_nat_grad(params,data,i):
    a = 1./(5+i)
    samples = sample_theta(params,500)
    H_val = np.array([H_i(samples,params,data,0),H_i(samples,params,data,1)])
    params = params-a*(params-np.dot(inv_fisher(params),H_val))
    all_gradients.append(params-np.dot(inv_fisher(params),H_val))
    return params

def loglognormal_np( logx, mu, stddev ):
  log_pdf = -np.log(stddev) - 0.5*pow( (logx-mu)/stddev, 2.0 )-logx-0.5*np.log(2*np.pi)
  return log_pdf

if __name__=='__main__':
    t_lambda = 1./10
    M = 15
    data = data_Sy(t_lambda,M)
    #params = np.array([10.,10.])
    params = np.random.uniform(0,1,2)
    for i in range(250):
        params = iterate_nat_grad(params,data,i)
        if i%10==0:
            alpha = np.exp(params[0])
            beta = np.exp(params[1])
            print "estimated params"
            print params
            print "estimated mean"
            print alpha/beta
            print "true params"
            print t_lambda+M, t_lambda+np.sum(data)
            print "true mean"
            print (t_lambda+M)/(t_lambda+np.sum(data))
    print 'final params'
    print params
    x = np.linspace(0,1,100)
    true_params = np.array([t_lambda+M,t_lambda+np.sum(data)])
    print 'true params'
    print true_params
    #plt.plot(x, gamma.pdf(x,true_params[0],scale=1/true_params[1]),'--', lw=2.5, label='true',color='red')
    #plt.plot(x, gamma.pdf(x,params[0],scale=1/params[1]),'r-', label='VBIL',color='green')
    lognormal=np.exp(loglognormal_np(np.log(x),-2.2787,0.208157)
)
    all_gradients = np.asarray(all_gradients)
    running_var = []
    for i in range(1,len(all_gradients)):
        running_var.append(np.var(all_gradients[0:i]))
    plt.plot(running_var)
    plt.show()
    #plt.plot(x,lognormal,'b',label='AD',color='blue')
    #plt.legend(loc=1)
    #plt.title('Exponential Problem for Sy=10,M=15')
    #plt.show()
    '''
    true_gamma_samples = np.random.gamma(500+t_lambda,1/(t_lambda+np.mean(data)*500),100000)
    recognition_gamma_samples = np.random.gamma(params[0],1/params[1],100000)
    sns.distplot(true_gamma_samples)
    sns.distplot(recognition_gamma_samples)
    plt.show()
    '''
