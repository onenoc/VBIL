import autograd.numpy as np
import math
from scipy import special
from scipy import stats
from scipy import misc
from matplotlib import pyplot as plt
import seaborn as sns

def H_i(samples,params,i):
    '''
    @summary: calculate H via sampling for parameter i
    @param params: the list of parameters for your recognition model
    @param S: number of samples
    '''
    H = 0
    S=len(samples)
    for theta in samples:
        h = h_s(theta)*gradient_log_recognition(params,theta,i)
        H+=h
    H = H/S
    print "H",i
    print H
    return H

#CHECKED
def sample_theta(params,S):
    '''
    @param params: list of parameters for recognition model, gamma
    @param S: number of samples
    '''
    return np.random.gamma(params[0],1/params[1],size=S)

#should be close to max at 0.1, which is the case
def h_s(theta):
    '''
    @summary: calculate h at a single sample point
    '''
    N = 10
    #print theta
    #print abc_log_likelihood(theta,N)
    h_s = math.log(prior_density(theta))+log_likelihood(theta,N)
    #abc_log_likelihood(theta,N)
    return h_s

def log_likelihood(theta,N):
    return 500*np.log(theta)-theta*500*data_Sy(0.1)

#CORRECT
def abc_log_likelihood(theta, N):
    '''
    @summary: calculate abc likelihood for a single
    value of theta and N samples from simulator
    '''
    log_kernels = np.zeros(N)
    for i in range(N):
        x = simulator(theta)
        log_kernels[i] = log_abc_kernel(x)
    log_likelihood = misc.logsumexp(log_kernels)
    log_likelihood = np.log(1./N)+log_likelihood
    return log_likelihood

#CORRECT
def log_abc_kernel(x):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    e=np.std(x)/np.sqrt(500)
    Sx = np.mean(x)
    Sy=data_Sy(0.1)
    #return stats.norm.pdf(Sy, loc=Sx, scale=e)
    return np.log(1/(e*np.sqrt(2*np.pi)))-(Sy-Sx)**2/(2*e**2)

#CORRECT
def simulator(theta):
    '''
    @given a parameter theta, simulate
    CORRECT
    '''
    w = np.random.exponential(1./theta,500)
    return w

#CHECKED
def prior_density(theta):
    '''
    @summary: calculate your prior density for some value of theta
    gamma prior
    '''
    alpha = 0.1
    beta = 0.1
    return stats.gamma.pdf(theta, alpha, scale=1/beta)

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

#CHECKED
def numerical_gradient_log_recognition(params,theta,i):
    alpha = params[0]
    beta = params[1]
    h=0.000001
    if i==0:
        f1=np.log(stats.gamma.pdf(theta,alpha,scale=1/beta))
        f2=np.log(stats.gamma.pdf(theta,alpha+h,scale=1/beta))
    else:
        f1=np.log(stats.gamma.pdf(theta,alpha,scale=1/beta))
        f2=np.log(stats.gamma.pdf(theta,alpha,scale=1/(beta+h)))
    return (f2-f1)/h

#CHECKED
def data_Sy(rate):
    '''
    @summary: the true data
    '''
    np.random.seed(0)
    w = np.random.exponential(1./rate,500)
    return np.mean(w)
'''
def c_i(i, params):
    @summary: control variate for single parameter
    return 0
'''

#CHECKED
def test_likelihood():
    theta = 1
    theta = 0.001*np.array([x+0.000001 for x in range(1000)])
    likelihoods=[]
    for i in theta:
        likelihoods.append(np.exp(abc_log_likelihood(i,100)))
    plt.plot(theta,likelihoods)
    plt.show()

#CHECKED, BUT WHAT ABOUT N?
def fisher_info(params,N):
    alpha = params[0]
    beta = params[1]
    I=np.zeros((2,2))
    I[0][0]=special.polygamma(1,alpha)
    I[0][1]=-1/beta
    I[1][0]=-1/beta
    I[1][1]=alpha/(beta**2)
    return I

def nat_grad(params):
    samples = sample_theta(params,5000)
    #samples = generate_theta_samples(params,100)
    H = np.array([H_i(samples,params,0),H_i(samples,params,1)])
    nat_grad = params-np.dot(np.linalg.inv(fisher_info(params,1)),H)
    return nat_grad

def iterate(params,i):
    a = 2./(1+i)
    params = params-a*nat_grad(params)
    return params

def grad_KL(params):
    samples = sample_theta(params,5000)
    #generate_theta_samples(params,100)
    H = np.array([H_i(samples,params,0), H_i(samples,params,1)])
    return np.dot(fisher_info(params,1),params)-H

def lower_bound(params):
    samples_1 = sample_theta(params,100)
    samples_2 = sample_theta(params,100)

if __name__=='__main__':
    params = np.array([100.,100.])
    for i in range(100000):
        if i%10000==0:
            params = iterate(params,i)
            alpha = params[0]
            beta = params[1]
            print "estimated params"
            print params
            print "estimated mean"
            print alpha/beta
            print "true params"
            print 0.1+500, 0.1+data_Sy(0.1)*500
            print "true mean"
            print (0.1+500)/(0.1+data_Sy(0.1)*500)
    true_gamma_samples = np.random.gamma(500.1,1/(0.1+data_Sy(0.1)*500),100000)
    recognition_gamma_samples = np.random.gamma(params[0],1/params[1],100000)
    sns.distplot(true_gamma_samples)
    sns.distplot(recognition_gamma_samples)
    plt.show()    
    #test_likelihood()
