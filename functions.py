import autograd.numpy as np
import math
from scipy import special
from scipy import stats
from matplotlib import pyplot as plt

def generate_theta_samples(logParams,S):
    params = np.exp(logParams)
    samples = []
    for s in range(S):
        samples.append(sample_theta(params))
    return samples

def H_i(samples,params,i):
    '''
    @summary: calculate H via sampling for parameter i
    @param params: the list of parameters for your recognition model
    @param S: number of samples
    '''
    H = 0
    S=len(samples)
    for theta in samples:
        h = h_s(theta)
        H+=h
    print H
    print gradient_log_recognition(theta, params,i)
    H = H/S*gradient_log_recognition(theta, params,i)
    return H

def sample_theta(params):
    '''
    @param params: list of parameters for recognition model, gamma
    '''
    return np.random.gamma(params[0],1/params[1])

def h_s(theta):
    '''
    @summary: calculate h at a single sample point
    '''
    N = 100
    h_s = math.log(prior_density(theta)*abc_likelihood(theta,N))
    return h_s

def abc_likelihood(theta, N):
    '''
    @summary: calculate abc likelihood for a single
    value of theta and N samples from simulator
    '''
    likelihood = 0
    for i in range(N):
        x = simulator(theta)
        K = abc_kernel(x)
        likelihood+=K
    likelihood = K/N
    return likelihood

def abc_kernel(x):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    e = 1
    #np.std(x)/50
    Sx = np.mean(x)
    Sy=data_Sy(0.1)
    return stats.norm.pdf(Sy, loc=Sx, scale=e)

def simulator(theta):
    '''
    @given a parameter theta, simulate
    '''
    w = np.random.exponential(1/theta,500)
    return w

def prior_density(theta):
    '''
    @summary: calculate your prior density for some value of theta
    For now, we assume standard normal prior
    '''
    alpha = 0.1
    beta = 0.1
    return stats.gamma.pdf(theta, alpha, scale=1/beta)

def gradient_log_recognition(theta, params,i):
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

def numerical_gradient_log_recognition(theta,params,i):
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

def data_Sy(rate):
    '''
    @summary: the true data
    '''
    w = np.random.exponential(1/rate,500)
    return np.mean(w)

'''
def c_i(i, params):
    @summary: control variate for single parameter

    return 0
'''

def test_likelihood():
    theta = 1
    abc_likelihood(theta, 100)
    theta = 0.001*np.array([x+0.000001 for x in range(1000)])
    likelihoods=[]
    for i in theta:
        likelihoods.append(abc_likelihood(i,100))
    plt.plot(theta,likelihoods)
    plt.show()

def fisher_info(params):
    alpha = params[0]
    beta = params[1]
    I=np.zeros((2,2))
    I[0][0]=special.polygamma(1,alpha)
    I[0][1]=-1/beta
    I[1][0]=-1/beta
    I[1][1]=alpha/(beta**2)
    return I

def iterate(logParams,learning_rate):
    a = learning_rate
    logParams = logParams-a*nat_grad(logParams)
    return logParams

def nat_grad(logParams):
    params = np.exp(logParams)
    samples = generate_theta_samples(params,100)
    H = np.array([0,0])
    H[0] = H_i(samples,params,0)
    H[1] = H_i(samples,params,1)
    nat_grad = params-np.dot(np.linalg.inv(fisher_info(params)),H)
    return nat_grad

if __name__=='__main__':
    logParams = np.array([0.1,0.1])
    learning_rate = 0.5
    for i in range(100):
        logParams = iterate(logParams,learning_rate)
        print np.exp(logParams)

