import autograd.numpy as np
import math
from scipy import stats

def H(i,params,s):
    '''
    @summary: calculate H via sampling for parameter i
    @param params: the list of parameters for your recognition model
    @param S: number of samples
    '''
        

def sample_theta(params, s):
    '''
    @param params: list of parameters for recognition model, gaussian
    '''
    mu = params[0]
    sigma = params[1]
    return np.random.normal(mu,sigma, s)

def h_s(theta):
    '''
    @summary: calculate h at a single sample point
    '''
    math.log()

def abc_likelihood(theta, N):
    '''
    @summary: calculate abc likelihood for a single
    value of theta and N samples from simulator
    '''

def abc_kernel(x, e):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    y=data_y()
    stats.norm.pdf(x, e)

def simulator(theta, ind):
    '''
    @given a parameter theta, simulate
    '''
    return theta*ind

def prior_density(theta):
    '''
    @summary: calculate your prior density for some value of theta
    For now, we assume standard normal prior
    '''
    return stats.norm.pdf(theta) 

def gradient_log_recognition(theta, params):
    '''
    @summary: calculate the gradient of the log of your
    recognition density.  For now we assume Gaussian
    @param theta: the value
    @param params: the parameters of your recognition model.
    For a Gaussian, this is mu and sigma
    '''

def normal_density(mu, sigma, theta):
    k=np.size(mu)
    if k>1:
        return (2*np.pi)**(-k/2)*np.linalg.det(sigma)**(-1/2.)*np.exp((-1/2)*(theta-mu).T*np.linalg.inv(sigma)*(theta-mu))
    else:
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(float(theta)-mu)**2/(2*sigma**2))

def data_y():
    '''
    @summary: the true data
    '''
    mu = 2
    sigma = 5
    np.random.seed(5)
    y = np.random.normal(mu,sigma,100)
    return y

'''
def c_i(i, params):
    @summary: control variate for single parameter
    return 0
'''

if __name__=='__main__':
    y = data_y()
