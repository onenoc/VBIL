import numpy as np
import math
from scipy import stats

def H(i,params):
    '''
    @summary: calculate H via sampling for parameter i
    @param params: the list of parameters for your recognition model
    '''

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

def abc_kernel(y, x, e):
    '''
    @summary: kernel density
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''

def simulator(theta):
    '''
    @given a parameter theta, 
    '''
    x=5
    return x

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

def c_i(i, params):
    '''
    '''
