import functions
import numpy as np

def test_data_Sy():
    rate = 0.1
    Sy = functions.data_Sy(rate)
    print "these should be approx equal"
    print Sy
    print 1/rate

def test_gradient_log_recognition():
    theta = 1
    params = np.array([2,0.5])
    print "these should be approx equal"
    print functions.gradient_log_recognition(theta,params,0)
    print functions.numerical_gradient_log_recognition(theta,params,0)
    print "these should be approx equal"
    print functions.gradient_log_recognition(theta,params,1)
    print functions.numerical_gradient_log_recognition(theta,params,1)

def test_fisher_info():
    params =np.zeros(2)
    params[0] = 0.19
    params[1] = 5.18
    print "fisher with 1 datapoint"
    print functions.fisher_info(params,1)
    true_fisher = 1*np.array([[28.983,-0.1983],[-0.193,0.007]])
    print true_fisher
    print "fisher with 500 datapoints"
    print functions.fisher_info(params,500)
    true_fisher = 500*np.array([[28.983,-0.1983],[-0.193,0.007]])
    print true_fisher

def test_generate_theta_samples():
    logParams = np.zeros(2)
    alpha = 0.1
    beta = 0.1
    logParams[0] = np.log(alpha)
    logParams[1] = np.log(beta)
    samples = functions.generate_theta_samples(logParams,50000)
    print "These should be close"
    print np.mean(samples)
    print alpha/beta

def test_prior_density():
    print "These should be equal"
    print "0.6566234388"
    print functions.prior_density(0.1) 

if __name__=='__main__':
    #test_data_Sy()
    #test_gradient_log_recognition()
    #test_fisher_info()
    test_generate_theta_samples()
    test_prior_density()
