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

def test_prior_density():
    print "These should be equal"
    print "0.6566234388"
    print functions.prior_density(0.1)

def test_h_s():
    print "h_s: 0.01,1,10"
    print functions.h_s(0.01)
    print functions.h_s(0.1)
    print functions.h_s(1)
    print functions.h_s(10)

def test_grad_KL():
    print "for parameters close to true value, these should be small"
    params = np.zeros(2)
    alpha = 50
    beta = 500
    params[0] = alpha
    params[1] = beta
    print functions.grad_KL(params)
    
if __name__=='__main__':
    #test_data_Sy()
    #test_gradient_log_recognition()
    #test_fisher_info()
    test_prior_density()
    test_grad_KL()
    print test_h_s()
