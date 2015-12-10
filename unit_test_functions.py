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

if __name__=='__main__':
    test_data_Sy()
    test_gradient_log_recognition()

