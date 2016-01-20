import paper_example_estimate as pe
from scipy import stats
import numpy as np

def test_log_recognition(params,theta):
    alpha = 0.1
    beta = 0.1
    theta = 0.5
    params = np.array([alpha,beta])
    print np.exp(pe.log_recognition(params, theta))
    
