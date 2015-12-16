import paper_example as pe
import numpy as np

def test_fisher_info(params):
    print "fisher: these should be same"
    print pe.fisher_info(params)
    true_fisher = np.zeros((2,2))
    true_fisher[0][0]=0.221322-0.181322
    true_fisher[0][1]=0.181322
    true_fisher[1][0]=0.181322
    true_fisher[1][1]=1.644933-0.181322
    print true_fisher

def test_inv_fisher(params):
    print "inverse fisher: should be same"
    print np.linalg.inv(pe.fisher_info(params))
    print pe.inv_fisher(params)

def test_H(params):
    print "H: should be same"
    print pe.H(params,1,1)
    print 0.221322-0.181322
    print -0.181322

if __name__=='__main__':
    params=np.zeros(2)
    params[0]=5
    params[1]=1
    test_fisher_info(params)
    test_inv_fisher(params)
    test_H(params)
