import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy import stats
from scipy.stats import beta
from scipy import misc
from matplotlib import pyplot as plt
import math

all_gradients = []
lower_bounds = []

def fisher_info(params):
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    I[0][0]=special.polygamma(1,alpha)-special.polygamma(1,alpha+beta)
    I[0][1]=-special.polygamma(1,alpha+beta)
    I[1][0]=-special.polygamma(1,alpha+beta)
    I[1][1]=special.polygamma(1,beta)-special.polygamma(1,alpha+beta)
    return I

def inv_fisher(params):
    alpha = params[0]
    beta = params[1]
    I = np.zeros((2,2))
    a=special.polygamma(1,alpha)-special.polygamma(1,alpha+beta)
    b=-special.polygamma(1,alpha+beta)
    c=-special.polygamma(1,alpha+beta)
    d=special.polygamma(1,beta)-special.polygamma(1,alpha+beta)
    const = 1./(a*d-b*c)
    I[0][0]=d
    I[0][1]=-b
    I[1][0]=-c
    I[1][1]=a
    I=const*I
    return I

def H(params,n,k):
    alpha = params[0]
    beta = params[1]
    H=np.zeros(2)
    H[0]=k*special.polygamma(1,alpha)-n*special.polygamma(1,alpha+beta)
    H[1]=(n-k)*special.polygamma(1,beta)-n*special.polygamma(1,alpha+beta)
    return H

def iterate_nat_grad(params,i,n,k,num_samples,num_particles):
    a = 1./(5+i)
    samples = sample_theta(params,num_samples)
    H_val = np.array([H_i(samples,params,n,k,0,num_particles),H_i(samples,params,n,k,1,num_particles)])
    H_true = H(params,n,k)
    params = params-a*(params-np.dot(inv_fisher(params),H_val))
    all_gradients.append(params-np.dot(inv_fisher(params),H_val))
    #lower bound
    #print samples, params[0], params[1]
    #lower_bound =np.mean(h_s(samples,n,k,num_particles)-np.log(stats.beta.pdf(samples,params[0],params[1])))
    #lower_bounds.append(lower_bound)
    #print 'lower bound'
    #print lower_bound
    return params

def H_i(samples,params,n,k,i,num_particles):
    H_i = 0
    S = len(samples)
    c = c_i(params,n,k,i,S,num_particles)
    inner = (h_s(samples,n,k,num_particles)-c)*gradient_log_recognition(params,samples,i)
    H_i = np.mean(inner)
    return H_i

def c_i(params,n,k,i,S,num_particles):
    first = np.zeros(S)
    second = np.zeros(S)
    samples = sample_theta(params,S)
    first = h_s(samples,n,k,num_particles)*gradient_log_recognition(params,samples,i)
    second = gradient_log_recognition(params,samples,i)
    return np.cov(first,second)[0][1]/np.cov(first,second)[1][1]


def sample_theta(params,S):
    '''
    @param params: list of parameters for recognition model, gamma
    @param S: number of samples
    '''
    return np.random.beta(params[0],params[1],size=S)

def h_s(theta,n,k,num_particles):
    h_s = np.log(prior_density(theta))+abc_log_likelihood(theta,n,k,num_particles)
    #print h_s
    return h_s

def abc_log_likelihood(samples,n,k,num_particles):
    N=num_particles
    S = len(samples)
    log_kernels = np.zeros(N)
    ll = np.zeros(S)
    for s in range(S):
        theta = samples[s]
        x = simulator(theta,n,N)
        log_kernels = log_abc_kernel(x,k)
        ll[s] = misc.logsumexp(log_kernels)
        ll[s] = np.log(1./N)+ll[s]
    return ll

def simulator(theta,n,N):
    return np.random.binomial(n,theta,size=N)

def log_abc_kernel(x,k):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    #e=np.std(x)/np.sqrt(len(data))
    e = 0.5
    Sx = x
    Sy = k
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

def log_likelihood(theta, n, k):
    return np.log(stats.binom.pmf(k,n,theta))
    
def gradient_log_recognition(params,theta,i):
    alpha = params[0]
    beta = params[1]
    if i==0:
        return np.log(theta)-special.polygamma(0,alpha)+special.polygamma(0,alpha+beta)
    if i==1:
        return np.log(1-theta)-special.polygamma(0,beta)+special.polygamma(0,alpha+beta)

def prior_density(theta):
    return stats.beta.pdf(theta,1,1)

def numerical_gradient_log_recognition(params,theta,i):
    alpha = params[0]
    beta = params[1]
    h = 0.000001
    if i==0:
        f1 = stats.beta.logpdf(theta,alpha,beta)
        f2 = stats.beta.logpdf(theta,alpha+h,beta)
    else:
        f1 = stats.beta.logpdf(theta,alpha,beta)
        f2 = stats.beta.logpdf(theta,alpha,beta+h)
    return (f2-f1)/h

def run_VBIL(start_params,n,k,num_samples,num_particles,num_iterations):
    true_alpha = k+1.
    true_beta = n-k+1.
    e_alpha = 0
    e_beta = 0
    params = start_params
    i=0
    '''
    convergence = 0
    while convergence==0:
        if i>1000:
            convergence = 1
    '''
    for i in range(1,num_iterations):
       params = iterate_nat_grad(params,i,n,k,num_samples,num_particles)
       if i%1==0:
           print "param estimates"
           print params
           e_alpha = params[0]
           e_beta = params[1]
           print "true params"
           print true_alpha, true_beta
           print "mean is %f" % (e_alpha/(e_alpha+e_beta))
           print "true mean is %f" % (true_alpha/(true_alpha+true_beta))
    true_params = np.array([true_alpha, true_beta])
    return params,true_params

if __name__=='__main__':
    #note that for n=100,k=80, we use 500,100,1000
    # for n=100,k=20, we use 500,300,300
    params = np.random.uniform(10,100,2)
    n=100
    k=20
    #samples, particles, iterations
    params,true_params=run_VBIL(params,n,k,500,300,100)
    x = np.linspace(0,1,100)
    #plt.plot(x, beta.pdf(x,true_params[0],true_params[1]),'--', lw=2.5, label='true',color='red')
    #plt.plot(x, beta.pdf(x,params[0],params[1]),'r-', label='VBIL',color='green')
    #plt.plot(x, kumaraswamy_pdf(x,params_ABC),'r-', label='AD',color='blue')
    #plt.legend(loc=2)
    #plt.title('Bernoulli Problem M=100,k=80')
    all_gradients = np.asarray(all_gradients)
    running_var = []
    for i in range(1,len(all_gradients)):
        running_var.append(np.var(all_gradients[0:i])/i)
    plt.hist(all_gradients)
    plt.show()

    plt.show()
