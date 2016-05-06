import autograd.numpy as np
from autograd import elementwise_grad, grad
from scipy import special
from scipy import stats
from scipy.stats import beta, gamma
from scipy import misc
import seaborn as sns
from matplotlib import pyplot as plt
import math
import pickle

all_gradients = []
lower_bounds = []
M=15
iteration = 1
def iterate(params,num_samples,num_particles,i,m,v):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    g = -grad_KL(params, num_samples,num_particles)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = 0.25
    params = params+a*m_h/(np.sqrt(v_h)+e)
    #params = params+a*g
    return params,m,v

def grad_KL(params, num_samples, num_particles):
    S = num_samples
    samples = sample_theta(params,S)
    #initialize KL to be this
    KL1 = gradient_log_variational(params,samples,0)
    KL1 *= log_variational(params,samples)-h_s(samples, num_particles)-c_i(params,0,S,num_particles)
    KL1 = np.sum(KL1)/S
    KL2 = gradient_log_variational(params,samples,1)
    KL2 *= log_variational(params,samples)-h_s(samples, num_particles)-c_i(params,1,S,num_particles)
    KL2 = np.sum(KL2)/S
    KL = np.array([KL1,KL2])
    return KL

def log_variational(params, theta):
    '''
    @summary: log-pdf of variational distribution
        '''
    mu=params[0]
    sigma=params[1]
    x = theta
    return loglognormal_np(np.log(x),mu,sigma)

#correct
def loglognormal_np( logx, mu, stddev ):
    log_pdf = -np.log(stddev) - 0.5*pow( (logx-mu)/stddev, 2.0 )-logx-0.5*np.log(2*np.pi)
    return log_pdf

#correct
def gradient_log_variational(params,theta, i):
    mu=params[0]
    sigma=params[1]
    x= theta
    if i==0:
        return (np.log(x)-mu)/(sigma**2)
    else:
        return (mu-np.log(x))**2/(sigma**3)-1/sigma

#correct
def gradient_check():
    params = np.array([2,2])
    h = np.array([1e-5,0])
    print (log_variational(params+h,0.5)-log_variational(params,0.5))/h[0]
    h = np.array([0,1e-5])
    print (log_variational(params+h,0.5)-log_variational(params,0.5))/h[1]
    print gradient_log_variational(params,0.5,0)
    print gradient_log_variational(params,0.5,1)

def h_s(theta,num_particles):
    h_s = log_prior_density(theta)+abc_log_likelihood(theta,num_particles)
    #h_s = abc_log_likelihood(theta,num_particles)
    #print h_s
    return h_s

#seems correct
def abc_log_likelihood(samples,num_particles):
    N=num_particles
    S = len(samples)
    log_kernels = np.zeros(N)
    ll = np.zeros(S)
    for s in range(S):
        theta = samples[s]
        x,std = simulator(theta,N)
        log_kernels = log_abc_kernel(x,std)
        ll[s] = misc.logsumexp(log_kernels)
        ll[s] = np.log(1./N)+ll[s]
    return ll

#correct
def simulator(theta,N):
    #get 500*N exponentials
    exponentials = np.random.exponential(1/theta,size=N*M)
    #reshape to Nx500
    exponentials = np.reshape(exponentials,(N,M))
    #get means of the rows
    summaries = np.mean(exponentials,1)
    std = np.std(exponentials,1)
    return summaries, std

#gets max likelihood at right point
def log_abc_kernel(x,std):
    '''
        @summary: kernel density, we use normal here
        @param x: simulator output, often the mean of kernel density
        @param e: bandwith of density
        '''
    
    e=std[0]/np.sqrt(M)
    #e = max(30./iteration,0.03)
    #e = 1
    Sx = x
    Sy = trueData()
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

def c_i(params,i,S,num_particles):
    return 0
    if S==1:
        return 0
    first = np.zeros(S)
    second = np.zeros(S)
    samples = sample_theta(params,S)
    first = h_s(samples,n,k,num_particles)*gradient_log_variational(params,samples,i)
    second = gradient_log_variational(params,samples,i)
    return np.cov(first,second)[0][1]/np.cov(first,second)[1][1]

#Correct
def sample_theta(params,S):
    '''
        @param params: list of parameters for recognition model, gamma
        @param S: number of samples
        '''
    return generate_lognormal(params,S)

#Correct
def log_prior_density(theta):
    alpha = 2
    beta = 0.5
    mu = np.log(alpha/beta)
    sigma = np.log(np.sqrt(alpha/(beta**2)))
    params = np.array([mu,sigma])
    return log_variational(params, theta)

#correct
def trueData():
    #np.random.seed(5)
    #true = np.mean(np.random.exponential(20,M))
    #np.random.seed()
    return 1

#Correct
def generate_lognormal(params,S):
    mu=params[0]
    sigma=params[1]
    Y = np.random.normal(mu,sigma,S)
    X = np.exp(Y)
    return X



if __name__=='__main__':
    #print trueData()
    #params = np.zeros(2)
    #params[0] = np.random.uniform(0,1)
    #params[1] = np.random.uniform(0,1)
    #m = np.array([0.,0.])
    #v = np.array([0.,0.])
    #lower_bounds = []
    #for i in range(500):
    #    params,m,v = iterate(params,50,50,i,m,v)
    #    iteration +=1
    #    if i%100==0:
    #        print params
    #print params
    #print "true mean"
    #print (M+1.)/(trueData()*M+1)
    #samples = generate_lognormal(params,10000)
    #print "estimated mean"
    #print np.mean(samples)
    #mu = params[0]
    #sigma = params[1]
    #x = np.linspace(0,3,100)
    ##fig, ax = plt.subplots(1, 1)
#plt#.plot(x, beta.pdf(x, a,b),'r-', lw=5, label='beta pdf',color='blue')
    ##plt.plot(x,np.exp(log_variational(params,x)),'r-', lw=5, label='variational',color='green')
    #plt.plot(x,stats.gamma.pdf(x,M+1,scale=1/(trueData()*M+1)),label='true exponential')
    #plt.legend()
    #plt.show()

    params = np.array([0.223545, 0.289477])
    #params = np.array([5,5])
    ted_gradients = pickle.load(open('gradients.pkl','rb'))
    num_samples = 10
    num_particles = 1
    for i in range(1000):
        all_gradients.append(grad_KL(params, num_samples,num_particles)[1])
    ted_gradients = np.asarray(ted_gradients)
    all_gradients = np.asarray(all_gradients)
    #plt.hist(all_gradients,color='orange')
    #sns.distplot(ted_gradients,hist=False)
    sns.kdeplot(ted_gradients,label='AVABC')
    #sns.distplot(all_gradients,hist=False)
    sns.kdeplot(all_gradients,label='VBIL')
    #print reduce(lambda x, y: x + y, all_gradients) / len(all_gradients)
    #plt.hist(ted_gradients)
    plt.legend()
    plt.show()

