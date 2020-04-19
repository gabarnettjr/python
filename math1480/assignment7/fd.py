import numpy as np
import matplotlib.pyplot as plt
from math import factorial

#######################################

def getWeights(z, x, m):
    
    stc = len(x)
    
    A = np.zeros((stc, stc))
    b = np.zeros((stc, 1))
    
    x = x - z
    
    if (m < 0) or (not isinstance(m, int)):
        raise ValueError("m should be a nonnegative integer.")
    else:
        b[m] = factorial(m)
    
    for i in range(stc):
        A[i,:] = x**i
    
    w = np.linalg.solve(A, b)
    
    return w.flatten()

#######################################

def test_getWeights():
    
    z = 0
    
    x = np.array([-1,0,1])
    
    m = 2
    
    w = getWeights(z, x, m)
    
    print(w)

#######################################
    
def getDM(X, m, stc):
    
    N = len(X)
    
    W = np.zeros((N, N))
    
    if np.mod(stc, 2) == 1:
        n = np.int((stc-1) / 2)
    else:
        raise ValueError("Please choose an odd number for stc.")
    
    for i in range(n):
        W[i,0:stc] = getWeights(X[i], X[0:stc], m)
    
    for i in range(n, N-n):
        W[i,i-n:i+n+1] = getWeights(X[i], X[i-n:i+n+1], m)
    
    for i in range(n):
        W[N-n+i, N-stc:N] = getWeights(X[N-n+i], X[N-stc:N], m)         
    
    return W

#######################################

def test_getDM(stc):
    
    x = np.linspace(-1, 1, 26)
    
    def f(x):
        return np.sin(np.pi * x)
    
    def fp(x):
        return np.pi * np.cos(np.pi * x)
    
    def fpp(x):
        return -np.pi**2 * np.sin(np.pi * x)
    
    exact = fpp(x[1:-1])
    
    W_2 = getDM(x, 2, stc)
    
    approximate = W_2.dot(f(x))[1:-1]
    
    plt.figure()
    plt.clf()
    plt.plot(x[1:-1], exact, x[1:-1], approximate)
    plt.show()

#######################################

def bvp(a, b, c, f, yL, yR, x, stc):
    
    W_1 = getDM(x, 1, stc)
    W_2 = getDM(x, 2, stc)
    
    A = np.diag(a(x))
    B = np.diag(b(x))
    C = np.diag(c(x))
    
    ode = A.dot(W_2) + B.dot(W_1) + C
    ode[0,:] = 0.
    ode[0,0] = 1.
    ode[-1,:] = 0.
    ode[-1,-1] = 1.
    
    rhs = f(x)
    rhs[0] = yL
    rhs[-1] = yR
    
    y = np.linalg.solve(ode, rhs)
    
    return y.flatten()

#######################################

def test_bvp(testCase=1, N=17, stc=5, ptb = 0):
    
    if testCase == 1:
        def a(x):
            return np.ones(np.shape(x))
        def b(x):
            return np.zeros(np.shape(x))
        def c(x):
            return np.zeros(np.shape(x))
        def f(x):
            return -np.sin(x)
        yL = 0.
        yR = 0.
        x = np.linspace(-np.pi, np.pi, N)
        def yExact(x):
            return np.sin(x)
    elif testCase == 2:
        def a(x):
            return np.ones(np.shape(x))
        def b(x):
            return 5. * np.ones(np.shape(x))
        def c(x):
            return 6. * np.ones(np.shape(x))
        def f(x):
            return np.zeros(np.shape(x))
        yL = np.exp(2) + np.exp(3)
        yR = np.exp(-2) + np.exp(-3)
        x = np.linspace(-1, 1, N)
        def yExact(x):
            return np.exp(-2.*x) + np.exp(-3.*x)
    elif testCase == 3:
        def a(x):
            return np.ones(np.shape(x))
        def b(x):
            return np.ones(np.shape(x))
        def c(x):
            return -2. * np.ones(np.shape(x))
        def f(x):
            return x**2 + x + 1.
        yL = 1./2.
        yR = np.exp(1) + np.exp(-2) - 3.
        x = np.linspace(0, 1, N)
        def yExact(x):
            return np.exp(x) + np.exp(-2.*x) - 1./2.*x**2. - x - 3./2.
    else:
        raise ValueError("testCase should be 1, 2, or 3 please.")
    
    # Perturb the nodes by the fraction ptb
    x[1:-1] = x[1:-1] + (-ptb+2*ptb*np.random.rand(N-2)) \
        * (x[-1]-x[1])/(N-1)
    
    y = bvp(a, b, c, f, yL, yR, x, stc)
    
    plt.figure()
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(x, yExact(x), x, y)
    plt.title("exact and approximate")
    plt.legend(["exact", "approximate"])
    plt.subplot(1,2,2)
    plt.plot(x, y-yExact(x))
    plt.title("difference")
    plt.show()

#######################################
















