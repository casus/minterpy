import MultiIndicesTree
from Solvers import interpol
import utils
import numpy as np
import time

from scipy.linalg import solve_triangular

import functools
import time


TIMING = True
TIMES = {}

def timer(func):
    """simple timing decorator"""
    if TIMING:
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start = time.time()    # 1
            value = func(*args, **kwargs)
            run_time = time.time() - start
            #print(f"%s finished in %1.2es"%(func.__name__,run_time))
            TIMES[func.__name__] = run_time
            return value
    else:
        wrapper_timer = func
    return wrapper_timer



class transform(interpol):
    @timer
    def __init__(self,M=2,n=2,lp=2):
        interpol.__init__(self,M,n,lp)
        self.__build_vandermonde_n2c()
        self.__build_transform_n2c()
        self.__build_vandermonde_l2n()
        self.__build_transform_l2n()
        self.__build_trans_Matrix()

    @timer
    def __build_vandermonde_n2c(self):
        self.init_gamma = np.zeros((self.M,1))
        self.trans_gamma = (utils.Gamma_lp(self.M, self.n, self.init_gamma, self.init_gamma.copy(), self.lp))
        self.V_n2c = np.ones((self.N,self.N))
        for i in np.arange(0,self.N):
            for j in np.arange(1,self.N):
                for d in np.arange(0,self.M):
                    self.V_n2c[i,j] = self.V_n2c[i,j]*self.tree.PP[d,i]**self.trans_gamma[d,j]

    @timer
    def __build_transform_n2c(self):
        self.Cmn_n2c = np.zeros((self.N,self.N))
        for j in np.arange(self.N):
            self.Cmn_n2c[:,j] =  self.run(self.M,self.N,self.tree.tree,self.V_n2c[:,j].copy(),self.tree.GP.copy(),self.gamma.copy(), 1, 1)
        self.inv_Cmn_n2c = solve_triangular(self.Cmn_n2c,np.identity(self.N))
    @timer
    def transform_n2c(self,d):
        return np.dot(self.inv_Cmn_n2c,d)

    @timer
    def __build_vandermonde_l2n(self):
        self.V_l2n=np.eye(self.V_n2c.shape[0])

    @timer
    def __build_transform_l2n(self):
        self.Cmn_l2n = np.zeros((self.N,self.N))
        for j in np.arange(self.N):
            self.Cmn_l2n[:,j] =  self.run(self.M,self.N,self.tree.tree,self.V_l2n[:,j].copy(),self.tree.GP.copy(),self.gamma.copy(), 1, 1)

    @timer
    def transform_l2n(self,l):
        #return solve_triangular(self.Cmn_l2n,l)
        return np.dot(self.Cmn_l2n,l)

    @timer
    def __build_trans_Matrix(self):
        self.trans_matrix = np.dot(self.inv_Cmn_n2c,self.Cmn_l2n)

    def transform_l2c(self,v):
        #return self.transform_n2c(self.transform_l2n(v))
        return np.dot(self.trans_matrix,v)




if __name__ == '__main__':
    import time
    times = {}
    np.random.seed(23232323)
    # M,n,lp
    input_para = (2,2,2)

    startFull = time.time()

    start = time.time()
    test_tr = transform(*input_para)
    times['init transform'] = time.time() - start


    lagrange_coefs = np.zeros(test_tr.N)
    base_coefs = np.random.uniform(-10,10,test_tr.N)
    start = time.time()
    for i in np.arange(test_tr.N):
        temp_lag = np.ones(test_tr.N)
        for j in np.arange(test_tr.N):
            for d in np.arange(test_tr.M):
                temp_lag[j] *= test_tr.tree.PP[d,i]**test_tr.trans_gamma[d,j]
            temp_lag[j]*=base_coefs[j]
        lagrange_coefs[i] = np.sum(temp_lag)
    times['build lagrange'] = time.time() - start

    start = time.time()
    newton = test_tr.transform_l2n(lagrange_coefs)
    times['transform to newton'] = time.time() - start

    start = time.time()
    #canon = test_tr.transform_n2c(newton)
    canon = test_tr.transform_l2c(lagrange_coefs)
    times['transform to canon'] = time.time() - start



    TIMES['full'] = time.time() - startFull

    print("---- results ----")
    #print('base',base_coefs)
    #print('lagrange',lagrange_coefs)
    #print('newton',newton)
    #print("base again", canon)
    abs_err = np.abs(base_coefs - canon)
    print("max abs_err",abs_err.max())
    rel_err = np.abs(abs_err/(base_coefs + canon))
    print("max rel_err",rel_err.max())

    print("---- times ---- ")
    for key in times.keys():
        print(key,"\n\t%1.2es"%times[key])

    print("---- internal times ----")
    for key in TIMES.keys():
        print(key,"\n\t%1.2es"%TIMES[key])

    #print("full time:",times['build lagrange'] + sum(TIMES.values()))

    #base_coefs = np.random.uniform(-10,10,6)
    #g = lambda x: np.dot(base_coefs,np.array([1,x[0],x[0]**2,x[1],x[0]*x[1],x[1]**2]))
    #test_inter(g)
