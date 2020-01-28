from argparse import ArgumentParser

import concurrent.futures

from mpl_toolkits.mplot3d import Axes3D

import MultiIndicesTree
import Solvers
import utils
import numpy as np
import time

import matplotlib.pyplot as plt

np.random.seed(23232323)

def run_test(M,n,lpDegree):
    K = n
    t1 = time.time()
    #tr.PP: interpolation points
    #tr.
    tr = MultiIndicesTree.MultiIndicesTree(M=M,K=n,lpDegree=lpDegree)
    print("Tree built in %1.2es" % (time.time() - t1))


    nn, N = tr.PP.shape
    print("  - degree: %d" % (K))
    print("  - no. coefficients: %d" % (N))
    #C = np.arange(1, N + 1)

    ## interpolate function values F given coefficients C
    gamma = np.zeros([M])

    # F: the function values that we want to fit using our polynomial
    # groundtruth: evaluate some polynomial at randomly chosen points
    C = 2 * np.random.rand(N) - 1
    #g = lambda i: tr.eval_lp(i, C.copy(), M, K, N, gamma.copy(), tr.GP.copy(), tr.lpDegree, 1, 1)
    # groundtruth:
    #g = lambda x: np.sin( (2*x[0]) / np.pi + x[1] )
    g = lambda x: x[0]+x[1]


    t1 = time.time()
    F = np.zeros([N])
    for i in range(N):
        F[i] = g(tr.PP[:, i])
    print("Groundtruth generated in %1.2es" % (time.time() - t1))

    ## estimate parameters D using Divided Differences Scheme
    t1 = time.time()
    dds = Solvers.DDS()
    D = dds.run(M,N,tr.tree, F.copy(), tr.GP.copy(), gamma.copy(), 1, 1)
    print("DDS took %1.2es" % (time.time() - t1))

    """
    Evaluate polynomial at interpolation points (sanity check)
    """
    PPx = tr.PP
    _, noElem = PPx.shape
    F_hat = np.zeros([noElem])
    F2 = np.zeros([noElem])
    for i in range(noElem):
        F_hat[i] = tr.eval_lp(PPx[:, i].copy(), D.copy(), M, K, N, gamma.copy(), tr.GP.copy(), tr.lpDegree, 1, 1)
        F2[i] = g(PPx[:, i])

    ## some diagnostics
    res = F2 - F_hat
    print("----- sanity check")
    print("L2 " + str(np.linalg.norm(res)))
    print("Linfty " + str(np.linalg.norm(res,ord=np.inf)))
    print("-----")

    if(M>2):
        return
    """
    Evaluate polynomial on uniformly sampled grid
    """
    x = np.arange(-1,1,step=0.1)
    y = np.arange(-1,1,step=0.1)
    x,y = np.meshgrid(x,y)
    PPx = np.stack([x.reshape(-1),y.reshape(-1)])
    _, noElem = PPx.shape
    # evaluate polynomial
    F_hat = np.zeros([noElem])
    F2 = np.zeros([noElem])
    for i in range(noElem):
        F_hat[i] = tr.eval_lp(PPx[:, i].copy(), D.copy(), M, K, N, gamma.copy(), tr.GP, tr.lpDegree, 1, 1)
        F2[i] = g(PPx[:, i])

    ## some diagnostics
    res = F2 - F_hat

    print("----- Uniform sampling at [-1,1]^m")
    print("L2 " + str(np.linalg.norm(res)))
    print("Linfty " + str(np.linalg.norm(res,ord=np.inf)))
    print("-----")


    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(F2.reshape([len(x),len(x)]))
    ax1.set_title("gt")
    ax2.imshow(F_hat.reshape([len(x),len(x)]))
    ax2.set_title("estimate")
    plt.show()

    #plt.plot(C-D)
    #plt.title('residual: groundtruth - estimate')
    #plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", dest="m", type=int, help="input dimension", default = 3)
    parser.add_argument("-n", dest="n", type=int, help="polynomial degree", default = 20)
    parser.add_argument("-lp", dest="lp", type=float, help="LP order", default = 2)
    args = parser.parse_args()

    n = args.n
    M = args.m
    lpDegree = args.lp

    assert (M>1),"M must be larger than 1"

    run_test(M,n,lpDegree)
