import numpy as np


def chebpoints(n):
    return np.cos(np.arange(n,dtype=np.float128)*np.pi/(n-1))

def Leja(n):
    # Leja Ordering Chebyshev nodes
    points = chebpoints(n + 1)[::-1]
    Points = points
    ord = np.arange(1, n + 1)

    LJ = np.zeros([1, n + 1])
    LJ[0] = 0
    M = 0

    for k in range(0, n):
        JJ = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(LJ[0, j])
                P = P * (points[idx_pts] - points[ord[i]])
            P = np.abs(P)
            if (P >= M):
                JJ = i
                M = P
        M = 0
        LJ[0, k + 1] = ord[JJ]
        ord = np.delete(ord, JJ)

    Leja_Points = np.zeros([1, n + 1])
    for i in range(n + 1):
        Leja_Points[0, i] = Points[int(LJ[0, i])]
    return Leja_Points


def Gamma_lp(m, n, gamma, gamma2, p):
        gamma0 = gamma.copy()
        gamma0[m - 1] = gamma0[m - 1] + 1

        norm = np.linalg.norm(gamma0.reshape(-1), p)
        if (norm < n and m > 1):
            o1 = Gamma_lp(m - 1, n, gamma.copy(), gamma.copy(), p)
            o2 = Gamma_lp(m, n, gamma0.copy(), gamma0.copy(), p)
            out = np.concatenate([o1, o2], axis=-1)
        elif (norm < n and m == 1):
            out = np.concatenate([gamma2, Gamma_lp(m, n, gamma0.copy(), gamma0.copy(), p)], axis=-1)
        elif (norm == n and m > 1):
            out = np.concatenate([Gamma_lp(m - 1, n, gamma.copy(), gamma.copy(), p), gamma0], axis=-1)
        elif (norm == n and m == 1):
            out = np.concatenate([gamma2, gamma0], axis=-1)
        elif (norm > n):
            norm_ = np.linalg.norm(gamma.reshape(-1), p)
            if (norm_ < n and m > 1):
                for j in range(1, m):
                    gamma0 = gamma.copy()
                    gamma0[j - 1] = gamma0[j - 1] + 1  # gamm0 -> 1121 broken
                    if (np.linalg.norm(gamma0.reshape(-1), p) <= n):
                        gamma2 = np.concatenate([gamma2, Gamma_lp(j, n, gamma0.copy(), gamma0.copy(), p)], axis=-1)
                out = gamma2
            elif (m == 1):
                out = gamma2
            elif (norm_ <= n):
                out = gamma
            else:
                out = []

        return out


if __name__ == '__main__':
    from scipy.special import roots_chebyt,roots_chebyu
    import scipy
    print("scipy version", scipy.__version__)
    import matplotlib.pylab as plt

    import h5py

    with h5py.File("chebpts.mat",'r') as chebpts:
        cp_10 = np.asarray(chebpts['cp_10'])[0]
        cp_50 = np.asarray(chebpts['cp_50'])[0]
        cp_100 = np.asarray(chebpts['cp_100'])[0]
        cp_500 = np.asarray(chebpts['cp_500'])[0]
        cp_1000 = np.asarray(chebpts['cp_1000'])[0]
        chebfun_points = [cp_10,cp_50,cp_100,cp_500,cp_1000]


    n_arr = np.array([10,50,100,500,1000])
    for i,n in enumerate(n_arr):
        a = chebfun_points[i]
        b = chebpoints(n)[::-1]#roots_chebyt(n)[0]
        #b=roots_chebyu(n)[0]
        abs_err = np.abs(a-b)
        rel_err = abs_err/(np.abs(a+b))
        print('mean abs err',abs_err.mean())
        print('mean rel err',rel_err.mean())
        plt.plot(n,abs_err.mean(),'or')
        plt.plot(n,rel_err.mean(),'Xk')
        plt.plot(n,abs_err.max(),'>g')

    plt.yscale('log')
    plt.show()
