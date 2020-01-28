import itertools as it
import numpy as np


class gaussPoints(object):
    r"""processed gauss points and weights

    Parameters
    ----------
    deg : int
        degree of gaussian quadrature (default=5)

    mode : str
        flag for the mode used

    Attributes
    ----------
    mode : str
        returns the mode of the instance

    points : np.ndarray
        array of gauss points (shape=(deg,))

    weights : np.ndarray
        array of gauss weights (shape=(deg,))

    bounds : tuple
        current boundaries of the gauss points

    Methods
    -------
    transfrom(low,up)
        transforms the gauss points and weights to the boundaries (low,up)

    Notes
    -----
    -   the modes are
         - gauss legendre: 'gaussLeg' (default)
         - gauss chebychev: 'gaussCheb' (not tested jet)
         - gauss laguerre: 'gaussLag' (not tested jet)
    """
    def __init__(self,deg=5,mode="gaussLeg"):
        self.__deg = deg
        self.mode=mode
        self.__setPoints()
        self.__resetPoints()

    def __setPoints(self):
        if self.mode=="gaussLeg":
            polynomial=np.polynomial.legendre.leggauss
        elif self.mode=="gaussCheb":
            polynomial=np.polynomial.chebyshev.chebgauss
        elif self.mode=="gaussLag":
            polynomial=numpy.polynomial.laguerre.laggauss
        else:
            raise ValueError("<%s> is not a mode of gaussPoints!")

        self.__initPoints, self.__initWeights=polynomial(self.__deg)

    def __resetPoints(self):
        self.points, self.weights = self.__initPoints,self.__initWeights
        self.bounds = (-1.0,1.0)

    def transform(self,low,up):
        r""" transformation of the boundaries

        Parameters
        ----------
        low : float
            lower boundary
        up : float
            upper boundary

        Returns
        -------
        None
            sets points and weights of the instance to the new boundaries

        Notes
        -----
            -   every call: calculation starts from (-1,1), internally saved
            -   new gauss points and weights are

                .. math:: \hat x_i = \frac{b-a}{2}x_i + \frac{a+b}{2}
                .. math:: \hat w_i = \frac{b-a}{2}w_i

                where :math:`\hat x_i,\hat w_i` are the gauss points/weights to boundaries :math:`(a,b)` and :math:`x_i,w_i` the gauss points/weights to boundaries :math:`(-1,1)`

        """
        self.points = (up-low)/2.0*self.__initPoints + (up+low)/2.0
        self.weights = (up-low)/2.0*self.__initWeights
        self.bounds = (low,up)




class ngauss_quad(object):
    def __init__(self,dim,num_pts):
        self.gauss = gaussPoints(deg = num_pts)
        self.gauss.transform(-1,1)
        self.pts_1d, self.weights_1d = self.gauss.points, self.gauss.weights
        self.pts_nd = np.array([t for t in it.product(*[self.pts_1d for _ in np.arange(dim)])]).T
        self.weights_nd = np.prod(np.array([t for t in it.product(*[self.weights_1d for _ in np.arange(dim)])]).T,axis=0)

    def integrate(self,func):
        vals = func(self.pts_nd)
        return np.dot(self.weights_nd,vals)
