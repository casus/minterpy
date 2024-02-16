"""
This module contains the concrete implementation of ordinary regression.

A polynomial regression model is used when the given dataset do not lie
on the unisolvent nodes of an interpolating polynomial.
In such cases, the function values at the unisolvent nodes are obtained via
least-squares.
"""
import numpy as np
import scipy
from typing import Optional, Type, Union, Tuple, Callable

from minterpy.transformations import LagrangeToNewton
from minterpy.utils import eval_newton_monomials, eval_newton_polynomials
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.polynomials import (
    LagrangePolynomial,
    NewtonPolynomial,
    CanonicalPolynomial,
)
from minterpy.core.multi_index import MultiIndexSet
from minterpy.core.grid import Grid

from .regression_abc import RegressionABC

__all__ = ["OrdinaryRegression"]


class OrdinaryRegression(RegressionABC):
    """Implementation of an ordinary (weighted/unweighted) poly. regression.

    ``OrdinaryRegression`` fits a polynomial model specified by a multi-index
    set on a given dataset.

    Parameters
    ----------
    multi_index : :py:class:`.MultiIndexSet`, optional
        The multi-index set that defines the underlying polynomial.
        This parameter is optional if a grid is specified; in that case,
        the multi-index set is set to the one attached to the grid.
    grid : :py:class:`.Grid`, optional
        The grid where the polynomial lives.
        This parameter is optional if a multi-index set is specified;
        in that case, the grid is constructed from the specified
        multi-index set.
    origin_poly : Type[:py:class:`.MultivariatePolynomialSingleABC`], optional
        The polynomial basis on which the regression is carried out.
        This parameter is optional and, by default, is set to the
        :py:class:`.LagrangePolynomial`.
    """

    def __init__(
        self,
        multi_index: Optional[MultiIndexSet] = None,
        grid: Optional[Grid] = None,
        origin_poly: Type[MultivariatePolynomialSingleABC] = LagrangePolynomial,
    ):
        if multi_index is None and grid is None:
            raise ValueError(
                "Either a multi-index set or a grid must be specified!"
            )

        # Initialize and verify the grid
        if grid is None:
            # Create a grid from a verified multi-index set
            _verify_multi_index(multi_index)
            self._grid = Grid(multi_index)
        else:
            _verify_grid(grid)
            self._grid = grid

        # Verify multi-index set
        if multi_index is None:
            self._multi_index = grid.multi_index
        else:
            if len(multi_index) == 0:
                raise ValueError("MultiIndexSet must not be empty!")
            # Verify the multi-index set and its relation with the grid
            _verify_multi_index(multi_index)
            _verify_grid(self.grid, multi_index)
            self._multi_index = multi_index

        # Initialize the origin polynomial basis
        self._origin_poly = origin_poly(
            multi_index=self._multi_index, grid=self._grid
        )

        # Initialize additional internal attributes
        self._regression_matrix = None
        self._coeffs = None
        self._loocv_error = None
        self._regfit_linf_error = None
        self._regfit_l2_error = None
        self._eval_poly = None

    @property
    def multi_index(self) -> MultiIndexSet:
        """Multi-index set that defines the polynomial."""
        return self._multi_index

    @property
    def loocv_error(self) -> Tuple[float, float]:
        r"""Leave-one-out CV error of the fitted polynomial (abs. and norm.).

        The leave-one-out (LOO) cross-validation (CV) error is defined
        as follows:

        .. math::

           \epsilon_{LOO} = \frac{1}{N} \sum_{i = 1}^N \left( y^{(i)} - \hat{f}_{\setminus i} \left(\boldsymbol{x}^{(i)}\right) \right)^2

        where:

        - :math:`N` is the number of data points;
        - :math:`y^{(i)}` is the :math:`i`-th response data point;
        - :math:`\boldsymbol{x}^{(i)}` is the :math:`i`-th input data point;
        - :math:`\hat{f}_{\setminus i}` is the polynomial fitted on the dataset
          excluding the $i$-th data point.

        The normalized LOO-CV error is defined as follows:

        .. math::

           \epsilon_{LOO, \text{norm}} = \frac{\epsilon_{LOO}}{\mathbb{V}[\boldsymbol{y}]}

        where :math:`\mathbb{V}[\boldsymbol{y}]` denotes the variance
        of the response data (:math:`\boldsymbol{y}`).
        """
        return self._loocv_error

    @property
    def regfit_linf_error(self) -> Tuple[float, float]:
        r""":math:`L_{\infty}` regression fit error (in abs. and norm. terms).

        The :math:`L_{\infty}` regression fit error is defined as follows:

        .. math::

           \epsilon_{L_{\infty}} = \max_{i} \; \lvert y^{(i)} - \hat{f}\left(\boldsymbol{x}^{(i)}\right) \rvert

        where:

        - :math:`y^{(i)}` is the :math:`i`-th response data point;
        - :math:`\boldsymbol{x}^{(i)}` is the :math:`i`-th input data point;
        - :math:`\hat{f}` is the fitted polynomial.

        The normalized :math:`L_{\infty}` regression fit error is defined as follows:

        .. math::

           \epsilon_{L_{\infty}, \text{norm}} = \frac{\epsilon_{L_1}}{\sqrt{\mathbb{V}[\boldsymbol{y}]}}

        where :math:`\mathbb{V}[\boldsymbol{y}]` denotes the variance
        of the response data (:math:`\boldsymbol{y}`).
        """
        return self._regfit_linf_error

    @property
    def regfit_l2_error(self) -> Tuple[float, float]:
        r""":math:`L_2` regression fit error (in abs. and norm. terms).

        The :math:`L_2` regression fit error is defined as follows:

        .. math::

           \epsilon_{L_2} = \frac{1}{N} \sum_{i = 1}^N \left( y^{(i)} - \hat{f}\left(\boldsymbol{x}^{(i)}\right) \right)^2

        where:

        - :math:`N` is the number of data points;
        - :math:`y^{(i)}` is the :math:`i`-th response data point;
        - :math:`\boldsymbol{x}^{(i)}` is the :math:`i`-th input data point;
        - :math:`\hat{f}` is the fitted polynomial.

        The normalized :math:`L_2` regression fit error is defined as follows:

        .. math::

           \epsilon_{L_2, \text{norm}} = \frac{\epsilon_{L_2}}{\mathbb{V}[\boldsymbol{y}]}

        where :math:`\mathbb{V}[\boldsymbol{y}]` denotes the variance
        of the response data (:math:`\boldsymbol{y}`).
        """
        return self._regfit_l2_error

    @property
    def grid(self) -> Grid:
        """Grid on which the polynomial lives."""
        return self._grid

    @property
    def origin_poly(
        self
    ) -> Union[LagrangePolynomial, NewtonPolynomial, CanonicalPolynomial]:
        """Polynomial basis on which the regression is carried out."""
        return self._origin_poly

    @property
    def coeffs(self):
        """The fitted coefficients with respect to the origin poly. basis."""
        return self._coeffs

    @property
    def eval_poly(
        self
    ) -> Union[NewtonPolynomial, CanonicalPolynomial]:
        """Polynomial basis use for the prediction.

        Notes
        -----
        - Because Minterpy cannot directly evaluate a polynomial
          in the Lagrange basis on a set of query points,
          the corresponding polynomial in the Newton basis is
          created and used for prediction.
        """
        return self._eval_poly

    def get_regression_matrix(self, xx: np.ndarray) -> np.ndarray:
        r"""Get the regression matrix on a set of query points.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            Evaluation points, an array of shape :math:`N_e \times M`, where
            :math:`N_e` and :math:`M` correspond to the number of evaluation
            points and the number of spatial dimensions, respectively.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            Regression matrix in the chosen polynomial basis,
            an array of shape :math:`N_e \times N_b`, where
            :math:`N_e` and :math:`N_b` correspond to the number of evaluation
            points and the number of monomials, respectively.
        """
        return compute_regression_matrix(self._origin_poly, xx)

    def fit(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        weights: np.ndarray = None,
        lstsq_solver: Union[str, Callable] = "lstsq",
        **kwargs,
    ) -> None:
        r"""Fit the ordinary polynomial regression model.

        Fitting an (unweighted) ordinary polynomial regression model solves
        for the set of coefficients :math:`\boldsymbol{c}` in the following
        least-squares problem:

        .. math::

           \hat{\boldsymbol{c}} = \underset{\boldsymbol{c}}{\mathrm{arg min}} \lVert \boldsymbol{R} \boldsymbol{c} - \boldsymbol{y} \rVert_2^2

        where:

        - :math:`\lVert \cdot \rVert_2^2` denotes the square of the Euclidian
          norm;
        - :math:`\boldsymbol{R}` is the regression matrix evaluated
          on the input data points in the chosen polynomial basis;
        - :math:`\boldsymbol{y}` is the response data points.

        Moreover, if the weights matrix :math:`\boldsymbol{W}` are provided,
        then fitting a weighted ordinary polynomial regression model obtains
        the fitted coefficients :math:`\hat{\boldsymbol{c}}`
        from the following relation:

        .. math ::

           (\boldsymbol{R}^T \boldsymbol{W} \boldsymbol{R}) \hat{\boldsymbol{c}} = \boldsymbol{R}^T \boldsymbol{W} \boldsymbol{y}

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            Input matrix, also known as the training inputs.
        yy : :class:`numpy:numpy.ndarray`
            Response vector, observed or evaluated at ``xx``.
        weights : :class:`numpy:numpy.ndarray`, optional
            Individual weights for each input points.
            The default is ``None``.
        lstsq_solver : Union[:py:class:`str`, ``Callable``], optional
            Least-square solver. The default is ``"lstsq"`` from `SciPy`_.
            The additional following linear solvers are available
            as pre-defined:

            - ``"inv"``: (Gram) matrix inversion (NumPy)
            - ``"pinv"``: pseudo-inversion of the regression matrix (NumPy)
            - ``"dgesv"``: LU with full pivoting solver (SciPy)
            - ``"dsysv"``: Diagonal pivoting solver (SciPy)
            - ``"dposv"``: Cholesky decomposition solver (SciPy)
            - ``"qr"``: QR-decomposition-based solver (NumPy)
            - ``"svd"``: SVD-based solver (NumPy)
        compute_loocv : :py:class:`bool`, optional
            Flag to compute the leave-one-out cross-validation (LOO-CV) error.
            For a problem of certain size, the LOO-CV computation
            may be costly.
            The default is set to ``True``.

        Returns
        -------
        ``None``
            The instance itself is updated with a fitted polynomial.
            After a successful fitting, the instance can be evaluated on
            a set of query points.

        Notes
        -----
        - ``**kwargs`` may take additional keyword arguments that are passed
          to the selected least-square solver. Refer to the documentation
          of the solver for the list of supported keyword arguments.

        .. _Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
        """

        # Parse 'kwargs'
        # NOTE: Use "pop", don't propagate this argument further downstream
        compute_loocv = kwargs.pop("compute_loocv", True)
        if not isinstance(compute_loocv, bool):
            raise ValueError(
                "'compute_loocv' argument value must be of 'bool' type! "
                f"Got instead {type(compute_loocv)!r} ."
            )

        # Get the regression matrix on the data points
        self._regression_matrix = self.get_regression_matrix(xx)

        # Solve the least-squares problem to obtain the coefficients
        self._coeffs = solve_least_squares(
            self._regression_matrix, yy, weights, lstsq_solver, **kwargs
        )

        # Compute some error metrics
        self._regfit_linf_error = compute_regfit_linf_error(
            self._regression_matrix, self._coeffs, yy
        )
        self._regfit_l2_error = compute_regfit_l2_error(
            self._regression_matrix, self._coeffs, yy
        )
        if compute_loocv:
            self._loocv_error = compute_loocv_error(
                self._regression_matrix, self._coeffs, yy, weights
            )
        else:
            # Reset the property to None
            self._loocv_error = None

        # Create a polynomial that can be evaluated
        if isinstance(self.origin_poly, LagrangePolynomial):
            # Lagrange can't be evaluated, transform to Newton first
            lag_poly = LagrangePolynomial(
                multi_index=self.multi_index,
                coeffs=self._coeffs,
                grid=self.grid,
            )
            l2n = LagrangeToNewton(lag_poly)
            eval_poly = l2n()
        else:
            # Other polynomials can be directly evaluated
            eval_poly = type(self.origin_poly)(
                multi_index=self.multi_index,
                coeffs=self._coeffs,
                grid=self.grid,
            )

        self._eval_poly = eval_poly

    def predict(self, xx: np.ndarray) -> np.ndarray:
        r"""Predict using the fitted polynomial model a set of points.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            Query points, a two-dimensional array of shape
            :math:`N_q \times M`, where :math:`N_q` and :math:`M` correspond to
            the number of query points and the number of spatial dimensions,
            respectively.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            Predicted response at the query points, a one-dimensional
            array of length :math:`N_q`, the number of query points.

        Notes
        -----
        - A fitted polynomial regression model can be directly called on the
          set of points without accessing the ``predict()`` method.
        """
        if self._eval_poly is None:
            raise TypeError(
                "Ordinary regression model is not fitted, "
                "regression polynomial can't be evaluated!"
            )
        else:
            return self._eval_poly(xx)

    def show(self) -> None:
        """Show the summary of the polynomial regression model."""
        output = (
            f"Ordinary Polynomial Regression\n"
            f"------------------------------\n"
            f"Spatial dimension: {self.multi_index.spatial_dimension}\n"
            f"Poly. degree     : {self.multi_index.poly_degree}\n"
            f"Lp-degree        : {self.multi_index.lp_degree}\n"
            f"Origin poly.     : {type(self.origin_poly)}\n"
        )
        if self.regfit_linf_error is not None:
            output += (
                f"{'Error':<21s} {'Absolute':>12s} {'Relative':>12s}\n"
                f"{'L-inf Regression Fit':<21s}  {self.regfit_linf_error[0]:10.5e}  "
                f"{self.regfit_linf_error[1]:10.5e}\n"
                f"{'L2 Regression Fit':<21s}  {self.regfit_l2_error[0]:10.5e}  "
                f"{self.regfit_l2_error[1]:10.5e}"
            )
            if self.loocv_error is not None:
                output += (
                    f"\n{'LOO CV':<21s}  {self.loocv_error[0]:10.5e}  "
                    f"{self.loocv_error[1]:10.5e}"
                )

        print(output)

def compute_regression_matrix(
    basis_poly: MultivariatePolynomialSingleABC, xx: np.ndarray
) -> np.ndarray:
    """Construct a regression matrix of the chosen polynomial basis.

    The regression matrix is constructed by evaluation the monomials of
    the chosen basis on all the input points.

    TODO: This is a temporary solution as ideally each polynomial basis
          can call its own "eval_monomials_on" method to construct
          the regression matrix on the training points.
    """
    exponents = basis_poly.multi_index.exponents
    generating_points = basis_poly.grid.generating_points

    regression_matrix = None
    # Evaluate the basis at the trainining points
    # TODO: should have a method basis_poly.eval_monomials(xx),
    #       below are temporary solutions.
    if isinstance(basis_poly, LagrangePolynomial):
        # Compute the coefficients of the Newton basis
        # for the Lagrange monomials
        newton_coeffs = LagrangeToNewton(
            basis_poly
        ).transformation_operator.array_repr_full

        # Evaluate the Newton polynomials representing the Lagrange monomials
        # on the input points
        regression_matrix = eval_newton_polynomials(
            xx, newton_coeffs, exponents, generating_points
        )

        # Edge case: only a single training point is given
        if xx.shape[0] == 1:
            regression_matrix = regression_matrix[np.newaxis, :]

        # Edge case: only a single Lagrange basis is given
        if regression_matrix.ndim == 1:
            regression_matrix = regression_matrix[:, np.newaxis]

    elif isinstance(basis_poly, NewtonPolynomial):
        regression_matrix = eval_newton_monomials(
            xx, exponents, generating_points
        )

    elif isinstance(basis_poly, CanonicalPolynomial):
        regression_matrix = np.prod(
            np.power(xx[:, None, :], exponents[None, :, :]), axis=-1
        )

    else:
        raise TypeError(f"Polynomial {type(basis_poly)} is not supported!")

    return regression_matrix


def solve_least_squares(
    regression_matrix: np.ndarray,
    yy: np.ndarray,
    weights: np.ndarray,
    lstsq_solver: Union[str, Callable],
    **kwargs,
) -> np.ndarray:
    """Solve the least-squares problem with a specified solver.

    The following linear solvers are available as pre-defined:

    - ``lstsq``: least-squares solver (SciPy)
    - ``inv``: (Gram) matrix inversion (NumPy)
    - ``pinv``: pseudo-inversion of the regression matrix (NumPy)
    - ``dgesv``: LU with full pivoting solver (SciPy)
    - ``dsysv``: Diagonal pivoting solver (SciPy)
    - ``dposv``: Cholesky decomposition solver (SciPy)
    - ``qr``: QR-decomposition-based solver (NumPy)
    - ``svd``: SVD-based solver (NumPy)

    Parameters
    ----------
    regression_matrix : np.ndarray
        Regression matrix constructed on the chosen polynomial basis and
        training points.
    yy : np.ndarray
        Function values that correspond to the training points.
    weights: np.ndarray
        The weights associated with the function values.
    lstsq_solver : Union[str, Callable]
        Chosen least-square solvers, may be chosen from a pre-defined solver
        using a string or a user-defined function that satisfies a common
        interface.

    Returns
    -------
    np.ndarray
        The coefficients of the regression polynomial.
    """
    # If the weights are given, make a diagonal matrix out of it
    if weights is not None:
        if weights.ndim == 1:
            weights = np.diag(weights)

    if isinstance(lstsq_solver, Callable):
        # Solve with a user-defined least-squares solver
        coeffs = lstsq_solver(regression_matrix, yy, weights, **kwargs)

        return coeffs

    lstsq_solver = lstsq_solver.lower()

    if lstsq_solver == "lstsq":
        # Least-squares solver from SciPy (based on SVD)
        coeffs = _solve_lstsq(regression_matrix, yy, weights, **kwargs)

    elif lstsq_solver == "inv":
        # Matrix inversion from NumPy (inversion of the Gram matrix)
        coeffs = _solve_inv(regression_matrix, yy, weights)

    elif lstsq_solver == "pinv":
        # Pseudo-inverse (of the regression matrix) from NumPy
        coeffs = _solve_pinv(regression_matrix, yy, weights, **kwargs)

    elif lstsq_solver == "dgesv":
        # LU w/ full pivoting solver from SciPy
        coeffs = _solve_dgesv(regression_matrix, yy, weights, **kwargs)

    elif lstsq_solver == "dsysv":
        # Diagonal pivoting solver from SciPy
        coeffs = _solve_dsysv(regression_matrix, yy, weights, **kwargs)

    elif lstsq_solver == "dposv":
        # Cholesky decomposition solver from SciPy
        coeffs = _solve_dposv(regression_matrix, yy, weights, **kwargs)

    elif lstsq_solver == "qr":
        # QR-decomposition-based solver from NumPy
        coeffs = _solve_qr(regression_matrix, yy, weights)

    elif lstsq_solver == "svd":
        # SVD-based solver from NumPy
        coeffs = _solve_svd(regression_matrix, yy, weights)

    else:
        raise NotImplementedError(f"{lstsq_solver} solver not implemented!")

    return coeffs


def compute_loocv_error(
    regression_matrix: np.ndarray,
    coeffs: np.ndarray,
    yy: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """Calculate the leave-one-out (LOO) cross-validation (CV) errors.

    Parameters
    ----------
    regression_matrix : np.ndarray
        Regression matrix constructed on the chosen polynomial basis and
        training points.
    coeffs : np.ndarray
        Coefficients of the regression polynomial.
    yy : np.ndarray
        Function values that correspond to the training points.
    weights: np.ndarray
        Weights associated with the function values.

    Returns
    -------
    Tuple[float, float]
        The leave-one-out cross-validation errors, in absolute
        and relative (normalized) terms. The normalization is with respect to
        the function values sample variance.
    """
    # If the weights are given, make a diagonal matrix out of it
    if weights is None:
        weights = np.eye(len(yy))
    else:
        if weights.ndim == 1:
            weights = np.diag(weights)

    if regression_matrix.shape[0] >= regression_matrix.shape[1]:
        # Determined or over-determined system
        hi = np.diag(
            regression_matrix
            @ np.linalg.pinv(regression_matrix.T @ weights @ regression_matrix)
            @ regression_matrix.T
            @ weights
        )
        loo_cv = ((yy - regression_matrix @ coeffs) / (1 - hi)) ** 2
        idx = np.where(np.isnan(loo_cv))
        loo_cv[idx] = np.infty
        loo_cv_error = np.mean(loo_cv)
    else:
        # NOTE: Under-determined system, unreliable results
        loo_cv_error = np.infty

    return loo_cv_error, loo_cv_error / np.var(yy)


def compute_regfit_linf_error(
    regression_matrix: np.ndarray,
    coeffs: np.ndarray,
    yy: np.ndarray,
) -> Tuple[float, float]:
    """Calculate the absolute and normalized L-inf regression fit error.

    Calculate the absolute and normalized L-inf regression fit error.

    Parameters
    ----------
    regression_matrix : np.ndarray
        Regression matrix constructed on the chosen polynomial basis and
        training points.
    coeffs : np.ndarray
        Coefficients of the regression polynomial.
    yy : np.ndarray
        Function values that correspond to the training points.

    Returns
    -------
    Tuple[float, float]
        The L-inf regression fit error (i.e., max. of absolute error),
        in absolute and relative (normalized) terms.
        The normalization is with respect
        to the function values sample standard deviation.
    """
    linf_error = np.max(np.abs(yy - regression_matrix @ coeffs))

    return linf_error, linf_error / np.std(yy)


def compute_regfit_l2_error(
    regression_matrix: np.ndarray,
    coeffs: np.ndarray,
    yy: np.ndarray,
) -> Tuple[float, float]:
    """Calculate the absolute and normalized L2 regression fit error.

    Parameters
    ----------
    regression_matrix : np.ndarray
        Regression matrix constructed on the chosen polynomial basis and
        training points.
    coeffs : np.ndarray
        Coefficients of the regression polynomial.
    yy : np.ndarray
        Function values that correspond to the training points.

    Returns
    -------
    Tuple[float, float]
        The L2 regression fit error (i.e., mean-squared error), in absolute
        and relative (normalized) terms. The normalization is with respect to
        the function values sample variance.
    """
    l2_error = float(np.mean((yy - regression_matrix @ coeffs) ** 2))

    return l2_error, l2_error / np.var(yy)


def _verify_multi_index(multi_index: MultiIndexSet):
    """Verify the instance of MultiIndexSet passed to the constructor."""
    if not isinstance(multi_index, MultiIndexSet):
        raise TypeError(
            f"Unexpected type {type(multi_index)} "
            "of the input multi-index set!"
        )


def _verify_grid(grid: Grid, multi_index: Optional[MultiIndexSet] = None):
    """Verify the instance of Grid passed to the constructor"""
    if not isinstance(grid, Grid):
        raise TypeError(f"Unexpected type {type(grid)} of the input grid!")

    if multi_index is not None:
        if not grid.multi_index.is_superset(multi_index):
            raise ValueError(
                "The multi-indices of a polynomial must be a subset of "
                "the multi-indices of the grid in use!"
            )


def _solve_lstsq(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Solve a least-squares problem using lstsq solver (SciPy)."""
    if ww is None:
        coeffs, _, _, _ = scipy.linalg.lstsq(rr, yy, **kwargs)
    else:
        coeffs, _, _, _ = scipy.linalg.lstsq(
            rr.T @ ww @ rr, rr.T @ ww @ yy, **kwargs
        )

    return coeffs


def _solve_inv(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
) -> np.ndarray:
    """Solve a system of linear equations via matrix inversion."""
    if ww is None:
        ww = np.eye(len(yy))

    coeffs = np.linalg.inv(rr.T @ ww @ rr) @ rr.T @ ww @ yy

    return coeffs


def _solve_pinv(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Solve a least-squares problem via pseudo-inverse of the regr. matrix."""
    if ww is None:
        coeffs = np.linalg.pinv(rr, **kwargs) @ yy
    else:
        coeffs = np.linalg.pinv(rr.T @ ww @ rr, **kwargs) @ rr.T @ ww @ yy

    return coeffs


def _solve_dgesv(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Solve a least-squares problem using LU w/ full pivoting solver."""
    if ww is None:
        ww = np.eye(len(yy))

    _, _, coeffs, _ = scipy.linalg.lapack.dgesv(
        rr.T @ ww @ rr, rr.T @ ww @ yy, **kwargs
    )

    return coeffs


def _solve_dsysv(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Solve a least-squares problem using a diagonal pivoting solver."""
    if ww is None:
        ww = np.eye(len(yy))

    _, _, coeffs, _ = scipy.linalg.lapack.dsysv(
        rr.T @ ww @ rr, rr.T @ ww @ yy, **kwargs
    )

    return coeffs


def _solve_dposv(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Solve a least-squares problem using a Cholesky decompos. solver."""
    if ww is None:
        ww = np.eye(len(yy))

    _, coeffs, _ = scipy.linalg.lapack.dposv(
        rr.T @ ww @ rr, rr.T @ ww @ yy, **kwargs
    )

    return coeffs


def _solve_qr(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
) -> np.ndarray:
    """Solve a least-squares problem using a QR-decomposition-based solver."""
    if ww is None:
        if rr.shape[0] >= rr.shape[1]:
            # Determined or over-determined system
            qq_star, rr_star = np.linalg.qr(rr)
            coeffs = np.linalg.solve(rr_star, qq_star.T @ yy)
        else:
            # Under-determined system
            qq_star, rr_star = np.linalg.qr(rr.T)
            coeffs = qq_star @ np.linalg.inv(rr_star.T) @ yy
    else:
        qq_star, rr_star = np.linalg.qr(rr.T @ ww @ rr)
        coeffs = scipy.linalg.solve_triangular(
            rr_star, qq_star.T @ rr.T @ ww @ yy
        )

    return coeffs


def _solve_svd(
    rr: np.ndarray,
    yy: np.ndarray,
    ww: np.ndarray,
) -> np.ndarray:
    """Solve a least-squares problem using an SVD-based solver."""
    if ww is None:
        uu, s, vv_t = np.linalg.svd(rr)

        # Compute the pseudo-inverse of the singular value matrix
        ss_inv = np.zeros((rr.shape[1], rr.shape[0]))
        np.fill_diagonal(ss_inv, 1 / s)

        coeffs = (vv_t.T @ ss_inv @ uu.T) @ yy
    else:
        rr_star = rr.T @ ww @ rr
        # SVD-based solver from NumPy
        uu, s, vv_t = np.linalg.svd(rr_star)

        # Compute the pseudo-inverse of the singular value matrix
        ss_inv = np.zeros((rr_star.shape[1], rr_star.shape[0]))
        np.fill_diagonal(ss_inv, 1 / s)

        yy_star = rr.T @ ww @ yy
        coeffs = (vv_t.T @ ss_inv @ uu.T) @ yy_star

    return coeffs
