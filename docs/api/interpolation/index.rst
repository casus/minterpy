.. currentmodule:: minterpy.interpolation

#############
Interpolation
#############

The main purpose of ``minterpy`` is the interpolation of given functions given as python-callables. For this application, we designed a easy to use one-call interface: ``minterpy.interpolate`` which returns a callable of type :class:`Interpolant` representing the given function as a multivariate (Newton) polynomial. Furthermore, one also can build an :class:`Interpolator` which precomputes and caches all necessary ingredients for the interpolation of arbitrary functions, defined on a domain of a given dimension.



.. toctree::
   :maxdepth: 2

   interpolate function <interpolate>
   Interpolant class <interpolant>
   Interpolator class <interpolator>
