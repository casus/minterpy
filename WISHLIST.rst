Implementation wish list
========================

General
-------
- agree on one shape of multi_indices: either (DIM,len_mi) (as used in the old code) or (len_mi,DIM) (more readable and used in MultivariatePolynomialABC)
- every setting of exponents in MultiIndex shall restore lex order in all related data (like Grid, coeffs, domains, ...)


class: CanonicalPolynomial
--------------------------
- Factorisation and Horner scheme for the eval function


class: MultivariatePolynomialABC
--------------------------------
- put in verification of the input (e.g. does user_domain fit the MultiIndex attributes)
- getter for exponents, lp_degree, spatial_dimesion, and poly_degree
- agree on the shape of user_domain, internal_domain: either (dim,2) or (2,dim)


class: Grid
-----------
- choose custom constructor depending on input type
- implement apply_func

class: MultiIndex
-----------------
- implement reduce function to remove dimensions where all exponents (along the resp. dim) are zero.
