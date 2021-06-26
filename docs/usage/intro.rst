============
Introduction
============

*Minterpy* is based on an optimised re-implementation of the multivariate interpolation prototype algorithm (**MIP**), given in [13]_ 
and thereby provides software solutions that lift the curse of dimensionality from interpolation tasks. 
While interpolation occurs as the bottleneck of most computational challenges, MINTERPY aims to free empirical sciences from their computational limitations. MINTERPY is continuously extended and improved by adding further functionality and modules that provide novel digital solutions to a broad field of computational challenges, including but not limited to:

*multivariate interpolation, non-linear polynomial regression, numerical integration, global (black box) optimisation, surface level set methods, non-periodic spectral PDE solvers on flat and complex geometries, 
machine learning regularisation, data reconstruction, computational solutions in algebraic geometry*

*Minterpy* is provided as an openSource PYTHON package that makes it easily accessable and allows further development and improvement by the PYTHON community. In the following short survey we give some crucial insights of the underlying theoretical aspects of multivariate polynomial interpolation.

Multivariate Polynomial Interpolation - A Short Survey
======================================================


Polynomial interpolation goes back to Newton, Lagrange, and others [14]_, and its fundamental importance for mathematics and computing is undisputed. 
Interpolation is based on the fact that, in 1D, one and only one polynomial :math:`Q_{f,n}` of degree :math:`n` can interpolate a function :math:`f : \mathbb{R} \longrightarrow \mathbb{R}` on :math:`n+1` distinct 
*unisolvent interpolation nodes*
:math:`P_n \subseteq \mathbb{R}` , i.e.,

.. math::

  Q_{f,n}(p_i) = f(p_i)\,, \quad \text{for all} \quad  p_i \in P_n \,, 0 \leq i \leq n\,.

This makes interpolation fundamentally different from approximation. For the latter, the famous *Weierstrass Approximation Theorem* [19]_ states that any continuous function 
:math:`f : \Omega =[-1,1]^m \longrightarrow \mathbb{R}` can be uniformly approximated by polynomials [8]_. However, the Weierstrass Approximation Theorem does not require the polynomials  to coincide with :math:`f` at all, i.e., it is possible that there is a sequence of multivariate polynomials :math:`Q_{f,n}`, :math:`n \in \mathbb{N}` with :math:`Q_{f,n}(x) \not = f(x)` for all :math:`x \in \Omega`, but still

.. math::

  Q_{f,n} \xrightarrow[n \rightarrow \infty]{} f \quad \text{uniformly on} \quad \Omega\,. 

Even though there are several constructive versions of the Weierstrass Approximation Theorem, with the earliest given by Serge Bernstein [1]_,
providing an algorithm for computing such approximations it only delivers a slow (inverse linear) convergence rate. In 1D, however, interpolation on **Chebyshev and Legendre nodes** is known to yield exponential approximation rates [18]_ for analytic functions, which is much faster than what has been shown possible by Weierstrass-type approximations [1]_. 
There has therefore been much research into extending :math:`1`\ D Newton or Lagrange interpolation schemes to multi–dimensions (mD) by maintaining their computational power. 
Any approach that addresses this problem has to guarantee uniform approximation of the interpolation target :math:`f` and resist the curse of dimensionality. That is:

.. figure:: mip_approximation.png
  :align: center
  
  Approximation errors rates for interpolating the Runge function in dimension m = 4.

We consider the :math:`l_p\text{-norm}\|x\|_p = \big(\sum_{i=1}^m x_i^p\big)^{1/p}`, :math:`x = (x_1,\dots,x_m) \in\mathbb{R}^m`, :math:`m \in \mathbb{N}` and the **lexicographical ordered multi-index sets**

.. math::
  :label: eq_A

  A=A_{m,n,p} = \left\{\alpha \in \mathbb{N}^m | \|\alpha\|_p \leq n \right\}\,, \quad m,n \in \mathbb{N}\,, p \geq 1\,. 


This notion generalises the 1D notion of polynomial degree to multi-dimensional :math:`l_p`-degree, i.e, we consider the polynomial spaces spanned by all monomials of bounded :math:`l_p`-degree 

.. math::
 
   \Pi_A = \mathrm{span} \left\{ x^\alpha = \prod_{i=1}^mx^{\alpha_i}\right\}\,, \quad \alpha = (\alpha_1,\dots,\alpha_m) \in A\,.


Given :math:`A=A_{m,n,p}` we ask for:

i) Unisolvent interpolation nodes :math:`P_A` that uniquely determine the interpolant :math:`Q_{f,A} \in \Pi_A` by satisfying :math:`Q_{f,A}(p_{\alpha}) = f(p_{\alpha})`, :math:`\forall p_{\alpha} \in P_A`, :math:`\alpha \in A`.
ii) An interpolation scheme **MIP/DDS** that computes the uniquely determined interpolant :math:`Q_{f,A} \in \Pi_A` efficiently and numerically accurate (with machine precision).
iii) The unisolvent nodes :math:`P_A` that scale sub-exponentially with the space dimension :math:`m \in \mathbb{N}`, :math:`|P_A| \in o(n^m)` and guarantee uniform approximation of even strongly varying functions (avoiding over fitting)  as the Runge function :math:`f_R(x) = 1/(1+\|x\|_2^2)` by fast (exponential) approximation rates.


In fact, the results of [13]_ suggest that the therein presented algorithm MIP resolves issues i) - iii) by choosing :math:`p=2`, i.e., yields :math:`|P_{A_{m,n,2}}| \approx \frac{(n+1)^m }{\sqrt{\pi m}} (\frac{\pi \mathrm{e}}{2m})^{m/2} \in o(n^m)` and 

.. math::

  Q_{f,A_{m,n,2}} \xrightarrow[n\rightarrow]{} f \quad \text{uniformly and fast (exponentially) on} \,\,\, \Omega\,.


Figure 1 shows the approximation rates of the classic Runge function [15]_ in dimension :math:`m=4`, which is known to cause over fitting when interpolated naively. There is an optimal approximation rate known [17]_, which we call the Trefethen rate. Spline-type interpolation is based on works of by Carl de Boor et al. [3]_, [4]_, [5]_, [6]_ is limited to reach only polynomial approximation rates [7]_. 
Similarly, interpolation by rational functions as in Floater-Hormann interpolation [2]_, [10]_ and tensorial Chebyshev interpolation, relying on :math:`l_{\infty}`-degree, [11]_ miss optimality. In contrast MIP reaches optimality. While relying on interpolating with respect to :math:`l_2`-degree instead of :math:`l_{\infty}`-degree MIP reduces the amount of samples needed to reach machine precision  compared to tensorial Chebyshev interpolation by about :math:`\sim 5 \cdot 10^7` samples in that case.





Newton and Lagrange Interpolation on Unisolvent Nodes
=====================================================

For :math:`A= A_{m,n,p}`, :math:`m,n \in \mathbb{N}`, :math:`p\geq1` we assign the **unisolvent nodes** :math:`P_A` given by choosing :math:`n+1` **genrerating nodes** :math:`P_i \subseteq \mathbb{R}`, :math:`|P_i| = n+1` for each dimension :math:`1 \leq i \leq m` and generate the non-tensorial (non-symmetric) grid 

.. math::
  :label: eq_PA

  P_A = \left\{  (p_{1,\alpha_1}, \dots, p_{m,\alpha_m}) \in \mathbb{R}^m  \mid  \alpha \in A \,, p_{i,\alpha_i}\in P_i\right\}\,.



By default the  :math:`P_i = (-1)^i\mathrm{Cheb}_n^{0}` are chosen as the Chebyshev extremes  [18]_, 

.. math::

  \mathrm{Cheb}_n^{0} = \left\{ \cos\Big(\frac{k\pi}{n}\Big) \mid 0 \leq k \leq n\right\}\,.

**Give an example of the nodes**

Polynomial interpolation goes back to Newton, Lagrange, and others [14]_, and its fundamental importance for mathematics and computing is undisputed. We derive a multivariate generalisation by defining:

**Definition 1 (Multivariate polynomials)** Let :math:`A= A_{m,n,p}` and :math:`P_A\subseteq \mathbb{R}^m` be as in Eq. :eq:`eq_A`, :eq:`eq_PA`. Then, we define the **multivariate Lagrange polynomials** as 

.. math::

  L_{\alpha} \in \Pi_{P_A}\ \quad \text{with}\quad L_{\alpha}(p_\beta)= \delta_{\alpha,\beta}\, , \,\,\, \alpha,\beta \in A\,,

where :math:`\delta_{\cdot,\cdot}` is the Kronecker delta. The **multivariate Newton polynomials** are given by

.. math::

  N_\alpha(x) = \prod_{i=1}^m\prod_{j=0}^{\alpha_i-1}(x_i-p_{j,i}) \,, \quad \alpha \in A\,.


Finally, we call the monomials :math:`x^\alpha = \prod_{i=1}^m x^{\alpha_i}_{i}`, :math:`\alpha \in A` the **canonical basis** of :math:`\Pi_{A}`.


Indeed, in dimension :math:`m=1` this reduces to the classic definition of Lagrange and Newton polynomials [12]_, [16]_, [18]_. Moreover, also the Newton and Lagrange polynomials are bases of :math:`\Pi_A` [13]_. 
Therefore, the unique  Lagrange interpolant :math:`Q_{f,A} \in \Pi_A` of a function :math:`f : \Omega \longrightarrow \mathbb{R}` on :math:`P_A` is given by 

.. math::

  Q_{f,A} = \sum_{\alpha \in A}f(p_{\alpha})L_{\alpha}(x)\,.

However, while the Lagrange polynomials are rather a mathematical concept this does not assert how to evaluate  the interpolant :math:`Q_{f,A}` on a point :math:`x_0 \not \in P_A \subseteq \mathbb{R}^m`. 
To resolve that problem we have generalised the classic Newton interpolation scheme to mD:



**Theorem 1 (Newton Interpolation)** Let :math:`A = A_{m,n,p}` and :math:`P_A\subseteq \mathbb{R}^m` be as in Eq. :eq:`eq_A`, :eq:`eq_PA` and let :math:`f : \Omega \subseteq \mathbb{R}^m \longrightarrow \mathbb{R}` be a function. 
Then, the Newton coefficients :math:`C = (c_{\alpha})_{\alpha \in A} \in \mathbb{R}^{|A|}` of the unique interpolant of :math:`f` in Newton form

.. math::

  Q_{f,A}(x) = \sum_{\alpha \in A} c_\alpha N_{\alpha} (x)\,, \quad Q_{f,A} \in \Pi_A

can be determined in :math:`\mathcal{O}(|A|^2)` operations requiring :math:`\mathcal{O}(|A|)` storage. 

Earlier versions of this statement were limited to the case where :math:`P_A` is given by a (sparse) tensorial grid [9]_. 
In contrast, Theorem 1 also holds for our generalised notion of non-tensorial unisolvent nodes. 
**The DDS** functions realises a concrete (recursive divided difference scheme) implementation  of the algorithm explicitly described in [13]_.


Once the interpolant :math:`Q_{f,A}` is given in Newton form the following crucial consequences applies. 

**Theorem 2 (Evaluation and Differentiation in Newton form)** Let :math:`A= A_{m,n,p}` and :math:`P_A\subseteq \mathbb{R}^m` be as in Eq. :eq:`eq_A`, :eq:`eq_PA`,  :math:`x_0 \in \mathbb{R}^m` 
Let :math:`Q(x) = \sum_{\alpha \in A}c_\alpha N_{\alpha} \in \Pi_A`,
:math:`C = (c_{\alpha})_{\alpha \in A} \in \mathbb{R}^{|A|}` be a polynomial in Newton form. Then: 

i) It requires :math:`\mathcal{O}(m|A|)` operations and :math:`\mathcal{O}(|A|)` storage to evaluate :math:`Q` at :math:`x_0`.
ii) It requires :math:`\mathcal{O}(nm|A|)` operations and :math:`\mathcal{O}(|A|)` storage to evaluate the partial derivative :math:`\partial_{x_j}Q`, :math:`1 \leq j \leq m` at :math:`x_0`.


In fact, all three basis  Newton, Lagrange and Canonical basis are inter-linked [13]_.

**Theorem 3 (Transformations)**  
Let :math:`A= A_{m,n,p}` and :math:`P_A\subseteq \mathbb{R}^m` be as in Eq. :eq:`eq_A`, :eq:`eq_PA`, :math:`f : \mathbb{R}^m \longrightarrow  \mathbb{R}` be a function and :math:`F=\big(f(p_\alpha)\big)_{\alpha \in A}\in \mathbb{R}^{|A|}`. Then:

i) Lower triangular matrices  :math:`\mathrm{NL}_A, \mathrm{LN}_A  \in \mathbb{R}^{|A|\times |A|}`  can be computed in :math:`\mathcal{O}(|A|^3)` operations, such that 

  .. math::

     \mathrm{LN}_A \cdot\mathrm{NL}_A = \mathrm{I} \,, \quad \mathrm{NL}_A  \cdot C_{\mathrm{Newt}} = C_{\mathrm{Lag}}\,, \,\,\,  \mathrm{LN}_A\cdot C_{\mathrm{Lag}} = C_{\mathrm{Newt}} \,,

 where :math:`C_{\mathrm{Lag}}=F \in \mathbb{R}^{|A|}` are the **Lagrange coefficients** and :math:`C_{\mathrm{Newt}} \in \mathbb{R}^A` the **Newton coefficients** of :math:`Q_{f,A} \in \Pi_A`.

ii) Upper triangular matrices :math:`\mathrm{CL}_A,\mathrm{CN}_A \in \mathbb{R}^{|A|\times |A|}` can be computed in :math:`\mathcal{O}(|A|^3)` operations, such that 

  .. math::

    \mathrm{CL}_A\cdot C_{\mathrm{can}} =C_{\mathrm{Lag}}\,, \quad \mathrm{CN}_A\cdot C_{\mathrm{can}} =C_{\mathrm{Newt}}\,,

 where :math:`C_{\mathrm{can}}=(d_{\alpha})_{\alpha \in A}  \in \mathbb{R}^{|A|}` denotes the  **canonical coefficients** of :math:`Q_{f,A}\in \Pi_A`.


**Remark 1** If :math:`P_A` is fixed, all matrices can be precomputed. In fact the columns of :math:`\mathrm{NL}_A` are given by **evaluating the Newton polynomials**, i.e., 
:math:`C_{\alpha} = (N_{\alpha}(p_\beta))_{\beta \in A} \in \mathbb{R}^{|A|}`. Thereby, Theorem 2 enables efficient and numerically accurate computation. 
Vice versa, the **DDS scheme** from Theorem 1 can be used to interpolate the 
**Lagrange polynomials** :math:`L_{\alpha}`, :math:`\alpha \in A` in Newton form, i.e, the resulting **Newton coefficients** :math:`C_\alpha=(c_{\alpha,\beta})_{\beta \in A} \in \mathbb{R}^{|A|}` are the columns of :math:`\mathrm{LN}_A`.  
In particular, :math:`\mathrm{CL}_A =(x^\alpha(p_{\beta}))_{\alpha,\beta \in A} \in \mathbb{R}^{|A|\times|A|}` coincides with the classic Vandermonde matrix and the columns of :math:`\mathrm{CN}_A` are given by applying **DDS** to the canonical basis :math:`x^\alpha`. 

**Remark 2** In fact, all matrices are of recursive triangular sparse structure, which allows numerical accurate precomputation of the occurring sub-matrices, avoiding storage issues. Consequently, the explicit structure of :math:`LN,NL` can be condensed into **barycentric transformations** performing much faster than classic matrix multiplication, resulting in 
fast interpolation, evaluation and even differentiation. A preliminary implementation of these
fast **barycentric transformations** is already used in the *minterpy* package. Current research aims to improve this technique and deliver further insights on the algorithmic optimality and complexity. 


.. [1] Serge Bernstein. Sur l’ordre de la meilleure approximation des fonctions continues par des polynômes de degré donné, volume 4. Hayez, imprimeur des académies royales, 1912.

.. [2] Emiliano Cirillo, Kai Hormann, and Jean Sidon. Convergence rates of derivatives of Floater–Hormann interpolants for well-spaced nodes. Applied Numerical Mathematics, 116:108–118, 2017.

.. [3] Carl De Boor. On calculating with B-splines. Journal of Approximation theory, 6(1):50–62, 1972.

.. [4] Carl De Boor. Efficient computer manipulation of tensor products. Technical report, Wisconsin Univ. Madison Mathematics Research Center, 1977.

.. [5] Carl De Boor. On wings of splines. In Creative Minds, Charmed Lives: Interviews at Institute for Mathematical Sciences, National University of Singapore, pages 50–57. World Scientific, 2010.

.. [6] Carl De Boor, Carl De Boor, Etats-Unis Mathématicien, Carl De Boor, and Carl De Boor. A practical guide to splines, volume 27. Springer-Verlag New York, 1978.

.. [7] Carl De Boor and Klaus Höllig. Approximation power of smooth bivariate pp functions. Mathematische Zeitschrift, 197(3):343–363, 1988.

.. [8] Louis De Branges. The Stone-Weierstrass Theorem. Proceedings of the American Mathematical Society, 10(5):822–824, 1959.

.. [9] Nira Dyn and Michael S Floater. Multivariate polynomial interpolation on lower sets. Journal of Approximation Theory, 177:34–42, 2014.

.. [10] Michael S Floater and Kai Hormann. Barycentric rational interpolation with no poles and high rates of approximation. Numerische Mathematik, 107(2):315–331, 2007.

.. [11]  Simen Gaure. Usage notes for package chebpol, https://cran.r-project.org/package=chebpol. 2018.

.. [12]  Walter Gautschi. Numerical analysis. Springer Science & Business Media, 2011.

.. [13] Michael Hecht, Kryzsztof Gonciarz, Jannik Michelfeit, Vladimir Sivkin, and Ivo F. Sbalzarini. Multivariate interpolation on unisolvent nodes, 2020.

.. [14] E Meijering. A chronology of interpolation: From ancient astronomy to modern signal and image processing. Proceedings of the IEEE, 90(3):319–342, March 2002.

.. [15] Carl Runge. Über empirische Funktionen und die Interpolation zwischen äquidistanten Ordinaten. Zeitschrift für Mathematik und Physik, 46(224-243):20, 1901.

.. [16] Josef Stoer, Roland Bulirsch, Richard H. Bartels, Walter Gautschi, and Christoph Witzgall. Introduction to numerical analysis. Texts in applied mathematics. Springer, New York, 2002.

.. [17] Lloyd N. Trefethen. Multivariate polynomial approximation in the hypercube. Proceedings of the American Mathematical Society, 145(11):4837–4844, 2017.

.. [18] Lloyd N. Trefethen. Approximation theory and approximation practice, volume 164. SIAM, 2019.

.. [19] Karl Weierstrass. Über die analytische Darstellbarkeit sogenannter willkürlicher Funktionen einer reellen Veränderlichen. *Sitzungsberichte der Königlich Preußischen Akademie der Wissenschaften zu Berlin*, 2:633–639, 1885.




