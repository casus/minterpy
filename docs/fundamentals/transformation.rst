############################
Transformation between bases
############################

.. todo::

   This page should explain the transformation between supported polynomial bases,
   some important relations about it, and the reasons to do it.









   Newton and Lagrange interpolation in Unisolvent Nodes
   #####################################################

   For :math:`A= A_{m,n,p}`, :math:`m,n \in \mathbb{N}`, :math:`p\geq1` we assign the **unisolvent nodes**
   (See :doc:`../api/core/grid`) :math:`P_A` given by choosing :math:`n+1` **genrerating nodes**
   :math:`P_i \subseteq \mathbb{R}`, :math:`|P_i| = n+1` for each dimension :math:`1 \leq i \leq m` and generate the
   non-tensorial (non-symmetric) grid

   .. math::
     :label: eq_PA

     P_A = \left\{  (p_{1,\alpha_1}, \dots, p_{m,\alpha_m}) \in \mathbb{R}^m  \mid  \alpha \in A \,, p_{i,\alpha_i}\in P_i\right\}\,.

   By default the  :math:`P_i = (-1)^i\mathrm{Cheb}_n^{0}` are chosen as the Chebyshev extremes\ :footcite:`trefethen2019`,

   .. math::

     \mathrm{Cheb}_n^{0} = \left\{ \cos\Big(\frac{k\pi}{n}\Big) \mid 0 \leq k \leq n\right\}\,.

   **Give an example of the nodes**

   Polynomial interpolation goes back to Newton, Lagrange, and others\ :footcite:`meijering2002`, and its fundamental
   importance for mathematics and computing is undisputed. We derive a multivariate generalisation by defining:

   **Definition 1 (Multivariate polynomials)** Let :math:`A= A_{m,n,p}` and :math:`P_A\subseteq \mathbb{R}^m` be as in Eq. :eq:`eq_A`, :eq:`eq_PA`. Then, we define the **multivariate Lagrange polynomials** as

   .. math::

     L_{\alpha} \in \Pi_{P_A}\ \quad \text{with}\quad L_{\alpha}(p_\beta)= \delta_{\alpha,\beta}\, , \,\,\, \alpha,\beta \in A\,,

   where :math:`\delta_{\cdot,\cdot}` is the Kronecker delta. The **multivariate Newton polynomials**
   (see :doc:`../api/polyBases/newton`) are given by

   .. math::

     N_\alpha(x) = \prod_{i=1}^m\prod_{j=0}^{\alpha_i-1}(x_i-p_{j,i}) \,, \quad \alpha \in A\,.


   Finally, we call the monomials :math:`x^\alpha = \prod_{i=1}^m x^{\alpha_i}_{i}`, :math:`\alpha \in A` the
   **canonical basis** (see :doc:`../api/polyBases/canonical`) of :math:`\Pi_{A}`.


   Indeed, in dimension :math:`m=1` this reduces to the classic definition of Lagrange and Newton polynomials\ :footcite:`gautschi2012, stoer2002, trefethen2019`.
   Moreover, also the Newton and Lagrange polynomials are bases of :math:`\Pi_A`\ :footcite:`Hecht2020`.
   Therefore, the unique Lagrange interpolant :math:`Q_{f,A} \in \Pi_A` of a function
   :math:`f : \Omega \longrightarrow \mathbb{R}` on :math:`P_A` is given by

   .. math::

     Q_{f,A} = \sum_{\alpha \in A}f(p_{\alpha})L_{\alpha}(x)\,.

   However, while the Lagrange polynomials (see :doc:`../api/polyBases/lagrange`) are rather a mathematical concept this
   does not assert how to evaluate the interpolant :math:`Q_{f,A}` on a point
   :math:`x_0 \not \in P_A \subseteq \mathbb{R}^m`. To resolve that problem we have generalised the classic Newton
   interpolation scheme to mD:

   **Theorem 1 (Newton Interpolation)** Let :math:`A = A_{m,n,p}` and :math:`P_A\subseteq \mathbb{R}^m` be as in Eq. :eq:`eq_A`, :eq:`eq_PA` and let :math:`f : \Omega \subseteq \mathbb{R}^m \longrightarrow \mathbb{R}` be a function.
   Then, the Newton coefficients :math:`C = (c_{\alpha})_{\alpha \in A} \in \mathbb{R}^{|A|}` of the unique interpolant of :math:`f` in Newton form

   .. math::

     Q_{f,A}(x) = \sum_{\alpha \in A} c_\alpha N_{\alpha} (x)\,, \quad Q_{f,A} \in \Pi_A

   can be determined in :math:`\mathcal{O}(|A|^2)` operations requiring :math:`\mathcal{O}(|A|)` storage.

   Earlier versions of this statement were limited to the case where :math:`P_A` is given by a (sparse) tensorial grid\ :footcite:`Dyn2014`.
   In contrast, Theorem 1 also holds for our generalised notion of non-tensorial unisolvent nodes.
   **The DDS** functions realises a concrete (recursive divided difference scheme) implementation  of the algorithm explicitly described in\ :footcite:`Hecht2020`.

   Once the interpolant :math:`Q_{f,A}` is given in Newton form the following crucial consequences applies.

   **Theorem 2 (Evaluation and Differentiation in Newton form)** Let :math:`A= A_{m,n,p}` and :math:`P_A\subseteq \mathbb{R}^m` be as in Eq. :eq:`eq_A`, :eq:`eq_PA`,  :math:`x_0 \in \mathbb{R}^m`
   Let :math:`Q(x) = \sum_{\alpha \in A}c_\alpha N_{\alpha} \in \Pi_A`,
   :math:`C = (c_{\alpha})_{\alpha \in A} \in \mathbb{R}^{|A|}` be a polynomial in Newton form. Then:

   i) It requires :math:`\mathcal{O}(m|A|)` operations and :math:`\mathcal{O}(|A|)` storage to evaluate :math:`Q` at :math:`x_0`.
   ii) It requires :math:`\mathcal{O}(nm|A|)` operations and :math:`\mathcal{O}(|A|)` storage to evaluate the partial derivative :math:`\partial_{x_j}Q`, :math:`1 \leq j \leq m` at :math:`x_0`.

   In fact, all three basis  Newton, Lagrange and Canonical basis are inter-linked\ :footcite:`Hecht2020`.

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
   fast **barycentric transformations** is already used in the `minterpy` package. Current research aims to improve this technique and deliver further insights on the algorithmic optimality and complexity.
