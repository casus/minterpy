############################
Transformation between bases
############################


In case the multivariate polynomial :math:`Q \in \Pi_A`, :math:`A \subseteq \mathbb{N}^m`, is given in either of the considered (canonical, Newton or Lagrange bases)
on may asks for transforming it into another representation in order to benefit from the corresponding properties e.g. interpretability of **canonical polynomials**, **fast evaluation of Newton polynomials**
and approximation analysis for **Lagrange polynomials**. Here, we sketch how the transformations arise and are realised in ``minterpy``.






**Theorem 1 (Transformations)** Let :math:`A= A_{m,n,p}\,, m,n \in \mathbb{N}\,, p\geq 1` be a **multi-index set** and :math:`P_A\subseteq \Omega` be  unisolvent nodes and :math:`Q\in \Pi_A` be a polynomial. Then:

:math:`i)` Lower triangular matrices  :math:`\mathrm{NL}_A, \mathrm{LN}_A  \in \mathbb{R}^{|A|\times |A|}`  can be computed in :math:`\mathcal{O}(|A|^3)` operations, such that

.. math::
  \mathrm{LN}_A \cdot\mathrm{NL}_A = \mathrm{I} \,, \quad \mathrm{NL}_A  \cdot C_{\mathrm{Newt}} = C_{\mathrm{Lag}}\,, \quad   \mathrm{LN}_A\cdot C_{\mathrm{Lag}} = C_{\mathrm{Newt}} \,,

where :math:`C_{\mathrm{Lag}}=(c_{\alpha})_{\alpha \in A}\in \mathbb{R}^{|A|}` are the **Lagrange coefficients** of :math:`Q` and :math:`C_{\mathrm{Newt}}=(d_\alpha)_{\alpha \in A} \in \mathbb{R}^A` are the **Newton coefficients** of :math:`Q`.

:math:`ii)` Upper triangular matrices :math:`\mathrm{CL}_A,\mathrm{CN}_A \in \mathbb{R}^{|A|\times |A|}` can be computed in :math:`\mathcal{O}(|A|^3)` operations, such that

.. math::
  \mathrm{CL}_A\cdot C_{\mathrm{can}} =C_{\mathrm{Lag}}\,, \quad \mathrm{CN}_A\cdot C_{\mathrm{can}} =C_{\mathrm{Newt}}\,,

where :math:`C_{\mathrm{can}}=(d_{\alpha})_{\alpha \in A}  \in \mathbb{R}^{|A|}` denotes the  **canonical coefficients** of :math:`Q`.


Due to their triangualrity, the inverse matrices :math:`\mathrm{NC}_A =\mathrm{CN}_A^{-1}`, :math:`\mathrm{LC}_A =\mathrm{CL}_A^{-1}` can be computed in :math:`\mathcal{O}(|A|^2)`,
efficiently and accurately, yielding realisations of the inverse transformations.


If the **unisolvent nodes** :math:`P_A` are fixed, all matrices can be precomputed.
The columns of :math:`\mathrm{NL}_A` are given by **evaluating the Newton polynomials** on the unisolvent nodes, i.e,

.. math::

  \mathrm{NL}_A = (N_{\alpha}(p_\beta))_{\beta,\alpha \in A} \in \mathbb{R}^{|A|\times |A|}\,.


Thereby, Theorem 1 enables efficient and numerically accurate computation.
Vice versa, the **multivariate DDS scheme** can be used to interpolate the
**Lagrange polynomials** :math:`L_{\alpha} = \sum_{\beta\in A}d_{\alpha,\beta}N_{\beta}`, :math:`\alpha \in A` in Newton form, yielding the columns of
:math:`\mathrm{LN}_A`, i.e.,

.. math::

  \mathrm{LN}_A = (d_{\alpha,\beta})_{\beta,\alpha \in A} \in \mathbb{R}^{|A|\times |A|}\,.


Further,

.. math::

  \mathrm{CL}_A =(x^\alpha(p_{\beta}))_{\alpha,\beta \in A} \in \mathbb{R}^{|A|\times|A|}

coincides with the classic *Vandermonde matrix*\ :footcite:`gautschi2012` and the columns of :math:`\mathrm{CN}_A` are given by applying **DDS** to the canonical basis :math:`x^\alpha`.
In fact, all matrices are of recursive triangular sparse structure, which allows numerical accurate precomputation of the occurring sub-matrices, avoiding storage issues.

Consequently, the explicit structure of the matrices can be condensed into **barycentric transformations** performing much faster than classic matrix multiplication, resulting in
fast polynomial interpolation and  polynomial evaluation. A preliminary implementation of these
fast **barycentric transformations** is already used in ``minterpy``.


References
##########

.. footbibliography::
