######################################
Evaluation of multivariate polynomials
######################################

..
    .. todo::

       This page should explain what it means (computationally) to evaluate
       interpolating polynomials and the algorithmic complexity associated with it.



Once a multivariate polynomial :math:`Q \in \Pi_A`, :math:`A\subseteq \mathbb{N}^m` in :math:`m \in \mathbb{N}` variables
is derived one may ask for its value :math:`Q(p) \in \mathbb{R}` at a given argument :math:`p \in \Omega=[-1,1]^m`.
In the following we sketch how this value can be computed depending on the representation of the given polynomial :math:`Q \in \Pi_A`.


Canonical evaluation
####################

The current ``minterpy`` implementation for evaluating a **canonical polynomial** in :math:`p \in \Omega=[-1,1]^m`

.. math::
  Q(x) = \sum_{\alpha \in A} c_\alpha x^\alpha \in \Pi_A\,, \quad c_{\alpha} \in \mathbb{R}

is realised naïvly by computing the value :math:`p^\alpha` for each monomial and determine :math:`\sum_{\alpha \in A} c_\alpha p^\alpha`
straight forward.
A much more advanced approach is given in\ :footcite:`Jannik` by realising a *multivariate Horner scheme*. A replacement of the current
naïv implementation by this better performing scheme is aimed to be realised for future enhancement. However, though the
multivariate Horner scheme performs much faster in practice
the runtime complexity of both approches is quadratic, i.e, :math:`\mathcal{O}(|A|^2)`.



Newton evaluation
#################

In case the multivariate polynomial :math:`Q \in \Pi_A`, :math:`A \subseteq \mathbb{N}^m`, is given in Newton form
the following crucial consequence applies\ :footcite:`Hecht2020`.

**Theorem 1 (Newton evaluation)** Let :math:`A \subseteq \mathbb{N}^m`, :math:`m\in \mathbb{N}` be a multi-index set and

.. math::

  Q(x) = \sum_{\alpha \in A} c_\alpha N_\alpha \in \Pi_A\,, \quad c_{\alpha} \in \mathbb{R}

be a polynomial in Newton form. Then it requires :math:`\mathcal{O}(m|A|)` operations and :math:`\mathcal{O}(|A|)` storage
to evaluate the value :math:`Q(p)` of :math:`Q` at :math:`p \in \mathbb{R}^m`.

The current ``minterpy`` implementation realises a generalised version of the classic
**Aitken-Neville algorithm**\ :footcite:`neville` on which Theorem 1 rests. Apart, from the runtime improvement compared to the
naïv (canonical) evaluation the Leja ordering of the underlying **unisolvent nodes** makes Newton evaluation a numerically robust
and highly accurate scheme.


Lagrange evaluation
###################

In case the multivariate polynomial :math:`Q \in \Pi_A`, :math:`A \subseteq \mathbb{N}^m`, is given in Lagrange form

.. math::

  Q(x) = \sum_{\alpha \in A} c_\alpha L_\alpha \in \Pi_A\,, \quad c_{\alpha} \in \mathbb{R}

there is no direct way to evaluate the polynomial. Instead one has to transform the polynomial into the **Newton or canonical basis**
and apply the available evalautiona schemes accordingly. Especially, the **transformation into Newton form**  is realised
by ``minterpy`` efficiently with machine precision (for reasonable instance size), thereby, enabling evaluation of polynomials in Lagrange
form in a suitable fashion.


References
##########

.. footbibliography::
