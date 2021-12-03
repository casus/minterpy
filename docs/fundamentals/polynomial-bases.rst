#############################
Multivariate polynomial bases
#############################

..
    .. todo::

       This page should introduce and define all the supported multivariate
       polynomial bases supported in ``minterpy``. Some pictures might help as well.

Multi-index sets and polynomial degree
######################################

Multi-index sets :math:`A\subseteq \mathbb{N}^m` generalise the notion of polynomial degree to multi-dimensions :math:`m \in \mathbb{N}`.
We call a multi-index set **downward closed** if and only if there is no :math:`\beta = (b_1,\dots,b_m) \in \mathbb{N}^m \setminus A`
with :math:`b_i \leq a_i`,  for all :math:`i=1,\dots,m` and some :math:`\alpha = (a_1,\dots,a_m) \in A`.
This follows the classic terminology introduced for instance by \ :footcite:`Cohen2018`.

Any (not necessarily downward closed) multi-index set :math:`A\subseteq \mathbb{N}^m` is assumed to be orderd with respect to the
**lexicographical order** :math:`\preceq` from the last entry to the first, e.g.,
:math:`(5,3,1)\preceq(1,0,3) \preceq(1,1,3)`.


For :math:`\alpha=(\alpha_1,\ldots,\alpha_m) \in \mathbb{N}^m` we consider the :math:`l_p`-norm

.. math::

  \|\alpha\|_p  = (\alpha_1^p + \cdots +\alpha_m^p)^{1/p}

and denote the multi-index sets of bounded :math:`l_p`-norm by

.. math::

  A_{m,n,p} = \{\alpha \in \mathbb{N}^m :  \|\alpha\|_p \leq n \}\,, \quad p>1 \,.

Indeed, the sets :math:`A_{m,n,p}` are downward closed and yield the multi-indices being relevant when considering polynomials of :math:`l_p`-degree
:math:`n \in \mathbb{N}` in dimension :math:`m \in \mathbb{N}`.


While :math:`A_{1,n,1}=A_{1,n,p} = A_{1,n,\infty}` for all :math:`p>1` the
generalised notion of polynomial degree induced by the multi-index sets :math:`A_{m,n,p}` just becomes observable for multi-dimensions.

Below the canonical, Lagrange \& Newton bases are introduced. In fact, all of them are bases of :math:`\Pi_A` (:eq:`eq_PiA`) whenever :math:`A`
is a downward closed multi-index set.


Canonical polynomials
#####################

Given a not necessarily downward closed multi-index set :math:`A\subseteq \mathbb{N}^m` in dimension  :math:`m \in \mathbb{N}` we consider the
**polynomial spaces**

.. math::
  :label: eq_PiA

  \Pi_A =\left<x^\alpha = x_1^{\alpha_1}\cdots x_m^{\alpha_m} : \alpha \in A\right>

spanned by the canonical monomials :math:`x^\alpha\,, \alpha \in A`. A polynomial in **canonical form**

.. math::

  Q(x) = \sum_{\alpha \in A} c_\alpha x^\alpha \in \Pi_A\,.

is determined by :math:`A` and its **canonical coefficients** :math:`(c_\alpha)_{\alpha \in A} \in \mathbb{R}^{|A|}`,
which are implemented as an array ordered with respect to the lexicographical order :math:`\preceq`.

The crucial point of our general setup of multi-index sets :math:`A \subseteq \mathbb{N}^m` can for instance be observed by realising that
:math:`\|(2,2)\|_1 = 4  > 3`, :math:`\|(2,2)\|_2 = \sqrt{8}  < 3`  implies

.. math::

  x^2y^2 \not \in \Pi_{A_{2,3,1}}\,, \quad \text{but}\quad x^2y^2  \in \Pi_{A_{2,3,2}}\,.

In other words: the choice of :math:`l_p`-degree constraints the combinations of considered monomials. This fact is crucial for the approximation
power of polynomials as asserted in the **introduction**.



Lagrange polynomials
####################

Given a multi-index set :math:`A\subseteq \mathbb{N}^m` in dimension :math:`m \in \mathbb{N}`
and a set of unisolvent nodes given as the sub-grid

.. math::

  P_A = \left\{ p_\alpha = (p_{\alpha_1,1},\ldots,p_{\alpha_m,m}) \in \Omega\subseteq \mathbb{R}^m : \alpha \in A\right\}\,, \quad p_{\alpha_i,i} \in P_i \subseteq [-1,1]\,.

that is specified by the chosen **generating nodes** :math:`\mathrm{GP} = \oplus_{i=1}^m P_i` the Lagrange polynomials
:math:`L_\alpha` are uniquely determined by being required to satisfy

.. math::

  L_{\alpha}(p_\beta) = \delta_{\alpha,\beta}\,,

where :math:`\delta_{\cdot,\cdot}` denotes the **Kronecker delta**. A polynomial in **Lagrange form**

.. math::

  Q(x) = \sum_{\alpha \in A} c_\alpha L_\alpha \in \Pi_A

is determined by the choice of :math:`A, \mathrm{GP}` and its **Lagrange coefficients** :math:`(c_\alpha)_{\alpha \in A} \in \mathbb{R}^{|A|}`,
which are implemented as an array ordered with respect to the lexicographical order :math:`\preceq`.


Newton polynomials
##################

Given a multi-index set :math:`A\subseteq \mathbb{N}^m` in dimension :math:`m \in \mathbb{N}`
and a set of unisolvent nodes given as the sub-grid

.. math::

  P_A = \left\{ p_\alpha = (p_{\alpha_1,1},\ldots,p_{\alpha_m,m}) \in \Omega\subseteq \mathbb{R}^m : \alpha \in A\right\}\,, \quad p_{\alpha_i,i} \in P_i\,.

that is specified by the chosen **generating nodes** :math:`\mathrm{GP} = \oplus_{i=1}^m P_i` the Newton polynomials
:math:`N_\alpha` are defined by

.. math::
  N_\alpha(x) = \prod_{i=1}^m\prod_{j=0}^{\alpha_i -1}(x- p_{j,i})\,,\quad  p_{j,i} \in P_i

generalising their classic notion from 1D to multi-dimensions, see e.g.\ :footcite:`stoer2002,gautschi2012`. A polynomial in **Newton form**

.. math::

  Q(x) = \sum_{\alpha \in A} c_\alpha N_\alpha \in \Pi_A

is determined by the choice of :math:`A, \mathrm{GP}` and its **Newton coefficients** :math:`(c_\alpha)_{\alpha \in A} \in \mathbb{R}^{|A|}`,
which are implemented as an array ordered with respect to the lexicographical order :math:`\preceq`.



References
##########

.. footbibliography::
