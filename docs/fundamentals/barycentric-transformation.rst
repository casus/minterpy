##########################
Barycentric transformation
##########################



In 1D, barycentric Lagrange interpolation is the most efficient scheme for fixed nodes\ :footcite:`berrut2004`. Both determining the interpolant
:math:`Q_{f,A}\,, A=\{0,\ldots,n\}\,, n\in \mathbb{N}` and evaluating :math:`Q_{f,A}` at any argument :math:`p \in \mathbb{R}`
require linear runtime :math:`\mathcal{O}(n)`. This is achieved by precomputing the constant *barycentric weights*
that only depend on the locations of the interpolation nodes, but not on the function :math:`f : \Omega \longrightarrow \mathbb{R}`.

``minterpy`` already partially extended the classic barycentric Lagrange interpolation to the multi-dimensional case.
This is based on the fact that the transformation from **Lagrange to Newton basis** consists of structured sparse triangular matrices.
Exploiting this structure accordingly to our preliminary results for the case of :math:`l_1`-degree\ :footcite:`sivkin` allow
inverting and multiplying of the corresponding matrices much faster than in the general case, see :footcite:`struct2,struct1` for similar approaches.

In summary, we expect to reduce the runtime complexity :math:`\mathcal{O}(|A|^2)\,, A =A_{m,n,p}\,, p>1`, for deriving and executing the transformations
to :math:`\mathcal{O}(mn|A|)` resulting in :math:`\mathcal{O}(\log(|A|)|A|)` for :math:`A=A_{m,n,p},p = 1,\infty`.

Research and implementation enhancement are ongoing towards reaching this goal.


References
##########

.. footbibliography::
