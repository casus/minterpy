############################
Fundamentals of ``minterpy``
############################

.. todo::

   The Fundamentals is a Guide about the underlying theory behind ``minterpy``
   and establishing the mathematical notions and conventions used in ``minterpy``.
   Interpolation problems are quite common (and simple its premise) across
   scientific and engineering disciplines and ``minterpy`` users most probably
   do not come from mathematics or disciplines that have particularly strong
   background in mathematics. However, it is also important for users
   to understand the underlying theory behind ``minterpy`` if they were to progress to, say,
   *power users* (or from users to *contributors*).

   Some general notes on how the Fundamentals should be written:

   - *Focus on introducing the mathematical notions or concepts required to understand*
     *``minterpy``*: minimize reference to ``minterpy`` usage and specific code
     implementation.
   - *Be consistent with mathematical notations*: define the terms and symbols
     and use the same terms and symbols again later on when required.
     Using mathematics is unavoidable in the Fundamental guide.
     Be mindful, however, of the target audience and keep the bar of the pre-requisites low
     (say university-level mathematics of applied science and engineering curricula,
     **not** of a pure mathematics curriculum).
   - *Be as verbose with as much details as necessary, when necessary*: users are
     coming here intentionally to look for an explanation and understanding
     so they are ready to invest some time.
   - *Clearly distinguish between mathematical concepts,*
     *algorithmic considerations, usage, and code implementation*:
     It is easy to confuse users when these different aspects of ``minterpy``
     are all mixed up in the same part of the documentation.
     If this is not always possible, make a clear transition or a notebox.
   - *Cross-reference other parts of the documentation* to minimize repetition:
     external cross-referencing (say, to Wikipedia) might be useful so as not to repeat oneself.
     But do this only if readers won't be confused with any changes of conventions and notations.

.. toctree::
   :maxdepth: 4

   introduction
   polynomial-bases
   interpolation-on-unisolvent-nodes
   multivariate-dds
   interpolation-evaluation
   transformation
   barycentric-transformation
