#!/usr/bin/env python
""" global settings of all tests
"""

import numpy as np

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"

np.random.seed(42)

DESIRED_PRECISION = 10  # decimals of maximally allowed errors
NR_SAMPLE_POINTS = 1000  # TODO dependent on m?
MIN_DEGREE = 1
MAX_DEGREE = 4
ONLY_UNEVEN_DEGREES = False  # test only uneven interpolation total_degrees (result in more symmetrical grid)
DEGREE_STEP = 1
if ONLY_UNEVEN_DEGREES:
    assert MIN_DEGREE % 2 == 1
    assert MAX_DEGREE % 2 == 1
else:
    DEGREE_STEP = 1
DEGREES2TEST = range(MIN_DEGREE, MAX_DEGREE + 1, DEGREE_STEP)

MIN_DIMENSION = 1
MAX_DIMENSION = 3
DIMENSIONS2TEST = range(MIN_DIMENSION, MAX_DIMENSION + 1)

# TODO allow other lp degrees: -> not supported yet. e.g. functions use linalg.norm!
LP_DEGREES = [1.0, 2.0, np.inf]
# LP_DEGREES = [1.0, 2.0,]
# LP_DEGREES = [1.0]
# LP_DEGREES = [2.0]
# LP_DEGREES = [np.inf]

RUNGE_FCT = lambda x: 1 / (1 + 25 * np.linalg.norm(x) ** 2)

# TODO more
TEST_FUNCTIONS = [  # ground truths:
    RUNGE_FCT,
    # lambda x: np.sin((2 * x[0]) / np.pi + x[1]), # TODO fixed 2D
    # lambda x: x[0] + x[1], # TODO fixed 2D
]

RUNGE_FCT_VECTORIZED = lambda eval_points: np.apply_along_axis(RUNGE_FCT, 0, eval_points)

# TODO
# TEST_FUNCTIONS_VECTORIZED = [lambda eval_points: np.apply_along_axis(g, 0, eval_points) for g in TEST_FUNCTIONS]
TEST_FUNCTIONS_VECTORIZED = [RUNGE_FCT_VECTORIZED]

TIME_FORMAT_STR = "{:1.2e}s"
