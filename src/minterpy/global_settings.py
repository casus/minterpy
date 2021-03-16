# -*- coding:utf-8 -*-
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import numba
# numpy dtypes
# for exponents:

# ATTENTION: when Numba JIT compilation is active. Some functions may crash silently
# without raising the expected errors! (and stack traces, e.g. just SIGSEGV)
# -> perhaps deactivate Numba for DEBUGGING!
DEBUG = True
# DEBUG = False

INT_DTYPE = np.int_
FLOAT_DTYPE = np.float_
DEFAULT_DOMAIN = np.array([-1, 1])

# Numba types. Must match the Numpy dtypes
FLOAT = numba.from_dtype(FLOAT_DTYPE)
# F_TYPE = f8
F_1D = FLOAT[:]
F_2D = FLOAT[:, :]
F_3D = FLOAT[:, :, :]
INT = numba.from_dtype(INT_DTYPE)
I_1D = INT[:]
I_2D = INT[:, :]
B_TYPE = numba.b1

DEFAULT_LP_DEG = 2.0

NOT_FOUND = -1  # meaning: exponent vector is not contained

ARRAY = np.ndarray
# TYPED_LIST = List[ARRAY]
TYPED_LIST = numba.typed.List
INT_TUPLE = Tuple[int, int]
ARRAY_DICT = Dict[Tuple[int, int, int], Optional[ARRAY]]
INT_SET = Set[int]