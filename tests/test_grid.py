import numpy as np
import pytest

from minterpy import Grid, MultiIndexSet

from conftest import LpDegree


@pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
def test_empty_set(spatial_dimension, LpDegree):
    """Test construction with an empty set."""
    # Create an empty set
    mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

    # Assertion
    with pytest.raises(ValueError):
        Grid(mi)