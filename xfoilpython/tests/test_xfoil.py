import numpy as np
import pytest
import os
import sys

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from utils.xfoil_utils import run_xfoil
from src.xfoil.test import naca0012
from data.naca0012 import expected_naca0012


@pytest.mark.parametrize("expected_values", expected_naca0012)
def test_run_xfoil(expected_values: np.ndarray[np.float64, ...]) -> None:
    """
    Test whether xfoil simulation results match expected values.
    """

    # Arrange
    alpha, lift, drag, drag_p, moment, top_xtr, bot_xtr = expected_values

    # Act
    _, cl, cd, cm, _ = run_xfoil(airfoil=naca0012, alpha=alpha, reynolds_number=1e6)

    # Assert
    np.testing.assert_almost_equal(lift, cl, decimal=2)
    np.testing.assert_almost_equal(drag, cd, decimal=2)
    np.testing.assert_almost_equal(moment, cm, decimal=2)
