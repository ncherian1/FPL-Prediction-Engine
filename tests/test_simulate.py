import numpy as np

from fpl_engine.models.simulate import SimBonusModel


def test_bonus_grid_index_matches_grid_order():
    """Every simulated event combination must map to its own row of the bonus grid."""
    g = SimBonusModel.GRID
    idx = SimBonusModel.grid_index(g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5])
    assert np.array_equal(idx, np.arange(len(g)))


def test_bonus_grid_index_caps_large_counts():
    """A hat-trick-plus maps to the 3-goal row (the grid caps goals at 3, assists at 2, save points at 3)."""
    capped = SimBonusModel.grid_index(np.array([2]), np.array([5]), np.array([4]), np.array([1]), np.array([1]),
                                      np.array([7]))
    exact = SimBonusModel.grid_index(np.array([2]), np.array([3]), np.array([2]), np.array([1]), np.array([1]),
                                     np.array([3]))
    assert capped[0] == exact[0]
