"""Tests for utils.py — angle math and helpers."""

import numpy as np
from utils import wrap_angle, angular_error, circular_mean


def test_wrap_angle_positive():
    assert wrap_angle(0.0) == 0.0
    assert abs(wrap_angle(2 * np.pi) - 0.0) < 1e-10


def test_wrap_angle_negative():
    result = wrap_angle(-np.pi / 2)
    assert abs(result - 3 * np.pi / 2) < 1e-10


def test_angular_error_same_direction():
    assert angular_error(1.0, 1.0) < 1e-10


def test_angular_error_opposite():
    assert abs(angular_error(0.0, np.pi) - np.pi) < 1e-10


def test_circular_mean_uniform():
    """Uniform angles around the circle should give ambiguous result."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    result = circular_mean(angles)
    # With uniform spacing, the mean is near-zero magnitude -> returns 0.0
    assert isinstance(result, float)
