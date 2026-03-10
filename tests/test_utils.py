"""Tests for utils.py — angle math, exports, and display helpers."""

import numpy as np
import pytest
from utils import (
    wrap_angle,
    wrap_angle_array,
    angular_error,
    angular_error_degrees,
    degrees_to_radians,
    radians_to_degrees,
    circular_mean,
    export_to_npz,
    export_to_csv,
    format_angle_display,
    generate_direction_labels,
)


# ── wrap_angle ──────────────────────────────────────────────────────────────

def test_wrap_angle_positive():
    assert wrap_angle(0.0) == 0.0
    assert abs(wrap_angle(2 * np.pi) - 0.0) < 1e-10


def test_wrap_angle_negative():
    result = wrap_angle(-np.pi / 2)
    assert abs(result - 3 * np.pi / 2) < 1e-10


def test_wrap_angle_large_positive():
    result = wrap_angle(5 * np.pi)
    assert abs(result - np.pi) < 1e-10


def test_wrap_angle_large_negative():
    result = wrap_angle(-3 * np.pi)
    assert abs(result - np.pi) < 1e-10


# ── wrap_angle_array ────────────────────────────────────────────────────────

def test_wrap_angle_array_basic():
    arr = np.array([-np.pi, 0.0, np.pi, 3 * np.pi])
    result = wrap_angle_array(arr)
    assert result.shape == (4,)
    assert np.all(result >= 0)
    assert np.all(result < 2 * np.pi)


def test_wrap_angle_array_matches_scalar():
    values = [-1.5, 0.0, 3.7, 7.0]
    arr_result = wrap_angle_array(np.array(values))
    scalar_results = np.array([wrap_angle(v) for v in values])
    np.testing.assert_allclose(arr_result, scalar_results)


# ── angular_error ───────────────────────────────────────────────────────────

def test_angular_error_same_direction():
    assert angular_error(1.0, 1.0) < 1e-10


def test_angular_error_opposite():
    assert abs(angular_error(0.0, np.pi) - np.pi) < 1e-10


def test_angular_error_symmetric():
    """angular_error(a, b) == angular_error(b, a)."""
    assert abs(angular_error(0.5, 2.0) - angular_error(2.0, 0.5)) < 1e-10


def test_angular_error_range():
    """Result should always be in [0, pi]."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        a, b = rng.uniform(0, 10, 2)
        err = angular_error(a, b)
        assert 0 <= err <= np.pi + 1e-10


def test_angular_error_near_wrap():
    """Small error across 0/2pi boundary."""
    err = angular_error(0.01, 2 * np.pi - 0.01)
    assert err < 0.03


# ── angular_error_degrees ──────────────────────────────────────────────────

def test_angular_error_degrees_same():
    assert angular_error_degrees(90.0, 90.0) < 1e-8


def test_angular_error_degrees_opposite():
    assert abs(angular_error_degrees(0.0, 180.0) - 180.0) < 1e-8


def test_angular_error_degrees_range():
    assert 0 <= angular_error_degrees(30.0, 200.0) <= 180.0


# ── degrees_to_radians / radians_to_degrees ─────────────────────────────────

def test_degrees_to_radians():
    assert abs(degrees_to_radians(180.0) - np.pi) < 1e-10
    assert abs(degrees_to_radians(0.0)) < 1e-10
    assert abs(degrees_to_radians(90.0) - np.pi / 2) < 1e-10


def test_radians_to_degrees():
    assert abs(radians_to_degrees(np.pi) - 180.0) < 1e-10
    assert abs(radians_to_degrees(0.0)) < 1e-10


def test_roundtrip_conversion():
    for deg in [0, 45, 90, 135, 180, 270, 360]:
        assert abs(radians_to_degrees(degrees_to_radians(deg)) - deg) < 1e-10


# ── circular_mean ───────────────────────────────────────────────────────────

def test_circular_mean_uniform():
    """Uniform angles around the circle should give ambiguous result."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    result = circular_mean(angles)
    assert isinstance(result, float)


def test_circular_mean_single():
    result = circular_mean(np.array([1.5]))
    assert abs(result - 1.5) < 1e-10


def test_circular_mean_two_close():
    """Mean of two close angles should be between them."""
    angles = np.array([0.1, 0.3])
    result = circular_mean(angles)
    assert abs(result - 0.2) < 1e-10


def test_circular_mean_weighted():
    """Weighted mean should shift toward heavily weighted angle."""
    angles = np.array([0.0, np.pi])
    weights = np.array([10.0, 1.0])
    result = circular_mean(angles, weights)
    # Should be much closer to 0.0 than to pi
    assert angular_error(result, 0.0) < angular_error(result, np.pi)


def test_circular_mean_zero_magnitude():
    """Exactly opposing equal-weight angles give 0.0 fallback."""
    angles = np.array([0.0, np.pi])
    weights = np.array([1.0, 1.0])
    result = circular_mean(angles, weights)
    assert result == 0.0


# ── export_to_npz ──────────────────────────────────────────────────────────

def test_export_to_npz_roundtrip():
    data = {'arr': np.array([1, 2, 3]), 'scalar': np.array(42.0)}
    blob = export_to_npz(data, 'test.npz')
    assert isinstance(blob, bytes)
    assert len(blob) > 0
    # Verify we can load it back
    import io
    loaded = np.load(io.BytesIO(blob))
    np.testing.assert_array_equal(loaded['arr'], data['arr'])


def test_export_to_npz_empty():
    blob = export_to_npz({}, 'empty.npz')
    assert isinstance(blob, bytes)


# ── export_to_csv ──────────────────────────────────────────────────────────

def test_export_to_csv_neuron_params():
    data = {
        'preferred_directions': np.array([0.0, np.pi / 2, np.pi]),
        'baseline_rate': 5.0,
        'modulation_depth': 15.0,
    }
    csv_str = export_to_csv(data, 'neurons.csv')
    assert 'neuron_id' in csv_str
    assert 'preferred_direction_rad' in csv_str
    lines = csv_str.strip().split('\n')
    assert len(lines) == 4  # header + 3 neurons


def test_export_to_csv_trial_data():
    data = {
        'spike_counts': np.array([[5, 10], [8, 3]]),
        'true_directions': np.array([0.0, np.pi]),
    }
    csv_str = export_to_csv(data, 'trials.csv')
    assert 'trial_id' in csv_str
    assert 'spike_count' in csv_str


def test_export_to_csv_empty():
    assert export_to_csv({}, 'empty.csv') == ""


# ── format_angle_display ───────────────────────────────────────────────────

def test_format_angle_display_degrees():
    result = format_angle_display(np.pi, use_degrees=True)
    assert '180' in result
    assert '°' in result


def test_format_angle_display_radians():
    result = format_angle_display(np.pi, use_degrees=False)
    assert 'rad' in result
    assert '3.14' in result


# ── generate_direction_labels ──────────────────────────────────────────────

def test_generate_direction_labels_8():
    angles, labels = generate_direction_labels(8)
    assert len(angles) == 8
    assert len(labels) == 8
    assert 'N' in labels
    assert 'E' in labels


def test_generate_direction_labels_4():
    angles, labels = generate_direction_labels(4)
    assert len(angles) == 4
    assert set(labels) == {'E', 'N', 'W', 'S'}


def test_generate_direction_labels_arbitrary():
    angles, labels = generate_direction_labels(12)
    assert len(angles) == 12
    assert len(labels) == 12
    assert all('°' in label for label in labels)
