"""Tests for config.py — verify all constants exist and have valid values."""

import numpy as np
from config import (
    LOG_EPSILON,
    ML_GRID_POINTS,
    KALMAN_DT,
    ADAPTATION_INCREMENT,
    BURST_INITIATION_PROB,
    MAX_SPIKE_PROB_PER_BIN,
    BCI_TARGET_RADIUS,
    BCI_CANVAS_SIZE,
    DIRECTION_HIGHLIGHT_THRESHOLD,
    RASTER_PX_PER_NEURON,
    RASTER_MIN_HEIGHT,
    MAX_LEADERBOARD_ENTRIES,
    DEFAULT_N_NEURONS,
    DEFAULT_DURATION_MS,
    DEFAULT_BASELINE_RATE,
    DEFAULT_MODULATION_DEPTH,
    DEFAULT_VARIANCE_SCALE,
    DEFAULT_N_TRIALS,
)


def test_log_epsilon_positive_tiny():
    assert 0 < LOG_EPSILON < 1e-5


def test_ml_grid_positive():
    assert ML_GRID_POINTS > 0
    assert isinstance(ML_GRID_POINTS, int)


def test_kalman_dt_positive():
    assert 0 < KALMAN_DT < 1.0


def test_probabilities_in_range():
    assert 0 < ADAPTATION_INCREMENT <= 1.0
    assert 0 < BURST_INITIATION_PROB <= 1.0
    assert 0 < MAX_SPIKE_PROB_PER_BIN <= 1.0


def test_bci_constants():
    assert BCI_TARGET_RADIUS > 0
    assert BCI_CANVAS_SIZE > 0


def test_visualization_constants():
    assert 0 < DIRECTION_HIGHLIGHT_THRESHOLD < np.pi
    assert RASTER_PX_PER_NEURON > 0
    assert RASTER_MIN_HEIGHT > 0


def test_leaderboard_entries():
    assert MAX_LEADERBOARD_ENTRIES > 0


def test_default_simulation_params():
    assert DEFAULT_N_NEURONS > 0
    assert DEFAULT_DURATION_MS > 0
    assert DEFAULT_BASELINE_RATE >= 0
    assert DEFAULT_MODULATION_DEPTH >= 0
    assert DEFAULT_VARIANCE_SCALE > 0
    assert DEFAULT_N_TRIALS > 0
