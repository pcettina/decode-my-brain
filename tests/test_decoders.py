"""Tests for decoders.py — direction decoding from spike counts."""

import numpy as np
import pytest
from simulation import simulate_trial
from decoders import (
    PopulationVectorDecoder,
    MaximumLikelihoodDecoder,
    KalmanFilterDecoder,
    evaluate_decoder,
)
from utils import angular_error


def test_pv_known_direction(medium_population):
    """PV decoder should roughly recover the true direction."""
    true_theta = np.pi / 2
    spikes = simulate_trial(true_theta, medium_population, duration_ms=1000, seed=0)
    pv = PopulationVectorDecoder()
    decoded = pv.decode(spikes, medium_population)
    error = angular_error(true_theta, decoded)
    assert error < np.pi / 4  # within 45 degrees


def test_pv_zero_spikes(small_population):
    """PV decoder should handle all-zero spike counts gracefully."""
    pv = PopulationVectorDecoder()
    result = pv.decode(np.zeros(small_population.n_neurons, dtype=int), small_population)
    assert result == 0.0


def test_ml_known_direction(medium_population):
    """ML decoder should recover direction within 30 degrees."""
    true_theta = np.pi
    spikes = simulate_trial(true_theta, medium_population, duration_ms=1000, seed=1)
    ml = MaximumLikelihoodDecoder()
    decoded = ml.decode(spikes, medium_population, duration_s=1.0)
    error = angular_error(true_theta, decoded)
    assert error < np.pi / 6  # within 30 degrees


def test_kalman_not_fitted():
    """KalmanFilterDecoder should raise if decode_step called before fit."""
    kf = KalmanFilterDecoder(n_neurons=10)
    with pytest.raises(RuntimeError, match="not fitted"):
        kf.decode_step(np.zeros(10))


def test_evaluate_decoder_range(medium_population):
    """Mean error should be in a reasonable range for PV decoder."""
    from simulation import simulate_random_trials
    spikes, dirs = simulate_random_trials(50, medium_population, seed=99)
    results = evaluate_decoder(
        PopulationVectorDecoder(), spikes, dirs, medium_population
    )
    assert 0 < results['mean_error_degrees'] < 90
