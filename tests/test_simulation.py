"""Tests for simulation.py — population generation and spike simulation."""

import numpy as np
from simulation import cosine_tuning, generate_neuron_population, simulate_trial


def test_cosine_tuning_peak():
    """Firing rate should be highest at preferred direction."""
    mu = np.pi / 4
    r0, k = 5.0, 15.0
    peak_rate = cosine_tuning(mu, mu, r0, k)
    off_rate = cosine_tuning(mu + np.pi, mu, r0, k)
    assert peak_rate > off_rate


def test_cosine_tuning_nonneg():
    """Firing rate should never be negative."""
    theta = np.linspace(0, 2 * np.pi, 100)
    rates = cosine_tuning(theta, 0.0, 5.0, 15.0)
    assert np.all(rates >= 0)


def test_generate_population_size():
    pop = generate_neuron_population(n_neurons=20, seed=0)
    assert pop.n_neurons == 20
    assert len(pop.preferred_directions) == 20


def test_simulate_trial_shape(small_population):
    spikes = simulate_trial(np.pi / 2, small_population, seed=0)
    assert spikes.shape == (small_population.n_neurons,)
    assert np.all(spikes >= 0)


def test_simulate_trial_reproducibility(small_population):
    s1 = simulate_trial(1.0, small_population, seed=42)
    s2 = simulate_trial(1.0, small_population, seed=42)
    np.testing.assert_array_equal(s1, s2)
