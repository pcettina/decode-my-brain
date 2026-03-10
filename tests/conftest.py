"""Shared fixtures for Decode My Brain test suite."""

import numpy as np
import pytest
from simulation import (
    generate_neuron_population,
    NeuronPopulation,
    simulate_trial,
    simulate_random_trials,
    TemporalParams,
)


@pytest.fixture
def rng():
    """Deterministic RNG for tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_population():
    """Small 8-neuron population for fast unit tests."""
    return generate_neuron_population(n_neurons=8, seed=42)


@pytest.fixture
def medium_population():
    """Medium 50-neuron population for integration tests."""
    return generate_neuron_population(n_neurons=50, seed=42)


@pytest.fixture
def large_population():
    """Large 100-neuron population for decoder accuracy tests."""
    return generate_neuron_population(n_neurons=100, seed=42)


@pytest.fixture
def single_trial_spikes(medium_population):
    """Single trial spike counts at theta=pi/2 for a 50-neuron population."""
    return simulate_trial(np.pi / 2, medium_population, duration_ms=500, seed=0)


@pytest.fixture
def multi_trial_data(medium_population):
    """50 random trials from a 50-neuron population."""
    spikes, dirs = simulate_random_trials(50, medium_population, seed=99)
    return spikes, dirs


@pytest.fixture
def temporal_params_default():
    """Default temporal dynamics parameters."""
    return TemporalParams()


@pytest.fixture
def temporal_params_no_dynamics():
    """Temporal params with all dynamics disabled."""
    return TemporalParams(
        adaptation_strength=0.0,
        refractory_abs_ms=0.0,
        refractory_rel_ms=0.0,
        burst_probability=0.0,
    )
