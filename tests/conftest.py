"""Shared fixtures for Decode My Brain test suite."""

import numpy as np
import pytest
from simulation import generate_neuron_population, NeuronPopulation


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
