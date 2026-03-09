"""Integration test — end-to-end simulation + decoding pipeline."""

import numpy as np
from simulation import generate_neuron_population, simulate_random_trials
from decoders import PopulationVectorDecoder, evaluate_decoder


def test_end_to_end_pv_accuracy():
    """50 neurons, 100 trials: PV mean error should be < 45 degrees."""
    neurons = generate_neuron_population(50, seed=123)
    spikes, dirs = simulate_random_trials(100, neurons, duration_ms=500, seed=456)
    results = evaluate_decoder(
        PopulationVectorDecoder(), spikes, dirs, neurons
    )
    assert results['mean_error_degrees'] < 45
