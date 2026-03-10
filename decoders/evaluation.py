"""
Decoder evaluation and comparison utilities.
"""

import numpy as np
from simulation.core import NeuronPopulation
from decoders.base import Decoder
from decoders.direction import PopulationVectorDecoder, MaximumLikelihoodDecoder
from decoders.kalman import KalmanFilterDecoder


def evaluate_decoder(
    decoder: Decoder,
    spike_counts: np.ndarray,
    true_directions: np.ndarray,
    neurons: NeuronPopulation,
    duration_s: float = 0.5
) -> dict:
    """Evaluate decoder performance on a set of trials."""
    from utils import angular_error

    n_trials = len(true_directions)
    decoded_directions = np.zeros(n_trials)
    errors = np.zeros(n_trials)

    for i in range(n_trials):
        decoded_directions[i] = decoder.decode(spike_counts[i], neurons, duration_s=duration_s)
        errors[i] = angular_error(true_directions[i], decoded_directions[i])

    return {
        'decoded_directions': decoded_directions,
        'errors': errors,
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'median_error': float(np.median(errors)),
        'mean_error_degrees': float(np.mean(errors) * 180 / np.pi),
        'std_error_degrees': float(np.std(errors) * 180 / np.pi)
    }


def compare_decoders(
    spike_counts: np.ndarray,
    true_directions: np.ndarray,
    neurons: NeuronPopulation,
    duration_s: float = 0.5
) -> dict:
    """Compare multiple decoders on the same data."""
    from utils import angular_error

    decoders = {
        'Population Vector': PopulationVectorDecoder(),
        'Maximum Likelihood': MaximumLikelihoodDecoder(),
        'Kalman Filter': KalmanFilterDecoder(neurons.n_neurons)
    }

    results = {}

    for name, decoder in decoders.items():
        if isinstance(decoder, KalmanFilterDecoder):
            decoder.fit_from_neurons(neurons)

        errors = []
        decoded = []

        for i in range(len(true_directions)):
            if isinstance(decoder, KalmanFilterDecoder):
                decoder.reset()
            d = decoder.decode(spike_counts[i], neurons, duration_s=duration_s)
            decoded.append(d)
            errors.append(angular_error(true_directions[i], d))

        results[name] = {
            'decoded': np.array(decoded),
            'errors': np.array(errors),
            'mean_error_deg': np.mean(errors) * 180 / np.pi,
            'std_error_deg': np.std(errors) * 180 / np.pi
        }

    return results
