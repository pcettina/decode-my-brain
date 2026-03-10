"""
Direction decoders: Population Vector, Maximum Likelihood, Naive Bayes.
"""

import numpy as np
from typing import Optional, Tuple

from simulation.core import NeuronPopulation
from utils import wrap_angle, circular_mean
from config import ML_GRID_POINTS

from decoders.base import (
    Decoder,
    _compute_poisson_log_likelihoods,
    _validate_decode_inputs,
)


class PopulationVectorDecoder(Decoder):
    """Population vector decoder: weighted vector sum of preferred directions."""

    @property
    def name(self) -> str:
        return "Population Vector"

    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        _validate_decode_inputs(spike_counts, neurons)

        if np.sum(spike_counts) == 0:
            return 0.0

        x = np.sum(spike_counts * np.cos(neurons.preferred_directions))
        y = np.sum(spike_counts * np.sin(neurons.preferred_directions))

        if np.abs(x) < 1e-10 and np.abs(y) < 1e-10:
            return 0.0

        theta_hat = np.arctan2(y, x)
        return wrap_angle(theta_hat)


class MaximumLikelihoodDecoder(Decoder):
    """Maximum likelihood decoder assuming independent Poisson neurons."""

    def __init__(self, n_grid_points: int = ML_GRID_POINTS):
        self.n_grid_points = n_grid_points
        self.theta_grid = np.linspace(0, 2 * np.pi, n_grid_points, endpoint=False)

    @property
    def name(self) -> str:
        return "Maximum Likelihood"

    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        _validate_decode_inputs(spike_counts, neurons)
        log_likelihoods = _compute_poisson_log_likelihoods(
            self.theta_grid, spike_counts, neurons, duration_s
        )
        best_idx = np.argmax(log_likelihoods)
        return self.theta_grid[best_idx]

    def get_likelihood_curve(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the full normalized likelihood curve for visualization."""
        log_likelihoods = _compute_poisson_log_likelihoods(
            self.theta_grid, spike_counts, neurons, duration_s
        )
        log_likelihoods -= np.max(log_likelihoods)
        likelihoods = np.exp(log_likelihoods)
        likelihoods /= np.sum(likelihoods)
        return self.theta_grid, likelihoods


class NaiveBayesDecoder(Decoder):
    """Naive Bayes / Maximum a Posteriori decoder."""

    def __init__(self, n_grid_points: int = 360, prior: Optional[np.ndarray] = None):
        self.n_grid_points = n_grid_points
        self.theta_grid = np.linspace(0, 2 * np.pi, n_grid_points, endpoint=False)

        if prior is None:
            self.log_prior = np.zeros(n_grid_points)
        else:
            self.log_prior = np.log(prior + 1e-10)

    @property
    def name(self) -> str:
        return "Naive Bayes (MAP)"

    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        _validate_decode_inputs(spike_counts, neurons)
        log_likelihoods = _compute_poisson_log_likelihoods(
            self.theta_grid, spike_counts, neurons, duration_s
        )
        log_posteriors = log_likelihoods + self.log_prior
        best_idx = np.argmax(log_posteriors)
        return self.theta_grid[best_idx]


def get_decoder(decoder_name: str) -> Decoder:
    """Factory function to get a decoder by name."""
    decoders = {
        'population_vector': PopulationVectorDecoder,
        'ml': MaximumLikelihoodDecoder,
        'maximum_likelihood': MaximumLikelihoodDecoder,
        'naive_bayes': NaiveBayesDecoder,
        'map': NaiveBayesDecoder
    }

    if decoder_name.lower() not in decoders:
        raise ValueError(f"Unknown decoder: {decoder_name}. "
                         f"Available: {list(decoders.keys())}")

    return decoders[decoder_name.lower()]()
