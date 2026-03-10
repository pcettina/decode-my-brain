"""
Base decoder class and shared utilities.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from simulation.core import NeuronPopulation, cosine_tuning
from config import LOG_EPSILON

logger = logging.getLogger(__name__)


def _compute_poisson_log_likelihoods(
    theta_grid: np.ndarray,
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    duration_s: float
) -> np.ndarray:
    """
    Compute Poisson log-likelihoods over a grid of candidate directions.

    Vectorized: evaluates all (theta, neuron) pairs simultaneously.
    """
    rates = cosine_tuning(
        theta_grid[:, np.newaxis],
        neurons.preferred_directions[np.newaxis, :],
        neurons.baseline_rate,
        neurons.modulation_depth
    )
    expected = np.maximum(rates * duration_s, LOG_EPSILON)
    return np.log(expected) @ spike_counts - expected.sum(axis=1)


def _validate_decode_inputs(spike_counts: np.ndarray, neurons: NeuronPopulation) -> None:
    """Validate inputs common to all decoders."""
    if len(spike_counts) == 0:
        raise ValueError("spike_counts must not be empty")
    if len(spike_counts) != neurons.n_neurons:
        raise ValueError(
            f"spike_counts length ({len(spike_counts)}) does not match "
            f"neurons.n_neurons ({neurons.n_neurons})"
        )


class Decoder(ABC):
    """Abstract base class for neural decoders."""

    @abstractmethod
    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        """
        Decode movement direction from spike counts.

        Returns:
            Estimated direction θ̂ in radians [0, 2π)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this decoder."""
        pass
