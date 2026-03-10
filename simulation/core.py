"""
Core neural population modeling: tuning curves, populations, spike generation.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class NeuronPopulation:
    """
    A population of direction-tuned neurons.

    Attributes:
        n_neurons: Number of neurons in the population
        preferred_directions: Array of preferred directions (radians) for each neuron
        baseline_rate: Baseline firing rate r₀ (Hz)
        modulation_depth: Modulation depth k (Hz)
    """
    n_neurons: int
    preferred_directions: np.ndarray
    baseline_rate: float
    modulation_depth: float

    def get_tuning_curve(self, theta: np.ndarray, neuron_idx: int) -> np.ndarray:
        """Get the tuning curve for a specific neuron."""
        mu = self.preferred_directions[neuron_idx]
        return cosine_tuning(theta, mu, self.baseline_rate, self.modulation_depth)

    def get_all_tuning_curves(self, theta: np.ndarray) -> np.ndarray:
        """Get tuning curves for all neurons. Returns (n_neurons, len(theta))."""
        curves = np.zeros((self.n_neurons, len(theta)))
        for i in range(self.n_neurons):
            curves[i] = self.get_tuning_curve(theta, i)
        return curves


def cosine_tuning(
    theta: float | np.ndarray,
    mu: float | np.ndarray,
    r0: float,
    k: float
) -> float | np.ndarray:
    """
    Compute firing rate using cosine tuning function.

    λ(θ) = max(0, r₀ + k · cos(θ - μ))
    """
    rate = r0 + k * np.cos(theta - mu)
    return np.maximum(0, rate)


def generate_neuron_population(
    n_neurons: int = 50,
    baseline_rate: float = 5.0,
    modulation_depth: float = 15.0,
    random_preferred: bool = False,
    seed: Optional[int] = None
) -> NeuronPopulation:
    """Create a population of direction-tuned neurons."""
    if n_neurons < 1:
        raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")
    if baseline_rate < 0:
        raise ValueError(f"baseline_rate must be >= 0, got {baseline_rate}")
    if modulation_depth < 0:
        raise ValueError(f"modulation_depth must be >= 0, got {modulation_depth}")

    logger.info("Generating population: %d neurons, r0=%.1f, k=%.1f",
                n_neurons, baseline_rate, modulation_depth)
    rng = np.random.default_rng(seed)

    if random_preferred:
        preferred_directions = rng.uniform(0, 2 * np.pi, n_neurons)
    else:
        preferred_directions = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)

    return NeuronPopulation(
        n_neurons=n_neurons,
        preferred_directions=preferred_directions,
        baseline_rate=baseline_rate,
        modulation_depth=modulation_depth
    )


def simulate_trial(
    theta: float,
    neurons: NeuronPopulation,
    duration_ms: float = 500.0,
    variance_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Simulate spike counts for a single trial."""
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")

    rng = np.random.default_rng(seed)
    duration_s = duration_ms / 1000.0

    rates = cosine_tuning(theta, neurons.preferred_directions,
                          neurons.baseline_rate, neurons.modulation_depth)
    expected_counts = rates * duration_s

    if variance_scale == 1.0:
        spike_counts = rng.poisson(expected_counts)
    elif variance_scale > 1:
        p = 1.0 / variance_scale
        p = np.clip(p, 0.01, 0.99)
        n = expected_counts * p / (1 - p)
        n = np.maximum(1, np.round(n).astype(int))
        spike_counts = rng.negative_binomial(n, p)
    else:
        std = np.sqrt(expected_counts * variance_scale)
        spike_counts = rng.normal(expected_counts, std)
        spike_counts = np.maximum(0, np.round(spike_counts)).astype(int)

    return spike_counts


def simulate_raster(
    theta: float,
    neurons: NeuronPopulation,
    duration_ms: float = 500.0,
    bin_size_ms: float = 10.0,
    variance_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate time-binned pseudo-raster data for a trial."""
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
    if bin_size_ms <= 0:
        raise ValueError(f"bin_size_ms must be > 0, got {bin_size_ms}")

    rng = np.random.default_rng(seed)
    n_bins = int(np.ceil(duration_ms / bin_size_ms))
    raster = np.zeros((neurons.n_neurons, n_bins))

    rates = cosine_tuning(theta, neurons.preferred_directions,
                          neurons.baseline_rate, neurons.modulation_depth)
    bin_duration_s = bin_size_ms / 1000.0
    expected_per_bin = rates * bin_duration_s

    for t in range(n_bins):
        if variance_scale == 1.0:
            raster[:, t] = rng.poisson(expected_per_bin)
        elif variance_scale > 1.0:
            p = 1.0 / variance_scale
            p = np.clip(p, 0.01, 0.99)
            n = expected_per_bin * p / (1 - p)
            n = np.maximum(1, np.round(n).astype(int))
            raster[:, t] = rng.negative_binomial(n, p)
        else:
            std = np.sqrt(expected_per_bin * variance_scale)
            raster[:, t] = np.maximum(0, np.round(
                rng.normal(expected_per_bin, std)
            ))

    return raster.astype(int)


def simulate_multiple_trials(
    thetas: np.ndarray,
    neurons: NeuronPopulation,
    duration_ms: float = 500.0,
    variance_scale: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate multiple trials with given directions."""
    rng = np.random.default_rng(seed)
    n_trials = len(thetas)
    spike_counts = np.zeros((n_trials, neurons.n_neurons), dtype=int)
    child_seeds = rng.integers(0, 2**31, size=n_trials)

    for i, theta in enumerate(thetas):
        spike_counts[i] = simulate_trial(
            theta, neurons, duration_ms, variance_scale, seed=int(child_seeds[i])
        )

    return spike_counts, thetas


def simulate_random_trials(
    n_trials: int,
    neurons: NeuronPopulation,
    duration_ms: float = 500.0,
    variance_scale: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate multiple trials with random directions."""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, n_trials)
    child_seed = int(rng.integers(0, 2**31))
    return simulate_multiple_trials(
        thetas, neurons, duration_ms, variance_scale, seed=child_seed
    )


def compute_firing_rate_stats(
    spike_counts: np.ndarray,
    duration_ms: float
) -> dict:
    """Compute summary statistics of firing rates."""
    duration_s = duration_ms / 1000.0
    firing_rates = spike_counts / duration_s

    return {
        'mean_rate': float(np.mean(firing_rates)),
        'std_rate': float(np.std(firing_rates)),
        'min_rate': float(np.min(firing_rates)),
        'max_rate': float(np.max(firing_rates)),
        'mean_per_neuron': np.mean(firing_rates, axis=0),
        'std_per_neuron': np.std(firing_rates, axis=0)
    }


def create_lesioned_population(
    neurons: NeuronPopulation,
    lesion_factor: float = 0.5,
    lesion_type: str = 'modulation'
) -> NeuronPopulation:
    """Create a 'lesioned' version of a neuron population."""
    if lesion_type == 'modulation':
        return NeuronPopulation(
            n_neurons=neurons.n_neurons,
            preferred_directions=neurons.preferred_directions.copy(),
            baseline_rate=neurons.baseline_rate,
            modulation_depth=neurons.modulation_depth * lesion_factor
        )
    elif lesion_type == 'baseline':
        return NeuronPopulation(
            n_neurons=neurons.n_neurons,
            preferred_directions=neurons.preferred_directions.copy(),
            baseline_rate=neurons.baseline_rate * lesion_factor,
            modulation_depth=neurons.modulation_depth
        )
    else:
        raise ValueError(f"Unknown lesion type: {lesion_type}")
