"""
Temporal dynamics models: adaptation, refractory periods, bursting.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from simulation.core import NeuronPopulation, cosine_tuning
from config import ADAPTATION_INCREMENT, BURST_INITIATION_PROB, MAX_SPIKE_PROB_PER_BIN

logger = logging.getLogger(__name__)


@dataclass
class TemporalParams:
    """
    Parameters for temporal neural dynamics.

    Attributes:
        adaptation_strength: How much firing rate decreases with sustained activity (0-1)
        adaptation_tau_ms: Time constant for adaptation recovery (ms)
        refractory_abs_ms: Absolute refractory period (ms)
        refractory_rel_ms: Relative refractory period (ms)
        burst_probability: Probability that a neuron is a "bursting" type (0-1)
        burst_spikes: Number of spikes in a burst
        burst_isi_ms: Inter-spike interval within a burst (ms)
    """
    adaptation_strength: float = 0.3
    adaptation_tau_ms: float = 100.0
    refractory_abs_ms: float = 2.0
    refractory_rel_ms: float = 5.0
    burst_probability: float = 0.2
    burst_spikes: int = 3
    burst_isi_ms: float = 3.0


def _temporal_step(
    rng: np.random.Generator,
    base_rates: np.ndarray,
    t_ms: float,
    dt_ms: float,
    temporal_params: TemporalParams,
    last_spike_time: np.ndarray,
    adaptation_level: np.ndarray,
    burst_remaining: np.ndarray,
    is_bursting: np.ndarray,
    spike_times: List[List[float]],
    variance_scale: float = 1.0,
    burst_prob_override: float = 1.0,
) -> np.ndarray:
    """
    Execute one time step of temporal spike simulation (shared helper).

    All state arrays are updated in-place. Returns adapted firing rates.
    """
    n_neurons = len(base_rates)

    # Adaptation decay
    decay = np.exp(-dt_ms / temporal_params.adaptation_tau_ms)
    adaptation_level *= decay

    # Compute adapted rates
    adapted_rates = base_rates * (1 - temporal_params.adaptation_strength * adaptation_level)
    adapted_rates = np.maximum(0, adapted_rates)

    # Refractory effects
    time_since_spike = t_ms - last_spike_time
    in_abs_refractory = time_since_spike < temporal_params.refractory_abs_ms
    in_rel_refractory = (
        (time_since_spike >= temporal_params.refractory_abs_ms)
        & (time_since_spike < temporal_params.refractory_abs_ms + temporal_params.refractory_rel_ms)
    )
    rel_factor = np.where(
        in_rel_refractory,
        (time_since_spike - temporal_params.refractory_abs_ms) / temporal_params.refractory_rel_ms,
        1.0,
    )

    # Spike probability
    spike_prob = adapted_rates * (dt_ms / 1000.0) * rel_factor * variance_scale
    spike_prob = np.where(in_abs_refractory, 0, spike_prob)
    spike_prob = np.clip(spike_prob, 0, MAX_SPIKE_PROB_PER_BIN)

    # Burst override
    in_burst = burst_remaining > 0
    spike_prob = np.where(in_burst & ~in_abs_refractory, burst_prob_override, spike_prob)

    # Generate spikes (vectorized)
    spikes = rng.random(n_neurons) < spike_prob

    # Vectorized state updates
    spiking_indices = np.where(spikes)[0]
    if len(spiking_indices) > 0:
        last_spike_time[spiking_indices] = t_ms
        adaptation_level[spiking_indices] = np.minimum(
            1.0, adaptation_level[spiking_indices] + ADAPTATION_INCREMENT
        )

        for idx in spiking_indices:
            spike_times[idx].append(t_ms)

        continuing_burst = spiking_indices[in_burst[spiking_indices]]
        burst_remaining[continuing_burst] -= 1

        new_spike_mask = spikes & is_bursting & ~in_burst
        new_burst_candidates = np.where(new_spike_mask)[0]
        if len(new_burst_candidates) > 0:
            initiate = rng.random(len(new_burst_candidates)) < BURST_INITIATION_PROB
            burst_remaining[new_burst_candidates[initiate]] = temporal_params.burst_spikes - 1

    return adapted_rates


def simulate_temporal_spikes(
    theta: float,
    neurons: NeuronPopulation,
    duration_ms: float = 500.0,
    dt_ms: float = 1.0,
    temporal_params: Optional[TemporalParams] = None,
    variance_scale: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[List[List[float]], np.ndarray]:
    """
    Simulate spike times with temporal dynamics (adaptation, refractory, bursting).

    Returns:
        Tuple of (spike_times, spike_counts)
    """
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

    rng = np.random.default_rng(seed)

    if temporal_params is None:
        temporal_params = TemporalParams()

    n_neurons = neurons.n_neurons
    n_steps = int(np.ceil(duration_ms / dt_ms))

    is_bursting = rng.random(n_neurons) < temporal_params.burst_probability
    base_rates = cosine_tuning(theta, neurons.preferred_directions,
                               neurons.baseline_rate, neurons.modulation_depth)

    spike_times: List[List[float]] = [[] for _ in range(n_neurons)]
    last_spike_time = np.full(n_neurons, -1000.0)
    adaptation_level = np.zeros(n_neurons)
    burst_remaining = np.zeros(n_neurons, dtype=int)

    for step in range(n_steps):
        t = step * dt_ms
        _temporal_step(
            rng, base_rates, t, dt_ms, temporal_params,
            last_spike_time, adaptation_level, burst_remaining,
            is_bursting, spike_times,
            variance_scale=variance_scale,
            burst_prob_override=1.0,
        )

    spike_counts = np.array([len(times) for times in spike_times])
    return spike_times, spike_counts


def simulate_continuous_activity(
    theta_func,
    neurons: NeuronPopulation,
    duration_ms: float = 2000.0,
    dt_ms: float = 1.0,
    temporal_params: Optional[TemporalParams] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[List[float]], np.ndarray]:
    """
    Simulate continuous neural activity with time-varying direction.

    Returns:
        Tuple of (time_array, spike_times, instantaneous_rates)
    """
    rng = np.random.default_rng(seed)

    if temporal_params is None:
        temporal_params = TemporalParams()

    n_neurons = neurons.n_neurons
    time_array = np.arange(0, duration_ms, dt_ms)
    instantaneous_rates = np.zeros((n_neurons, len(time_array)))

    is_bursting = rng.random(n_neurons) < temporal_params.burst_probability
    spike_times: List[List[float]] = [[] for _ in range(n_neurons)]
    last_spike_time = np.full(n_neurons, -1000.0)
    adaptation_level = np.zeros(n_neurons)
    burst_remaining = np.zeros(n_neurons, dtype=int)

    for step, t in enumerate(time_array):
        theta = theta_func(t)
        base_rates = cosine_tuning(theta, neurons.preferred_directions,
                                   neurons.baseline_rate, neurons.modulation_depth)

        adapted_rates = _temporal_step(
            rng, base_rates, t, dt_ms, temporal_params,
            last_spike_time, adaptation_level, burst_remaining,
            is_bursting, spike_times,
            variance_scale=1.0,
            burst_prob_override=0.9,
        )

        instantaneous_rates[:, step] = adapted_rates

    return time_array, spike_times, instantaneous_rates


def spike_times_to_binned(
    spike_times: List[List[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0
) -> np.ndarray:
    """Convert spike times to binned spike counts."""
    n_neurons = len(spike_times)
    n_bins = int(np.ceil(duration_ms / bin_size_ms))
    binned = np.zeros((n_neurons, n_bins), dtype=int)

    for i, times in enumerate(spike_times):
        for t in times:
            bin_idx = int(t / bin_size_ms)
            if 0 <= bin_idx < n_bins:
                binned[i, bin_idx] += 1

    return binned
