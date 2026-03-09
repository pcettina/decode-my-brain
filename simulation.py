"""
Neural population simulation for the Decode My Brain app.

Implements cosine-tuned neurons and Poisson spike generation.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from utils import wrap_angle
from config import ADAPTATION_INCREMENT, BURST_INITIATION_PROB, MAX_SPIKE_PROB_PER_BIN

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
        """
        Get the tuning curve for a specific neuron.
        
        Args:
            theta: Array of directions to evaluate (radians)
            neuron_idx: Index of the neuron
            
        Returns:
            Firing rates at each direction
        """
        mu = self.preferred_directions[neuron_idx]
        return cosine_tuning(theta, mu, self.baseline_rate, self.modulation_depth)
    
    def get_all_tuning_curves(self, theta: np.ndarray) -> np.ndarray:
        """
        Get tuning curves for all neurons.
        
        Args:
            theta: Array of directions to evaluate (radians)
            
        Returns:
            2D array of shape (n_neurons, len(theta)) with firing rates
        """
        curves = np.zeros((self.n_neurons, len(theta)))
        for i in range(self.n_neurons):
            curves[i] = self.get_tuning_curve(theta, i)
        return curves


def generate_neuron_population(
    n_neurons: int = 50,
    baseline_rate: float = 5.0,
    modulation_depth: float = 15.0,
    random_preferred: bool = False,
    seed: Optional[int] = None
) -> NeuronPopulation:
    """
    Create a population of direction-tuned neurons.
    
    Args:
        n_neurons: Number of neurons (default 50)
        baseline_rate: Baseline firing rate r₀ in Hz (default 5.0)
        modulation_depth: Modulation depth k in Hz (default 15.0)
        random_preferred: If True, randomize preferred directions; 
                         if False, space uniformly
        seed: Random seed for reproducibility
        
    Returns:
        NeuronPopulation object with neuron parameters
    """
    if n_neurons < 1:
        raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")
    if baseline_rate < 0:
        raise ValueError(f"baseline_rate must be >= 0, got {baseline_rate}")
    if modulation_depth < 0:
        raise ValueError(f"modulation_depth must be >= 0, got {modulation_depth}")

    logger.info("Generating population: %d neurons, r0=%.1f, k=%.1f", n_neurons, baseline_rate, modulation_depth)
    rng = np.random.default_rng(seed)

    if random_preferred:
        # Random preferred directions
        preferred_directions = rng.uniform(0, 2 * np.pi, n_neurons)
    else:
        # Uniformly spaced preferred directions
        preferred_directions = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
    
    return NeuronPopulation(
        n_neurons=n_neurons,
        preferred_directions=preferred_directions,
        baseline_rate=baseline_rate,
        modulation_depth=modulation_depth
    )


def cosine_tuning(
    theta: float | np.ndarray,
    mu: float | np.ndarray,
    r0: float,
    k: float
) -> float | np.ndarray:
    """
    Compute firing rate using cosine tuning function.

    λ(θ) = max(0, r₀ + k · cos(θ - μ))

    Args:
        theta: Movement direction (radians), scalar or array
        mu: Preferred direction (radians), scalar or array (broadcasts with theta)
        r0: Baseline firing rate (Hz)
        k: Modulation depth (Hz)

    Returns:
        Firing rate(s) in Hz, enforced non-negative
    """
    rate = r0 + k * np.cos(theta - mu)
    return np.maximum(0, rate)


def simulate_trial(
    theta: float,
    neurons: NeuronPopulation,
    duration_ms: float = 500.0,
    variance_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate spike counts for a single trial.
    
    Args:
        theta: True movement direction (radians)
        neurons: NeuronPopulation object
        duration_ms: Trial duration in milliseconds
        variance_scale: Scale factor for Poisson variance (1.0 = standard Poisson)
        seed: Random seed for reproducibility
        
    Returns:
        Array of spike counts for each neuron
    """
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")

    rng = np.random.default_rng(seed)

    # Convert duration to seconds
    duration_s = duration_ms / 1000.0

    # Compute expected spike counts for each neuron (vectorized over neurons)
    rates = cosine_tuning(theta, neurons.preferred_directions,
                          neurons.baseline_rate, neurons.modulation_depth)
    expected_counts = rates * duration_s

    # Generate spike counts
    if variance_scale == 1.0:
        # Standard Poisson
        spike_counts = rng.poisson(expected_counts)
    else:
        # Scaled variance using negative binomial approximation
        # For variance_scale > 1: overdispersed
        # For variance_scale < 1: underdispersed (use clipped Gaussian)
        if variance_scale > 1:
            # Negative binomial: variance = mean * variance_scale
            # NB params: n (failures), p (success prob)
            # mean = n*(1-p)/p, var = n*(1-p)/p^2
            # So: var/mean = 1/p, p = mean/var = 1/variance_scale
            p = 1.0 / variance_scale
            p = np.clip(p, 0.01, 0.99)
            n = expected_counts * p / (1 - p)
            n = np.maximum(1, np.round(n).astype(int))
            spike_counts = rng.negative_binomial(n, p)
        else:
            # Underdispersed: use Gaussian with reduced variance, clipped to non-negative
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
    """
    Generate time-binned pseudo-raster data for a trial.
    
    Args:
        theta: True movement direction (radians)
        neurons: NeuronPopulation object
        duration_ms: Trial duration in milliseconds
        bin_size_ms: Size of each time bin in milliseconds
        variance_scale: Scale factor for variance
        seed: Random seed for reproducibility
        
    Returns:
        2D array of shape (n_neurons, n_bins) with spike counts per bin
    """
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
    if bin_size_ms <= 0:
        raise ValueError(f"bin_size_ms must be > 0, got {bin_size_ms}")

    rng = np.random.default_rng(seed)

    n_bins = int(np.ceil(duration_ms / bin_size_ms))
    raster = np.zeros((neurons.n_neurons, n_bins))

    # Compute rates for each neuron (vectorized)
    rates = cosine_tuning(theta, neurons.preferred_directions,
                          neurons.baseline_rate, neurons.modulation_depth)

    # Expected counts per bin
    bin_duration_s = bin_size_ms / 1000.0
    expected_per_bin = rates * bin_duration_s

    # Generate spikes for each bin
    for t in range(n_bins):
        if variance_scale == 1.0:
            raster[:, t] = rng.poisson(expected_per_bin)
        elif variance_scale > 1.0:
            # Negative binomial for overdispersion (matches simulate_trial)
            p = 1.0 / variance_scale
            p = np.clip(p, 0.01, 0.99)
            n = expected_per_bin * p / (1 - p)
            n = np.maximum(1, np.round(n).astype(int))
            raster[:, t] = rng.negative_binomial(n, p)
        else:
            # Underdispersed: Gaussian with reduced variance
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
    """
    Simulate multiple trials with given directions.
    
    Args:
        thetas: Array of movement directions for each trial (radians)
        neurons: NeuronPopulation object
        duration_ms: Trial duration in milliseconds
        variance_scale: Scale factor for variance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (spike_counts, thetas) where spike_counts has shape 
        (n_trials, n_neurons)
    """
    rng = np.random.default_rng(seed)

    n_trials = len(thetas)
    spike_counts = np.zeros((n_trials, neurons.n_neurons), dtype=int)

    # Generate child seeds for each trial to preserve reproducibility
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
    """
    Simulate multiple trials with random directions.
    
    Args:
        n_trials: Number of trials to simulate
        neurons: NeuronPopulation object
        duration_ms: Trial duration in milliseconds
        variance_scale: Scale factor for variance
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (spike_counts, thetas) where spike_counts has shape 
        (n_trials, n_neurons)
    """
    rng = np.random.default_rng(seed)

    # Generate random directions
    thetas = rng.uniform(0, 2 * np.pi, n_trials)

    # Generate child seed for simulate_multiple_trials
    child_seed = int(rng.integers(0, 2**31))

    return simulate_multiple_trials(
        thetas, neurons, duration_ms, variance_scale, seed=child_seed
    )


def compute_firing_rate_stats(
    spike_counts: np.ndarray,
    duration_ms: float
) -> dict:
    """
    Compute summary statistics of firing rates.
    
    Args:
        spike_counts: Array of spike counts, shape (n_trials, n_neurons)
        duration_ms: Trial duration in milliseconds
        
    Returns:
        Dictionary with mean, std, min, max firing rates
    """
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
    """
    Create a "lesioned" version of a neuron population.
    
    Args:
        neurons: Original NeuronPopulation
        lesion_factor: Factor to reduce the affected parameter (0-1)
        lesion_type: 'modulation' reduces k, 'baseline' reduces r0
        
    Returns:
        New NeuronPopulation with reduced parameters
    """
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


# =============================================================================
# Temporal Dynamics Models
# =============================================================================

@dataclass
class TemporalParams:
    """
    Parameters for temporal neural dynamics.
    
    Attributes:
        adaptation_strength: How much firing rate decreases with sustained activity (0-1)
        adaptation_tau_ms: Time constant for adaptation recovery (ms)
        refractory_abs_ms: Absolute refractory period (ms) - no spikes possible
        refractory_rel_ms: Relative refractory period (ms) - reduced firing probability
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
    
    Args:
        theta: Movement direction (radians)
        neurons: NeuronPopulation object
        duration_ms: Trial duration in milliseconds
        dt_ms: Time step for simulation (ms)
        temporal_params: TemporalParams object (uses defaults if None)
        variance_scale: Additional noise scaling
        seed: Random seed
        
    Returns:
        Tuple of (spike_times, spike_counts) where:
        - spike_times: List of lists, spike_times[i] contains spike times for neuron i
        - spike_counts: Array of total spike counts per neuron
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

    # Determine which neurons are bursting type
    is_bursting = rng.random(n_neurons) < temporal_params.burst_probability

    # Base firing rates for each neuron (vectorized)
    base_rates = cosine_tuning(theta, neurons.preferred_directions,
                               neurons.baseline_rate, neurons.modulation_depth)
    
    # Initialize state variables
    spike_times = [[] for _ in range(n_neurons)]
    last_spike_time = np.full(n_neurons, -1000.0)  # Last spike time per neuron
    adaptation_level = np.zeros(n_neurons)  # Current adaptation state
    burst_remaining = np.zeros(n_neurons, dtype=int)  # Spikes remaining in current burst
    
    # Simulate each time step
    for step in range(n_steps):
        t = step * dt_ms
        
        # Update adaptation (exponential decay)
        decay = np.exp(-dt_ms / temporal_params.adaptation_tau_ms)
        adaptation_level *= decay
        
        # Compute current firing rates with adaptation
        adapted_rates = base_rates * (1 - temporal_params.adaptation_strength * adaptation_level)
        adapted_rates = np.maximum(0, adapted_rates)
        
        # Apply refractory period effects
        time_since_spike = t - last_spike_time
        
        # Absolute refractory - zero probability
        in_abs_refractory = time_since_spike < temporal_params.refractory_abs_ms
        
        # Relative refractory - reduced probability
        in_rel_refractory = (time_since_spike >= temporal_params.refractory_abs_ms) & \
                           (time_since_spike < temporal_params.refractory_abs_ms + temporal_params.refractory_rel_ms)
        rel_refractory_factor = np.where(
            in_rel_refractory,
            (time_since_spike - temporal_params.refractory_abs_ms) / temporal_params.refractory_rel_ms,
            1.0
        )
        
        # Final firing probability for this time step
        # P(spike) = rate * dt / 1000 (convert Hz*ms to probability)
        spike_prob = adapted_rates * (dt_ms / 1000.0) * rel_refractory_factor * variance_scale
        spike_prob = np.where(in_abs_refractory, 0, spike_prob)
        spike_prob = np.clip(spike_prob, 0, MAX_SPIKE_PROB_PER_BIN)
        
        # Handle bursting neurons - if in burst, force spike
        in_burst = burst_remaining > 0
        spike_prob = np.where(in_burst & ~in_abs_refractory, 1.0, spike_prob)
        
        # Generate spikes
        spikes = rng.random(n_neurons) < spike_prob

        # Record spike times and update state
        for i in range(n_neurons):
            if spikes[i]:
                spike_times[i].append(t)
                last_spike_time[i] = t
                adaptation_level[i] = min(1.0, adaptation_level[i] + ADAPTATION_INCREMENT)

                # Handle burst initiation/continuation
                if in_burst[i]:
                    burst_remaining[i] -= 1
                elif is_bursting[i] and rng.random() < BURST_INITIATION_PROB:
                    # Start a new burst
                    burst_remaining[i] = temporal_params.burst_spikes - 1

    # Convert to spike counts
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
    
    Args:
        theta_func: Function that takes time (ms) and returns direction (radians)
        neurons: NeuronPopulation object
        duration_ms: Total duration in milliseconds
        dt_ms: Time step (ms)
        temporal_params: TemporalParams for dynamics
        seed: Random seed
        
    Returns:
        Tuple of (time_array, spike_times, instantaneous_rates)
        - time_array: Array of time points
        - spike_times: List of spike time lists per neuron
        - instantaneous_rates: Array of shape (n_neurons, n_time_points)
    """
    rng = np.random.default_rng(seed)

    if temporal_params is None:
        temporal_params = TemporalParams()

    n_neurons = neurons.n_neurons
    n_steps = int(np.ceil(duration_ms / dt_ms))
    time_array = np.arange(0, duration_ms, dt_ms)

    # Track instantaneous rates for visualization
    instantaneous_rates = np.zeros((n_neurons, len(time_array)))

    # Determine bursting neurons
    is_bursting = rng.random(n_neurons) < temporal_params.burst_probability
    
    # State variables
    spike_times = [[] for _ in range(n_neurons)]
    last_spike_time = np.full(n_neurons, -1000.0)
    adaptation_level = np.zeros(n_neurons)
    burst_remaining = np.zeros(n_neurons, dtype=int)
    
    for step, t in enumerate(time_array):
        # Get current direction
        theta = theta_func(t)
        
        # Compute base rates (vectorized over neurons)
        base_rates = cosine_tuning(theta, neurons.preferred_directions,
                                   neurons.baseline_rate, neurons.modulation_depth)
        
        # Apply adaptation
        decay = np.exp(-dt_ms / temporal_params.adaptation_tau_ms)
        adaptation_level *= decay
        adapted_rates = base_rates * (1 - temporal_params.adaptation_strength * adaptation_level)
        adapted_rates = np.maximum(0, adapted_rates)
        
        # Store instantaneous rates
        instantaneous_rates[:, step] = adapted_rates
        
        # Refractory effects
        time_since_spike = t - last_spike_time
        in_abs_refractory = time_since_spike < temporal_params.refractory_abs_ms
        in_rel_refractory = (time_since_spike >= temporal_params.refractory_abs_ms) & \
                           (time_since_spike < temporal_params.refractory_abs_ms + temporal_params.refractory_rel_ms)
        rel_factor = np.where(
            in_rel_refractory,
            (time_since_spike - temporal_params.refractory_abs_ms) / temporal_params.refractory_rel_ms,
            1.0
        )
        
        # Spike probability
        spike_prob = adapted_rates * (dt_ms / 1000.0) * rel_factor
        spike_prob = np.where(in_abs_refractory, 0, spike_prob)
        
        # Clip before burst override (burst should bypass cap)
        spike_prob = np.clip(spike_prob, 0, MAX_SPIKE_PROB_PER_BIN)

        # Bursting — applied after clip so burst probability is preserved
        in_burst = burst_remaining > 0
        spike_prob = np.where(in_burst & ~in_abs_refractory, 0.9, spike_prob)
        
        # Generate spikes
        spikes = rng.random(n_neurons) < spike_prob

        for i in range(n_neurons):
            if spikes[i]:
                spike_times[i].append(t)
                last_spike_time[i] = t
                adaptation_level[i] = min(1.0, adaptation_level[i] + ADAPTATION_INCREMENT)

                if in_burst[i]:
                    burst_remaining[i] -= 1
                elif is_bursting[i] and rng.random() < BURST_INITIATION_PROB:
                    burst_remaining[i] = temporal_params.burst_spikes - 1

    return time_array, spike_times, instantaneous_rates


def spike_times_to_binned(
    spike_times: List[List[float]],
    duration_ms: float,
    bin_size_ms: float = 10.0
) -> np.ndarray:
    """
    Convert spike times to binned spike counts.
    
    Args:
        spike_times: List of spike time lists per neuron
        duration_ms: Total duration
        bin_size_ms: Bin size in ms
        
    Returns:
        Array of shape (n_neurons, n_bins) with spike counts
    """
    n_neurons = len(spike_times)
    n_bins = int(np.ceil(duration_ms / bin_size_ms))
    binned = np.zeros((n_neurons, n_bins), dtype=int)
    
    for i, times in enumerate(spike_times):
        for t in times:
            bin_idx = int(t / bin_size_ms)
            if 0 <= bin_idx < n_bins:
                binned[i, bin_idx] += 1
    
    return binned



# =============================================================================
# Multi-Brain-Area Hierarchical Network
# =============================================================================

@dataclass
class BrainArea:
    """
    A brain area containing a population of neurons with specific properties.
    
    Attributes:
        name: Name of the brain area (e.g., 'M1', 'PMd', 'PPC')
        neurons: NeuronPopulation in this area
        modulation_gain: Amplitude scaling factor for modulation depth (>1 = stronger)
        noise_level: Variability in neural responses
        delay_ms: Processing delay for this area
        description: Human-readable description
    """
    name: str
    neurons: NeuronPopulation
    modulation_gain: float = 1.0
    noise_level: float = 1.0
    delay_ms: float = 0.0
    description: str = ""


class HierarchicalNetwork:
    """
    A hierarchical network of interconnected brain areas.
    
    Simulates information flow through multiple brain regions with
    feedforward and feedback connections.
    """
    
    def __init__(
        self,
        n_neurons_per_area: int = 50,
        baseline_rate: float = 5.0,
        modulation_depth: float = 15.0,
        seed: Optional[int] = None
    ):
        """
        Initialize a hierarchical network with default brain areas.

        Args:
            n_neurons_per_area: Number of neurons in each area
            baseline_rate: Baseline firing rate (Hz)
            modulation_depth: Modulation depth (Hz)
            seed: Random seed for reproducibility
        """
        self.n_neurons_per_area = n_neurons_per_area
        self.baseline_rate = baseline_rate
        self.modulation_depth = modulation_depth
        self._rng = np.random.default_rng(seed)

        # Create brain areas with different properties
        self.areas = self._create_default_areas()

        # Connection weights between areas (feedforward and feedback)
        self.connections = self._create_default_connections()
    
    def _create_default_areas(self) -> dict:
        """Create the default set of brain areas."""
        areas = {}
        
        # M1 - Primary Motor Cortex: Sharp tuning, direct motor output
        m1_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate,
            modulation_depth=self.modulation_depth * 1.2  # Strong modulation
        )
        areas['M1'] = BrainArea(
            name='M1',
            neurons=m1_neurons,
            modulation_gain=1.5,  # Sharp tuning
            noise_level=0.8,       # Low noise
            delay_ms=0.0,          # No delay (output area)
            description="Primary Motor Cortex - Direct movement execution"
        )
        
        # PMd - Dorsal Premotor: Broader tuning, movement planning
        pmd_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate * 0.8,
            modulation_depth=self.modulation_depth * 0.9
        )
        areas['PMd'] = BrainArea(
            name='PMd',
            neurons=pmd_neurons,
            modulation_gain=1.0,   # Moderate tuning
            noise_level=1.0,        # Moderate noise
            delay_ms=20.0,          # 20ms ahead of M1
            description="Dorsal Premotor - Movement planning and preparation"
        )
        
        # PPC - Posterior Parietal Cortex: Broad tuning, goal encoding
        ppc_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate * 0.6,
            modulation_depth=self.modulation_depth * 0.7
        )
        areas['PPC'] = BrainArea(
            name='PPC',
            neurons=ppc_neurons,
            modulation_gain=0.7,   # Broad tuning
            noise_level=1.2,        # Higher noise
            delay_ms=50.0,          # 50ms ahead of M1
            description="Posterior Parietal - Spatial goals and intentions"
        )
        
        # SMA - Supplementary Motor Area: Sequence encoding
        sma_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate * 0.7,
            modulation_depth=self.modulation_depth * 0.8
        )
        areas['SMA'] = BrainArea(
            name='SMA',
            neurons=sma_neurons,
            modulation_gain=0.9,
            noise_level=1.1,
            delay_ms=30.0,
            description="Supplementary Motor Area - Movement sequences"
        )
        
        return areas
    
    def _create_default_connections(self) -> dict:
        """Create connection weight matrices between areas."""
        n = self.n_neurons_per_area
        connections = {}
        
        # Feedforward connections (information flows toward M1)
        # PPC -> PMd (strong)
        connections[('PPC', 'PMd')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.3 + 0.5,
            'type': 'feedforward',
            'strength': 0.6
        }

        # PMd -> M1 (very strong)
        connections[('PMd', 'M1')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.2 + 0.7,
            'type': 'feedforward',
            'strength': 0.8
        }

        # SMA -> M1 (moderate)
        connections[('SMA', 'M1')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.3 + 0.4,
            'type': 'feedforward',
            'strength': 0.5
        }

        # PPC -> SMA
        connections[('PPC', 'SMA')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.3 + 0.4,
            'type': 'feedforward',
            'strength': 0.4
        }

        # Feedback connections
        # M1 -> PMd (feedback for error correction)
        connections[('M1', 'PMd')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.2 + 0.3,
            'type': 'feedback',
            'strength': 0.3
        }

        # PMd -> PPC (feedback)
        connections[('PMd', 'PPC')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.2 + 0.2,
            'type': 'feedback',
            'strength': 0.2
        }
        
        return connections
    
    def get_area(self, name: str) -> BrainArea:
        """Get a brain area by name."""
        return self.areas.get(name)
    
    def get_area_names(self) -> List[str]:
        """Get list of all area names."""
        return list(self.areas.keys())
    
    def simulate_area(
        self,
        area_name: str,
        theta: float,
        duration_ms: float = 500.0,
        variance_scale: float = 1.0
    ) -> np.ndarray:
        """
        Simulate activity in a single brain area.
        
        Args:
            area_name: Name of the area to simulate
            theta: Movement direction (radians)
            duration_ms: Trial duration
            variance_scale: Additional noise scaling
            
        Returns:
            Spike counts for each neuron
        """
        area = self.areas[area_name]
        
        # Apply tuning sharpness by modifying effective modulation depth
        effective_neurons = NeuronPopulation(
            n_neurons=area.neurons.n_neurons,
            preferred_directions=area.neurons.preferred_directions,
            baseline_rate=area.neurons.baseline_rate,
            modulation_depth=area.neurons.modulation_depth * area.modulation_gain
        )
        
        # Simulate with area-specific noise
        spikes = simulate_trial(
            theta,
            effective_neurons,
            duration_ms=duration_ms,
            variance_scale=variance_scale * area.noise_level
        )
        
        return spikes
    
    def simulate_hierarchy(
        self,
        theta: float,
        duration_ms: float = 500.0,
        variance_scale: float = 1.0,
        include_dynamics: bool = True
    ) -> dict:
        """
        Simulate activity across all brain areas with inter-area dynamics.
        
        Args:
            theta: Movement direction (radians)
            duration_ms: Trial duration
            variance_scale: Noise scaling
            include_dynamics: Whether to include inter-area interactions
            
        Returns:
            Dictionary mapping area names to spike count arrays
        """
        results = {}
        
        # Simulate each area independently first
        for name, area in self.areas.items():
            results[name] = self.simulate_area(
                name, theta, duration_ms, variance_scale
            )
        
        if include_dynamics:
            # Apply feedforward influences
            for (source, target), conn in self.connections.items():
                if conn['type'] == 'feedforward':
                    # Modulate target based on source activity
                    source_activity = results[source].astype(float)
                    influence = conn['weights'] @ source_activity * conn['strength']
                    
                    # Add influence to target (with some noise)
                    noise = self._rng.standard_normal(len(results[target])) * 0.1
                    modulation = np.clip(1 + influence * 0.1 + noise, 0.5, 2.0)
                    results[target] = np.round(results[target] * modulation).astype(int)
        
        return results
    
    def get_connectivity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get the connectivity matrix between all areas.
        
        Returns:
            Tuple of (connectivity_matrix, area_names)
        """
        names = list(self.areas.keys())
        n_areas = len(names)
        matrix = np.zeros((n_areas, n_areas))
        
        for (source, target), conn in self.connections.items():
            i = names.index(source)
            j = names.index(target)
            matrix[i, j] = conn['strength']
        
        return matrix, names
    
    def get_hierarchy_order(self) -> List[str]:
        """Get areas ordered from highest to lowest in hierarchy."""
        # Order by delay (higher delay = earlier in processing)
        sorted_areas = sorted(
            self.areas.items(),
            key=lambda x: x[1].delay_ms,
            reverse=True
        )
        return [name for name, _ in sorted_areas]


def create_hierarchical_network(
    n_neurons_per_area: int = 50,
    baseline_rate: float = 5.0,
    modulation_depth: float = 15.0
) -> HierarchicalNetwork:
    """
    Factory function to create a hierarchical brain network.
    
    Args:
        n_neurons_per_area: Neurons per brain area
        baseline_rate: Baseline firing rate
        modulation_depth: Modulation depth
        
    Returns:
        Configured HierarchicalNetwork
    """
    return HierarchicalNetwork(
        n_neurons_per_area=n_neurons_per_area,
        baseline_rate=baseline_rate,
        modulation_depth=modulation_depth
    )


def simulate_hierarchical_trial(
    network: HierarchicalNetwork,
    theta: float,
    duration_ms: float = 500.0,
    variance_scale: float = 1.0
) -> dict:
    """
    Convenience function to simulate a trial across the hierarchy.
    
    Args:
        network: HierarchicalNetwork instance
        theta: Movement direction
        duration_ms: Trial duration
        variance_scale: Noise scaling
        
    Returns:
        Dictionary of area_name -> spike_counts
    """
    return network.simulate_hierarchy(
        theta, duration_ms, variance_scale, include_dynamics=True
    )

