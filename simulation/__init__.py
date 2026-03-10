"""
Neural population simulation subpackage.

Re-exports all public symbols so existing ``from simulation import X`` imports
continue to work unchanged.
"""

from simulation.core import (
    NeuronPopulation,
    cosine_tuning,
    generate_neuron_population,
    simulate_trial,
    simulate_raster,
    simulate_multiple_trials,
    simulate_random_trials,
    compute_firing_rate_stats,
    create_lesioned_population,
)
from simulation.temporal import (
    TemporalParams,
    simulate_temporal_spikes,
    simulate_continuous_activity,
    spike_times_to_binned,
    _temporal_step,
)
from simulation.hierarchy import (
    BrainArea,
    HierarchicalNetwork,
    create_hierarchical_network,
    simulate_hierarchical_trial,
)

__all__ = [
    # core
    'NeuronPopulation',
    'cosine_tuning',
    'generate_neuron_population',
    'simulate_trial',
    'simulate_raster',
    'simulate_multiple_trials',
    'simulate_random_trials',
    'compute_firing_rate_stats',
    'create_lesioned_population',
    # temporal
    'TemporalParams',
    'simulate_temporal_spikes',
    'simulate_continuous_activity',
    'spike_times_to_binned',
    '_temporal_step',
    # hierarchy
    'BrainArea',
    'HierarchicalNetwork',
    'create_hierarchical_network',
    'simulate_hierarchical_trial',
]
