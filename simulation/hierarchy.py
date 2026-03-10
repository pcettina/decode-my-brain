"""
Multi-brain-area hierarchical network simulation.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from simulation.core import (
    NeuronPopulation,
    generate_neuron_population,
    simulate_trial,
)

logger = logging.getLogger(__name__)


@dataclass
class BrainArea:
    """
    A brain area containing a population of neurons with specific properties.
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
    """

    def __init__(
        self,
        n_neurons_per_area: int = 50,
        baseline_rate: float = 5.0,
        modulation_depth: float = 15.0,
        seed: Optional[int] = None
    ):
        self.n_neurons_per_area = n_neurons_per_area
        self.baseline_rate = baseline_rate
        self.modulation_depth = modulation_depth
        self._rng = np.random.default_rng(seed)

        self.areas = self._create_default_areas()
        self.connections = self._create_default_connections()

    def _create_default_areas(self) -> dict:
        """Create the default set of brain areas."""
        areas = {}

        m1_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate,
            modulation_depth=self.modulation_depth * 1.2
        )
        areas['M1'] = BrainArea(
            name='M1', neurons=m1_neurons,
            modulation_gain=1.5, noise_level=0.8, delay_ms=0.0,
            description="Primary Motor Cortex - Direct movement execution"
        )

        pmd_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate * 0.8,
            modulation_depth=self.modulation_depth * 0.9
        )
        areas['PMd'] = BrainArea(
            name='PMd', neurons=pmd_neurons,
            modulation_gain=1.0, noise_level=1.0, delay_ms=20.0,
            description="Dorsal Premotor - Movement planning and preparation"
        )

        ppc_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate * 0.6,
            modulation_depth=self.modulation_depth * 0.7
        )
        areas['PPC'] = BrainArea(
            name='PPC', neurons=ppc_neurons,
            modulation_gain=0.7, noise_level=1.2, delay_ms=50.0,
            description="Posterior Parietal - Spatial goals and intentions"
        )

        sma_neurons = generate_neuron_population(
            n_neurons=self.n_neurons_per_area,
            baseline_rate=self.baseline_rate * 0.7,
            modulation_depth=self.modulation_depth * 0.8
        )
        areas['SMA'] = BrainArea(
            name='SMA', neurons=sma_neurons,
            modulation_gain=0.9, noise_level=1.1, delay_ms=30.0,
            description="Supplementary Motor Area - Movement sequences"
        )

        return areas

    def _create_default_connections(self) -> dict:
        """Create connection weight matrices between areas."""
        n = self.n_neurons_per_area
        connections = {}

        connections[('PPC', 'PMd')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.3 + 0.5,
            'type': 'feedforward', 'strength': 0.6
        }
        connections[('PMd', 'M1')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.2 + 0.7,
            'type': 'feedforward', 'strength': 0.8
        }
        connections[('SMA', 'M1')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.3 + 0.4,
            'type': 'feedforward', 'strength': 0.5
        }
        connections[('PPC', 'SMA')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.3 + 0.4,
            'type': 'feedforward', 'strength': 0.4
        }
        connections[('M1', 'PMd')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.2 + 0.3,
            'type': 'feedback', 'strength': 0.3
        }
        connections[('PMd', 'PPC')] = {
            'weights': self._rng.standard_normal((n, n)) * 0.2 + 0.2,
            'type': 'feedback', 'strength': 0.2
        }

        return connections

    def get_area(self, name: str) -> BrainArea:
        """Get a brain area by name."""
        return self.areas.get(name)

    def get_area_names(self) -> List[str]:
        """Get list of all area names."""
        return list(self.areas.keys())

    def simulate_area(
        self, area_name: str, theta: float,
        duration_ms: float = 500.0, variance_scale: float = 1.0
    ) -> np.ndarray:
        """Simulate activity in a single brain area."""
        area = self.areas[area_name]

        effective_neurons = NeuronPopulation(
            n_neurons=area.neurons.n_neurons,
            preferred_directions=area.neurons.preferred_directions,
            baseline_rate=area.neurons.baseline_rate,
            modulation_depth=area.neurons.modulation_depth * area.modulation_gain
        )

        return simulate_trial(
            theta, effective_neurons,
            duration_ms=duration_ms,
            variance_scale=variance_scale * area.noise_level
        )

    def simulate_hierarchy(
        self, theta: float, duration_ms: float = 500.0,
        variance_scale: float = 1.0, include_dynamics: bool = True
    ) -> dict:
        """Simulate activity across all brain areas with inter-area dynamics."""
        results = {}

        for name in self.areas:
            results[name] = self.simulate_area(name, theta, duration_ms, variance_scale)

        if include_dynamics:
            for (source, target), conn in self.connections.items():
                if conn['type'] == 'feedforward':
                    source_activity = results[source].astype(float)
                    influence = conn['weights'] @ source_activity * conn['strength']
                    noise = self._rng.standard_normal(len(results[target])) * 0.1
                    modulation = np.clip(1 + influence * 0.1 + noise, 0.5, 2.0)
                    results[target] = np.round(results[target] * modulation).astype(int)

        return results

    def get_connectivity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Get the connectivity matrix between all areas."""
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
    """Factory function to create a hierarchical brain network."""
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
    """Convenience function to simulate a trial across the hierarchy."""
    return network.simulate_hierarchy(
        theta, duration_ms, variance_scale, include_dynamics=True
    )
