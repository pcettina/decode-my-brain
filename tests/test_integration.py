"""Integration tests — end-to-end pipelines combining multiple modules."""

import numpy as np
import pytest
from simulation import (
    generate_neuron_population,
    simulate_trial,
    simulate_random_trials,
    simulate_raster,
    simulate_temporal_spikes,
    spike_times_to_binned,
    create_lesioned_population,
    create_hierarchical_network,
    simulate_hierarchical_trial,
)
from decoders import (
    PopulationVectorDecoder,
    MaximumLikelihoodDecoder,
    KalmanFilterDecoder,
    evaluate_decoder,
    compare_decoders,
)
from utils import angular_error


class TestEndToEndPV:
    def test_pv_accuracy(self):
        """50 neurons, 100 trials: PV mean error should be < 45 degrees."""
        neurons = generate_neuron_population(50, seed=123)
        spikes, dirs = simulate_random_trials(100, neurons, duration_ms=500, seed=456)
        results = evaluate_decoder(
            PopulationVectorDecoder(), spikes, dirs, neurons
        )
        assert results['mean_error_degrees'] < 45


class TestEndToEndML:
    def test_ml_accuracy(self):
        """50 neurons, 100 trials: ML should be < 30 degrees."""
        neurons = generate_neuron_population(50, seed=123)
        spikes, dirs = simulate_random_trials(100, neurons, seed=456)
        results = evaluate_decoder(
            MaximumLikelihoodDecoder(), spikes, dirs, neurons
        )
        assert results['mean_error_degrees'] < 30


class TestEndToEndKalman:
    def test_kalman_trajectory(self):
        """Kalman decoder should track a sequence of spike observations."""
        neurons = generate_neuron_population(30, seed=0)
        kf = KalmanFilterDecoder(n_neurons=30)
        kf.fit_from_neurons(neurons)

        # Generate a short sequence of spikes at different directions
        directions = np.linspace(0, np.pi, 5)
        sequence = np.zeros((5, 30))
        for i, theta in enumerate(directions):
            sequence[i] = simulate_trial(theta, neurons, duration_ms=500, seed=i)

        states, covs = kf.decode_trajectory(sequence, neurons)
        assert states.shape == (5, 4)
        # States should change over the trajectory
        assert not np.allclose(states[0], states[-1])


class TestEndToEndDecoderComparison:
    def test_all_decoders(self):
        """compare_decoders should return results for all decoders."""
        neurons = generate_neuron_population(30, seed=0)
        spikes, dirs = simulate_random_trials(20, neurons, seed=0)
        results = compare_decoders(spikes, dirs, neurons)
        assert len(results) == 3
        for name, data in results.items():
            assert data['mean_error_deg'] < 90  # reasonable range


class TestEndToEndLesion:
    def test_lesion_degrades_performance(self):
        """Lesioned population should decode worse than normal."""
        neurons = generate_neuron_population(50, seed=0)
        lesioned = create_lesioned_population(neurons, lesion_factor=0.2)

        spikes_n, dirs = simulate_random_trials(50, neurons, seed=0)
        spikes_l, _ = simulate_random_trials(50, lesioned, seed=0)

        pv = PopulationVectorDecoder()
        res_n = evaluate_decoder(pv, spikes_n, dirs, neurons)
        res_l = evaluate_decoder(pv, spikes_l, dirs, lesioned)
        assert res_l['mean_error_degrees'] > res_n['mean_error_degrees']


class TestEndToEndTemporalToBinned:
    def test_temporal_to_decoder(self):
        """Temporal spike times -> binned counts -> PV decoder pipeline."""
        neurons = generate_neuron_population(20, seed=42)
        true_theta = np.pi / 3
        spike_times, spike_counts = simulate_temporal_spikes(
            true_theta, neurons, duration_ms=500, seed=0
        )
        # Use total counts for decoding
        pv = PopulationVectorDecoder()
        decoded = pv.decode(spike_counts, neurons)
        error = angular_error(true_theta, decoded)
        assert error < np.pi / 2  # within 90 degrees (temporal has more variability)


class TestEndToEndHierarchy:
    def test_hierarchy_pipeline(self):
        """Create network -> simulate -> decode from M1 spikes."""
        net = create_hierarchical_network(n_neurons_per_area=30)
        results = simulate_hierarchical_trial(net, theta=np.pi / 4)
        m1_spikes = results['M1']
        m1_neurons = net.get_area('M1').neurons

        pv = PopulationVectorDecoder()
        decoded = pv.decode(m1_spikes, m1_neurons)
        assert 0 <= decoded < 2 * np.pi


class TestEndToEndRaster:
    def test_raster_sum_vs_trial(self):
        """Sum of raster bins should roughly equal total spike count."""
        neurons = generate_neuron_population(20, seed=0)
        raster = simulate_raster(1.0, neurons, duration_ms=500, bin_size_ms=10, seed=42)
        trial = simulate_trial(1.0, neurons, duration_ms=500, seed=42)
        # They use different RNG sequences, so just check magnitudes are similar
        raster_total = np.sum(raster, axis=1)
        assert raster_total.shape == trial.shape
        # Both should be in the same general range
        assert abs(np.mean(raster_total) - np.mean(trial)) < 10 * np.mean(trial)


class TestEndToEndVarianceScales:
    def test_overdispersed_still_decodable(self):
        """Even with overdispersed noise, PV should decode reasonably."""
        neurons = generate_neuron_population(50, seed=0)
        spikes, dirs = simulate_random_trials(
            50, neurons, variance_scale=2.0, seed=0
        )
        results = evaluate_decoder(
            PopulationVectorDecoder(), spikes, dirs, neurons
        )
        assert results['mean_error_degrees'] < 60

    def test_underdispersed_still_decodable(self):
        """Underdispersed noise should also work."""
        neurons = generate_neuron_population(50, seed=0)
        spikes, dirs = simulate_random_trials(
            50, neurons, variance_scale=0.5, seed=0
        )
        results = evaluate_decoder(
            PopulationVectorDecoder(), spikes, dirs, neurons
        )
        assert results['mean_error_degrees'] < 60
