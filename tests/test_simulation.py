"""Tests for simulation.py — population generation, spike simulation, and hierarchy."""

import numpy as np
import pytest
from simulation import (
    NeuronPopulation,
    cosine_tuning,
    generate_neuron_population,
    simulate_trial,
    simulate_raster,
    simulate_multiple_trials,
    simulate_random_trials,
    compute_firing_rate_stats,
    create_lesioned_population,
    TemporalParams,
    simulate_temporal_spikes,
    simulate_continuous_activity,
    spike_times_to_binned,
    BrainArea,
    HierarchicalNetwork,
    create_hierarchical_network,
    simulate_hierarchical_trial,
)


# ── cosine_tuning ───────────────────────────────────────────────────────────

class TestCosineTuning:
    def test_peak_at_preferred(self):
        """Rate highest at preferred direction."""
        mu = np.pi / 4
        peak = cosine_tuning(mu, mu, 5.0, 15.0)
        off = cosine_tuning(mu + np.pi, mu, 5.0, 15.0)
        assert peak > off

    def test_nonnegative(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        rates = cosine_tuning(theta, 0.0, 5.0, 15.0)
        assert np.all(rates >= 0)

    def test_peak_value(self):
        """At preferred direction, rate = r0 + k."""
        assert cosine_tuning(1.0, 1.0, 5.0, 15.0) == pytest.approx(20.0)

    def test_antipreferred_clamped(self):
        """At anti-preferred direction, r0 + k*cos(pi) = r0 - k, clamped to 0."""
        rate = cosine_tuning(0.0, np.pi, 2.0, 10.0)
        assert rate == 0.0  # 2 - 10 = -8, clamped

    def test_broadcast_array_mu(self):
        """cosine_tuning should broadcast theta scalar with array mu."""
        mus = np.array([0.0, np.pi / 2, np.pi])
        rates = cosine_tuning(0.0, mus, 5.0, 15.0)
        assert rates.shape == (3,)
        # First neuron (mu=0) should fire fastest at theta=0
        assert rates[0] > rates[2]

    def test_broadcast_2d(self):
        """theta (n,1) x mu (1,m) should produce (n,m) rates."""
        theta = np.linspace(0, 2 * np.pi, 10)[:, np.newaxis]
        mu = np.linspace(0, 2 * np.pi, 5)[np.newaxis, :]
        rates = cosine_tuning(theta, mu, 5.0, 15.0)
        assert rates.shape == (10, 5)
        assert np.all(rates >= 0)


# ── NeuronPopulation ───────────────────────────────────────────────────────

class TestNeuronPopulation:
    def test_get_tuning_curve(self, small_population):
        theta = np.linspace(0, 2 * np.pi, 50)
        curve = small_population.get_tuning_curve(theta, 0)
        assert curve.shape == (50,)
        assert np.all(curve >= 0)

    def test_get_all_tuning_curves(self, small_population):
        theta = np.linspace(0, 2 * np.pi, 50)
        curves = small_population.get_all_tuning_curves(theta)
        assert curves.shape == (8, 50)

    def test_preferred_directions_uniform(self):
        pop = generate_neuron_population(n_neurons=4, seed=0)
        expected = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        np.testing.assert_allclose(pop.preferred_directions, expected)


# ── generate_neuron_population ─────────────────────────────────────────────

class TestGeneratePopulation:
    def test_size(self):
        pop = generate_neuron_population(n_neurons=20, seed=0)
        assert pop.n_neurons == 20
        assert len(pop.preferred_directions) == 20

    def test_random_preferred(self):
        pop = generate_neuron_population(n_neurons=10, random_preferred=True, seed=0)
        assert pop.n_neurons == 10
        assert np.all(pop.preferred_directions >= 0)
        assert np.all(pop.preferred_directions < 2 * np.pi)

    def test_custom_params(self):
        pop = generate_neuron_population(
            n_neurons=5, baseline_rate=10.0, modulation_depth=20.0
        )
        assert pop.baseline_rate == 10.0
        assert pop.modulation_depth == 20.0

    def test_validation_n_neurons(self):
        with pytest.raises(ValueError, match="n_neurons"):
            generate_neuron_population(n_neurons=0)

    def test_validation_baseline_rate(self):
        with pytest.raises(ValueError, match="baseline_rate"):
            generate_neuron_population(baseline_rate=-1.0)

    def test_validation_modulation_depth(self):
        with pytest.raises(ValueError, match="modulation_depth"):
            generate_neuron_population(modulation_depth=-5.0)

    def test_reproducibility(self):
        p1 = generate_neuron_population(n_neurons=10, random_preferred=True, seed=42)
        p2 = generate_neuron_population(n_neurons=10, random_preferred=True, seed=42)
        np.testing.assert_array_equal(p1.preferred_directions, p2.preferred_directions)


# ── simulate_trial ─────────────────────────────────────────────────────────

class TestSimulateTrial:
    def test_shape(self, small_population):
        spikes = simulate_trial(np.pi / 2, small_population, seed=0)
        assert spikes.shape == (small_population.n_neurons,)
        assert np.all(spikes >= 0)

    def test_reproducibility(self, small_population):
        s1 = simulate_trial(1.0, small_population, seed=42)
        s2 = simulate_trial(1.0, small_population, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_poisson_path(self, small_population):
        """Default variance_scale=1.0 uses Poisson."""
        spikes = simulate_trial(0.0, small_population, duration_ms=500, seed=0)
        assert spikes.dtype in (np.int64, np.int32, np.intp)

    def test_negative_binomial_path(self, small_population):
        """variance_scale > 1.0 uses negative binomial (overdispersed)."""
        spikes = simulate_trial(0.0, small_population, variance_scale=2.0, seed=0)
        assert np.all(spikes >= 0)

    def test_gaussian_path(self, small_population):
        """variance_scale < 1.0 uses clipped Gaussian (underdispersed)."""
        spikes = simulate_trial(0.0, small_population, variance_scale=0.5, seed=0)
        assert np.all(spikes >= 0)

    def test_overdispersion_higher_variance(self, medium_population):
        """Overdispersed spikes should have higher variance than Poisson."""
        rng = np.random.default_rng(42)
        poisson_vars, nb_vars = [], []
        for _ in range(200):
            seed = int(rng.integers(0, 2**31))
            p = simulate_trial(1.0, medium_population, variance_scale=1.0, seed=seed)
            n = simulate_trial(1.0, medium_population, variance_scale=3.0, seed=seed + 1)
            poisson_vars.append(np.var(p))
            nb_vars.append(np.var(n))
        assert np.mean(nb_vars) > np.mean(poisson_vars)

    def test_validation_duration(self, small_population):
        with pytest.raises(ValueError, match="duration_ms"):
            simulate_trial(0.0, small_population, duration_ms=0)

    def test_longer_duration_more_spikes(self, medium_population):
        """Longer trials should produce more spikes on average."""
        short = simulate_trial(0.0, medium_population, duration_ms=100, seed=0)
        long = simulate_trial(0.0, medium_population, duration_ms=2000, seed=0)
        assert np.sum(long) > np.sum(short)


# ── simulate_raster ────────────────────────────────────────────────────────

class TestSimulateRaster:
    def test_shape(self, small_population):
        raster = simulate_raster(0.0, small_population, duration_ms=100, bin_size_ms=10, seed=0)
        assert raster.shape == (8, 10)  # 8 neurons, 10 bins

    def test_nonnegative(self, small_population):
        raster = simulate_raster(0.0, small_population, seed=0)
        assert np.all(raster >= 0)

    def test_integer_counts(self, small_population):
        raster = simulate_raster(0.0, small_population, seed=0)
        assert raster.dtype in (np.int64, np.int32, np.intp)

    def test_overdispersed_path(self, small_population):
        raster = simulate_raster(0.0, small_population, variance_scale=2.0, seed=0)
        assert np.all(raster >= 0)

    def test_underdispersed_path(self, small_population):
        raster = simulate_raster(0.0, small_population, variance_scale=0.5, seed=0)
        assert np.all(raster >= 0)

    def test_validation_duration(self, small_population):
        with pytest.raises(ValueError, match="duration_ms"):
            simulate_raster(0.0, small_population, duration_ms=-1)

    def test_validation_bin_size(self, small_population):
        with pytest.raises(ValueError, match="bin_size_ms"):
            simulate_raster(0.0, small_population, bin_size_ms=0)

    def test_reproducibility(self, small_population):
        r1 = simulate_raster(0.0, small_population, seed=42)
        r2 = simulate_raster(0.0, small_population, seed=42)
        np.testing.assert_array_equal(r1, r2)


# ── simulate_multiple_trials / simulate_random_trials ──────────────────────

class TestMultipleTrials:
    def test_simulate_multiple_trials_shape(self, small_population):
        thetas = np.array([0.0, np.pi / 2, np.pi])
        spikes, dirs = simulate_multiple_trials(thetas, small_population, seed=0)
        assert spikes.shape == (3, 8)
        np.testing.assert_array_equal(dirs, thetas)

    def test_simulate_random_trials_shape(self, medium_population):
        spikes, dirs = simulate_random_trials(20, medium_population, seed=0)
        assert spikes.shape == (20, 50)
        assert dirs.shape == (20,)
        assert np.all(dirs >= 0)
        assert np.all(dirs < 2 * np.pi)

    def test_simulate_random_trials_reproducibility(self, small_population):
        s1, d1 = simulate_random_trials(10, small_population, seed=42)
        s2, d2 = simulate_random_trials(10, small_population, seed=42)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(d1, d2)


# ── compute_firing_rate_stats ──────────────────────────────────────────────

class TestFiringRateStats:
    def test_basic(self, medium_population, multi_trial_data):
        spikes, _ = multi_trial_data
        stats = compute_firing_rate_stats(spikes, duration_ms=500)
        assert 'mean_rate' in stats
        assert 'std_rate' in stats
        assert 'min_rate' in stats
        assert 'max_rate' in stats
        assert stats['mean_rate'] > 0
        assert stats['min_rate'] >= 0
        assert stats['max_rate'] >= stats['mean_rate']

    def test_per_neuron_shape(self, medium_population, multi_trial_data):
        spikes, _ = multi_trial_data
        stats = compute_firing_rate_stats(spikes, duration_ms=500)
        assert stats['mean_per_neuron'].shape == (50,)
        assert stats['std_per_neuron'].shape == (50,)


# ── create_lesioned_population ─────────────────────────────────────────────

class TestLesionedPopulation:
    def test_modulation_lesion(self, medium_population):
        lesioned = create_lesioned_population(medium_population, lesion_factor=0.5)
        assert lesioned.modulation_depth == medium_population.modulation_depth * 0.5
        assert lesioned.baseline_rate == medium_population.baseline_rate

    def test_baseline_lesion(self, medium_population):
        lesioned = create_lesioned_population(
            medium_population, lesion_factor=0.3, lesion_type='baseline'
        )
        assert lesioned.baseline_rate == pytest.approx(medium_population.baseline_rate * 0.3)
        assert lesioned.modulation_depth == medium_population.modulation_depth

    def test_unknown_lesion_type(self, small_population):
        with pytest.raises(ValueError, match="Unknown lesion"):
            create_lesioned_population(small_population, lesion_type='unknown')

    def test_preserves_preferred_directions(self, medium_population):
        lesioned = create_lesioned_population(medium_population)
        np.testing.assert_array_equal(
            lesioned.preferred_directions, medium_population.preferred_directions
        )

    def test_independent_copy(self, medium_population):
        """Modifying lesioned pop should not affect original."""
        lesioned = create_lesioned_population(medium_population)
        lesioned.preferred_directions[0] = 999.0
        assert medium_population.preferred_directions[0] != 999.0


# ── simulate_temporal_spikes ───────────────────────────────────────────────

class TestTemporalSpikes:
    def test_basic_shape(self, small_population):
        spike_times, spike_counts = simulate_temporal_spikes(
            0.0, small_population, duration_ms=100, seed=0
        )
        assert len(spike_times) == small_population.n_neurons
        assert spike_counts.shape == (small_population.n_neurons,)

    def test_spike_times_within_range(self, small_population):
        spike_times, _ = simulate_temporal_spikes(
            0.0, small_population, duration_ms=200, dt_ms=1.0, seed=0
        )
        for times in spike_times:
            for t in times:
                assert 0 <= t < 200

    def test_spike_counts_match_times(self, small_population):
        spike_times, spike_counts = simulate_temporal_spikes(
            0.0, small_population, duration_ms=100, seed=0
        )
        for i, times in enumerate(spike_times):
            assert len(times) == spike_counts[i]

    def test_nonneg_counts(self, small_population):
        _, counts = simulate_temporal_spikes(0.0, small_population, seed=0)
        assert np.all(counts >= 0)

    def test_validation_duration(self, small_population):
        with pytest.raises(ValueError, match="duration_ms"):
            simulate_temporal_spikes(0.0, small_population, duration_ms=0)

    def test_validation_dt(self, small_population):
        with pytest.raises(ValueError, match="dt_ms"):
            simulate_temporal_spikes(0.0, small_population, dt_ms=0)

    def test_adaptation_reduces_firing(self, medium_population):
        """With adaptation, neurons should fire less than without."""
        params_adapt = TemporalParams(adaptation_strength=0.8, burst_probability=0.0)
        params_none = TemporalParams(adaptation_strength=0.0, burst_probability=0.0)
        _, counts_adapt = simulate_temporal_spikes(
            0.0, medium_population, duration_ms=500, temporal_params=params_adapt, seed=0
        )
        _, counts_none = simulate_temporal_spikes(
            0.0, medium_population, duration_ms=500, temporal_params=params_none, seed=0
        )
        assert np.sum(counts_adapt) <= np.sum(counts_none)

    def test_reproducibility(self, small_population):
        t1, c1 = simulate_temporal_spikes(0.0, small_population, seed=42)
        t2, c2 = simulate_temporal_spikes(0.0, small_population, seed=42)
        np.testing.assert_array_equal(c1, c2)
        for a, b in zip(t1, t2):
            assert a == b


# ── simulate_continuous_activity ───────────────────────────────────────────

class TestContinuousActivity:
    def test_basic_shape(self, small_population):
        theta_func = lambda t: 0.0
        time_arr, spike_times, rates = simulate_continuous_activity(
            theta_func, small_population, duration_ms=100, seed=0
        )
        assert len(time_arr) == 100  # 100ms / 1ms dt
        assert len(spike_times) == small_population.n_neurons
        assert rates.shape == (small_population.n_neurons, len(time_arr))

    def test_rates_nonnegative(self, small_population):
        theta_func = lambda t: np.sin(t / 100)
        _, _, rates = simulate_continuous_activity(
            theta_func, small_population, duration_ms=200, seed=0
        )
        assert np.all(rates >= 0)

    def test_time_varying_direction(self, small_population):
        """Spike times list should have entries for active neurons."""
        theta_func = lambda t: (t / 1000) * 2 * np.pi
        _, spike_times, _ = simulate_continuous_activity(
            theta_func, small_population, duration_ms=500, seed=0
        )
        total_spikes = sum(len(st) for st in spike_times)
        assert total_spikes > 0


# ── spike_times_to_binned ──────────────────────────────────────────────────

class TestSpikeTimesToBinned:
    def test_basic(self):
        spike_times = [[5.0, 15.0, 25.0], [50.0]]
        binned = spike_times_to_binned(spike_times, duration_ms=100, bin_size_ms=10)
        assert binned.shape == (2, 10)
        assert binned[0, 0] == 1  # t=5 -> bin 0
        assert binned[0, 1] == 1  # t=15 -> bin 1
        assert binned[0, 2] == 1  # t=25 -> bin 2
        assert binned[1, 5] == 1  # t=50 -> bin 5

    def test_empty(self):
        spike_times = [[], []]
        binned = spike_times_to_binned(spike_times, duration_ms=100, bin_size_ms=10)
        assert binned.shape == (2, 10)
        assert np.all(binned == 0)

    def test_multiple_spikes_per_bin(self):
        spike_times = [[1.0, 2.0, 3.0]]  # all in bin 0
        binned = spike_times_to_binned(spike_times, duration_ms=50, bin_size_ms=10)
        assert binned[0, 0] == 3

    def test_consistency_with_temporal(self, small_population):
        """Binned counts from temporal sim should match spike_counts."""
        spike_times, spike_counts = simulate_temporal_spikes(
            0.0, small_population, duration_ms=200, seed=0
        )
        binned = spike_times_to_binned(spike_times, duration_ms=200, bin_size_ms=200)
        # Single bin should equal total counts
        np.testing.assert_array_equal(binned[:, 0], spike_counts)


# ── HierarchicalNetwork ───────────────────────────────────────────────────

class TestHierarchicalNetwork:
    def test_creation(self):
        net = HierarchicalNetwork(n_neurons_per_area=20, seed=42)
        assert len(net.get_area_names()) == 4
        assert set(net.get_area_names()) == {'M1', 'PMd', 'PPC', 'SMA'}

    def test_get_area(self):
        net = HierarchicalNetwork(seed=0)
        m1 = net.get_area('M1')
        assert isinstance(m1, BrainArea)
        assert m1.name == 'M1'

    def test_get_area_missing(self):
        net = HierarchicalNetwork(seed=0)
        assert net.get_area('V1') is None

    def test_simulate_area(self):
        net = HierarchicalNetwork(n_neurons_per_area=10, seed=0)
        spikes = net.simulate_area('M1', theta=0.0)
        assert spikes.shape == (10,)
        assert np.all(spikes >= 0)

    def test_simulate_hierarchy_keys(self):
        net = HierarchicalNetwork(n_neurons_per_area=10, seed=0)
        results = net.simulate_hierarchy(theta=np.pi / 4)
        assert set(results.keys()) == {'M1', 'PMd', 'PPC', 'SMA'}

    def test_simulate_hierarchy_no_dynamics(self):
        net = HierarchicalNetwork(n_neurons_per_area=10, seed=0)
        results = net.simulate_hierarchy(theta=0.0, include_dynamics=False)
        for name, spikes in results.items():
            assert spikes.shape == (10,)
            assert np.all(spikes >= 0)

    def test_simulate_hierarchy_with_dynamics(self):
        net = HierarchicalNetwork(n_neurons_per_area=10, seed=0)
        results = net.simulate_hierarchy(theta=0.0, include_dynamics=True)
        for name, spikes in results.items():
            assert spikes.shape == (10,)

    def test_connectivity_matrix(self):
        net = HierarchicalNetwork(n_neurons_per_area=10, seed=0)
        matrix, names = net.get_connectivity_matrix()
        assert matrix.shape == (4, 4)
        assert len(names) == 4
        # There should be some non-zero connections
        assert np.sum(matrix > 0) > 0

    def test_hierarchy_order(self):
        net = HierarchicalNetwork(seed=0)
        order = net.get_hierarchy_order()
        assert len(order) == 4
        # PPC has highest delay (50ms), should be first
        assert order[0] == 'PPC'
        # M1 has 0 delay, should be last
        assert order[-1] == 'M1'

    def test_reproducibility(self):
        net1 = HierarchicalNetwork(n_neurons_per_area=10, seed=42)
        net2 = HierarchicalNetwork(n_neurons_per_area=10, seed=42)
        m1, _ = net1.get_connectivity_matrix()
        m2, _ = net2.get_connectivity_matrix()
        np.testing.assert_array_equal(m1, m2)


# ── Factory / convenience functions ────────────────────────────────────────

class TestFactoryFunctions:
    def test_create_hierarchical_network(self):
        net = create_hierarchical_network(n_neurons_per_area=10)
        assert isinstance(net, HierarchicalNetwork)
        assert len(net.get_area_names()) == 4

    def test_simulate_hierarchical_trial(self):
        net = create_hierarchical_network(n_neurons_per_area=10)
        results = simulate_hierarchical_trial(net, theta=1.0)
        assert isinstance(results, dict)
        assert 'M1' in results
