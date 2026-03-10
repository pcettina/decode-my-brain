"""Tests for decoders.py — all decoder types, evaluation, and comparison."""

import numpy as np
import pytest
from simulation import simulate_trial, simulate_random_trials, generate_neuron_population
from decoders import (
    _compute_poisson_log_likelihoods,
    _validate_decode_inputs,
    PopulationVectorDecoder,
    MaximumLikelihoodDecoder,
    NaiveBayesDecoder,
    KalmanFilterDecoder,
    get_decoder,
    evaluate_decoder,
    compare_decoders,
)
from utils import angular_error


# ── _validate_decode_inputs ────────────────────────────────────────────────

class TestValidation:
    def test_empty_spike_counts(self, small_population):
        with pytest.raises(ValueError, match="empty"):
            _validate_decode_inputs(np.array([]), small_population)

    def test_length_mismatch(self, small_population):
        with pytest.raises(ValueError, match="does not match"):
            _validate_decode_inputs(np.zeros(3), small_population)

    def test_valid_input(self, small_population):
        _validate_decode_inputs(np.zeros(small_population.n_neurons), small_population)


# ── _compute_poisson_log_likelihoods ───────────────────────────────────────

class TestLogLikelihoods:
    def test_shape(self, medium_population, single_trial_spikes):
        theta_grid = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        ll = _compute_poisson_log_likelihoods(
            theta_grid, single_trial_spikes, medium_population, 0.5
        )
        assert ll.shape == (36,)

    def test_peak_near_true(self, medium_population):
        """Log-likelihood should peak near the true direction."""
        true_theta = np.pi / 3
        spikes = simulate_trial(true_theta, medium_population, duration_ms=1000, seed=0)
        theta_grid = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        ll = _compute_poisson_log_likelihoods(theta_grid, spikes, medium_population, 1.0)
        best_theta = theta_grid[np.argmax(ll)]
        assert angular_error(true_theta, best_theta) < np.pi / 4

    def test_finite(self, medium_population, single_trial_spikes):
        theta_grid = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        ll = _compute_poisson_log_likelihoods(
            theta_grid, single_trial_spikes, medium_population, 0.5
        )
        assert np.all(np.isfinite(ll))


# ── PopulationVectorDecoder ────────────────────────────────────────────────

class TestPVDecoder:
    def test_known_direction(self, medium_population):
        true_theta = np.pi / 2
        spikes = simulate_trial(true_theta, medium_population, duration_ms=1000, seed=0)
        pv = PopulationVectorDecoder()
        decoded = pv.decode(spikes, medium_population)
        assert angular_error(true_theta, decoded) < np.pi / 4

    def test_zero_spikes(self, small_population):
        pv = PopulationVectorDecoder()
        result = pv.decode(np.zeros(small_population.n_neurons, dtype=int), small_population)
        assert result == 0.0

    def test_name(self):
        assert PopulationVectorDecoder().name == "Population Vector"

    def test_output_range(self, medium_population):
        """Decoded direction should be in [0, 2pi)."""
        pv = PopulationVectorDecoder()
        for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            spikes = simulate_trial(theta, medium_population, duration_ms=500, seed=0)
            decoded = pv.decode(spikes, medium_population)
            assert 0 <= decoded < 2 * np.pi

    def test_validation(self, small_population):
        pv = PopulationVectorDecoder()
        with pytest.raises(ValueError):
            pv.decode(np.array([]), small_population)


# ── MaximumLikelihoodDecoder ───────────────────────────────────────────────

class TestMLDecoder:
    def test_known_direction(self, medium_population):
        true_theta = np.pi
        spikes = simulate_trial(true_theta, medium_population, duration_ms=1000, seed=1)
        ml = MaximumLikelihoodDecoder()
        decoded = ml.decode(spikes, medium_population, duration_s=1.0)
        assert angular_error(true_theta, decoded) < np.pi / 6

    def test_name(self):
        assert MaximumLikelihoodDecoder().name == "Maximum Likelihood"

    def test_likelihood_curve(self, medium_population, single_trial_spikes):
        ml = MaximumLikelihoodDecoder()
        theta_grid, likelihoods = ml.get_likelihood_curve(
            single_trial_spikes, medium_population
        )
        assert len(theta_grid) == 360
        assert len(likelihoods) == 360
        assert np.all(likelihoods >= 0)
        assert abs(np.sum(likelihoods) - 1.0) < 1e-6  # normalized

    def test_likelihood_peak_near_true(self, medium_population):
        true_theta = 2.0
        spikes = simulate_trial(true_theta, medium_population, duration_ms=1000, seed=5)
        ml = MaximumLikelihoodDecoder()
        theta_grid, likelihoods = ml.get_likelihood_curve(
            spikes, medium_population, duration_s=1.0
        )
        peak_theta = theta_grid[np.argmax(likelihoods)]
        assert angular_error(true_theta, peak_theta) < np.pi / 4

    def test_custom_grid(self, medium_population, single_trial_spikes):
        ml = MaximumLikelihoodDecoder(n_grid_points=36)
        decoded = ml.decode(single_trial_spikes, medium_population)
        assert 0 <= decoded < 2 * np.pi

    def test_ml_better_than_pv_large_pop(self, large_population):
        """ML should be at least as accurate as PV with many neurons."""
        spikes, dirs = simulate_random_trials(100, large_population, seed=10)
        pv_res = evaluate_decoder(PopulationVectorDecoder(), spikes, dirs, large_population)
        ml_res = evaluate_decoder(MaximumLikelihoodDecoder(), spikes, dirs, large_population)
        # ML should generally be better or comparable
        assert ml_res['mean_error_degrees'] < pv_res['mean_error_degrees'] + 10


# ── NaiveBayesDecoder ──────────────────────────────────────────────────────

class TestNaiveBayes:
    def test_uniform_prior_matches_ml(self, medium_population, single_trial_spikes):
        """With uniform prior, NB should match ML."""
        ml = MaximumLikelihoodDecoder()
        nb = NaiveBayesDecoder()
        ml_result = ml.decode(single_trial_spikes, medium_population)
        nb_result = nb.decode(single_trial_spikes, medium_population)
        assert angular_error(ml_result, nb_result) < 0.02  # nearly identical

    def test_nonuniform_prior(self, medium_population, single_trial_spikes):
        """Non-uniform prior should shift the result."""
        nb_uniform = NaiveBayesDecoder()
        # Strong prior favoring theta=0
        prior = np.zeros(360)
        prior[0] = 1.0  # All mass at theta=0
        prior = prior / prior.sum()
        nb_biased = NaiveBayesDecoder(prior=prior)

        result_uniform = nb_uniform.decode(single_trial_spikes, medium_population)
        result_biased = nb_biased.decode(single_trial_spikes, medium_population)
        # Biased result should be closer to 0 than uniform
        assert angular_error(result_biased, 0.0) <= angular_error(result_uniform, 0.0) + 0.1

    def test_name(self):
        assert NaiveBayesDecoder().name == "Naive Bayes (MAP)"


# ── KalmanFilterDecoder ────────────────────────────────────────────────────

class TestKalmanDecoder:
    def test_not_fitted(self):
        kf = KalmanFilterDecoder(n_neurons=10)
        with pytest.raises(RuntimeError, match="not fitted"):
            kf.decode_step(np.zeros(10))

    def test_name(self):
        assert KalmanFilterDecoder(n_neurons=10).name == "Kalman Filter"

    def test_fit_from_neurons(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        assert kf.is_fitted
        assert kf.H.shape == (50, 4)

    def test_decode_step_after_fit(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        spikes = simulate_trial(0.0, medium_population, seed=0)
        bin_counts = spikes.astype(float) * (kf.dt / 0.5)
        state = kf.decode_step(bin_counts)
        assert state.shape == (4,)
        assert np.all(np.isfinite(state))

    def test_decode_interface(self, medium_population):
        """KalmanFilterDecoder.decode() should auto-fit and return direction."""
        kf = KalmanFilterDecoder(n_neurons=50)
        spikes = simulate_trial(np.pi / 2, medium_population, duration_ms=500, seed=0)
        result = kf.decode(spikes, medium_population, duration_s=0.5)
        assert 0 <= result < 2 * np.pi

    def test_reset(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        spikes = simulate_trial(0.0, medium_population, seed=0)
        kf.decode(spikes, medium_population)
        # State should be non-zero after decode
        assert np.sum(np.abs(kf.x)) > 0
        kf.reset()
        np.testing.assert_array_equal(kf.x, np.zeros(4))

    def test_reset_with_initial_state(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        init = np.array([1.0, 2.0, 0.5, -0.5])
        kf.reset(initial_state=init)
        np.testing.assert_array_equal(kf.x, init)

    def test_predict(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        x_pred, P_pred = kf.predict()
        assert x_pred.shape == (4,)
        assert P_pred.shape == (4, 4)
        # P_pred should be symmetric positive definite
        assert np.allclose(P_pred, P_pred.T)

    def test_joseph_form_symmetry(self, medium_population):
        """Covariance should remain symmetric after update."""
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        for _ in range(20):
            spikes = simulate_trial(0.0, medium_population, seed=None)
            bin_counts = spikes.astype(float) * (kf.dt / 0.5)
            kf.decode_step(bin_counts)
            assert np.allclose(kf.P, kf.P.T, atol=1e-10), "Covariance became asymmetric"

    def test_decode_trajectory(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        # Create a short sequence of spike observations
        rng = np.random.default_rng(42)
        sequence = np.zeros((5, 50))
        for i in range(5):
            sequence[i] = simulate_trial(0.0, medium_population, seed=int(rng.integers(2**31)))
        states, covs = kf.decode_trajectory(sequence, medium_population)
        assert states.shape == (5, 4)
        assert covs.shape == (5, 4, 4)

    def test_get_uncertainty_ellipse(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        semi_major, semi_minor, angle = kf.get_uncertainty_ellipse(confidence=0.95)
        assert semi_major > 0
        assert semi_minor > 0
        assert semi_major >= semi_minor

    def test_get_state(self, medium_population):
        kf = KalmanFilterDecoder(n_neurons=50)
        kf.fit_from_neurons(medium_population)
        state = kf.get_state()
        assert 'pos_x' in state
        assert 'vel_x' in state
        assert 'speed' in state
        assert 'direction' in state
        assert 'position_uncertainty' in state
        assert state['speed'] >= 0
        assert state['position_uncertainty'] >= 0

    def test_fit_with_training_data(self, medium_population):
        """fit() with spike_data and kinematics should set H and R."""
        kf = KalmanFilterDecoder(n_neurons=50)
        rng = np.random.default_rng(0)
        spike_data = rng.poisson(5, (20, 50))
        kinematics = rng.standard_normal((20, 4))
        kf.fit(spike_data, kinematics, medium_population)
        assert kf.is_fitted
        assert kf.H.shape == (50, 4)
        assert kf.R.shape == (50, 50)


# ── get_decoder factory ────────────────────────────────────────────────────

class TestGetDecoder:
    def test_population_vector(self):
        d = get_decoder('population_vector')
        assert isinstance(d, PopulationVectorDecoder)

    def test_ml(self):
        d = get_decoder('ml')
        assert isinstance(d, MaximumLikelihoodDecoder)

    def test_maximum_likelihood(self):
        d = get_decoder('maximum_likelihood')
        assert isinstance(d, MaximumLikelihoodDecoder)

    def test_naive_bayes(self):
        d = get_decoder('naive_bayes')
        assert isinstance(d, NaiveBayesDecoder)

    def test_map(self):
        d = get_decoder('map')
        assert isinstance(d, NaiveBayesDecoder)

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown decoder"):
            get_decoder('unknown')

    def test_case_insensitive(self):
        d = get_decoder('ML')
        assert isinstance(d, MaximumLikelihoodDecoder)


# ── evaluate_decoder ───────────────────────────────────────────────────────

class TestEvaluateDecoder:
    def test_result_keys(self, medium_population, multi_trial_data):
        spikes, dirs = multi_trial_data
        results = evaluate_decoder(
            PopulationVectorDecoder(), spikes, dirs, medium_population
        )
        assert 'decoded_directions' in results
        assert 'errors' in results
        assert 'mean_error' in results
        assert 'std_error' in results
        assert 'median_error' in results
        assert 'mean_error_degrees' in results

    def test_error_range(self, medium_population, multi_trial_data):
        spikes, dirs = multi_trial_data
        results = evaluate_decoder(
            PopulationVectorDecoder(), spikes, dirs, medium_population
        )
        assert 0 < results['mean_error_degrees'] < 90

    def test_error_arrays_shape(self, medium_population, multi_trial_data):
        spikes, dirs = multi_trial_data
        results = evaluate_decoder(
            PopulationVectorDecoder(), spikes, dirs, medium_population
        )
        assert results['errors'].shape == (len(dirs),)
        assert results['decoded_directions'].shape == (len(dirs),)


# ── compare_decoders ───────────────────────────────────────────────────────

class TestCompareDecoders:
    def test_all_decoders_present(self, medium_population):
        spikes, dirs = simulate_random_trials(20, medium_population, seed=0)
        results = compare_decoders(spikes, dirs, medium_population)
        assert 'Population Vector' in results
        assert 'Maximum Likelihood' in results
        assert 'Kalman Filter' in results

    def test_result_structure(self, medium_population):
        spikes, dirs = simulate_random_trials(10, medium_population, seed=0)
        results = compare_decoders(spikes, dirs, medium_population)
        for name, data in results.items():
            assert 'decoded' in data
            assert 'errors' in data
            assert 'mean_error_deg' in data
            assert data['mean_error_deg'] >= 0
