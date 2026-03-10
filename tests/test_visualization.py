"""Tests for visualization.py — smoke tests for all plot functions.

These tests verify that each function:
1. Returns a valid Plotly Figure object
2. Contains at least one trace (where applicable)
3. Handles typical inputs without errors
4. Handles edge cases (empty data, single neuron) gracefully
"""

import numpy as np
import pytest
import plotly.graph_objects as go
from simulation import (
    generate_neuron_population,
    simulate_trial,
    simulate_random_trials,
    simulate_temporal_spikes,
    simulate_raster,
    HierarchicalNetwork,
)
from decoders import PopulationVectorDecoder, MaximumLikelihoodDecoder


@pytest.fixture
def viz_population():
    """8-neuron pop for fast viz tests."""
    return generate_neuron_population(n_neurons=8, seed=42)


@pytest.fixture
def viz_spikes(viz_population):
    return simulate_trial(np.pi / 4, viz_population, seed=0)


@pytest.fixture
def viz_multi(viz_population):
    return simulate_random_trials(10, viz_population, seed=0)


def _import_viz():
    """Lazy import to avoid Streamlit init at module level."""
    import visualization as viz
    return viz


# ── plot_tuning_curves ─────────────────────────────────────────────────────

class TestPlotTuningCurves:
    def test_returns_figure(self, viz_population):
        viz = _import_viz()
        fig = viz.plot_tuning_curves(viz_population)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_with_highlight(self, viz_population):
        viz = _import_viz()
        fig = viz.plot_tuning_curves(viz_population, highlight_theta=np.pi / 2)
        assert isinstance(fig, go.Figure)

    def test_no_highlight(self, viz_population):
        viz = _import_viz()
        fig = viz.plot_tuning_curves(viz_population, show_highlight=False)
        assert isinstance(fig, go.Figure)


# ── plot_raster_heatmap ────────────────────────────────────────────────────

class TestPlotRasterHeatmap:
    def test_returns_figure_1d(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.plot_raster_heatmap(viz_spikes, viz_population)
        assert isinstance(fig, go.Figure)

    def test_returns_figure_2d(self, viz_population):
        viz = _import_viz()
        raster = simulate_raster(0.0, viz_population, duration_ms=100, bin_size_ms=10, seed=0)
        fig = viz.plot_raster_heatmap(raster, viz_population)
        assert isinstance(fig, go.Figure)

    def test_with_true_theta(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.plot_raster_heatmap(viz_spikes, viz_population, true_theta=np.pi / 4)
        assert isinstance(fig, go.Figure)


# ── plot_population_bar ────────────────────────────────────────────────────

class TestPlotPopulationBar:
    def test_returns_figure(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.plot_population_bar(viz_spikes, viz_population)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_zero_spikes(self, viz_population):
        viz = _import_viz()
        fig = viz.plot_population_bar(
            np.zeros(viz_population.n_neurons, dtype=int), viz_population
        )
        assert isinstance(fig, go.Figure)


# ── plot_polar_comparison ──────────────────────────────────────────────────

class TestPlotPolarComparison:
    def test_returns_figure(self):
        viz = _import_viz()
        fig = viz.plot_polar_comparison(np.pi / 4, user_theta=np.pi / 3, model_theta=np.pi / 2)
        assert isinstance(fig, go.Figure)

    def test_true_only(self):
        viz = _import_viz()
        fig = viz.plot_polar_comparison(np.pi / 4)
        assert isinstance(fig, go.Figure)


# ── plot_decoder_performance_vs_noise ──────────────────────────────────────

class TestPlotDecoderPerformanceVsNoise:
    def test_returns_figure(self):
        viz = _import_viz()
        scales = np.array([0.5, 1.0, 2.0, 3.0])
        errors = np.array([20.0, 25.0, 35.0, 45.0])
        fig = viz.plot_decoder_performance_vs_noise(scales, errors)
        assert isinstance(fig, go.Figure)

    def test_with_std(self):
        viz = _import_viz()
        scales = np.array([0.5, 1.0, 2.0])
        errors = np.array([20.0, 25.0, 35.0])
        stds = np.array([5.0, 8.0, 12.0])
        fig = viz.plot_decoder_performance_vs_noise(scales, errors, std_errors=stds)
        assert isinstance(fig, go.Figure)


# ── plot_condition_comparison ──────────────────────────────────────────────

class TestPlotConditionComparison:
    def test_returns_figure(self):
        viz = _import_viz()
        normal = np.array([15.0, 20.0, 18.0, 22.0])
        lesioned = np.array([30.0, 35.0, 40.0, 28.0])
        fig = viz.plot_condition_comparison(normal, lesioned)
        assert isinstance(fig, go.Figure)


# ── plot_likelihood_curve ──────────────────────────────────────────────────

class TestPlotLikelihoodCurve:
    def test_returns_figure(self):
        viz = _import_viz()
        theta = np.linspace(0, 2 * np.pi, 360)
        likelihoods = np.sin(theta) ** 2
        likelihoods /= likelihoods.sum()
        fig = viz.plot_likelihood_curve(theta, likelihoods, true_theta=np.pi / 2)
        assert isinstance(fig, go.Figure)

    def test_with_decoded(self):
        viz = _import_viz()
        theta = np.linspace(0, 2 * np.pi, 360)
        likelihoods = np.sin(theta) ** 2
        likelihoods /= likelihoods.sum()
        fig = viz.plot_likelihood_curve(theta, likelihoods, true_theta=1.0, decoded_theta=1.1)
        assert isinstance(fig, go.Figure)


# ── create_scoreboard_table ────────────────────────────────────────────────

class TestCreateScoreboardTable:
    def test_returns_figure(self):
        viz = _import_viz()
        rounds = [
            {'true_deg': 90.0, 'user_deg': 85.0, 'model_deg': 88.0,
             'user_error': 5.0, 'model_error': 2.0, 'winner': 'Model'},
        ]
        fig = viz.create_scoreboard_table(rounds)
        assert isinstance(fig, go.Figure)

    def test_empty(self):
        viz = _import_viz()
        fig = viz.create_scoreboard_table([])
        assert isinstance(fig, go.Figure)


# ── create_spike_raster_snapshot ───────────────────────────────────────────

class TestCreateSpikeRasterSnapshot:
    def test_returns_figure(self, viz_population):
        viz = _import_viz()
        spike_times, _ = simulate_temporal_spikes(
            0.0, viz_population, duration_ms=200, seed=0
        )
        fig = viz.create_spike_raster_snapshot(
            spike_times, viz_population, current_time_ms=200.0
        )
        assert isinstance(fig, go.Figure)

    def test_with_true_theta(self, viz_population):
        viz = _import_viz()
        spike_times, _ = simulate_temporal_spikes(
            np.pi / 2, viz_population, duration_ms=200, seed=0
        )
        fig = viz.create_spike_raster_snapshot(
            spike_times, viz_population, current_time_ms=200.0, true_theta=np.pi / 2
        )
        assert isinstance(fig, go.Figure)


# ── create_bci_canvas ──────────────────────────────────────────────────────

class TestCreateBCICanvas:
    def test_returns_figure(self):
        viz = _import_viz()
        fig = viz.create_bci_canvas(
            cursor_pos=(100.0, 100.0), target_pos=(50.0, 50.0)
        )
        assert isinstance(fig, go.Figure)


# ── create_bci_metrics_display ─────────────────────────────────────────────

class TestCreateBCIMetrics:
    def test_returns_figure(self):
        viz = _import_viz()
        fig = viz.create_bci_metrics_display(
            time_elapsed=10.5,
            distance_to_target=25.0,
            path_length=150.0,
            n_targets_hit=3,
            n_attempts=5,
        )
        assert isinstance(fig, go.Figure)


# ── create_pv_decoder_step ─────────────────────────────────────────────────

class TestCreatePVDecoderStep:
    def test_returns_figure(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.create_pv_decoder_step(viz_spikes, viz_population, step=1)
        assert isinstance(fig, go.Figure)

    def test_with_true_theta(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.create_pv_decoder_step(
            viz_spikes, viz_population, step=3, true_theta=np.pi / 4
        )
        assert isinstance(fig, go.Figure)


# ── create_ml_decoder_step ─────────────────────────────────────────────────

class TestCreateMLDecoderStep:
    def test_returns_figure(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.create_ml_decoder_step(
            viz_spikes, viz_population, step=1
        )
        assert isinstance(fig, go.Figure)

    def test_with_true_theta(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.create_ml_decoder_step(
            viz_spikes, viz_population, step=2, true_theta=np.pi / 4
        )
        assert isinstance(fig, go.Figure)


# ── create_vector_animation_polar ──────────────────────────────────────────

class TestCreateVectorAnimationPolar:
    def test_returns_figure(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.create_vector_animation_polar(
            viz_spikes, viz_population, n_vectors_shown=4
        )
        assert isinstance(fig, go.Figure)

    def test_with_true_theta(self, viz_population, viz_spikes):
        viz = _import_viz()
        fig = viz.create_vector_animation_polar(
            viz_spikes, viz_population, n_vectors_shown=8, true_theta=np.pi / 4
        )
        assert isinstance(fig, go.Figure)


# ── compute_neural_manifold ────────────────────────────────────────────────

class TestComputeNeuralManifold:
    def test_basic(self, viz_multi):
        viz = _import_viz()
        spikes, _ = viz_multi
        pca_data, pca_model, explained_var = viz.compute_neural_manifold(spikes, n_components=3)
        assert pca_data.shape == (10, 3)
        assert hasattr(pca_model, 'explained_variance_ratio_')
        assert len(explained_var) == 3

    def test_2_components(self, viz_multi):
        viz = _import_viz()
        spikes, _ = viz_multi
        pca_data, _, _ = viz.compute_neural_manifold(spikes, n_components=2)
        assert pca_data.shape == (10, 2)


# ── plot_neural_manifold_3d / 2d ───────────────────────────────────────────

class TestPlotNeuralManifold:
    def test_3d(self, viz_multi):
        viz = _import_viz()
        spikes, dirs = viz_multi
        pca_data, _, _ = viz.compute_neural_manifold(spikes, n_components=3)
        fig = viz.plot_neural_manifold_3d(pca_data, dirs)
        assert isinstance(fig, go.Figure)

    def test_2d(self, viz_multi):
        viz = _import_viz()
        spikes, dirs = viz_multi
        pca_data, _, _ = viz.compute_neural_manifold(spikes, n_components=3)
        fig = viz.plot_neural_manifold_2d(pca_data, dirs)
        assert isinstance(fig, go.Figure)


# ── plot_variance_explained ────────────────────────────────────────────────

class TestPlotVarianceExplained:
    def test_returns_figure(self, viz_multi):
        viz = _import_viz()
        spikes, _ = viz_multi
        _, _, explained_var = viz.compute_neural_manifold(spikes, n_components=3)
        fig = viz.plot_variance_explained(explained_var)
        assert isinstance(fig, go.Figure)


# ── plot_brain_connectivity ────────────────────────────────────────────────

class TestPlotBrainConnectivity:
    def test_returns_figure(self):
        viz = _import_viz()
        net = HierarchicalNetwork(n_neurons_per_area=5, seed=0)
        matrix, names = net.get_connectivity_matrix()
        fig = viz.plot_brain_connectivity(matrix, names)
        assert isinstance(fig, go.Figure)


# ── plot_area_comparison ───────────────────────────────────────────────────

class TestPlotAreaComparison:
    def test_returns_figure(self):
        viz = _import_viz()
        net = HierarchicalNetwork(n_neurons_per_area=5, seed=0)
        area_data = {}
        neurons_dict = {}
        for name in net.get_area_names():
            area = net.get_area(name)
            area_data[name] = np.random.default_rng(0).poisson(5, 5)
            neurons_dict[name] = area.neurons
        fig = viz.plot_area_comparison(area_data, neurons_dict, true_theta=0.0)
        assert isinstance(fig, go.Figure)


# ── plot_leaderboard ───────────────────────────────────────────────────────

class TestPlotLeaderboard:
    def test_returns_figure(self):
        viz = _import_viz()
        data = [
            {'rank': 1, 'name': 'Alice', 'score': 100, 'trials': 10, 'date': '03/09'},
        ]
        fig = viz.plot_leaderboard(data)
        assert isinstance(fig, go.Figure)

    def test_empty(self):
        viz = _import_viz()
        fig = viz.plot_leaderboard([])
        assert isinstance(fig, go.Figure)


# ── get_direction_color ────────────────────────────────────────────────────

class TestGetDirectionColor:
    def test_returns_hsl(self):
        viz = _import_viz()
        color = viz.get_direction_color(0.0)
        assert 'hsl' in color

    def test_different_directions(self):
        viz = _import_viz()
        c1 = viz.get_direction_color(0.0)
        c2 = viz.get_direction_color(np.pi)
        assert c1 != c2


# ── Color constants ────────────────────────────────────────────────────────

def test_color_constants():
    viz = _import_viz()
    assert viz.TRUE_COLOR == '#3498db'
    assert viz.USER_COLOR == '#2ecc71'
    assert viz.MODEL_COLOR == '#e67e22'
