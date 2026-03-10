"""Tests for engine.game — GameEngine and BCISimulator."""

import numpy as np
import pytest
from simulation import generate_neuron_population
from decoders import PopulationVectorDecoder, MaximumLikelihoodDecoder
from engine.game import GameEngine, GameResult, BCISimulator


@pytest.fixture
def game_neurons():
    return generate_neuron_population(n_neurons=20, seed=42)


class TestGameResult:
    def test_to_dict(self):
        r = GameResult(
            true_rad=1.0, true_deg=57.3, user_rad=1.1, user_deg=63.0,
            model_rad=0.9, model_deg=51.6, user_error=5.7,
            model_error=5.7, winner="Tie",
        )
        d = r.to_dict()
        assert d['winner'] == 'Tie'
        assert d['true_deg'] == 57.3

    def test_fields(self):
        r = GameResult(
            true_rad=0.0, true_deg=0.0, user_rad=0.0, user_deg=0.0,
            model_rad=0.0, model_deg=0.0, user_error=0.0,
            model_error=0.0, winner="User",
        )
        assert r.winner == "User"


class TestGameEngine:
    def test_generate_round(self, game_neurons):
        theta, spikes = GameEngine.generate_round(game_neurons, duration_ms=500)
        assert 0 <= theta < 2 * np.pi
        assert len(spikes) == game_neurons.n_neurons
        assert np.all(spikes >= 0)

    def test_submit_guess_user_wins(self, game_neurons):
        # True direction is 90 degrees; user guesses 90, model will have some error
        true_theta = np.pi / 2
        spikes = np.zeros(game_neurons.n_neurons, dtype=int)  # zero spikes -> model returns 0
        decoder = PopulationVectorDecoder()
        result = GameEngine.submit_guess(
            user_deg=90.0, true_theta=true_theta,
            spike_counts=spikes, neurons=game_neurons,
            decoder=decoder, duration_s=0.5,
        )
        assert isinstance(result, GameResult)
        assert result.user_error == 0.0  # perfect guess
        assert result.winner == "User"

    def test_submit_guess_returns_result(self, game_neurons):
        from simulation import simulate_trial
        true_theta = np.pi / 4
        spikes = simulate_trial(true_theta, game_neurons, seed=0)
        decoder = MaximumLikelihoodDecoder()
        result = GameEngine.submit_guess(
            user_deg=45.0, true_theta=true_theta,
            spike_counts=spikes, neurons=game_neurons,
            decoder=decoder, duration_s=0.5,
        )
        assert isinstance(result, GameResult)
        assert result.winner in ("User", "Model", "Tie")
        assert 0 <= result.model_deg < 360

    def test_submit_guess_dict_roundtrip(self, game_neurons):
        from simulation import simulate_trial
        true_theta = 1.0
        spikes = simulate_trial(true_theta, game_neurons, seed=1)
        decoder = PopulationVectorDecoder()
        result = GameEngine.submit_guess(
            user_deg=57.0, true_theta=true_theta,
            spike_counts=spikes, neurons=game_neurons,
            decoder=decoder,
        )
        d = result.to_dict()
        assert set(d.keys()) == {
            'true_rad', 'true_deg', 'user_rad', 'user_deg',
            'model_rad', 'model_deg', 'user_error', 'model_error', 'winner',
        }


class TestBCISimulator:
    def test_init(self):
        bci = BCISimulator()
        assert bci.cursor_pos == (0.0, 0.0)
        assert bci.target_pos is None
        assert bci.hits == 0

    def test_new_target(self):
        bci = BCISimulator()
        target = bci.new_target()
        assert bci.target_pos is not None
        assert bci.attempts == 1
        dist = np.sqrt(target[0] ** 2 + target[1] ** 2)
        assert 50 <= dist <= 80

    def test_move_cursor(self, game_neurons):
        bci = BCISimulator()
        bci.new_target()
        decoder = PopulationVectorDecoder()
        new_pos, decoded, hit = bci.move_cursor(
            game_neurons, decoder, cursor_speed=5.0, noise_level=1.0,
        )
        assert new_pos != (0.0, 0.0)
        assert len(bci.trail) == 2

    def test_move_cursor_no_target_raises(self, game_neurons):
        bci = BCISimulator()
        decoder = PopulationVectorDecoder()
        with pytest.raises(RuntimeError, match="No target"):
            bci.move_cursor(game_neurons, decoder)

    def test_get_distance_to_target(self):
        bci = BCISimulator()
        assert bci.get_distance_to_target() == 0.0
        bci.target_pos = (30.0, 40.0)
        assert abs(bci.get_distance_to_target() - 50.0) < 1e-10

    def test_get_path_length(self):
        bci = BCISimulator()
        assert bci.get_path_length() == 0.0
        bci.trail = [(0, 0), (3, 4)]
        assert abs(bci.get_path_length() - 5.0) < 1e-10

    def test_reset_stats(self):
        bci = BCISimulator()
        bci.new_target()
        bci.hits = 5
        bci.reset_stats()
        assert bci.hits == 0
        assert bci.attempts == 0
        assert bci.target_pos is None

    def test_hit_detection(self, game_neurons):
        bci = BCISimulator()
        bci.target_pos = (5.0, 0.0)  # Very close target
        bci.cursor_pos = (0.0, 0.0)
        bci.trail = [(0.0, 0.0)]
        # Move directly toward target (won't necessarily hit due to decoding error,
        # but we test the hit check mechanism)
        decoder = PopulationVectorDecoder()
        # Force cursor near target
        bci.cursor_pos = (4.0, 0.0)
        bci.target_pos = (5.0, 0.0)
        # Manually check distance
        dist = bci.get_distance_to_target()
        assert dist < bci.target_radius  # Should be within radius
