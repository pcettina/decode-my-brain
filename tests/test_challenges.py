"""Tests for challenges.py — scoring, lifecycle, leaderboards, and achievements."""

import json
import pytest
import numpy as np
from pathlib import Path
from challenges import (
    ChallengeMode,
    ChallengeConfig,
    ChallengeResult,
    ChallengeManager,
    AchievementManager,
    create_default_achievements,
    CHALLENGE_CONFIGS,
    score_speed_trial,
    score_precision,
    score_noise_gauntlet,
    score_streak,
    score_area_expert,
)


# ── Scoring Functions ──────────────────────────────────────────────────────

class TestScoreSpeedTrial:
    def test_empty(self):
        assert score_speed_trial([], 0.0) == 0.0

    def test_perfect(self):
        errors = [5.0] * 10
        score = score_speed_trial(errors, 30.0)
        assert score > 0

    def test_penalties_for_large_errors(self):
        good = score_speed_trial([10.0] * 5, 30.0)
        bad = score_speed_trial([60.0] * 5, 30.0)
        assert good > bad

    def test_more_trials_better(self):
        few = score_speed_trial([15.0] * 5, 60.0)
        many = score_speed_trial([15.0] * 20, 60.0)
        assert many > few


class TestScorePrecision:
    def test_empty(self):
        assert score_precision([], 0.0) == 0.0

    def test_low_error_bonus(self):
        """Mean error < 10 should get 20-point bonus."""
        score = score_precision([5.0] * 10, 60.0)
        assert score > 100  # 100 - 5 + 20 = 115

    def test_medium_error_bonus(self):
        score = score_precision([12.0] * 10, 60.0)
        # 100 - 12 + 10 = 98
        assert abs(score - 98.0) < 1e-6

    def test_high_error_no_bonus(self):
        score = score_precision([50.0] * 10, 60.0)
        assert score == 50.0  # 100 - 50 + 0 = 50

    def test_nonnegative(self):
        score = score_precision([200.0] * 10, 60.0)
        assert score >= 0


class TestScoreNoiseGauntlet:
    def test_empty(self):
        assert score_noise_gauntlet([], []) == 0.0

    def test_basic(self):
        errors = [20.0, 25.0, 30.0, 50.0]
        noise = [1.0, 1.0, 1.5, 2.0]
        score = score_noise_gauntlet(errors, noise)
        assert score > 0

    def test_higher_noise_better_score(self):
        low = score_noise_gauntlet([20.0, 20.0], [1.0, 1.0])
        high = score_noise_gauntlet([20.0, 20.0, 20.0], [1.0, 1.0, 3.0])
        assert high > low


class TestScoreStreak:
    def test_empty(self):
        assert score_streak([], 25.0) == 0.0

    def test_perfect_streak(self):
        errors = [10.0] * 10
        score = score_streak(errors, 25.0)
        assert score >= 100  # 10 * 10 = 100 + accuracy_bonus

    def test_broken_streak(self):
        errors = [10.0, 10.0, 10.0, 50.0, 10.0]
        score = score_streak(errors, 25.0)
        # Max streak = 3
        assert score >= 30  # 3 * 10
        assert score < 50

    def test_no_streak(self):
        errors = [50.0] * 5
        score = score_streak(errors, 25.0)
        assert score == 0.0  # no streak, negative accuracy bonus clamped to 0


class TestScoreAreaExpert:
    def test_empty(self):
        assert score_area_expert([], {}) == 0.0

    def test_all_areas(self):
        errors = [15.0] * 20
        area_errors = {
            'M1': [10.0] * 5,
            'PMd': [15.0] * 5,
            'PPC': [20.0] * 5,
            'SMA': [12.0] * 5,
        }
        score = score_area_expert(errors, area_errors)
        assert score > 25  # Should get completion bonus

    def test_completion_bonus(self):
        """4+ areas should give completion bonus of 25."""
        area_errors_3 = {'M1': [10.0], 'PMd': [10.0], 'PPC': [10.0]}
        area_errors_4 = {**area_errors_3, 'SMA': [10.0]}
        s3 = score_area_expert([10.0] * 3, area_errors_3)
        s4 = score_area_expert([10.0] * 4, area_errors_4)
        assert s4 - s3 >= 20  # roughly 25 point bonus


# ── ChallengeResult ────────────────────────────────────────────────────────

class TestChallengeResult:
    def test_to_dict(self):
        r = ChallengeResult(
            mode=ChallengeMode.PRECISION,
            score=85.0,
            trials_completed=10,
            mean_error=15.0,
            best_error=5.0,
            worst_error=30.0,
            time_taken=45.0,
        )
        d = r.to_dict()
        assert d['mode'] == 'precision'
        assert d['score'] == 85.0
        assert d['trials_completed'] == 10

    def test_roundtrip(self):
        r = ChallengeResult(
            mode=ChallengeMode.SPEED_TRIAL,
            score=100.0,
            trials_completed=20,
            mean_error=10.0,
            best_error=2.0,
            worst_error=25.0,
            time_taken=55.0,
            player_name="TestPlayer",
            streak_length=5,
            noise_level_reached=2.0,
        )
        d = r.to_dict()
        r2 = ChallengeResult.from_dict(d)
        assert r2.mode == r.mode
        assert r2.score == r.score
        assert r2.player_name == r.player_name
        assert r2.streak_length == r.streak_length
        assert r2.noise_level_reached == r.noise_level_reached


# ── CHALLENGE_CONFIGS ──────────────────────────────────────────────────────

class TestChallengeConfigs:
    def test_all_modes_configured(self):
        for mode in ChallengeMode:
            assert mode in CHALLENGE_CONFIGS

    def test_config_structure(self):
        for mode, config in CHALLENGE_CONFIGS.items():
            assert config.mode == mode
            assert config.name
            assert config.description
            assert config.scoring_func


# ── ChallengeManager lifecycle ─────────────────────────────────────────────

class TestChallengeManager:
    def test_lifecycle(self):
        mgr = ChallengeManager()
        config = mgr.start_challenge(ChallengeMode.PRECISION)
        assert config.mode == ChallengeMode.PRECISION

        for _ in range(10):
            mgr.record_trial(error=15.0)

        result = mgr.finish_challenge(player_name="Test")
        assert result.trials_completed == 10
        assert result.score > 0

    def test_double_finish(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.STREAK)
        mgr.record_trial(error=10.0)
        mgr.finish_challenge()
        with pytest.raises(RuntimeError):
            mgr.finish_challenge()

    def test_record_without_start(self):
        mgr = ChallengeManager()
        with pytest.raises(RuntimeError, match="No active"):
            mgr.record_trial(error=10.0)

    def test_get_state_no_challenge(self):
        mgr = ChallengeManager()
        state = mgr.get_state()
        assert state['active'] is False

    def test_get_state_active(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.PRECISION)
        mgr.record_trial(error=20.0)
        state = mgr.get_state()
        assert state['active'] is True
        assert state['trials_completed'] == 1
        assert state['mean_error'] == 20.0

    def test_noise_gauntlet_progression(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.NOISE_GAUNTLET)
        # Record 6 trials (should advance noise level)
        for _ in range(6):
            mgr.record_trial(error=20.0)
        # After 6 trials, level_idx = 6//3 = 2, noise = 1.0 (index 2 of progression)
        assert mgr.current_noise_level == 1.0
        assert len(mgr.noise_levels) == 6

    def test_streak_tracking(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.STREAK)
        mgr.record_trial(error=10.0)
        mgr.record_trial(error=15.0)
        mgr.record_trial(error=10.0)
        assert mgr.current_streak == 3
        assert mgr.best_streak == 3
        mgr.record_trial(error=50.0)  # break streak
        assert mgr.current_streak == 0
        assert mgr.best_streak == 3

    def test_area_error_tracking(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.AREA_EXPERT)
        mgr.record_trial(error=10.0, brain_area='M1')
        mgr.record_trial(error=20.0, brain_area='M1')
        mgr.record_trial(error=15.0, brain_area='PMd')
        assert len(mgr.area_errors['M1']) == 2
        assert len(mgr.area_errors['PMd']) == 1

    def test_is_challenge_over_trial_limit(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.PRECISION)  # 10 trial limit
        for _ in range(10):
            mgr.record_trial(error=15.0)
        assert mgr.is_challenge_over()

    def test_is_challenge_over_streak_broken(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.STREAK)
        mgr.record_trial(error=10.0)
        assert not mgr.is_challenge_over()
        mgr.record_trial(error=50.0)  # breaks streak
        assert mgr.is_challenge_over()

    def test_is_challenge_over_noise_gauntlet(self):
        mgr = ChallengeManager()
        mgr.start_challenge(ChallengeMode.NOISE_GAUNTLET)
        mgr.record_trial(error=20.0)
        assert not mgr.is_challenge_over()
        mgr.record_trial(error=60.0)  # over 45 threshold
        assert mgr.is_challenge_over()

    def test_is_challenge_over_no_active(self):
        mgr = ChallengeManager()
        assert mgr.is_challenge_over()

    def test_all_challenge_modes_start(self):
        """Every mode should be startable."""
        for mode in ChallengeMode:
            mgr = ChallengeManager()
            config = mgr.start_challenge(mode)
            assert config.mode == mode


# ── Leaderboard persistence ───────────────────────────────────────────────

class TestLeaderboard:
    @staticmethod
    def _fresh_mgr(tmp_path):
        """ChallengeManager with isolated temp leaderboard path."""
        mgr = ChallengeManager()
        mgr.LEADERBOARD_PATH = tmp_path / "test_lb.json"
        mgr.leaderboards = mgr._load_leaderboard()
        return mgr

    def test_get_leaderboard_empty(self, tmp_path):
        mgr = self._fresh_mgr(tmp_path)
        lb = mgr.get_leaderboard(ChallengeMode.PRECISION)
        assert lb == []

    def test_get_leaderboard_after_finish(self, tmp_path):
        mgr = self._fresh_mgr(tmp_path)
        mgr.start_challenge(ChallengeMode.PRECISION)
        for _ in range(10):
            mgr.record_trial(error=15.0)
        mgr.finish_challenge(player_name="Alice")
        lb = mgr.get_leaderboard(ChallengeMode.PRECISION)
        assert len(lb) == 1
        assert lb[0]['name'] == 'Alice'

    def test_get_personal_best(self, tmp_path):
        mgr = self._fresh_mgr(tmp_path)
        mgr.start_challenge(ChallengeMode.SPEED_TRIAL)
        for _ in range(5):
            mgr.record_trial(error=10.0)
        mgr.finish_challenge(player_name="Bob")
        pb = mgr.get_personal_best(ChallengeMode.SPEED_TRIAL)
        assert pb is not None
        assert pb.player_name == "Bob"

    def test_get_personal_best_none(self, tmp_path):
        mgr = self._fresh_mgr(tmp_path)
        assert mgr.get_personal_best(ChallengeMode.STREAK) is None

    def test_save_and_load_roundtrip(self, tmp_path):
        """Leaderboard should persist and reload correctly."""
        mgr = self._fresh_mgr(tmp_path)

        mgr.start_challenge(ChallengeMode.PRECISION)
        for _ in range(10):
            mgr.record_trial(error=12.0)
        mgr.finish_challenge(player_name="SaveTest")

        assert mgr.LEADERBOARD_PATH.exists()

        # Load into a new manager
        mgr2 = ChallengeManager()
        mgr2.LEADERBOARD_PATH = mgr.LEADERBOARD_PATH
        mgr2.leaderboards = mgr2._load_leaderboard()
        lb = mgr2.get_leaderboard(ChallengeMode.PRECISION)
        assert len(lb) == 1
        assert lb[0]['name'] == 'SaveTest'

    def test_load_missing_file(self, tmp_path):
        mgr = ChallengeManager()
        mgr.LEADERBOARD_PATH = tmp_path / "nonexistent.json"
        result = mgr._load_leaderboard()
        assert all(mode in result for mode in ChallengeMode)
        assert all(len(v) == 0 for v in result.values())

    def test_load_corrupt_file(self, tmp_path):
        lb_path = tmp_path / "corrupt.json"
        lb_path.write_text("not valid json {{{")
        mgr = ChallengeManager()
        mgr.LEADERBOARD_PATH = lb_path
        result = mgr._load_leaderboard()
        # Should return empty defaults
        assert all(len(v) == 0 for v in result.values())

    def test_leaderboard_limit(self, tmp_path):
        """Leaderboard should cap at MAX_LEADERBOARD_ENTRIES."""
        from config import MAX_LEADERBOARD_ENTRIES
        mgr = self._fresh_mgr(tmp_path)

        for i in range(MAX_LEADERBOARD_ENTRIES + 5):
            mgr.start_challenge(ChallengeMode.SPEED_TRIAL)
            mgr.record_trial(error=float(i + 1))
            mgr.finish_challenge(player_name=f"P{i}")

        lb = mgr.get_leaderboard(ChallengeMode.SPEED_TRIAL)
        assert len(lb) <= MAX_LEADERBOARD_ENTRIES


# ── AchievementManager ─────────────────────────────────────────────────────

class TestAchievementManager:
    def _make_result(self, mode=ChallengeMode.PRECISION, **kwargs):
        defaults = dict(
            mode=mode, score=50.0, trials_completed=10,
            mean_error=15.0, best_error=5.0, worst_error=30.0,
            time_taken=45.0,
        )
        defaults.update(kwargs)
        return ChallengeResult(**defaults)

    def test_first_decode(self):
        mgr = AchievementManager()
        result = self._make_result(trials_completed=1)
        earned = mgr.check_achievements(result)
        ids = [a.id for a in earned]
        assert 'first_decode' in ids

    def test_precision_master(self):
        mgr = AchievementManager()
        result = self._make_result(mode=ChallengeMode.PRECISION, mean_error=8.0)
        earned = mgr.check_achievements(result)
        ids = [a.id for a in earned]
        assert 'precision_master' in ids

    def test_speed_demon(self):
        mgr = AchievementManager()
        result = self._make_result(
            mode=ChallengeMode.SPEED_TRIAL, trials_completed=25
        )
        earned = mgr.check_achievements(result)
        ids = [a.id for a in earned]
        assert 'speed_demon' in ids

    def test_perfect_trial(self):
        mgr = AchievementManager()
        result = self._make_result(best_error=3.0)
        earned = mgr.check_achievements(result)
        ids = [a.id for a in earned]
        assert 'perfect_trial' in ids

    def test_noise_survivor(self):
        mgr = AchievementManager()
        result = self._make_result(
            mode=ChallengeMode.NOISE_GAUNTLET, noise_level_reached=3.5
        )
        earned = mgr.check_achievements(result)
        ids = [a.id for a in earned]
        assert 'noise_survivor' in ids

    def test_streak_10(self):
        mgr = AchievementManager()
        result = self._make_result(
            mode=ChallengeMode.STREAK, streak_length=12
        )
        earned = mgr.check_achievements(result)
        ids = [a.id for a in earned]
        assert 'streak_10' in ids

    def test_centurion(self):
        """100 total trials should earn centurion."""
        mgr = AchievementManager()
        # Add 90 trials first
        mgr.check_achievements(self._make_result(trials_completed=90))
        # Then 10 more
        earned = mgr.check_achievements(self._make_result(trials_completed=10))
        ids = [a.id for a in earned]
        assert 'centurion' in ids

    def test_all_modes(self):
        """Completing all 5 modes should earn 'all_modes'."""
        mgr = AchievementManager()
        for mode in ChallengeMode:
            earned = mgr.check_achievements(self._make_result(mode=mode))
        ids = [a.id for a in earned]
        assert 'all_modes' in ids

    def test_no_double_earn(self):
        """Achievement should not be earned twice."""
        mgr = AchievementManager()
        r = self._make_result(trials_completed=1)
        earned1 = mgr.check_achievements(r)
        earned2 = mgr.check_achievements(r)
        assert len(earned2) == 0  # already earned

    def test_get_all_achievements(self):
        mgr = AchievementManager()
        all_a = mgr.get_all_achievements()
        assert len(all_a) == 10
        assert all('id' in a for a in all_a)
        assert all('earned' in a for a in all_a)

    def test_get_earned_count(self):
        mgr = AchievementManager()
        earned, total = mgr.get_earned_count()
        assert earned == 0
        assert total == 10
        mgr.check_achievements(self._make_result(trials_completed=1))
        earned, total = mgr.get_earned_count()
        assert earned >= 1


# ── create_default_achievements ────────────────────────────────────────────

def test_default_achievements_count():
    achievements = create_default_achievements()
    assert len(achievements) == 10
    ids = [a.id for a in achievements]
    assert len(set(ids)) == 10  # all unique
