"""Tests for challenges.py — scoring and challenge lifecycle."""

import pytest
from challenges import (
    ChallengeManager,
    ChallengeMode,
    score_precision,
)


def test_scoring_empty():
    """Scoring with no errors should return 0."""
    assert score_precision([], 0.0) == 0.0


def test_challenge_lifecycle():
    """Start -> record trials -> finish should produce a valid result."""
    mgr = ChallengeManager()
    config = mgr.start_challenge(ChallengeMode.PRECISION)
    assert config.mode == ChallengeMode.PRECISION

    for _ in range(10):
        mgr.record_trial(error=15.0)

    result = mgr.finish_challenge(player_name="Test")
    assert result.trials_completed == 10
    assert result.score > 0


def test_double_finish():
    """Finishing twice should raise RuntimeError."""
    mgr = ChallengeManager()
    mgr.start_challenge(ChallengeMode.STREAK)
    mgr.record_trial(error=10.0)
    mgr.finish_challenge()

    with pytest.raises(RuntimeError):
        mgr.finish_challenge()
