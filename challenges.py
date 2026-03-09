"""
Challenge modes and scoring system for Decode My Brain.

Implements various competitive game modes with scoring, timing, and leaderboards.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Callable, Tuple
from enum import Enum

from config import MAX_LEADERBOARD_ENTRIES

logger = logging.getLogger(__name__)


class ChallengeMode(Enum):
    """Available challenge modes."""
    SPEED_TRIAL = "speed_trial"
    PRECISION = "precision"
    NOISE_GAUNTLET = "noise_gauntlet"
    STREAK = "streak"
    AREA_EXPERT = "area_expert"


@dataclass
class ChallengeConfig:
    """Configuration for a challenge mode."""
    mode: ChallengeMode
    name: str
    description: str
    time_limit_s: Optional[float]  # None = no time limit
    trial_limit: Optional[int]     # None = no trial limit
    target_error: float            # Target error to beat (degrees)
    difficulty: str                # 'easy', 'medium', 'hard'
    scoring_func: str              # Name of scoring function to use
    
    # Mode-specific parameters
    noise_progression: Optional[List[float]] = None  # For noise gauntlet
    streak_threshold: float = 20.0  # Max error to continue streak
    brain_area: Optional[str] = None  # For area expert


@dataclass
class ChallengeResult:
    """Result from a completed challenge."""
    mode: ChallengeMode
    score: float
    trials_completed: int
    mean_error: float
    best_error: float
    worst_error: float
    time_taken: float
    timestamp: datetime = field(default_factory=datetime.now)
    player_name: str = "Player"
    
    # Mode-specific results
    streak_length: int = 0
    noise_level_reached: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'mode': self.mode.value,
            'score': self.score,
            'trials_completed': self.trials_completed,
            'mean_error': self.mean_error,
            'best_error': self.best_error,
            'worst_error': self.worst_error,
            'time_taken': self.time_taken,
            'timestamp': self.timestamp.isoformat(),
            'player_name': self.player_name,
            'streak_length': self.streak_length,
            'noise_level_reached': self.noise_level_reached
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChallengeResult':
        """Create from dictionary."""
        return cls(
            mode=ChallengeMode(data['mode']),
            score=data['score'],
            trials_completed=data['trials_completed'],
            mean_error=data['mean_error'],
            best_error=data['best_error'],
            worst_error=data['worst_error'],
            time_taken=data['time_taken'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            player_name=data.get('player_name', 'Player'),
            streak_length=data.get('streak_length', 0),
            noise_level_reached=data.get('noise_level_reached', 1.0)
        )


# =============================================================================
# Challenge Configurations
# =============================================================================

CHALLENGE_CONFIGS = {
    ChallengeMode.SPEED_TRIAL: ChallengeConfig(
        mode=ChallengeMode.SPEED_TRIAL,
        name="Speed Trial",
        description="Decode as many directions as possible in 60 seconds! Accuracy matters - errors add time penalties.",
        time_limit_s=60.0,
        trial_limit=None,
        target_error=30.0,
        difficulty='medium',
        scoring_func='score_speed_trial'
    ),
    
    ChallengeMode.PRECISION: ChallengeConfig(
        mode=ChallengeMode.PRECISION,
        name="Precision Mode",
        description="Complete 10 trials with the lowest total error. Take your time and be accurate!",
        time_limit_s=None,
        trial_limit=10,
        target_error=15.0,
        difficulty='hard',
        scoring_func='score_precision'
    ),
    
    ChallengeMode.NOISE_GAUNTLET: ChallengeConfig(
        mode=ChallengeMode.NOISE_GAUNTLET,
        name="Noise Gauntlet",
        description="Neural noise increases every 3 trials. How far can you go before accuracy drops below 45°?",
        time_limit_s=None,
        trial_limit=None,
        target_error=45.0,
        difficulty='hard',
        scoring_func='score_noise_gauntlet',
        noise_progression=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    ),
    
    ChallengeMode.STREAK: ChallengeConfig(
        mode=ChallengeMode.STREAK,
        name="Streak Challenge",
        description="Build the longest streak of accurate decodes! Streak breaks if error exceeds 25°.",
        time_limit_s=None,
        trial_limit=None,
        target_error=25.0,
        difficulty='medium',
        scoring_func='score_streak',
        streak_threshold=25.0
    ),
    
    ChallengeMode.AREA_EXPERT: ChallengeConfig(
        mode=ChallengeMode.AREA_EXPERT,
        name="Area Expert",
        description="Master decoding from each brain area. Complete 5 trials per area!",
        time_limit_s=180.0,
        trial_limit=20,  # 5 trials × 4 areas
        target_error=25.0,
        difficulty='hard',
        scoring_func='score_area_expert'
    )
}


# =============================================================================
# Scoring Functions
# =============================================================================

def score_speed_trial(errors: List[float], time_taken: float, **kwargs) -> float:
    """
    Score for speed trial mode.
    
    Score = trials_completed - (mean_error / 10) - (time_penalties)
    """
    if not errors:
        return 0.0
    
    trials = len(errors)
    mean_error = np.mean(errors)
    
    # Penalty for errors over 45 degrees
    penalties = sum(max(0, e - 45) / 20 for e in errors)
    
    score = trials * 10 - mean_error - penalties
    return max(0, score)


def score_precision(errors: List[float], time_taken: float, **kwargs) -> float:
    """
    Score for precision mode.
    
    Score = 100 - mean_error (lower error = higher score)
    """
    if not errors:
        return 0.0
    
    mean_error = np.mean(errors)
    
    # Bonus for very low errors
    bonus = 0
    if mean_error < 10:
        bonus = 20
    elif mean_error < 15:
        bonus = 10
    elif mean_error < 20:
        bonus = 5
    
    score = max(0, 100 - mean_error + bonus)
    return score


def score_noise_gauntlet(errors: List[float], noise_levels: List[float], **kwargs) -> float:
    """
    Score for noise gauntlet mode.
    
    Score based on highest noise level reached while maintaining accuracy.
    """
    if not errors or not noise_levels:
        return 0.0
    
    # Find highest noise level where accuracy was maintained
    max_noise_reached = noise_levels[0]
    for i, (error, noise) in enumerate(zip(errors, noise_levels)):
        if error <= 45:  # Accuracy threshold
            max_noise_reached = noise
    
    # Score = noise level × 100 + accuracy bonus
    mean_error = np.mean(errors)
    accuracy_bonus = max(0, 45 - mean_error)
    
    score = max_noise_reached * 100 + accuracy_bonus
    return score


def score_streak(errors: List[float], streak_threshold: float = 25.0, **kwargs) -> float:
    """
    Score for streak mode.
    
    Score = longest streak × 10 + accuracy bonus
    """
    if not errors:
        return 0.0
    
    # Find longest streak
    current_streak = 0
    max_streak = 0
    
    for error in errors:
        if error <= streak_threshold:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    # Accuracy bonus
    mean_error = np.mean(errors)
    accuracy_bonus = max(0, (streak_threshold - mean_error) / 2)
    
    score = max_streak * 10 + accuracy_bonus
    return score


def score_area_expert(errors: List[float], area_errors: Dict[str, List[float]], **kwargs) -> float:
    """
    Score for area expert mode.
    
    Score based on performance across all brain areas.
    """
    if not errors or not area_errors:
        return 0.0
    
    area_scores = []
    for area_name, area_err in area_errors.items():
        if area_err:
            area_mean = np.mean(area_err)
            area_scores.append(max(0, 100 - area_mean * 2))
    
    if not area_scores:
        return 0.0
    
    # Average across areas with bonus for completing all
    base_score = np.mean(area_scores)
    completion_bonus = 25 if len(area_errors) >= 4 else 0
    
    return base_score + completion_bonus


# =============================================================================
# Challenge Manager
# =============================================================================

class ChallengeManager:
    """
    Manages challenge state, scoring, and leaderboards.
    """
    
    LEADERBOARD_PATH = Path("data/leaderboards.json")

    def __init__(self):
        """Initialize the challenge manager."""
        self.active_challenge: Optional[ChallengeConfig] = None
        self.errors: List[float] = []
        self.noise_levels: List[float] = []
        self.area_errors: Dict[str, List[float]] = {}
        self.start_time: Optional[datetime] = None
        self.current_streak: int = 0
        self.best_streak: int = 0
        self.current_noise_level: float = 1.0
        self.trials_completed: int = 0

        # Load persisted leaderboards or start fresh
        self.leaderboards = self._load_leaderboard()
    
    def start_challenge(self, mode: ChallengeMode) -> ChallengeConfig:
        """
        Start a new challenge.
        
        Args:
            mode: Which challenge mode to start
            
        Returns:
            Configuration for the challenge
        """
        self.active_challenge = CHALLENGE_CONFIGS[mode]
        self.errors = []
        self.noise_levels = []
        self.area_errors = {}
        self.start_time = datetime.now()
        self.current_streak = 0
        self.best_streak = 0
        self.current_noise_level = 1.0
        self.trials_completed = 0
        
        return self.active_challenge
    
    def record_trial(
        self,
        error: float,
        brain_area: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Record the result of a trial.
        
        Args:
            error: Angular error in degrees
            brain_area: Which brain area (for area expert mode)
            
        Returns:
            Dict with updated state info
        """
        if self.active_challenge is None:
            raise RuntimeError("No active challenge")
        
        self.errors.append(error)
        self.trials_completed += 1
        
        # Update noise level for gauntlet mode
        if self.active_challenge.mode == ChallengeMode.NOISE_GAUNTLET:
            progression = self.active_challenge.noise_progression
            if progression:
                level_idx = min(self.trials_completed // 3, len(progression) - 1)
                self.current_noise_level = progression[level_idx]
            self.noise_levels.append(self.current_noise_level)
        
        # Update streak
        if error <= self.active_challenge.streak_threshold:
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = 0
        
        # Record area-specific errors
        if brain_area:
            if brain_area not in self.area_errors:
                self.area_errors[brain_area] = []
            self.area_errors[brain_area].append(error)
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, any]:
        """Get current challenge state."""
        if self.active_challenge is None:
            return {'active': False}
        
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        time_remaining = None
        if self.active_challenge.time_limit_s:
            time_remaining = max(0, self.active_challenge.time_limit_s - elapsed)
        
        trials_remaining = None
        if self.active_challenge.trial_limit:
            trials_remaining = max(0, self.active_challenge.trial_limit - self.trials_completed)
        
        return {
            'active': True,
            'mode': self.active_challenge.mode.value,
            'trials_completed': self.trials_completed,
            'current_streak': self.current_streak,
            'best_streak': self.best_streak,
            'mean_error': np.mean(self.errors) if self.errors else 0,
            'best_error': min(self.errors) if self.errors else 0,
            'worst_error': max(self.errors) if self.errors else 0,
            'elapsed_time': elapsed,
            'time_remaining': time_remaining,
            'trials_remaining': trials_remaining,
            'current_noise_level': self.current_noise_level,
            'errors': self.errors.copy()
        }
    
    def is_challenge_over(self) -> bool:
        """Check if the current challenge is complete."""
        if self.active_challenge is None:
            return True
        
        state = self.get_state()
        
        # Time limit reached
        if state['time_remaining'] is not None and state['time_remaining'] <= 0:
            return True
        
        # Trial limit reached
        if state['trials_remaining'] is not None and state['trials_remaining'] <= 0:
            return True
        
        # Streak broken (for streak mode)
        if self.active_challenge.mode == ChallengeMode.STREAK:
            if self.current_streak == 0 and self.trials_completed > 0:
                return True
        
        # Noise gauntlet: accuracy dropped too low
        if self.active_challenge.mode == ChallengeMode.NOISE_GAUNTLET:
            if self.errors and self.errors[-1] > self.active_challenge.target_error:
                return True
        
        return False
    
    def finish_challenge(self, player_name: str = "Player") -> ChallengeResult:
        """
        Finish the current challenge and compute final score.
        
        Args:
            player_name: Name for the leaderboard
            
        Returns:
            Challenge result with final score
        """
        if self.active_challenge is None:
            raise RuntimeError("No active challenge")
        
        state = self.get_state()
        
        # Calculate score based on mode
        scoring_funcs = {
            'score_speed_trial': score_speed_trial,
            'score_precision': score_precision,
            'score_noise_gauntlet': score_noise_gauntlet,
            'score_streak': score_streak,
            'score_area_expert': score_area_expert
        }
        
        func = scoring_funcs.get(self.active_challenge.scoring_func, score_precision)
        score = func(
            errors=self.errors,
            time_taken=state['elapsed_time'],
            noise_levels=self.noise_levels,
            streak_threshold=self.active_challenge.streak_threshold,
            area_errors=self.area_errors
        )
        
        result = ChallengeResult(
            mode=self.active_challenge.mode,
            score=score,
            trials_completed=self.trials_completed,
            mean_error=state['mean_error'],
            best_error=state['best_error'],
            worst_error=state['worst_error'],
            time_taken=state['elapsed_time'],
            player_name=player_name,
            streak_length=self.best_streak,
            noise_level_reached=self.current_noise_level
        )
        
        # Add to leaderboard
        self.add_to_leaderboard(result)
        
        # Reset state
        self.active_challenge = None
        
        return result
    
    def add_to_leaderboard(self, result: ChallengeResult) -> None:
        """Add a result to the leaderboard and persist."""
        leaderboard = self.leaderboards[result.mode]
        leaderboard.append(result)

        if result.mode == ChallengeMode.PRECISION:
            leaderboard.sort(key=lambda x: x.mean_error)
        else:
            leaderboard.sort(key=lambda x: x.score, reverse=True)

        self.leaderboards[result.mode] = leaderboard[:MAX_LEADERBOARD_ENTRIES]
        self._save_leaderboard()

    def _save_leaderboard(self) -> None:
        """Persist leaderboards to disk."""
        try:
            self.LEADERBOARD_PATH.parent.mkdir(exist_ok=True)
            data = {
                mode.value: [r.to_dict() for r in results]
                for mode, results in self.leaderboards.items()
            }
            with open(self.LEADERBOARD_PATH, 'w') as f:
                json.dump(data, f)
        except Exception:
            logger.warning("Failed to save leaderboard", exc_info=True)

    def _load_leaderboard(self) -> Dict[ChallengeMode, List[ChallengeResult]]:
        """Load leaderboards from disk, or return empty ones."""
        default = {mode: [] for mode in ChallengeMode}
        if not self.LEADERBOARD_PATH.exists():
            return default
        try:
            with open(self.LEADERBOARD_PATH) as f:
                data = json.load(f)
            return {
                ChallengeMode(k): [ChallengeResult.from_dict(r) for r in v]
                for k, v in data.items()
            }
        except Exception:
            logger.warning("Failed to load leaderboard, starting fresh", exc_info=True)
            return default
    
    def get_leaderboard(self, mode: ChallengeMode, n: int = 10) -> List[Dict]:
        """
        Get the leaderboard for a mode.
        
        Args:
            mode: Which challenge mode
            n: Number of entries to return
            
        Returns:
            List of score dictionaries
        """
        leaderboard = self.leaderboards.get(mode, [])[:n]
        
        return [
            {
                'rank': i + 1,
                'name': r.player_name,
                'score': r.score if mode != ChallengeMode.PRECISION else r.mean_error,
                'trials': r.trials_completed,
                'date': r.timestamp.strftime('%m/%d %H:%M')
            }
            for i, r in enumerate(leaderboard)
        ]
    
    def get_personal_best(self, mode: ChallengeMode) -> Optional[ChallengeResult]:
        """Get personal best for a mode."""
        leaderboard = self.leaderboards.get(mode, [])
        return leaderboard[0] if leaderboard else None


# =============================================================================
# Achievements System
# =============================================================================

@dataclass
class Achievement:
    """An achievement that can be earned."""
    id: str
    name: str
    description: str
    icon: str
    condition: Callable[[ChallengeResult], bool]
    earned: bool = False


def create_default_achievements() -> List[Achievement]:
    """Create the default set of achievements."""
    return [
        Achievement(
            id='first_decode',
            name='First Decode',
            description='Complete your first decode',
            icon='🎯',
            condition=lambda r: r.trials_completed >= 1
        ),
        Achievement(
            id='speed_demon',
            name='Speed Demon',
            description='Complete 20+ trials in Speed Trial mode',
            icon='⚡',
            condition=lambda r: r.mode == ChallengeMode.SPEED_TRIAL and r.trials_completed >= 20
        ),
        Achievement(
            id='precision_master',
            name='Precision Master',
            description='Achieve mean error under 10° in Precision mode',
            icon='🎯',
            condition=lambda r: r.mode == ChallengeMode.PRECISION and r.mean_error < 10
        ),
        Achievement(
            id='noise_survivor',
            name='Noise Survivor',
            description='Reach noise level 3.0 in Noise Gauntlet',
            icon='📡',
            condition=lambda r: r.mode == ChallengeMode.NOISE_GAUNTLET and r.noise_level_reached >= 3.0
        ),
        Achievement(
            id='streak_10',
            name='Hot Streak',
            description='Build a streak of 10+ in Streak Challenge',
            icon='🔥',
            condition=lambda r: r.mode == ChallengeMode.STREAK and r.streak_length >= 10
        ),
        Achievement(
            id='streak_20',
            name='On Fire!',
            description='Build a streak of 20+ in Streak Challenge',
            icon='🔥🔥',
            condition=lambda r: r.mode == ChallengeMode.STREAK and r.streak_length >= 20
        ),
        Achievement(
            id='perfect_trial',
            name='Perfect Shot',
            description='Achieve an error under 5° on any trial',
            icon='💎',
            condition=lambda r: r.best_error < 5
        ),
        Achievement(
            id='area_master',
            name='Area Master',
            description='Complete Area Expert mode',
            icon='🧠',
            condition=lambda r: r.mode == ChallengeMode.AREA_EXPERT and r.trials_completed >= 20
        ),
        Achievement(
            id='centurion',
            name='Centurion',
            description='Complete 100 total trials across all modes',
            icon='💯',
            condition=lambda r: False  # Special: checked externally
        ),
        Achievement(
            id='all_modes',
            name='Well Rounded',
            description='Complete a challenge in each mode',
            icon='🌟',
            condition=lambda r: False  # Special: checked externally
        )
    ]


class AchievementManager:
    """Manages achievements and tracks progress."""
    
    def __init__(self):
        """Initialize with default achievements."""
        self.achievements = create_default_achievements()
        self.total_trials = 0
        self.modes_completed = set()
    
    def check_achievements(self, result: ChallengeResult) -> List[Achievement]:
        """
        Check which achievements were earned from a result.
        
        Args:
            result: The challenge result to check
            
        Returns:
            List of newly earned achievements
        """
        self.total_trials += result.trials_completed
        self.modes_completed.add(result.mode)
        
        newly_earned = []
        
        for achievement in self.achievements:
            if achievement.earned:
                continue
            
            # Special achievements
            if achievement.id == 'centurion':
                if self.total_trials >= 100:
                    achievement.earned = True
                    newly_earned.append(achievement)
            elif achievement.id == 'all_modes':
                if len(self.modes_completed) >= 5:
                    achievement.earned = True
                    newly_earned.append(achievement)
            else:
                # Standard achievements
                if achievement.condition(result):
                    achievement.earned = True
                    newly_earned.append(achievement)
        
        return newly_earned
    
    def get_all_achievements(self) -> List[Dict]:
        """Get all achievements as dictionaries."""
        return [
            {
                'id': a.id,
                'name': a.name,
                'description': a.description,
                'icon': a.icon,
                'earned': a.earned
            }
            for a in self.achievements
        ]
    
    def get_earned_count(self) -> Tuple[int, int]:
        """Get count of earned vs total achievements."""
        earned = sum(1 for a in self.achievements if a.earned)
        return earned, len(self.achievements)

