"""
Game and BCI business logic for the Decode My Brain app.

Extracted from app.py to separate UI concerns from game rules.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from simulation import NeuronPopulation, simulate_trial
from decoders import Decoder
from utils import angular_error_degrees, radians_to_degrees, degrees_to_radians, wrap_angle
from config import BCI_TARGET_RADIUS, BCI_CANVAS_SIZE

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game round."""
    true_rad: float
    true_deg: float
    user_rad: float
    user_deg: float
    model_rad: float
    model_deg: float
    user_error: float
    model_error: float
    winner: str  # "User" | "Model" | "Tie"

    def to_dict(self) -> dict:
        """Convert to dictionary for Streamlit session state."""
        return {
            'true_rad': self.true_rad,
            'true_deg': self.true_deg,
            'user_rad': self.user_rad,
            'user_deg': self.user_deg,
            'model_rad': self.model_rad,
            'model_deg': self.model_deg,
            'user_error': self.user_error,
            'model_error': self.model_error,
            'winner': self.winner,
        }


class GameEngine:
    """Handles the decode-the-direction game logic."""

    @staticmethod
    def generate_round(
        neurons: NeuronPopulation,
        duration_ms: float,
        variance_scale: float = 1.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Generate a new game round.

        Args:
            neurons: NeuronPopulation to simulate
            duration_ms: Trial duration in ms
            variance_scale: Noise level

        Returns:
            Tuple of (true_theta_radians, spike_counts)
        """
        true_theta = np.random.uniform(0, 2 * np.pi)
        spikes = simulate_trial(
            true_theta, neurons,
            duration_ms=duration_ms,
            variance_scale=variance_scale,
        )
        return true_theta, spikes

    @staticmethod
    def submit_guess(
        user_deg: float,
        true_theta: float,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        decoder: Decoder,
        duration_s: float = 0.5,
    ) -> GameResult:
        """
        Process a user's guess and compute the game result.

        Args:
            user_deg: User's guess in degrees
            true_theta: True direction in radians
            spike_counts: Spike counts for this trial
            neurons: NeuronPopulation
            decoder: Decoder instance to use for model
            duration_s: Trial duration in seconds

        Returns:
            GameResult with errors and winner
        """
        user_theta = degrees_to_radians(user_deg)
        model_theta = decoder.decode(spike_counts, neurons, duration_s=duration_s)

        user_error = angular_error_degrees(user_deg, radians_to_degrees(true_theta))
        model_error = angular_error_degrees(
            radians_to_degrees(model_theta),
            radians_to_degrees(true_theta),
        )

        if user_error < model_error:
            winner = "User"
        elif model_error < user_error:
            winner = "Model"
        else:
            winner = "Tie"

        return GameResult(
            true_rad=true_theta,
            true_deg=radians_to_degrees(true_theta),
            user_rad=user_theta,
            user_deg=user_deg,
            model_rad=model_theta,
            model_deg=radians_to_degrees(model_theta),
            user_error=user_error,
            model_error=model_error,
            winner=winner,
        )


class BCISimulator:
    """Simulates brain-computer interface cursor control."""

    def __init__(self, canvas_size: float = BCI_CANVAS_SIZE):
        self.canvas_size = canvas_size
        self.cursor_pos = (0.0, 0.0)
        self.target_pos: Optional[Tuple[float, float]] = None
        self.trail: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.hits = 0
        self.attempts = 0
        self.target_radius = BCI_TARGET_RADIUS

    def new_target(self) -> Tuple[float, float]:
        """Generate a new random target position."""
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(50, 80)
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        self.target_pos = (target_x, target_y)
        self.cursor_pos = (0.0, 0.0)
        self.trail = [(0.0, 0.0)]
        self.attempts += 1
        return self.target_pos

    def move_cursor(
        self,
        neurons: NeuronPopulation,
        decoder: Decoder,
        cursor_speed: float = 5.0,
        noise_level: float = 1.0,
    ) -> Tuple[Tuple[float, float], float, bool]:
        """
        Move cursor one step based on neural decoding.

        Args:
            neurons: NeuronPopulation
            decoder: Decoder for direction estimation
            cursor_speed: Step size per move
            noise_level: Variance scale for simulation

        Returns:
            Tuple of (new_position, decoded_angle, hit_target)
        """
        if self.target_pos is None:
            raise RuntimeError("No target set. Call new_target() first.")

        cx, cy = self.cursor_pos
        tx, ty = self.target_pos

        # True angle to target
        true_angle = np.arctan2(ty - cy, tx - cx)

        # Simulate neural activity
        spikes = simulate_trial(
            true_angle, neurons,
            duration_ms=100,
            variance_scale=noise_level,
        )

        # Decode direction
        decoded_angle = decoder.decode(spikes, neurons)

        # Move cursor
        new_x = cx + cursor_speed * np.cos(decoded_angle)
        new_y = cy + cursor_speed * np.sin(decoded_angle)
        self.cursor_pos = (new_x, new_y)
        self.trail.append((new_x, new_y))

        # Check if target reached
        dist = np.sqrt((new_x - tx) ** 2 + (new_y - ty) ** 2)
        hit = dist < self.target_radius
        if hit:
            self.hits += 1

        return self.cursor_pos, decoded_angle, hit

    def get_distance_to_target(self) -> float:
        """Get current distance from cursor to target."""
        if self.target_pos is None:
            return 0.0
        cx, cy = self.cursor_pos
        tx, ty = self.target_pos
        return float(np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2))

    def get_path_length(self) -> float:
        """Get total path length traversed."""
        if len(self.trail) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.trail)):
            dx = self.trail[i][0] - self.trail[i - 1][0]
            dy = self.trail[i][1] - self.trail[i - 1][1]
            total += np.sqrt(dx ** 2 + dy ** 2)
        return total

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.cursor_pos = (0.0, 0.0)
        self.target_pos = None
        self.trail = [(0.0, 0.0)]
        self.hits = 0
        self.attempts = 0
