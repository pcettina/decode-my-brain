"""
Kalman Filter decoder for continuous state estimation.
"""

import logging
import numpy as np
from typing import Optional, Tuple

from simulation.core import NeuronPopulation, simulate_trial
from utils import wrap_angle
from config import KALMAN_DT

from decoders.base import Decoder, _validate_decode_inputs

logger = logging.getLogger(__name__)


class KalmanFilterDecoder(Decoder):
    """
    Kalman Filter decoder for continuous state estimation.

    State vector: x = [pos_x, pos_y, vel_x, vel_y]
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = KALMAN_DT,
        process_noise: float = 0.1,
        observation_noise: float = 1.0
    ):
        self.n_neurons = n_neurons
        self.dt = dt
        self.state_dim = 4

        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.Q = np.eye(4) * process_noise
        self.Q[0, 0] = process_noise * 0.1
        self.Q[1, 1] = process_noise * 0.1

        self.R = np.eye(n_neurons) * observation_noise
        self.H = None
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
        self.baseline_rates = None
        self.preferred_directions = None
        self.is_fitted = False

    @property
    def name(self) -> str:
        return "Kalman Filter"

    def fit(
        self,
        spike_data: np.ndarray,
        kinematics: np.ndarray,
        neurons: NeuronPopulation
    ) -> None:
        """Fit the Kalman Filter parameters from training data."""
        self.preferred_directions = neurons.preferred_directions
        self.baseline_rates = np.array([
            neurons.baseline_rate for _ in range(neurons.n_neurons)
        ])

        n_neurons = len(self.preferred_directions)
        self.H = np.zeros((n_neurons, 4))

        for i, mu in enumerate(self.preferred_directions):
            self.H[i, 2] = np.cos(mu)
            self.H[i, 3] = np.sin(mu)

        self.H *= neurons.modulation_depth * self.dt

        if len(spike_data) > 0 and len(kinematics) > 0:
            predicted_rates = self.H @ kinematics.T
            residuals = spike_data.T - predicted_rates - self.baseline_rates[:, np.newaxis] * self.dt
            self.R = np.cov(residuals) + np.eye(n_neurons) * 0.1

        self.is_fitted = True

    def fit_from_neurons(self, neurons: NeuronPopulation) -> None:
        """Quick fit using just neuron tuning properties."""
        self.preferred_directions = neurons.preferred_directions
        self.baseline_rates = np.full(neurons.n_neurons, neurons.baseline_rate * self.dt)

        n_neurons = neurons.n_neurons
        self.H = np.zeros((n_neurons, 4))

        for i, mu in enumerate(self.preferred_directions):
            self.H[i, 2] = np.cos(mu) * neurons.modulation_depth * self.dt
            self.H[i, 3] = np.sin(mu) * neurons.modulation_depth * self.dt

        self.R = np.eye(n_neurons) * (neurons.baseline_rate * self.dt + 1)
        self.is_fitted = True

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """Reset the filter state."""
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step."""
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        return x_pred, P_pred

    def update(
        self,
        spike_counts: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update step with Joseph form covariance."""
        y_pred = self.H @ x_pred + self.baseline_rates
        y = spike_counts.astype(float)
        innovation = y - y_pred

        S = self.H @ P_pred @ self.H.T + self.R

        try:
            K = P_pred @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance, using pseudoinverse")
            K = P_pred @ self.H.T @ np.linalg.pinv(S)

        x_new = x_pred + K @ innovation

        IKH = np.eye(self.state_dim) - K @ self.H
        P_new = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        P_new = (P_new + P_new.T) / 2

        return x_new, P_new, K

    def decode_step(self, spike_counts: np.ndarray) -> np.ndarray:
        """Perform one decode step (predict + update)."""
        if not self.is_fitted:
            raise RuntimeError("Decoder not fitted. Call fit() or fit_from_neurons() first.")

        x_pred, P_pred = self.predict()
        self.x, self.P, _ = self.update(spike_counts, x_pred, P_pred)
        return self.x.copy()

    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        """Decode direction from spike counts."""
        _validate_decode_inputs(spike_counts, neurons)
        if not self.is_fitted:
            self.fit_from_neurons(neurons)

        bin_counts = spike_counts.astype(float) * (self.dt / duration_s)
        state = self.decode_step(bin_counts)

        vel_x, vel_y = state[2], state[3]
        if abs(vel_x) < 1e-10 and abs(vel_y) < 1e-10:
            return 0.0

        direction = np.arctan2(vel_y, vel_x)
        return wrap_angle(direction)

    def decode_trajectory(
        self,
        spike_sequence: np.ndarray,
        neurons: NeuronPopulation
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode a trajectory from a sequence of spike counts."""
        if not self.is_fitted:
            self.fit_from_neurons(neurons)

        self.reset()
        n_steps = len(spike_sequence)
        states = np.zeros((n_steps, 4))
        covariances = np.zeros((n_steps, 4, 4))

        for t in range(n_steps):
            states[t] = self.decode_step(spike_sequence[t])
            covariances[t] = self.P.copy()

        return states, covariances

    def get_uncertainty_ellipse(self, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Get the uncertainty ellipse parameters for position."""
        from scipy import stats

        pos_cov = self.P[:2, :2]
        eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)
        chi2_val = stats.chi2.ppf(confidence, 2)

        semi_major = np.sqrt(chi2_val * max(eigenvalues))
        semi_minor = np.sqrt(chi2_val * min(eigenvalues))
        major_idx = np.argmax(eigenvalues)
        angle = np.arctan2(eigenvectors[1, major_idx], eigenvectors[0, major_idx])

        return semi_major, semi_minor, angle

    def get_state(self) -> dict:
        """Get current state as a dictionary."""
        return {
            'pos_x': self.x[0],
            'pos_y': self.x[1],
            'vel_x': self.x[2],
            'vel_y': self.x[3],
            'speed': np.sqrt(self.x[2]**2 + self.x[3]**2),
            'direction': np.arctan2(self.x[3], self.x[2]),
            'position_uncertainty': np.sqrt(self.P[0, 0] + self.P[1, 1]),
            'velocity_uncertainty': np.sqrt(self.P[2, 2] + self.P[3, 3])
        }
