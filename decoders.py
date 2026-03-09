"""
Neural decoders for the Decode My Brain app.

Implements population vector and maximum likelihood decoders
for inferring movement direction from spike counts.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from simulation import NeuronPopulation, cosine_tuning
from utils import wrap_angle, circular_mean
from config import LOG_EPSILON, ML_GRID_POINTS, KALMAN_DT

logger = logging.getLogger(__name__)


def _compute_poisson_log_likelihoods(
    theta_grid: np.ndarray,
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    duration_s: float
) -> np.ndarray:
    """
    Compute Poisson log-likelihoods over a grid of candidate directions.

    Vectorized: evaluates all (theta, neuron) pairs simultaneously.

    Args:
        theta_grid: 1D array of candidate directions (radians)
        spike_counts: 1D array of spike counts per neuron
        neurons: NeuronPopulation with tuning parameters
        duration_s: Trial duration in seconds

    Returns:
        1D array of log-likelihoods for each candidate direction
    """
    # rates shape: (n_directions, n_neurons)
    rates = cosine_tuning(
        theta_grid[:, np.newaxis],
        neurons.preferred_directions[np.newaxis, :],
        neurons.baseline_rate,
        neurons.modulation_depth
    )
    expected = np.maximum(rates * duration_s, LOG_EPSILON)
    # log P(n|θ) = Σᵢ [nᵢ log(λᵢ) - λᵢ]
    return np.log(expected) @ spike_counts - expected.sum(axis=1)


def _validate_decode_inputs(spike_counts: np.ndarray, neurons: NeuronPopulation) -> None:
    """Validate inputs common to all decoders."""
    if len(spike_counts) == 0:
        raise ValueError("spike_counts must not be empty")
    if len(spike_counts) != neurons.n_neurons:
        raise ValueError(
            f"spike_counts length ({len(spike_counts)}) does not match "
            f"neurons.n_neurons ({neurons.n_neurons})"
        )


class Decoder(ABC):
    """Abstract base class for neural decoders."""

    @abstractmethod
    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        """
        Decode movement direction from spike counts.

        Args:
            spike_counts: Array of spike counts for each neuron
            neurons: NeuronPopulation object with neuron parameters
            duration_s: Trial duration in seconds

        Returns:
            Estimated direction θ̂ in radians [0, 2π)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this decoder."""
        pass


class PopulationVectorDecoder(Decoder):
    """
    Population vector decoder.
    
    Estimates direction by computing the weighted vector sum of
    preferred directions, where weights are spike counts.
    """
    
    @property
    def name(self) -> str:
        return "Population Vector"
    
    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        """
        Decode using population vector method.

        θ̂ = atan2(Σ nᵢ sin(μᵢ), Σ nᵢ cos(μᵢ))

        Args:
            spike_counts: Array of spike counts for each neuron
            neurons: NeuronPopulation object
            duration_s: Trial duration in seconds (unused, for interface compatibility)

        Returns:
            Estimated direction in radians [0, 2π)
        """
        _validate_decode_inputs(spike_counts, neurons)

        # Handle edge case of zero total spikes
        if np.sum(spike_counts) == 0:
            return 0.0
        
        # Compute weighted sum of unit vectors
        x = np.sum(spike_counts * np.cos(neurons.preferred_directions))
        y = np.sum(spike_counts * np.sin(neurons.preferred_directions))
        
        # Handle near-zero magnitude (ambiguous direction)
        if np.abs(x) < 1e-10 and np.abs(y) < 1e-10:
            return 0.0
        
        # Compute angle from vector components
        theta_hat = np.arctan2(y, x)
        
        return wrap_angle(theta_hat)


class MaximumLikelihoodDecoder(Decoder):
    """
    Maximum likelihood decoder assuming independent Poisson neurons.
    
    Computes log-likelihood over a grid of candidate directions
    and returns the direction with maximum likelihood.
    """
    
    def __init__(self, n_grid_points: int = ML_GRID_POINTS):
        """
        Initialize the ML decoder.

        Args:
            n_grid_points: Number of candidate directions to evaluate
        """
        self.n_grid_points = n_grid_points
        self.theta_grid = np.linspace(0, 2 * np.pi, n_grid_points, endpoint=False)
    
    @property
    def name(self) -> str:
        return "Maximum Likelihood"
    
    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        """
        Decode using maximum likelihood estimation.
        
        For Poisson: log P(n|θ) = Σᵢ [nᵢ log λᵢ(θ) - λᵢ(θ)] + const
        
        Args:
            spike_counts: Array of spike counts for each neuron
            neurons: NeuronPopulation object
            duration_s: Trial duration in seconds (default 0.5)
            
        Returns:
            Estimated direction in radians [0, 2π)
        """
        _validate_decode_inputs(spike_counts, neurons)
        log_likelihoods = _compute_poisson_log_likelihoods(
            self.theta_grid, spike_counts, neurons, duration_s
        )
        best_idx = np.argmax(log_likelihoods)
        return self.theta_grid[best_idx]
    
    def get_likelihood_curve(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the full likelihood curve for visualization.
        
        Args:
            spike_counts: Array of spike counts
            neurons: NeuronPopulation object
            duration_s: Trial duration in seconds
            
        Returns:
            Tuple of (theta_grid, normalized_likelihoods)
        """
        log_likelihoods = _compute_poisson_log_likelihoods(
            self.theta_grid, spike_counts, neurons, duration_s
        )

        # Convert to normalized probability (for visualization)
        # Subtract max for numerical stability before exp
        log_likelihoods -= np.max(log_likelihoods)
        likelihoods = np.exp(log_likelihoods)
        likelihoods /= np.sum(likelihoods)  # Normalize
        
        return self.theta_grid, likelihoods


class NaiveBayesDecoder(Decoder):
    """
    Naive Bayes / Maximum a Posteriori decoder.
    
    Similar to ML decoder but incorporates a prior over directions.
    With uniform prior, equivalent to ML decoder.
    """
    
    def __init__(self, n_grid_points: int = 360, prior: Optional[np.ndarray] = None):
        """
        Initialize the Naive Bayes decoder.
        
        Args:
            n_grid_points: Number of candidate directions
            prior: Prior probability over directions (uniform if None)
        """
        self.n_grid_points = n_grid_points
        self.theta_grid = np.linspace(0, 2 * np.pi, n_grid_points, endpoint=False)
        
        if prior is None:
            self.log_prior = np.zeros(n_grid_points)  # Uniform = constant log prior
        else:
            self.log_prior = np.log(prior + 1e-10)
    
    @property
    def name(self) -> str:
        return "Naive Bayes (MAP)"
    
    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        """
        Decode using maximum a posteriori estimation.
        
        θ̂ = argmax P(θ|n) = argmax P(n|θ) P(θ)
        
        Args:
            spike_counts: Array of spike counts
            neurons: NeuronPopulation object
            duration_s: Trial duration in seconds
            
        Returns:
            Estimated direction in radians [0, 2π)
        """
        _validate_decode_inputs(spike_counts, neurons)
        log_likelihoods = _compute_poisson_log_likelihoods(
            self.theta_grid, spike_counts, neurons, duration_s
        )
        log_posteriors = log_likelihoods + self.log_prior

        best_idx = np.argmax(log_posteriors)
        return self.theta_grid[best_idx]


def get_decoder(decoder_name: str) -> Decoder:
    """
    Factory function to get a decoder by name.
    
    Args:
        decoder_name: Name of the decoder ('population_vector', 'ml', 'naive_bayes')
        
    Returns:
        Decoder instance
    """
    decoders = {
        'population_vector': PopulationVectorDecoder,
        'ml': MaximumLikelihoodDecoder,
        'maximum_likelihood': MaximumLikelihoodDecoder,
        'naive_bayes': NaiveBayesDecoder,
        'map': NaiveBayesDecoder
    }
    
    if decoder_name.lower() not in decoders:
        raise ValueError(f"Unknown decoder: {decoder_name}. "
                        f"Available: {list(decoders.keys())}")
    
    return decoders[decoder_name.lower()]()


def evaluate_decoder(
    decoder: Decoder,
    spike_counts: np.ndarray,
    true_directions: np.ndarray,
    neurons: NeuronPopulation,
    duration_s: float = 0.5
) -> dict:
    """
    Evaluate decoder performance on a set of trials.
    
    Args:
        decoder: Decoder instance
        spike_counts: Array of shape (n_trials, n_neurons)
        true_directions: Array of true directions for each trial
        neurons: NeuronPopulation object
        duration_s: Trial duration in seconds
        
    Returns:
        Dictionary with performance metrics
    """
    from utils import angular_error
    
    n_trials = len(true_directions)
    decoded_directions = np.zeros(n_trials)
    errors = np.zeros(n_trials)
    
    for i in range(n_trials):
        decoded_directions[i] = decoder.decode(spike_counts[i], neurons, duration_s=duration_s)
        
        errors[i] = angular_error(true_directions[i], decoded_directions[i])
    
    return {
        'decoded_directions': decoded_directions,
        'errors': errors,
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'median_error': float(np.median(errors)),
        'mean_error_degrees': float(np.mean(errors) * 180 / np.pi),
        'std_error_degrees': float(np.std(errors) * 180 / np.pi)
    }


# =============================================================================
# Kalman Filter Decoder
# =============================================================================

class KalmanFilterDecoder(Decoder):
    """
    Kalman Filter decoder for continuous state estimation.
    
    Uses a state-space model to decode position and velocity from neural activity.
    This is the approach used in real brain-computer interfaces.
    
    State vector: x = [pos_x, pos_y, vel_x, vel_y]
    Observation: y = spike_counts
    
    State equation: x[t] = A @ x[t-1] + w, w ~ N(0, Q)
    Observation equation: y[t] = H @ x[t] + v, v ~ N(0, R)
    """
    
    def __init__(
        self,
        n_neurons: int,
        dt: float = KALMAN_DT,
        process_noise: float = 0.1,
        observation_noise: float = 1.0
    ):
        """
        Initialize the Kalman Filter decoder.
        
        Args:
            n_neurons: Number of neurons
            dt: Time step in seconds
            process_noise: Process noise magnitude
            observation_noise: Observation noise magnitude
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.state_dim = 4  # [pos_x, pos_y, vel_x, vel_y]
        
        # State transition matrix (constant velocity model)
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        self.Q[0, 0] = process_noise * 0.1  # Position has less process noise
        self.Q[1, 1] = process_noise * 0.1
        
        # Observation noise covariance (will be set during fit)
        self.R = np.eye(n_neurons) * observation_noise
        
        # Observation matrix (maps state to expected spike rates)
        # Will be learned during fit()
        self.H = None
        
        # Current state estimate and covariance
        self.x = np.zeros(4)  # [pos_x, pos_y, vel_x, vel_y]
        self.P = np.eye(4) * 1.0  # Initial uncertainty
        
        # Store baseline rates for decoding
        self.baseline_rates = None
        self.preferred_directions = None
        
        # Flag for whether decoder has been fitted
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
        """
        Fit the Kalman Filter parameters from training data.
        
        Args:
            spike_data: Array of shape (n_trials, n_neurons) with spike counts
            kinematics: Array of shape (n_trials, 4) with [pos_x, pos_y, vel_x, vel_y]
            neurons: NeuronPopulation for tuning information
        """
        self.preferred_directions = neurons.preferred_directions
        self.baseline_rates = np.array([
            neurons.baseline_rate for _ in range(neurons.n_neurons)
        ])
        
        # Build observation matrix H that maps velocity to expected spike rates
        # Each neuron's contribution is based on its preferred direction
        n_neurons = len(self.preferred_directions)
        self.H = np.zeros((n_neurons, 4))
        
        for i, mu in enumerate(self.preferred_directions):
            # Neurons encode velocity direction
            # H maps [pos_x, pos_y, vel_x, vel_y] to firing rate
            # We focus on velocity encoding
            self.H[i, 2] = np.cos(mu)  # vel_x contribution
            self.H[i, 3] = np.sin(mu)  # vel_y contribution
        
        # Scale H by modulation depth
        self.H *= neurons.modulation_depth * self.dt
        
        # Estimate observation noise from residuals
        if len(spike_data) > 0 and len(kinematics) > 0:
            predicted_rates = self.H @ kinematics.T
            residuals = spike_data.T - predicted_rates - self.baseline_rates[:, np.newaxis] * self.dt
            self.R = np.cov(residuals) + np.eye(n_neurons) * 0.1
        
        self.is_fitted = True
    
    def fit_from_neurons(self, neurons: NeuronPopulation) -> None:
        """
        Quick fit using just neuron tuning properties (no training data).
        
        Args:
            neurons: NeuronPopulation with preferred directions
        """
        self.preferred_directions = neurons.preferred_directions
        self.baseline_rates = np.full(neurons.n_neurons, neurons.baseline_rate * self.dt)
        
        # Build H matrix from preferred directions
        n_neurons = neurons.n_neurons
        self.H = np.zeros((n_neurons, 4))
        
        for i, mu in enumerate(self.preferred_directions):
            self.H[i, 2] = np.cos(mu) * neurons.modulation_depth * self.dt
            self.H[i, 3] = np.sin(mu) * neurons.modulation_depth * self.dt
        
        self.R = np.eye(n_neurons) * (neurons.baseline_rate * self.dt + 1)
        self.is_fitted = True
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """
        Reset the filter state.
        
        Args:
            initial_state: Optional initial state [pos_x, pos_y, vel_x, vel_y]
        """
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of Kalman filter.
        
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # State prediction
        x_pred = self.A @ self.x
        
        # Covariance prediction
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        return x_pred, P_pred
    
    def update(
        self,
        spike_counts: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step of Kalman filter.
        
        Args:
            spike_counts: Observed spike counts
            x_pred: Predicted state
            P_pred: Predicted covariance
            
        Returns:
            Tuple of (updated_state, updated_covariance, kalman_gain)
        """
        # Expected observation
        y_pred = self.H @ x_pred + self.baseline_rates
        
        # Innovation (measurement residual)
        y = spike_counts.astype(float)
        innovation = y - y_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        try:
            K = P_pred @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance, using pseudoinverse")
            K = P_pred @ self.H.T @ np.linalg.pinv(S)
        
        # State update
        x_new = x_pred + K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        IKH = np.eye(self.state_dim) - K @ self.H
        P_new = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        P_new = (P_new + P_new.T) / 2  # enforce symmetry

        return x_new, P_new, K
    
    def decode_step(self, spike_counts: np.ndarray) -> np.ndarray:
        """
        Perform one decode step (predict + update).
        
        Args:
            spike_counts: Current spike counts
            
        Returns:
            Updated state estimate [pos_x, pos_y, vel_x, vel_y]
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder not fitted. Call fit() or fit_from_neurons() first.")
        
        # Predict
        x_pred, P_pred = self.predict()
        
        # Update
        self.x, self.P, _ = self.update(spike_counts, x_pred, P_pred)
        
        return self.x.copy()
    
    def decode(
        self,
        spike_counts: np.ndarray,
        neurons: NeuronPopulation,
        duration_s: float = 0.5
    ) -> float:
        """
        Decode direction from spike counts (for compatibility with other decoders).

        Args:
            spike_counts: Spike counts per neuron
            neurons: NeuronPopulation (used to fit if not already)
            duration_s: Trial duration in seconds (for unit scaling)

        Returns:
            Decoded direction in radians
        """
        _validate_decode_inputs(spike_counts, neurons)
        if not self.is_fitted:
            self.fit_from_neurons(neurons)

        # Scale full-trial counts to per-bin counts matching Kalman dt
        bin_counts = spike_counts.astype(float) * (self.dt / duration_s)

        # Decode one step
        state = self.decode_step(bin_counts)
        
        # Extract direction from velocity
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
        """
        Decode a trajectory from a sequence of spike counts.
        
        Args:
            spike_sequence: Array of shape (n_timesteps, n_neurons)
            neurons: NeuronPopulation
            
        Returns:
            Tuple of (states, covariances) where states is (n_timesteps, 4)
        """
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
        """
        Get the uncertainty ellipse parameters for position.
        
        Args:
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (semi_major_axis, semi_minor_axis, angle_radians)
        """
        from scipy import stats
        
        # Extract position covariance
        pos_cov = self.P[:2, :2]
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)
        
        # Scale for confidence level (chi-squared distribution with 2 DOF)
        chi2_val = stats.chi2.ppf(confidence, 2)
        
        # Semi-axes
        semi_major = np.sqrt(chi2_val * max(eigenvalues))
        semi_minor = np.sqrt(chi2_val * min(eigenvalues))
        
        # Angle of major axis
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


def compare_decoders(
    spike_counts: np.ndarray,
    true_directions: np.ndarray,
    neurons: NeuronPopulation,
    duration_s: float = 0.5
) -> dict:
    """
    Compare multiple decoders on the same data.
    
    Args:
        spike_counts: Array of shape (n_trials, n_neurons)
        true_directions: True directions for each trial
        neurons: NeuronPopulation
        duration_s: Trial duration
        
    Returns:
        Dictionary with results for each decoder
    """
    from utils import angular_error
    
    decoders = {
        'Population Vector': PopulationVectorDecoder(),
        'Maximum Likelihood': MaximumLikelihoodDecoder(),
        'Kalman Filter': KalmanFilterDecoder(neurons.n_neurons)
    }
    
    results = {}
    
    for name, decoder in decoders.items():
        if isinstance(decoder, KalmanFilterDecoder):
            decoder.fit_from_neurons(neurons)
        
        errors = []
        decoded = []
        
        for i in range(len(true_directions)):
            if isinstance(decoder, KalmanFilterDecoder):
                decoder.reset()
            d = decoder.decode(spike_counts[i], neurons, duration_s=duration_s)
            
            decoded.append(d)
            errors.append(angular_error(true_directions[i], d))
        
        results[name] = {
            'decoded': np.array(decoded),
            'errors': np.array(errors),
            'mean_error_deg': np.mean(errors) * 180 / np.pi,
            'std_error_deg': np.std(errors) * 180 / np.pi
        }
    
    return results

