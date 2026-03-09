"""
Utility functions for the Decode My Brain app.

Contains angle math, unit conversions, and data export helpers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import io


def wrap_angle(theta: float) -> float:
    """
    Wrap an angle to the range [0, 2π).
    
    Args:
        theta: Angle in radians
        
    Returns:
        Angle wrapped to [0, 2π)
    """
    return theta % (2 * np.pi)


def wrap_angle_array(theta: np.ndarray) -> np.ndarray:
    """
    Wrap an array of angles to the range [0, 2π).
    
    Args:
        theta: Array of angles in radians
        
    Returns:
        Angles wrapped to [0, 2π)
    """
    return theta % (2 * np.pi)


def angular_error(theta1: float, theta2: float) -> float:
    """
    Compute the minimum angular distance between two angles.
    
    Args:
        theta1: First angle in radians
        theta2: Second angle in radians
        
    Returns:
        Absolute angular error in radians, in range [0, π]
    """
    diff = wrap_angle(theta1 - theta2)
    # Take the shorter arc
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return abs(diff)


def angular_error_degrees(theta1_deg: float, theta2_deg: float) -> float:
    """
    Compute the minimum angular distance between two angles in degrees.
    
    Args:
        theta1_deg: First angle in degrees
        theta2_deg: Second angle in degrees
        
    Returns:
        Absolute angular error in degrees, in range [0, 180]
    """
    theta1 = degrees_to_radians(theta1_deg)
    theta2 = degrees_to_radians(theta2_deg)
    return radians_to_degrees(angular_error(theta1, theta2))


def degrees_to_radians(deg: float) -> float:
    """
    Convert degrees to radians.
    
    Args:
        deg: Angle in degrees
        
    Returns:
        Angle in radians
    """
    return deg * np.pi / 180.0


def radians_to_degrees(rad: float) -> float:
    """
    Convert radians to degrees.
    
    Args:
        rad: Angle in radians
        
    Returns:
        Angle in degrees
    """
    return rad * 180.0 / np.pi


def circular_mean(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Compute the circular (weighted) mean of angles.
    
    Args:
        angles: Array of angles in radians
        weights: Optional weights for each angle
        
    Returns:
        Circular mean angle in radians [0, 2π)
    """
    if weights is None:
        weights = np.ones_like(angles)
    
    # Convert to unit vectors and compute weighted sum
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    
    # Handle zero-magnitude case
    if np.abs(x) < 1e-10 and np.abs(y) < 1e-10:
        return 0.0
    
    return wrap_angle(np.arctan2(y, x))


def export_to_npz(data: Dict[str, Any], filename: str) -> bytes:
    """
    Export simulation data to NPZ format (in memory).
    
    Args:
        data: Dictionary containing simulation data
        filename: Base filename (not used, for API consistency)
        
    Returns:
        Bytes of the NPZ file
    """
    buffer = io.BytesIO()
    np.savez(buffer, **data)
    buffer.seek(0)
    return buffer.getvalue()


def export_to_csv(data: Dict[str, Any], filename: str) -> str:
    """
    Export simulation data to CSV format.
    
    Args:
        data: Dictionary containing simulation data
        filename: Base filename (not used, for API consistency)
        
    Returns:
        CSV string representation
    """
    # Flatten the data for CSV export
    rows = []
    
    # Handle neuron parameters
    if 'preferred_directions' in data:
        n_neurons = len(data['preferred_directions'])
        for i in range(n_neurons):
            row = {
                'neuron_id': i,
                'preferred_direction_rad': data['preferred_directions'][i],
                'preferred_direction_deg': radians_to_degrees(data['preferred_directions'][i]),
            }
            if 'baseline_rate' in data:
                row['baseline_rate'] = data['baseline_rate']
            if 'modulation_depth' in data:
                row['modulation_depth'] = data['modulation_depth']
            rows.append(row)
    
    # Handle trial data if present
    if 'spike_counts' in data and 'true_directions' in data:
        spike_counts = data['spike_counts']
        true_directions = data['true_directions']
        
        for trial_idx in range(len(true_directions)):
            for neuron_idx in range(spike_counts.shape[1]):
                rows.append({
                    'trial_id': trial_idx,
                    'neuron_id': neuron_idx,
                    'true_direction_rad': true_directions[trial_idx],
                    'true_direction_deg': radians_to_degrees(true_directions[trial_idx]),
                    'spike_count': spike_counts[trial_idx, neuron_idx]
                })
    
    if rows:
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    else:
        return ""


def format_angle_display(angle_rad: float, use_degrees: bool = True) -> str:
    """
    Format an angle for display.
    
    Args:
        angle_rad: Angle in radians
        use_degrees: If True, display in degrees; otherwise radians
        
    Returns:
        Formatted string representation
    """
    if use_degrees:
        return f"{radians_to_degrees(angle_rad):.1f}°"
    else:
        return f"{angle_rad:.3f} rad"


def generate_direction_labels(n_points: int = 8) -> tuple:
    """
    Generate direction labels for polar plots (N, NE, E, SE, S, SW, W, NW).
    
    Args:
        n_points: Number of direction labels (default 8 for cardinal + ordinal)
        
    Returns:
        Tuple of (angles in degrees, labels)
    """
    if n_points == 8:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        labels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    elif n_points == 4:
        angles = [0, 90, 180, 270]
        labels = ['E', 'N', 'W', 'S']
    else:
        angles = list(np.linspace(0, 360, n_points, endpoint=False))
        labels = [f"{int(a)}°" for a in angles]
    
    return angles, labels

