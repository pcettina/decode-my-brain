"""Color constants and direction-to-color mapping."""

from utils import radians_to_degrees

# Colorblind-safe color scheme
NEURON_COLORSCALE = 'HSL'
TRUE_COLOR = '#3498db'   # Blue for true direction
USER_COLOR = '#2ecc71'   # Green for user guess
MODEL_COLOR = '#e67e22'  # Orange for model decode


def get_direction_color(theta: float) -> str:
    """Get a color based on direction (using HSL color wheel)."""
    hue = (radians_to_degrees(theta) % 360)
    return f'hsl({hue}, 70%, 50%)'
