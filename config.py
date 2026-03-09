"""
Centralized configuration constants for Decode My Brain.

All magic numbers and default parameters in one place.
"""

import numpy as np

# Numerical stability
LOG_EPSILON = 1e-10

# Decoder settings
ML_GRID_POINTS = 360
KALMAN_DT = 0.05

# Temporal dynamics constants
ADAPTATION_INCREMENT = 0.2
BURST_INITIATION_PROB = 0.3
MAX_SPIKE_PROB_PER_BIN = 0.5

# BCI simulator
BCI_TARGET_RADIUS = 20
BCI_CANVAS_SIZE = 200.0

# Visualization
DIRECTION_HIGHLIGHT_THRESHOLD = np.pi / 8
RASTER_PX_PER_NEURON = 8
RASTER_MIN_HEIGHT = 400

# Leaderboard
MAX_LEADERBOARD_ENTRIES = 10

# Default simulation parameters
DEFAULT_N_NEURONS = 50
DEFAULT_DURATION_MS = 500
DEFAULT_BASELINE_RATE = 5.0
DEFAULT_MODULATION_DEPTH = 15.0
DEFAULT_VARIANCE_SCALE = 1.0
DEFAULT_N_TRIALS = 20
