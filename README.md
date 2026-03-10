# Decode My Brain

Interactive neuroscience education app for exploring neural population coding, decoding, and brain-computer interfaces through gamified challenges.

Built with Streamlit, NumPy, SciPy, and Plotly.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/decode-my-brain.git
cd decode-my-brain

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Opens at `http://localhost:8501`. A demo simulation (50 neurons, 20 trials) loads automatically.

### Docker

```bash
docker build -t decode-my-brain .
docker run -p 8501:8501 decode-my-brain
```

## What It Does

Simulates direction-tuned neural populations using cosine tuning curves and lets users explore how the brain encodes and decodes movement direction.

**Core model:** Each neuron fires according to `lambda(theta) = r0 + k * cos(theta - mu)`, where `r0` is baseline rate, `k` is modulation depth, and `mu` is the neuron's preferred direction. Spikes are generated via Poisson process with optional overdispersion (negative binomial) or underdispersion (clipped Gaussian).

## Features

### 10 Interactive Tabs

| Tab | What it does |
|-----|-------------|
| **Setup** | View simulation summary stats (controls are in the sidebar) |
| **Visualize** | Tuning curves, spike heatmaps, population bar/polar plots |
| **Learn Decoding** | Step-by-step walkthrough of Population Vector and ML decoders |
| **Game** | Guess movement direction from spike patterns, compete against the model |
| **Challenges** | 5 timed/scored modes: Speed Trial, Precision, Noise Gauntlet, Streak, Area Expert |
| **Live Activity** | Real-time temporal spike dynamics with adaptation, refractory periods, bursting |
| **BCI Simulator** | Kalman filter-based cursor control from neural activity |
| **Brain Areas** | Hierarchical network (M1, PMd, PPC, SMA) with inter-area connectivity |
| **Neural Manifold** | PCA dimensionality reduction of population activity |
| **Analysis** | Noise vs. performance curves, normal vs. lesioned populations, data export |

### Decoders

- **Population Vector** -- weighted vector sum of preferred directions
- **Maximum Likelihood** -- Poisson log-likelihood maximization over direction grid (vectorized)
- **Kalman Filter** -- state-space model for continuous BCI decoding with Joseph-form covariance update

### Simulation Features

- Configurable population size (10-200 neurons), firing rates, modulation depth
- Poisson, negative binomial (overdispersed), or Gaussian (underdispersed) spike generation
- Temporal dynamics: spike-rate adaptation, absolute/relative refractory periods, burst firing
- Hierarchical brain network with feedforward/feedback connections
- Reproducible via `np.random.default_rng()` with child-seed threading

## Project Structure

```
decode-my-brain/
├── app.py                  # Streamlit UI (10-tab interface, sidebar controls)
├── simulation.py           # Neural population modeling & spike generation
├── decoders.py             # Decoder implementations (PV, ML, Kalman)
├── visualization.py        # Plotly interactive plots
├── challenges.py           # Gamification (5 modes, scoring, leaderboards)
├── utils.py                # Circular angle math, data export
├── config.py               # Centralized constants
├── requirements.txt        # Production dependencies (pinned with ~=)
├── requirements-dev.txt    # Dev dependencies (pytest, ruff)
├── runtime.txt             # Python version specification
├── Dockerfile              # Container deployment
├── .dockerignore
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit server config
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions (lint + test)
└── tests/
    ├── conftest.py          # Shared fixtures
    ├── test_utils.py        # 5 tests: angle math
    ├── test_simulation.py   # 5 tests: tuning, population, trials
    ├── test_decoders.py     # 5 tests: PV, ML, Kalman, evaluation
    ├── test_challenges.py   # 3 tests: scoring, lifecycle
    ├── test_visualization.py # 1 test: plot smoke test
    └── test_integration.py  # 1 test: end-to-end pipeline
```

## Development

### Install dev dependencies

```bash
pip install -r requirements-dev.txt
```

### Run tests

```bash
pytest tests/ -v
```

20 tests covering utils, simulation, decoders, challenges, visualization, and end-to-end integration.

### Lint

```bash
ruff check .
```

### CI

GitHub Actions runs `ruff check` and `pytest` on every push and pull request.

## Requirements

- Python 3.12+
- Streamlit ~= 1.28
- NumPy ~= 1.24
- SciPy ~= 1.10
- Plotly ~= 5.18
- Pandas ~= 2.0
- scikit-learn ~= 1.3

## For Educators

1. **Start simple** -- begin with 20-30 neurons to see clear tuning patterns
2. **Explore tuning** -- use the Visualize tab to explain population coding
3. **Add noise** -- increase variance scale to show how noise degrades decoding
4. **Compare decoders** -- use Learn Decoding to walk through PV vs ML step by step
5. **Lesion effects** -- Analysis tab demonstrates population coding redundancy
6. **Game mode** -- let students compete to internalize the decoding challenge
7. **BCI demo** -- show how Kalman filtering enables real-time cursor control

## License

MIT License
