# Decode My Brain

Interactive neuroscience education app for exploring neural population coding, decoding, and brain-computer interfaces through gamified challenges.

Built with Streamlit, NumPy, SciPy, and Plotly.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/pcettina/decode-my-brain.git
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

### 5 Pages (Multipage Navigation)

| Page | What it does |
|------|-------------|
| **Setup** | View simulation summary stats (controls are in the sidebar) |
| **Learn** | Tuning curves, spike heatmaps, population bar/polar plots; step-by-step PV and ML decoder walkthrough |
| **Play** | Guess movement direction from spike patterns; 4 challenge modes (Speed Trial, Precision, Noise Gauntlet, Streak) with leaderboards |
| **Explore** | Live temporal spike dynamics; BCI cursor control simulator; hierarchical brain network (M1, PMd, PPC, SMA); PCA neural manifold |
| **Analyze** | Noise vs. performance curves, normal vs. lesioned populations, data export (NPZ/CSV) |

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
├── app.py                      # Streamlit entry point (multipage nav, sidebar controls)
├── config.py                   # Centralized constants
├── challenges.py               # Game modes, scoring, achievements, leaderboards
├── utils.py                    # Circular angle math, data export
├── pages/
│   ├── setup.py                # Simulation overview
│   ├── learn.py                # Tuning curve visualization + decoder walkthrough
│   ├── play.py                 # Game mode + challenge modes
│   ├── explore.py              # Live activity, BCI, brain areas, neural manifold
│   └── analyze.py              # Noise analysis, lesion comparison, data export
├── simulation/
│   ├── __init__.py             # Re-exports all public symbols
│   ├── core.py                 # NeuronPopulation, tuning, spike generation
│   ├── temporal.py             # Temporal dynamics (adaptation, refractory, bursting)
│   └── hierarchy.py            # BrainArea, HierarchicalNetwork
├── decoders/
│   ├── __init__.py             # Re-exports all public symbols
│   ├── base.py                 # Decoder ABC, shared log-likelihood computation
│   ├── direction.py            # PopulationVector, MaximumLikelihood, NaiveBayes
│   ├── kalman.py               # KalmanFilterDecoder (state-space model)
│   └── evaluation.py           # evaluate_decoder, compare_decoders
├── visualization/
│   ├── __init__.py             # Re-exports all public symbols
│   ├── colors.py               # Color constants, direction-based HSL mapping
│   ├── tuning.py               # Tuning curves, population bar/polar
│   ├── raster.py               # Spike heatmaps, temporal raster snapshots
│   ├── bci.py                  # BCI canvas, metrics display
│   ├── walkthrough.py          # Step-by-step PV/ML decoder visualizations
│   ├── manifold.py             # PCA, neural manifold 2D/3D
│   ├── network.py              # Brain connectivity, area comparison, leaderboard
│   └── analysis.py             # Noise curves, condition comparison, likelihood
├── engine/
│   ├── __init__.py
│   └── game.py                 # GameEngine, GameResult, BCISimulator
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_config.py          # 8 tests: centralized constants
│   ├── test_utils.py           # 30 tests: angle math, exports, formatting
│   ├── test_simulation.py      # 67 tests: tuning, populations, trials, temporal, hierarchy
│   ├── test_decoders.py        # 47 tests: PV, ML, NaiveBayes, Kalman, evaluation
│   ├── test_challenges.py      # 64 tests: scoring, lifecycle, leaderboard I/O, achievements
│   ├── test_visualization.py   # 52 tests: plot smoke tests, edge cases, manifold, colors
│   ├── test_engine.py          # 14 tests: game logic, BCI simulator
│   ├── test_integration.py     # 10 tests: end-to-end pipelines
│   └── test_pages.py           # 5 tests: Streamlit AppTest page smoke tests
├── requirements.txt            # Production dependencies (pinned with ~=)
├── requirements-dev.txt        # Dev dependencies (pytest, pytest-cov, ruff)
├── runtime.txt                 # Python version specification
├── Dockerfile                  # Container deployment
├── .dockerignore
├── .gitignore
├── .streamlit/
│   └── config.toml             # Streamlit server config
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions (lint + test + coverage)
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

297 tests covering config, utils, simulation, decoders, challenges, visualization, engine, page rendering, and end-to-end integration.

### Lint

```bash
ruff check .
```

### CI

GitHub Actions runs `ruff check`, `pytest` with coverage (`--cov-fail-under=60`), on every push and pull request.

## Requirements

- Python 3.12+
- Streamlit ~= 1.36
- NumPy ~= 1.24
- SciPy ~= 1.10
- Plotly ~= 5.18
- Pandas ~= 2.0
- scikit-learn ~= 1.3

## For Educators

1. **Start simple** -- begin with 20-30 neurons to see clear tuning patterns
2. **Explore tuning** -- use the Learn page to explain population coding
3. **Add noise** -- increase variance scale to show how noise degrades decoding
4. **Compare decoders** -- use the Learn Decoding tab to walk through PV vs ML step by step
5. **Lesion effects** -- Analyze page demonstrates population coding redundancy
6. **Game mode** -- let students compete to internalize the decoding challenge
7. **BCI demo** -- Explore page shows how Kalman filtering enables real-time cursor control

## License

MIT License
