# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Decode My Brain** — an interactive Streamlit web app for neuroscience education. Simulates direction-tuned neural populations and teaches population coding, decoding, and brain-computer interfaces through gamified challenges.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

No external data files required — all neural activity is procedurally generated.

## Architecture

Multipage Streamlit app using `st.navigation` with subpackage modules:

### Entry Point & Pages

| File | Purpose |
|---|---|
| `app.py` | Thin orchestrator: page config, session state init, sidebar controls, `st.navigation` routing. |
| `pages/setup.py` | Simulation overview and parameter summary. |
| `pages/learn.py` | Visualize neural code (tuning curves, rasters) + step-by-step decoder walkthrough. |
| `pages/play.py` | Game mode (decode vs model) + challenge modes (speed, precision, noise, streak). |
| `pages/explore.py` | Live activity, BCI simulator, multi-brain-area simulation, neural manifold (PCA). |
| `pages/analyze.py` | Noise vs performance analysis, normal vs lesioned comparison, data export. |

### Core Modules

| Module | Purpose |
|---|---|
| `simulation/` | Subpackage: `core.py` (NeuronPopulation, cosine tuning, spike generation), `temporal.py` (adaptation, refractory, bursting), `hierarchy.py` (BrainArea, HierarchicalNetwork). |
| `decoders/` | Subpackage: `base.py` (Decoder ABC), `direction.py` (PopulationVector, MaximumLikelihood, NaiveBayes), `kalman.py` (KalmanFilter), `evaluation.py` (evaluate/compare). |
| `visualization/` | Subpackage: `colors.py`, `tuning.py`, `raster.py`, `bci.py`, `walkthrough.py`, `manifold.py`, `network.py`, `analysis.py`. |
| `engine/` | Game logic: `game.py` (GameEngine, GameResult, BCISimulator). |
| `challenges.py` | Gamification: 4 modes (Speed, Precision, Noise Gauntlet, Streak). ChallengeManager + AchievementManager. |
| `utils.py` | Circular angle math, data export (NPZ/CSV), direction labeling. |
| `config.py` | Centralized constants (LOG_EPSILON, ML_GRID_POINTS, KALMAN_DT, etc.). |

All subpackages use `__init__.py` re-exports so `from simulation import X` works.

## Key Design Patterns

- **Dataclasses** for immutable config: `NeuronPopulation`, `TemporalParams`, `BrainArea`, `ChallengeConfig`, `ChallengeResult`
- **ABC + Factory**: `Decoder` base class, `get_decoder()` factory function
- **Enum**: `ChallengeMode` for type-safe challenge selection
- **Streamlit session state** for challenge persistence; `@st.cache_data` for pure visualization functions
- **Multipage navigation**: `st.navigation` with grouped pages (Getting Started, Interactive, Tools)

## Scientific Conventions

- Angular errors in **degrees**, firing rates in **Hz**, durations in **ms**
- Core tuning model: `λ(θ) = r₀ + k·cos(θ - μ)` (cosine tuning)
- Spike generation: Poisson process with variance scaling options (Poisson, negative binomial, underdispersed)
- Decoders operate on spike count vectors across the population

## Visualization Stack

This project uses **Plotly** (not matplotlib) for all interactive visualizations. Maintain Plotly conventions when adding or modifying plots.

## Tech Stack

Streamlit 1.36+, NumPy 1.24+, SciPy 1.10+, Plotly 5.18+, Pandas 2.0+, scikit-learn 1.3.0+. Python 3.8+.

## Testing

- 273 tests in `tests/` — run with `pytest tests/ -v`
- CI via `.github/workflows/ci.yml` (ruff lint + pytest)
- Coverage: simulation, decoders, challenges, visualization, utils, config, engine, integration
