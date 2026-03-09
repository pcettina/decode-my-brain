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

Six-module flat architecture (no subdirectories):

| Module | Purpose |
|---|---|
| `app.py` | Streamlit UI — 7-tab interface (Simulation, Visualize, Game, Analysis, Networks, Advanced Decoders, Achievements). Orchestrates all other modules. |
| `simulation.py` | Neural population modeling: cosine tuning curves, Poisson spike generation, temporal dynamics (adaptation, refractory, bursting), hierarchical brain networks (`BrainArea`, `HierarchicalNetwork`). |
| `decoders.py` | Abstract `Decoder` base class with implementations: Population Vector, Maximum Likelihood, Naive Bayes/MAP, Kalman Filter. Factory via `get_decoder(name)`. |
| `visualization.py` | Plotly-based interactive plots: tuning curves, rasters, polar comparisons, neural manifolds (PCA), likelihood surfaces, BCI displays, connectivity matrices. |
| `challenges.py` | Gamification: 5 modes (Speed Trial, Precision, Noise Gauntlet, Streak, Area Expert). `ChallengeManager` handles state/scoring, `AchievementManager` tracks 10 achievements. |
| `utils.py` | Circular angle math (`wrap_angle`, `angular_error`, `circular_mean`), data export (NPZ/CSV), direction labeling. |

## Key Design Patterns

- **Dataclasses** for immutable config: `NeuronPopulation`, `TemporalParams`, `BrainArea`, `ChallengeConfig`, `ChallengeResult`
- **ABC + Factory**: `Decoder` base class, `get_decoder()` factory function
- **Enum**: `ChallengeMode` for type-safe challenge selection
- **Streamlit session state** for challenge persistence; `@st.cache_resource` for expensive computations

## Scientific Conventions

- Angular errors in **degrees**, firing rates in **Hz**, durations in **ms**
- Core tuning model: `λ(θ) = r₀ + k·cos(θ - μ)` (cosine tuning)
- Spike generation: Poisson process with variance scaling options (Poisson, negative binomial, underdispersed)
- Decoders operate on spike count vectors across the population

## Visualization Stack

This project uses **Plotly** (not matplotlib) for all interactive visualizations. Maintain Plotly conventions when adding or modifying plots.

## Tech Stack

Streamlit 1.28+, NumPy 1.24+, SciPy 1.10+, Plotly 5.18+, Pandas 2.0+, scikit-learn 1.3.0+. Python 3.8+.

## Current Gaps

- No test suite — add `pytest` tests when modifying core logic
- No CI/CD pipeline
- Several modules exceed 600 lines (`app.py`, `simulation.py`, `visualization.py`) — consider splitting when making significant additions
- Uses `print`/`st.write` instead of `logging` module for diagnostics
