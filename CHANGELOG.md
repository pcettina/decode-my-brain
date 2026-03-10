# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-03-09

Complete rebuild from initial prototype through a 35-step improvement plan.

### Added

- **Git repository** with `.gitignore`, CI/CD pipeline, and Docker support
- **35-step improvement plan** (PROJECT_AUDIT_REPORT.md) covering security, code quality, architecture, scientific correctness, UX, performance, testing, and DevOps
- **273-test suite** covering all modules: simulation, decoders, challenges, visualization, engine, config, utils, pages, and integration
- **CI/CD pipeline** via GitHub Actions (`ruff check` + `pytest --cov-fail-under=60`)
- **Docker deployment** with `Dockerfile`, `.dockerignore`, and health check
- **Streamlit config** (`.streamlit/config.toml`) with XSRF protection and no CORS
- **Centralized config** (`config.py`) for all magic numbers (LOG_EPSILON, ML_GRID_POINTS, KALMAN_DT, etc.)
- **Game engine** (`engine/game.py`) with `GameEngine`, `GameResult`, and `BCISimulator` classes extracted from app.py
- **Leaderboard persistence** via `data/leaderboards.json` with automatic save/load
- **Scoring formula display** before challenge start and score breakdown after completion
- **Input validation** on simulation parameters, decoder inputs, and challenge state
- **Error boundaries** (`try/except`) around simulation, decoding, and PCA computations
- **Logging** via `logging` module throughout all source files
- **Auto-generated demo** on first load (50 neurons, 20 trials, seed=42)
- **Multipage navigation** using `st.navigation` with 5 grouped pages
- **Stale state clearing** when re-simulating — clears derived data (hierarchy, manifold, game, BCI)

### Fixed

- **`np.random.seed()` global state** replaced with `np.random.default_rng()` + child-seed threading for reproducibility without side effects
- **Negative binomial +1 bias** — changed from `n + 1` to `np.maximum(1, np.round(n).astype(int))`
- **Kalman covariance update** — switched from naive `(I - KH)P` to Joseph form `(I-KH)P(I-KH)^T + KRK^T` with symmetry enforcement
- **Kalman observation units mismatch** — added `bin_counts = spike_counts * (dt / duration_s)` scaling
- **Raster variance branching** — added negative binomial path for `variance_scale > 1.0` (was Poisson-only)
- **Burst clip order** — moved clip before burst override in `simulate_continuous_activity`
- **`__code__.co_varnames` introspection** — replaced with unified `duration_s` parameter on Decoder ABC
- **Log-likelihood duplication** (4 copies) — consolidated into single `_compute_poisson_log_likelihoods()` vectorized function
- **`tuning_sharpness` misnomer** — renamed to `modulation_gain` throughout

### Changed

- **Architecture**: split 3 monolithic files into subpackages
  - `simulation.py` (996 lines) -> `simulation/core.py`, `temporal.py`, `hierarchy.py`
  - `decoders.py` (736 lines) -> `decoders/base.py`, `direction.py`, `kalman.py`, `evaluation.py`
  - `visualization.py` (1,780 lines) -> `visualization/colors.py`, `tuning.py`, `raster.py`, `bci.py`, `walkthrough.py`, `manifold.py`, `network.py`, `analysis.py`
- **App layout**: converted from 10-tab `st.tabs` to `st.navigation` with 5 pages (Setup, Learn, Play, Explore, Analyze)
- **Simulation controls** moved to sidebar (accessible from any page)
- **Dependencies** pinned with `~=` (compatible release): `streamlit~=1.36`, `numpy~=1.24`, etc.
- **Colors**: switched to colorblind-safe palette (blue/green/orange instead of green/red)
- **Cosine tuning call sites** vectorized (4 list comprehensions replaced with array operations)
- **Temporal simulation** deduplicated via shared `_temporal_step()` helper
- **Temporal inner loop** vectorized with `np.where()` instead of per-neuron Python loop
- **Plotly traces** optimized: tuning curves grouped into 8 direction bins; rasters use single trace with per-point colors
- **Game direction input** consolidated from dual slider+number_input to single number_input
- **KalmanFilterDecoder** now inherits from `Decoder` ABC
- **10 visualization functions** cached with `@st.cache_data`

### Removed

- Dead code: `compute_isi_statistics`, `create_animated_raster_frames`, `plot_challenge_progress`, `plot_achievement_badges`, unused imports
- `NaiveBayesDecoder` from UI (kept in codebase for potential future use)
