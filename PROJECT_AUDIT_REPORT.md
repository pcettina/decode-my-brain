# Decode My Brain -- Comprehensive Project Audit Report

**Generated**: 2026-03-09T12:32:15-05:00
**Codebase**: 6 modules, ~6,360 lines of application code
**Audit Scope**: Security, Code Quality, Architecture, Scientific Correctness, UX/UI, Performance, Testing, DevOps

---

## Executive Summary

**Decode My Brain** is a well-conceived educational neuroscience app with strong scientific foundations and a clean modular structure. However, it has significant gaps across infrastructure, performance, and reliability that must be addressed before production deployment. The codebase has **zero tests, zero CI/CD, no git repository, no caching, and no logging**. Several scientific implementation bugs (notably in the negative binomial spike model and Kalman filter) would produce silently incorrect results for users.

### Severity Distribution Across All Audits

| Severity | Count | Key Themes |
|----------|-------|------------|
| CRITICAL | 6 | No git repo; no caching; ML decoder Python loops; NB spike model bias; global RNG mutation; no test suite |
| HIGH | 24 | Kalman filter bugs; stale state; module oversize; code duplication; colorblind-unsafe plots; no onboarding; tab overload |
| MEDIUM | 42 | Magic numbers; missing validation; scoring opacity; mobile layout; scattered config; ephemeral leaderboards |
| LOW | 29 | Minor naming, dead code, import organization, accessibility polish |
| INFO | 14 | Confirmed-correct implementations (cosine tuning, PV decoder, angular error, etc.) |

---

## SECTOR 1: SECURITY

**Overall Risk: LOW** (for local/classroom use) / **MEDIUM** (for public deployment)

### Critical & High Findings

| ID | Severity | Finding | Location |
|----|----------|---------|----------|
| SEC-1 | HIGH | No authentication -- any user on the network can consume compute resources | Entire app |
| SEC-2 | MEDIUM | Unpinned dependency versions (`>=` with no upper bound) expose to future CVEs | `requirements.txt` |
| SEC-3 | MEDIUM | Unhandled exceptions leak full filesystem paths in stack traces | `app.py` (no try/except) |
| SEC-4 | MEDIUM | Bounded but non-trivial compute budget -- 200 neurons x 100 trials x 8 noise levels x 360 grid points | `app.py:1560-1613` |

### Safe Areas (Confirmed)

- No `eval()`, `exec()`, `pickle`, or dynamic imports anywhere
- No hardcoded secrets, API keys, or credentials
- Data exports use in-memory buffers (no path traversal risk)
- Streamlit enforces slider bounds server-side
- `unsafe_allow_html=True` used only with static CSS (no user data interpolated)
- WebSocket transport is inherently CSRF-resistant

### Recommendations

1. For public deployment: add authentication (Streamlit built-in auth or reverse proxy)
2. Set `server.showErrorDetails = false` in `.streamlit/config.toml`
3. Pin dependency versions with upper bounds; add `pip-audit` to CI
4. Add `max_chars=50` to player name `st.text_input`
5. Cap BCI trail list to 500 entries (currently unbounded)

---

## SECTOR 2: CODE QUALITY & BEST PRACTICES

**Overall Grade: C+** -- Good docstrings and type hints, but oversized modules, pervasive duplication, and zero logging

### Critical Findings

| ID | Severity | Finding | Location |
|----|----------|---------|----------|
| CQ-1 | CRITICAL | `np.random.seed()` mutates global RNG state (7 occurrences) -- breaks reproducibility and concurrency | `simulation.py:81,143,205,254,291,416,525` |
| CQ-2 | HIGH | `visualization.py` is 1,963 lines (3x guideline of 600) | `visualization.py` |
| CQ-3 | HIGH | `app.py` is 1,763 lines (3x guideline) | `app.py` |
| CQ-4 | HIGH | `simulation.py` is 1,008 lines (1.7x guideline) | `simulation.py` |
| CQ-5 | HIGH | Poisson log-likelihood loop copy-pasted 3 times | `decoders.py:130-148`, `decoders.py:170-188`, `decoders.py:240-252` |
| CQ-6 | HIGH | Temporal simulation code duplicated between two functions (~80% shared) | `simulation.py:389-496` vs `simulation.py:499-599` |
| CQ-7 | HIGH | `decoder.decode.__code__.co_varnames` introspection hack for parameter dispatch | `decoders.py:308,700` |

### Medium Findings

| ID | Finding | Location |
|----|---------|----------|
| CQ-8 | Zero `import logging` in entire codebase | All files |
| CQ-9 | Magic numbers: adaptation increment `0.2`, burst prob `0.3`, spike cap `0.5`, target radius `20`, epsilon `1e-10` | Multiple files |
| CQ-10 | `init_session_state()` is 100 lines of repetitive boilerplate | `app.py:137-237` |
| CQ-11 | Dead code: `NaiveBayesDecoder` never used in app, `get_decoder()` factory never called, `compute_isi_statistics()` never called, unused imports (`pandas`, `Optional`, `BrainArea`, `plotly.express`) | Multiple files |
| CQ-12 | Type hint errors: `Dict[str, any]` (lowercase `any`), missing `Callable` annotation on `theta_func` | `challenges.py:331`, `simulation.py:499` |
| CQ-13 | Internal function imports scattered inline instead of at module top | `decoders.py:301,679`, `visualization.py:1160` |

### Top 5 Remediation Priorities

1. **Replace `np.random.seed()` with `np.random.default_rng(seed)`** -- global state mutation is the single most impactful correctness issue
2. **Split oversized modules** -- `visualization/` subpackage, `tabs/` subpackage, `simulation/` subpackage
3. **Extract shared log-likelihood computation** -- deduplicate the 3 copies into one method
4. **Fix decoder interface** -- standardize `decode()` signature with `duration_s` as optional kwarg; eliminate `__code__.co_varnames` hack
5. **Add logging** -- `import logging; logger = logging.getLogger(__name__)` in every module

---

## SECTOR 3: ARCHITECTURE & DESIGN

**Overall Grade: B-** -- Clean dependency graph and good ABC usage, but business logic leaks into UI layer and stale state bugs lurk

### Critical & High Findings

| ID | Severity | Finding | Impact |
|----|----------|---------|--------|
| AR-1 | HIGH | Business logic (error computation, winner determination, area decoding) embedded in `app.py` | Cannot unit-test game logic without Streamlit |
| AR-2 | HIGH | PCA computation lives in `visualization.py` instead of an analysis module | Coupled analysis and rendering |
| AR-3 | HIGH | `KalmanFilterDecoder` does NOT inherit from `Decoder` ABC -- breaks polymorphism | Every consumer must special-case it with `isinstance` checks |
| AR-4 | HIGH | Stale derived state after re-simulation -- manifold, hierarchy, walkthrough data reference OLD population | Misleading visualizations without warning |
| AR-5 | HIGH | `co_varnames` introspection hack for decoder dispatch | Breaks with any decorator, wrapper, or signature change |

### Design Pattern Assessment

| Pattern | Status | Notes |
|---------|--------|-------|
| Strategy (Decoder ABC) | Well implemented | Clean abstract base with factory |
| Factory (`get_decoder`) | Defined but unused | All call sites construct directly |
| Dataclasses | Excellent | `NeuronPopulation`, `TemporalParams`, `BrainArea`, `ChallengeConfig`, `ChallengeResult` |
| State management | Fragile | No invalidation cascade when params change |
| Observer/Event | Missing | Would solve stale state problem |
| Configuration | Missing | Magic numbers scattered across files; no centralized config |

### Recommended Architecture Improvements

1. **Extract business logic from `app.py`** into a `GameEngine` class -- `engine.submit_guess(user_deg, decoder_choice) -> GameResult`
2. **Move PCA/manifold to `analysis.py`** -- visualization should only receive pre-computed data
3. **Make `KalmanFilterDecoder` inherit from `Decoder`** or create a `StatefulDecoder` subclass with `fit()`/`reset()` lifecycle
4. **Add state invalidation** -- when "Simulate" is pressed, clear all derived state (manifold, hierarchy, walkthrough)
5. **Centralize configuration** -- create `config.py` with all tunable constants (target radius, grid points, epsilon, adaptation params)
6. **Adopt factory pattern consistently** -- use `get_decoder()` everywhere instead of direct construction

### Future Structure (when > 8 files)

```
neurovis/
    __init__.py
    app.py              (thin orchestrator)
    config.py           (centralized constants)
    models/
        simulation.py, temporal.py, hierarchy.py
    decoders/
        base.py, direction.py, kalman.py
    views/
        tuning.py, raster.py, bci.py, manifold.py, walkthrough.py
    engine/
        game.py, bci_simulator.py
    tabs/
        simulation_tab.py, visualize_tab.py, game_tab.py, ...
    utils.py
    challenges.py
```

---

## SECTOR 4: SCIENTIFIC CORRECTNESS

**Overall Grade: C** -- Core tuning and PV decoder are correct, but multiple implementation bugs in advanced models would produce silently wrong results

### Critical & High Findings

| ID | Severity | Finding | Scientific Impact |
|----|----------|---------|-------------------|
| SCI-1 | CRITICAL | Negative binomial `n.astype(int) + 1` introduces systematic positive bias to mean spike counts | Overcounting corrupts any overdispersed analysis. Anti-preferred neurons (low expected count) are disproportionately affected. |
| SCI-2 | HIGH | Kalman filter covariance update uses naive form `P = (I-KH)P_pred` instead of Joseph stabilized form | Covariance can lose positive-semidefiniteness over many iterations, causing divergent estimates |
| SCI-3 | HIGH | Kalman observation model units mismatch -- H matrix scaled by `dt=0.05s` but `decode()` receives full-trial counts (~500ms) | Factor-of-10 mismatch produces erratic Kalman estimates in single-trial compatibility mode |
| SCI-4 | HIGH | `simulate_raster` always uses Gaussian for `variance_scale != 1.0`, while `simulate_trial` uses negative binomial for `>1` | Raster and trial data come from different generative models under overdispersion |
| SCI-5 | HIGH | `tuning_sharpness` in hierarchical network only scales modulation depth (amplitude), not bandwidth | Parameter name is misleading; does not implement actual sharper tuning (would need von Mises model) |
| SCI-6 | HIGH | Global `np.random.seed()` usage across all simulation functions | Non-reproducible in multi-session deployments; fragile call-chain seeding |

### Medium Findings

| ID | Finding | Location |
|----|---------|----------|
| SCI-7 | PV decoder returns `0.0` for ambiguous cases (zero spikes, antipodal cancellation) instead of NaN | `decoders.py:73-74,80-82` |
| SCI-8 | `variance_scale` used as rate multiplier in temporal model but as dispersion in `simulate_trial` -- inconsistent semantics | `simulation.py:468` |
| SCI-9 | `simulate_continuous_activity` clips burst probability to 0.5 AFTER setting it to 0.9, defeating the burst mechanism | `simulation.py:582-583` |
| SCI-10 | Hierarchical feedforward ordering is snapshot-inconsistent -- M1 gets unmodulated SMA but PPC-modulated PMd | `simulation.py:921-931` |
| SCI-11 | Connection weights can go negative for "excitatory" feedforward connections | `simulation.py:800-843` |
| SCI-12 | `std_error` uses linear standard deviation on angular errors instead of circular variance | `decoders.py:evaluate_decoder` |
| SCI-13 | Kalman `fit()` assumes linear observation model but generative model uses rectified cosine -- inflates R | `decoders.py:437-439` |

### Confirmed Correct Implementations

- Cosine tuning model `lambda(theta) = r0 + k*cos(theta - mu)` with proper rectification
- Population Vector decoder circular mean formula
- ML decoder Poisson log-likelihood with numerically stable log-sum-exp
- Angular error shortest-arc computation
- Adaptation/refractory period values (biologically plausible)
- Naive Bayes posterior = log-likelihood + log-prior

### Top Priority Scientific Fixes

1. **Fix negative binomial parameterization** -- use `np.maximum(1, np.round(n).astype(int))` instead of `int + 1`
2. **Fix Kalman covariance update** -- use Joseph form: `IKH = I - K@H; P = IKH @ P_pred @ IKH.T + K @ R @ K.T`
3. **Fix Kalman observation units** -- rescale spike counts to per-bin in `decode()` compatibility method
4. **Fix raster variance branching** -- mirror `simulate_trial`'s three-path logic
5. **Rename `tuning_sharpness` to `modulation_gain`** or implement von Mises tuning for actual bandwidth control
6. **Fix burst clip order in `simulate_continuous_activity`** -- move clip before burst override

---

## SECTOR 5: UX / UI DESIGN

**Overall Grade: C** -- Functional but overwhelming for target audience; strong educational content buried under navigational complexity

### Critical & High Findings

| ID | Severity | Finding | User Impact |
|----|----------|---------|-------------|
| UX-1 | HIGH | 10 tabs exceed cognitive limits (Miller's Law: 5-7 items) | First-time users face choice paralysis; right-end tabs may never be discovered |
| UX-2 | HIGH | No guided onboarding or interactive tutorial | Target audience (students) sees abstract math before any concrete example |
| UX-3 | HIGH | `st.rerun()` called 12+ times causing full-page flash on every interaction | Jarring in Game and BCI tabs where rapid interaction is expected |
| UX-4 | HIGH | Green-red color pair for true vs. model is worst-case for colorblind users (~8% of males) | Polar comparison plot is unreadable for deuteranopes/protanopes |
| UX-5 | HIGH | Slider-only direction input in Game mode -- imprecise for 1-degree accuracy task | Motor-impaired users and touchscreen users struggle; default 180 biases guesses |
| UX-6 | HIGH | Multi-column layouts stack poorly on mobile -- 10 tabs wrap into multiple rows | Mobile users cannot navigate effectively |
| UX-7 | HIGH | No Streamlit caching (`@st.cache_data`) anywhere -- everything recomputes on every widget interaction | Perceptible lag on every button press |
| UX-8 | HIGH | Tab order does not follow pedagogical progression -- Game (Tab 3) before Learn Decoding (Tab 6) | Students play before understanding; learning scaffold is inverted |
| UX-9 | HIGH | Scoring formulas are completely opaque to users | Cannot strategize; undermines gamification motivation |

### Medium Findings

| ID | Finding |
|----|---------|
| UX-10 | All tabs dead-end with "Please run simulation first" -- no auto-demo or pre-loaded data |
| UX-11 | Sidebar is read-only -- simulation controls trapped in Tab 1 instead of globally accessible |
| UX-12 | Leaderboard is ephemeral (in-memory only) -- always empty on first visit; no social proof |
| UX-13 | Achievement progress not visible during gameplay -- only on Challenges tab |
| UX-14 | Streak challenge ends on very first error -- no warm-up period |
| UX-15 | 200-neuron heatmap is 1600px tall -- pushes content far below fold |
| UX-16 | Decoder selection inconsistent -- some tabs hardcode PV, others offer choice |
| UX-17 | No "next steps" bridges between tabs -- each is a silo with no narrative flow |
| UX-18 | All 10 tabs render on every rerun (Streamlit's `st.tabs` behavior) |

### Recommended UX Improvements (Priority Order)

1. **Reorder tabs pedagogically**: Setup -> Learn Tuning -> Learn Decoding -> Play Game -> Live Activity -> BCI -> Brain Areas -> Manifold -> Challenges -> Analysis
2. **Reduce to 4-5 top-level pages** using `st.navigation` multipage pattern -- group advanced features
3. **Auto-generate demo simulation on first load** -- no tab should be a dead end
4. **Add `st.number_input` alongside slider** for precise direction guessing
5. **Replace green/red with blue/orange** for true vs. model colors (colorblind-safe)
6. **Move core parameters to sidebar** so they persist across all pages
7. **Add `@st.cache_data`** to all pure computation and visualization functions
8. **Use `st.fragment`** (Streamlit 1.33+) to isolate interactive sections from full-page reruns
9. **Display score breakdown** after each challenge round
10. **Add "What is this?" expanders** with plain-language explanations on each section

---

## SECTOR 6: PERFORMANCE

**Overall Grade: D+** -- Multiple O(n^2) Python loops, zero caching, redundant computation on every rerun

### Critical Findings

| ID | Severity | Finding | Expected Speedup |
|----|----------|---------|-----------------|
| PF-1 | CRITICAL | ML decoder: O(360 x N) nested Python loops -- should be single NumPy broadcast | **50-100x** from vectorization |
| PF-2 | CRITICAL | Zero `@st.cache_data` or `@st.cache_resource` -- every widget interaction recomputes everything | **Eliminates redundant work** |
| PF-3 | HIGH | Temporal simulation inner Python loop per neuron per timestep -- O(T x N) scalar ops | **10-30x** from vectorization |
| PF-4 | HIGH | `cosine_tuning` called in scalar loops via list comprehension despite supporting arrays | **5-10x** per call site (6 sites) |
| PF-5 | HIGH | Plotly `plot_tuning_curves` creates N separate traces (200 at max) -- browser rendering degrades above ~50 | **5-10x** render speed from single trace |
| PF-6 | HIGH | All 10 tabs execute on every widget interaction regardless of which tab is active | Migrate to multipage app |

### Performance Hotspot Ranking

| Rank | Function | Cost (200 neurons, defaults) | Vectorization Potential |
|------|----------|------------------------------|------------------------|
| 1 | `MaximumLikelihoodDecoder.decode` | O(360 x N) Python | Full vectorize: `rates = r0 + k * cos(grid[:, None] - prefs[None, :])` |
| 2 | `simulate_continuous_activity` | O(T x N) Python, T=3000 | Vectorize inner spike loop |
| 3 | `simulate_temporal_spikes` | O(T x N) Python, T=1500 | Vectorize inner spike loop |
| 4 | `evaluate_decoder` (with ML) | O(trials x 360 x N) | Inherits ML fix |
| 5 | `KalmanFilterDecoder.update` | O(N^3) matrix inverse | Use `np.linalg.solve` (2-3x) |
| 6 | `plot_tuning_curves` | 200 Plotly traces x 361 points | Single trace with None separators |
| 7 | `create_spike_raster_snapshot` | N traces per neuron | Merge into single scatter |

### Caching Strategy Recommendations

| Function | Decorator | Cache Key |
|----------|-----------|-----------|
| All `plot_*` functions | `@st.cache_data` | Input arrays (hash) |
| `evaluate_decoder` | `@st.cache_data` | Decoder name, spike counts, params |
| `compute_neural_manifold` | `@st.cache_data` | Spike counts, n_components |
| `PopulationVectorDecoder()` | `@st.cache_resource` | N/A (stateless) |
| `MaximumLikelihoodDecoder()` | `@st.cache_resource` | N/A (stateless) |
| `HierarchicalNetwork` | `@st.cache_resource` | Seed, n_neurons |

---

## SECTOR 7: TESTING & RELIABILITY

**Overall Grade: F** -- Zero tests, zero test infrastructure, multiple crash-paths and state corruption risks

### Top 10 Crash Risks (Untested)

| Priority | Risk | Trigger | Consequence |
|----------|------|---------|-------------|
| P0 | `duration_ms=0` | Slider edge case | `ZeroDivisionError` in `compute_firing_rate_stats` |
| P0 | `n_neurons=0` | Should not happen via UI but no guard | `IndexError` in all downstream code |
| P0 | Double `finish_challenge()` | Race between auto-finish and manual "End Challenge" button | `RuntimeError`, possible leaderboard corruption |
| P0 | All spike counts zero | Low-rate neurons, anti-preferred direction | Division by zero in `create_pv_decoder_step` and `create_vector_animation_polar` |
| P1 | Kalman filter not fitted before `decode_step` | Direct API usage | `RuntimeError` (correct behavior but no UI guard) |
| P1 | Kalman state not reset between trials in `evaluate_decoder` | Always | Order-dependent, incorrect performance metrics |
| P1 | `simulate_temporal_spikes` with `dt_ms=0` | Edge case | Infinite loop / `ZeroDivisionError` |
| P1 | `PCA(n_components=3)` with fewer than 3 trials | User runs manifold with 1-2 trials | sklearn raises error |
| P2 | Negative spike counts from Gaussian underdispersed path | `variance_scale < 1` with low expected counts | Flips ML likelihood contribution sign |
| P2 | Streak challenge ends on first trial | First guess > threshold | Frustrating UX, possibly unintended |

### Minimum Viable Test Suite (20 tests)

| # | Test | Catches |
|---|------|---------|
| 1 | `wrap_angle(-pi)` returns `pi` | Angle wrapping foundation |
| 2 | `wrap_angle(2*pi)` returns `0.0` | Boundary behavior |
| 3 | `angular_error(x, x) == 0` | Base scoring case |
| 4 | `angular_error(0, pi) == pi` | Maximum error case |
| 5 | `angular_error(a, b) == angular_error(b, a)` | Symmetry |
| 6 | `cosine_tuning` peak at preferred direction = `r0 + k` | Fundamental equation |
| 7 | `cosine_tuning` output always >= 0 | Prevents negative Poisson lambda |
| 8 | `generate_neuron_population` uniform spacing spans `[0, 2*pi)` | Population structure |
| 9 | `simulate_trial` output shape matches n_neurons, all >= 0 | Data integrity |
| 10 | `simulate_trial` same seed -> same output | Reproducibility |
| 11 | PV decode with one dominant neuron -> approximately correct | Core decoder |
| 12 | PV decode with zero spikes -> no crash | Edge case |
| 13 | ML decode with seeded sim -> within 30 degrees of truth | ML correctness |
| 14 | Kalman `decode_step` before `fit` raises RuntimeError | State guard |
| 15 | `evaluate_decoder` all errors in `[0, pi]` | Metric validity |
| 16 | `score_speed_trial` with empty errors returns 0.0 | Guard clause |
| 17 | Challenge lifecycle: start -> 5 trials -> finish -> score > 0 | State machine |
| 18 | Double `finish_challenge` raises RuntimeError | Corruption prevention |
| 19 | `plot_tuning_curves` returns `go.Figure` with traces | Viz smoke test |
| 20 | End-to-end: 50 neurons, 100 trials, PV mean error < 45 deg | Pipeline validation |

### Recommended Test Infrastructure

```
tests/
    conftest.py           # Shared fixtures
    test_utils.py         # ~25 tests: angle math, exports
    test_simulation.py    # ~30 tests: tuning, trial gen, temporal, hierarchy
    test_decoders.py      # ~30 tests: PV, ML, Kalman, evaluation
    test_challenges.py    # ~25 tests: scoring, state machine, achievements
    test_visualization.py # ~20 tests: smoke tests for all plot functions
    test_integration.py   # ~10 tests: end-to-end pipelines
```

---

## SECTOR 8: DEVOPS & DEPLOYMENT

**Overall Grade: F** -- No git, no CI/CD, no Docker, no deployment target, no logging

### Critical & High Findings

| ID | Severity | Finding | Impact |
|----|----------|---------|--------|
| DO-1 | CRITICAL | **Not a git repository** -- no `.git/`, no `.gitignore`, `__pycache__/` in project root | No version history, no rollback, no collaboration, blocks all CI/CD |
| DO-2 | HIGH | Loose dependency pins (`>=` with no upper bound), no lock file | Non-reproducible builds; numpy 2.x (allowed) has breaking changes |
| DO-3 | HIGH | No CI/CD pipeline (no `.github/workflows/`) | No automated quality gate |
| DO-4 | HIGH | No deployment target configured -- local-only | Inaccessible to target audience |
| DO-5 | HIGH | Environment not reproducible -- no lock file, no Python version spec, `__pycache__` shows 3.13 but claims 3.8+ compat | Two installs at different times produce different environments |

### Medium Findings

| ID | Finding |
|----|---------|
| DO-6 | No Dockerfile |
| DO-7 | No `.streamlit/config.toml` -- server settings unconfigured (CORS, XSRF, telemetry) |
| DO-8 | Zero logging or health checks |
| DO-9 | README lacks deployment documentation |
| DO-10 | All state (leaderboards, achievements) is ephemeral -- lost on session end |
| DO-11 | Python version claim (3.8+) is inaccurate -- Streamlit has dropped 3.8 support |

### Recommended Implementation Order

| Step | Action | Blocks |
|------|--------|--------|
| 1 | `git init` + `.gitignore` | Everything else |
| 2 | Pin dependencies properly + generate lock file | Reproducibility |
| 3 | Add `runtime.txt` + `.streamlit/config.toml` | Deployment |
| 4 | Deploy to Streamlit Community Cloud | Public access |
| 5 | Add GitHub Actions CI (lint + build) | Quality gate |
| 6 | Add `logging` module across codebase | Observability |
| 7 | Create Dockerfile | Alternative deployment |
| 8 | Build test suite | Meaningful CI |
| 9 | Add leaderboard/achievement persistence | UX completion |
| 10 | Add `@st.cache_data` caching | Performance |

---

## CROSS-SECTOR PRIORITY MATRIX

The following table ranks ALL findings across all sectors by impact, forming the basis for the project improvement plan.

### Tier 1: Foundational (Must Do First)

| # | Finding | Sectors | Effort |
|---|---------|---------|--------|
| 1 | **Initialize git repository** + `.gitignore` | DevOps | 15 min |
| 2 | **Pin dependencies** with upper bounds + lock file | DevOps, Security | 30 min |
| 3 | **Replace `np.random.seed()` with `default_rng()`** across all simulation functions | Code Quality, Scientific, Testing | 2 hr |
| 4 | **Fix negative binomial `+1` bias** in `simulate_trial` | Scientific | 30 min |
| 5 | **Fix Kalman covariance update** to Joseph stabilized form | Scientific | 30 min |
| 6 | **Fix Kalman observation units mismatch** in `decode()` compatibility method | Scientific | 1 hr |

### Tier 2: Structural (High-Leverage Refactors)

| # | Finding | Sectors | Effort |
|---|---------|---------|--------|
| 7 | **Add `@st.cache_data`** to all pure computation and visualization functions | Performance, UX | 2 hr |
| 8 | **Vectorize ML decoder** -- replace nested Python loops with NumPy broadcasting | Performance, Code Quality | 2 hr |
| 9 | **Standardize Decoder interface** -- add `duration_s` as optional kwarg to ABC; make Kalman inherit from Decoder | Architecture, Code Quality | 3 hr |
| 10 | **Extract log-likelihood computation** -- deduplicate 3 copies into one method | Code Quality | 1 hr |
| 11 | **Fix stale state** -- clear derived state (manifold, hierarchy, walkthrough) when "Simulate" is pressed | Architecture, UX | 1 hr |
| 12 | **Add input validation** -- guard `n_neurons > 0`, `duration_ms > 0`, `dt_ms > 0` in all public functions | Testing, Reliability | 2 hr |
| 13 | **Fix raster variance branching** -- mirror `simulate_trial`'s three-path logic | Scientific | 1 hr |

### Tier 3: UX & Deployment

| # | Finding | Sectors | Effort |
|---|---------|---------|--------|
| 14 | **Reorder tabs pedagogically** and reduce to 4-5 top-level pages (multipage app) | UX | 4 hr |
| 15 | **Deploy to Streamlit Community Cloud** | DevOps | 1 hr |
| 16 | **Add GitHub Actions CI** (lint + test stages) | DevOps | 2 hr |
| 17 | **Replace green/red with colorblind-safe palette** (blue/orange) | UX, Accessibility | 30 min |
| 18 | **Add `st.number_input` for direction guessing** alongside slider | UX, Accessibility | 30 min |
| 19 | **Add logging** (`import logging`) to every module | DevOps, Code Quality | 2 hr |
| 20 | **Auto-generate demo simulation** on first load | UX | 1 hr |
| 21 | **Move simulation controls to sidebar** | UX | 1 hr |
| 22 | **Add try/except with `st.error()`** around simulation/decode calls | Reliability, UX | 2 hr |

### Tier 4: Polish & Scale

| # | Finding | Sectors | Effort |
|---|---------|---------|--------|
| 23 | **Split oversized modules** into subpackages | Code Quality, Architecture | 6 hr |
| 24 | **Build minimum viable test suite** (20 tests) | Testing | 4 hr |
| 25 | **Vectorize temporal simulation** inner loops | Performance | 4 hr |
| 26 | **Create `config.py`** with centralized constants | Architecture | 2 hr |
| 27 | **Display score breakdowns** in challenge mode | UX | 2 hr |
| 28 | **Add leaderboard persistence** (JSON file or lightweight DB) | UX, DevOps | 3 hr |
| 29 | **Extract business logic from `app.py`** into `GameEngine` class | Architecture | 4 hr |
| 30 | **Reduce Plotly trace count** -- merge per-neuron traces into single traces | Performance | 3 hr |
| 31 | **Create Dockerfile** | DevOps | 1 hr |
| 32 | **Fix burst clip order** in `simulate_continuous_activity` | Scientific | 15 min |
| 33 | **Rename `tuning_sharpness` to `modulation_gain`** or implement von Mises | Scientific | 1-4 hr |
| 34 | **Add `.streamlit/config.toml`** with security settings | DevOps, Security | 15 min |
| 35 | **Remove dead code** (unused imports, uncalled functions) | Code Quality | 30 min |

---

## Estimated Total Effort

| Tier | Items | Estimated Hours |
|------|-------|-----------------|
| Tier 1: Foundational | 6 | ~5 hours |
| Tier 2: Structural | 7 | ~12 hours |
| Tier 3: UX & Deployment | 9 | ~13 hours |
| Tier 4: Polish & Scale | 13 | ~30 hours |
| **Total** | **35** | **~60 hours** |

---

*This report was generated by 8 parallel specialist audit agents (Security, Code Quality, Architecture, Scientific Correctness, UX/UI, Performance, Testing, DevOps) analyzing the complete codebase.*
