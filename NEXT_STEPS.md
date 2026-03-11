# Decode My Brain — Next Steps

Generated 2026-03-09, updated 2026-03-10.
314 tests passing (297 unit + 17 UI interaction), 13 E2E browser tests ready.

---

## Tier 1: Documentation (Quick wins, high impact) — DONE

- [x] **Update README.md architecture section** — now shows subpackage structure (`simulation/`, `decoders/`, `visualization/`, `engine/`, `pages/`)
- [x] **Update README.md test counts** — 297 tests across 10 test files
- [x] **Create CHANGELOG.md** — documents all 35-step outcomes with Added/Fixed/Changed/Removed sections
- [x] **Add docstrings to page files** — all 5 page files have detailed module docstrings describing tabs and content

---

## Tier 2: Testing (Medium effort, high safety value) — DONE

- [x] **Add page workflow integration tests** — 5 Streamlit `AppTest` smoke tests in `tests/test_pages.py`
- [x] **Add visualization edge case tests** — 12 tests: single neuron, empty data, zero spikes, uniform spikes, max neurons (200), empty spike times, single-trial manifold, polar variants
- [x] **Enable coverage reporting in CI** — `pytest --cov=. --cov-fail-under=60` in `.github/workflows/ci.yml`
- [x] **Add Streamlit `AppTest` tests** — `tests/test_pages.py` using `streamlit.testing.v1.AppTest`
- [x] **Add UI interaction tests** — 17 AppTest tests covering sidebar controls, widget manipulation, cross-page state persistence (`tests/test_ui_interactions.py`)
- [x] **Add E2E browser tests** — 13 Playwright tests for navigation, widget clicks, visual regression screenshots (`tests/e2e/`)

---

## Tier 3: UX Polish (Medium effort, user-facing) — DONE

- [x] **Consolidate game direction input** — replaced dual slider+number_input with single `st.number_input` (both game and challenge modes)
- [x] **Cap player name length** — `max_chars=30` on `st.text_input`
- [x] **Add scoring formula display** — `SCORING_DESCRIPTIONS` dict shown via `st.info` before challenge start
- [x] **Show score breakdown after challenge** — `score_breakdown()` function displays base/penalties/bonuses as `st.metric` cards
- [x] **Fix color contrast** — `.winner-model` updated from `#e74c3c` (3.6:1) to `#c0392b` (5.2:1) for WCAG AA
- [x] **Add reduced-motion support** — `@media (prefers-reduced-motion)` disables animations
- [x] **Mobile tab responsive CSS** — reduced gap/padding/font-size on screens < 768px
- [ ] **Add keyboard shortcuts** — submit guess on Enter key for faster gameplay

---

## Tier 4: Deployment Infrastructure — DONE

- [x] **`pyproject.toml`** — consolidated project metadata, pytest markers, ruff/mypy config
- [x] **Enhanced CI pipeline** — pip caching, Python 3.11+3.12 matrix, coverage XML artifacts, Docker build+healthcheck verification
- [x] **Deployment workflow** — `.github/workflows/deploy.yml` pushes Docker image to ghcr.io on main push/tags
- [x] **`docker-compose.yml`** — dev environment with volume mount live-reload + test service profile
- [x] **`.pre-commit-config.yaml`** — ruff lint+format, mypy on core modules
- [x] **E2E test infrastructure** — Playwright conftest with Streamlit server fixture, visual regression screenshots
- [x] **Updated `.gitignore` / `.dockerignore`** — E2E screenshots, coverage.xml, deployment artifacts excluded

---

## Tier 5: Code Quality (Low urgency, maintainability)

- [ ] **Add type hints to page files** — `pages/*.py` and `app.py` callbacks lack return type annotations; visualization functions should annotate `-> go.Figure`
- [ ] **Split `challenges.py`** — at 760 lines, exceeds 600-line guideline; could split scoring functions into `challenges/scoring.py`
- [ ] **Run `ruff format`** — apply consistent formatting across all files
- [ ] **Standardize heading hierarchy** — some pages use `st.markdown("###")` where `st.subheader` is more appropriate

---

## Tier 6: Features & Scale (Future, larger effort)

- [ ] **Add first-visit onboarding** — `st.info` or toast on first load directing users to Learn page
- [ ] **Add tutorial tooltips** — guided onboarding for first-time users (Streamlit `st.popover` or custom tour)
- [ ] **Add more decoder options to game** — Kalman Filter decoder as a selectable model opponent
- [ ] **Add data upload** — let users upload their own spike count matrices for decoding
- [ ] **Add real dataset examples** — bundled sample data from published motor cortex studies
- [ ] **Add authentication** — for public deployment, protect leaderboard with user accounts
- [ ] **Add neural network decoder** — simple MLP or RNN decoder for comparison with classical methods
- [ ] **Performance: cache simulation results** — if user re-runs same sidebar config, avoid resimulating from scratch
- [ ] **Performance: cache PCA in analysis tab** — `compute_neural_manifold()` recomputes every render
- [ ] **Custom font integration** — Fira Sans / Fira Code for science-dashboard identity
- [ ] **Brand color palette refinement** — consider indigo (`#6366F1`) + emerald (`#10B981`) as brand palette

---

## Tier 7: UI/UX Improvements (Design polish)

- [ ] **Replace emoji page icon** — `🧠` renders inconsistently; use Material icon or SVG favicon
- [ ] **Remove `.metric-card` CSS** — unused (all pages use Streamlit's built-in `st.metric`)
- [ ] **Add skeleton loading states** — use `st.spinner` + placeholder layouts during heavy computations
- [ ] **Convert sidebar guide to interactive stepper** — let users track progress through the 5-step guide
- [ ] **Add Plotly chart accessibility** — ensure all charts have descriptive titles and axis labels for screen readers
- [ ] **Dark mode compatibility audit** — test all custom CSS with Streamlit dark theme

---

## Priority Matrix

| Priority | Category | Status | Items |
|----------|----------|--------|-------|
| P0 | Documentation | DONE | README updated, CHANGELOG created, page docstrings added |
| P1 | Testing | DONE | 30 new tests (12 viz, 7 scoring, 5 AppTest, 17 UI, 13 E2E) |
| P2 | UX | DONE | Input consolidated, contrast fixed, reduced-motion, responsive CSS |
| P3 | Deployment | DONE | CI matrix, Docker deploy to ghcr.io, pre-commit, Playwright E2E |
| P4 | Code Quality | TODO | Type hints, ruff format, heading standardization |
| P5 | Features | TODO | Onboarding, new decoders, data upload, tutorials |
| P6 | Design Polish | TODO | Favicon, fonts, dark mode, loading states |

---

## How to Run

```bash
# Unit + UI tests (default)
pytest tests/ -m "not e2e" -v

# E2E browser tests (requires Playwright)
pip install -r requirements-e2e.txt
playwright install chromium
pytest tests/e2e/ -m e2e --browser chromium

# Docker dev
docker compose up app

# Pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
