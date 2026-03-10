# Decode My Brain — Next Steps

Generated 2026-03-09 after completing all 35 steps of the improvement plan.
297 tests passing, multipage app with subpackage modules.

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

---

## Tier 3: UX Polish (Medium effort, user-facing) — DONE

- [x] **Consolidate game direction input** — replaced dual slider+number_input with single `st.number_input` (both game and challenge modes)
- [x] **Cap player name length** — `max_chars=30` on `st.text_input`
- [x] **Add scoring formula display** — `SCORING_DESCRIPTIONS` dict shown via `st.info` before challenge start
- [x] **Show score breakdown after challenge** — `score_breakdown()` function displays base/penalties/bonuses as `st.metric` cards
- [ ] **Add keyboard shortcuts** — submit guess on Enter key for faster gameplay
- [ ] **Mobile layout testing** — `layout="wide"` may overflow on small screens; consider responsive fallback

---

## Tier 4: Code Quality (Low urgency, maintainability)

- [ ] **Add type hints to page files** — `pages/*.py` and `app.py` callbacks lack return type annotations; visualization functions should annotate `-> go.Figure`
- [ ] **Split `challenges.py`** — at 760 lines, exceeds 600-line guideline; could split scoring functions into `challenges/scoring.py`
- [ ] **Add `mypy` to CI** — static type checking to catch annotation issues; start with `--ignore-missing-imports`
- [ ] **Add `ruff format`** — consistent code formatting across all files
- [ ] **Add pre-commit hooks** — auto-run ruff + mypy on staged files

---

## Tier 5: Features & Scale (Future, larger effort)

- [ ] **Add more decoder options to game** — Kalman Filter decoder as a selectable model opponent
- [ ] **Add tutorial tooltips** — guided onboarding for first-time users (Streamlit `st.popover` or custom tour)
- [ ] **Add data upload** — let users upload their own spike count matrices for decoding
- [ ] **Add real dataset examples** — bundled sample data from published motor cortex studies
- [ ] **Add docker-compose.yml** — local dev setup with volume mounts for live reload
- [ ] **Add authentication** — for public deployment, protect leaderboard with user accounts
- [ ] **Add neural network decoder** — simple MLP or RNN decoder for comparison with classical methods
- [ ] **Performance: cache simulation results** — if user re-runs same sidebar config, avoid resimulating from scratch
- [ ] **Performance: cache PCA in analysis tab** — `compute_neural_manifold()` recomputes every render

---

## Priority Matrix

| Priority | Category | Status | Items |
|----------|----------|--------|-------|
| P0 | Documentation | DONE | README updated, CHANGELOG created, page docstrings added |
| P1 | Testing | DONE | 24 new tests (12 viz edge cases, 7 scoring/breakdown, 5 AppTest), coverage gate in CI |
| P2 | UX | DONE | Input consolidated, name capped, scoring formula + breakdown displayed |
| P3 | Code Quality | TODO | Type hints, mypy, ruff format |
| P4 | Features | TODO | New decoders, data upload, tutorials |
