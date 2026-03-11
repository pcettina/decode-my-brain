"""
Decode My Brain - Interactive Neuroscience Visualization App

A Streamlit app for exploring neural coding and decoding concepts.
Simulates direction-tuned neurons and lets users compete against
a model decoder to guess movement direction from spike patterns.

To run:
    pip install -r requirements.txt
    streamlit run app.py

Author: Decode My Brain Team
"""

import logging
import streamlit as st

from simulation import (
    generate_neuron_population,
    simulate_random_trials,
)
from engine.game import BCISimulator
from challenges import ChallengeManager, AchievementManager


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Decode My Brain",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

# Custom CSS — fonts, tab styling, dark mode support, accessibility
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .stCode, [data-testid="stMetricValue"] {
        font-family: 'Fira Code', 'Source Code Pro', monospace !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 16px;
    }
    .winner-user {
        color: #27ae60;
        font-weight: bold;
    }
    .winner-model {
        color: #c0392b;
        font-weight: bold;
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            padding-left: 10px; padding-right: 10px; font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""

    # Simulation parameters (defaults)
    if 'n_neurons' not in st.session_state:
        st.session_state.n_neurons = 50
    if 'duration_ms' not in st.session_state:
        st.session_state.duration_ms = 500
    if 'baseline_rate' not in st.session_state:
        st.session_state.baseline_rate = 5.0
    if 'modulation_depth' not in st.session_state:
        st.session_state.modulation_depth = 15.0
    if 'variance_scale' not in st.session_state:
        st.session_state.variance_scale = 1.0
    if 'n_trials' not in st.session_state:
        st.session_state.n_trials = 20

    # Simulation results
    if 'neurons' not in st.session_state:
        st.session_state.neurons = None
    if 'spike_counts' not in st.session_state:
        st.session_state.spike_counts = None
    if 'true_directions' not in st.session_state:
        st.session_state.true_directions = None
    if 'simulated' not in st.session_state:
        st.session_state.simulated = False

    # Game state
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'game_theta' not in st.session_state:
        st.session_state.game_theta = None
    if 'game_spikes' not in st.session_state:
        st.session_state.game_spikes = None
    if 'game_rounds' not in st.session_state:
        st.session_state.game_rounds = []
    if 'round_submitted' not in st.session_state:
        st.session_state.round_submitted = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # Live Activity state
    if 'live_spike_times' not in st.session_state:
        st.session_state.live_spike_times = None
    if 'live_time_ms' not in st.session_state:
        st.session_state.live_time_ms = 0
    if 'live_theta' not in st.session_state:
        st.session_state.live_theta = None
    if 'live_running' not in st.session_state:
        st.session_state.live_running = False

    # BCI Simulator state (uses BCISimulator engine)
    if 'bci_sim' not in st.session_state:
        st.session_state.bci_sim = BCISimulator()
    if 'bci_running' not in st.session_state:
        st.session_state.bci_running = False

    # Decoder walkthrough state
    if 'walkthrough_step' not in st.session_state:
        st.session_state.walkthrough_step = 0
    if 'walkthrough_spikes' not in st.session_state:
        st.session_state.walkthrough_spikes = None
    if 'walkthrough_theta' not in st.session_state:
        st.session_state.walkthrough_theta = None

    # Hierarchical network state
    if 'hierarchy' not in st.session_state:
        st.session_state.hierarchy = None
    if 'hierarchy_data' not in st.session_state:
        st.session_state.hierarchy_data = None

    # Neural manifold state
    if 'manifold_data' not in st.session_state:
        st.session_state.manifold_data = None
    if 'manifold_model' not in st.session_state:
        st.session_state.manifold_model = None
    if 'explained_variance' not in st.session_state:
        st.session_state.explained_variance = None

    # Challenge mode state
    if 'challenge_manager' not in st.session_state:
        st.session_state.challenge_manager = ChallengeManager()
    if 'achievement_manager' not in st.session_state:
        st.session_state.achievement_manager = AchievementManager()
    if 'challenge_active' not in st.session_state:
        st.session_state.challenge_active = False
    if 'challenge_theta' not in st.session_state:
        st.session_state.challenge_theta = None
    if 'challenge_spikes' not in st.session_state:
        st.session_state.challenge_spikes = None

    # Onboarding / page visit tracking
    if 'onboarded' not in st.session_state:
        st.session_state.onboarded = False
    for visit_key in ['visited_setup', 'visited_learn', 'visited_play',
                      'visited_explore', 'visited_analyze']:
        if visit_key not in st.session_state:
            st.session_state[visit_key] = False

    # Auto-generate demo simulation so the app isn't empty on first load
    if not st.session_state.simulated:
        demo_neurons = generate_neuron_population(50, seed=42)
        demo_spikes, demo_dirs = simulate_random_trials(
            20, demo_neurons, duration_ms=500, seed=42
        )
        st.session_state.update(
            neurons=demo_neurons,
            spike_counts=demo_spikes,
            true_directions=demo_dirs,
            simulated=True
        )
        logger.info("Auto-generated demo simulation: 50 neurons, 20 trials")


init_session_state()

# First-visit onboarding
if not st.session_state.onboarded:
    st.toast("Welcome! Start with **Learn** to explore neural coding, or jump to **Play** to test your skills.", icon=":material/waving_hand:")
    st.session_state.onboarded = True


# =============================================================================
# Sidebar - About & Simulation Controls
# =============================================================================

with st.sidebar:
    st.title("Decode My Brain")
    st.markdown("---")

    with st.expander("About This App", expanded=False):
        st.markdown("""
        This interactive tool demonstrates how the brain might encode
        and decode movement direction using populations of neurons.

        **Key concepts:**
        - **Tuning curves**: Each neuron responds most strongly to
          a preferred direction
        - **Population coding**: Direction is represented by the
          combined activity of many neurons
        - **Decoding**: We can infer direction from neural activity
        """)

    with st.expander("Quick Start Guide", expanded=not st.session_state.get('onboarded', False)):
        guide_steps = [
            (":material/settings: **Setup** — Review your simulation parameters", "visited_setup"),
            (":material/school: **Learn** — Explore tuning curves and walk through decoders", "visited_learn"),
            (":material/videogame_asset: **Play** — Compete against the model decoder", "visited_play"),
            (":material/explore: **Explore** — Live activity, BCI, brain areas, manifolds", "visited_explore"),
            (":material/analytics: **Analyze** — Investigate noise effects and lesions", "visited_analyze"),
        ]
        for step_text, step_key in guide_steps:
            checked = st.session_state.get(step_key, False)
            st.checkbox(step_text, value=checked, key=f"guide_{step_key}", disabled=True)

        completed = sum(1 for _, k in guide_steps if st.session_state.get(k, False))
        st.progress(completed / len(guide_steps))
        if completed == len(guide_steps):
            st.success("You've explored all sections!")

    st.markdown("---")
    st.subheader("Simulation Controls")

    n_neurons = st.slider(
        "Number of Neurons (N)",
        min_value=10, max_value=200, value=st.session_state.n_neurons,
        help="More neurons = better direction representation"
    )
    baseline_rate = st.slider(
        "Baseline Rate r0 (Hz)",
        min_value=1.0, max_value=20.0, value=st.session_state.baseline_rate, step=0.5,
        help="Firing rate when stimulus is opposite to preferred direction"
    )
    modulation_depth = st.slider(
        "Modulation Depth k (Hz)",
        min_value=5.0, max_value=40.0, value=st.session_state.modulation_depth, step=1.0,
        help="How much firing rate increases at preferred direction"
    )
    duration_ms = st.slider(
        "Trial Duration (ms)",
        min_value=100, max_value=2000, value=st.session_state.duration_ms, step=50,
        help="Longer trials = more spikes = better signal"
    )
    variance_scale = st.slider(
        "Variance Scale",
        min_value=0.5, max_value=3.0, value=st.session_state.variance_scale, step=0.1,
        help="1.0 = standard Poisson, >1 = overdispersed, <1 = underdispersed"
    )
    n_trials = st.slider(
        "Number of Trials",
        min_value=1, max_value=100, value=st.session_state.n_trials,
        help="Number of trials to simulate"
    )

    if st.button("Simulate Trials", type="primary", use_container_width=True):
        st.session_state.n_neurons = n_neurons
        st.session_state.duration_ms = duration_ms
        st.session_state.baseline_rate = baseline_rate
        st.session_state.modulation_depth = modulation_depth
        st.session_state.variance_scale = variance_scale
        st.session_state.n_trials = n_trials

        try:
            neurons = generate_neuron_population(
                n_neurons=n_neurons,
                baseline_rate=baseline_rate,
                modulation_depth=modulation_depth
            )
            st.session_state.neurons = neurons

            spike_counts, true_directions = simulate_random_trials(
                n_trials=n_trials,
                neurons=neurons,
                duration_ms=duration_ms,
                variance_scale=variance_scale
            )
            st.session_state.spike_counts = spike_counts
            st.session_state.true_directions = true_directions
            st.session_state.simulated = True

            for key in ['hierarchy', 'hierarchy_data', 'manifold_data',
                        'manifold_model', 'explained_variance',
                        'live_spike_times', 'live_time_ms', 'live_theta',
                        'walkthrough_spikes', 'walkthrough_theta', 'walkthrough_step',
                        'game_active', 'game_theta', 'game_spikes', 'game_rounds',
                        'round_submitted', 'last_result',
                        'bci_sim', 'bci_running']:
                st.session_state.pop(key, None)

            logger.info("Simulated %d trials with %d neurons", n_trials, n_neurons)
            st.rerun()
        except Exception:
            logger.error("Simulation failed", exc_info=True)
            st.error("Simulation failed. Try different parameters.")


# =============================================================================
# Navigation
# =============================================================================

setup_page = st.Page("pages/setup.py", title="Setup", icon=":material/settings:")
learn_page = st.Page("pages/learn.py", title="Learn", icon=":material/school:")
play_page = st.Page("pages/play.py", title="Play", icon=":material/videogame_asset:")
explore_page = st.Page("pages/explore.py", title="Explore", icon=":material/explore:")
analyze_page = st.Page("pages/analyze.py", title="Analyze", icon=":material/analytics:")

pg = st.navigation({
    "Getting Started": [setup_page, learn_page],
    "Interactive": [play_page, explore_page],
    "Tools": [analyze_page],
})

pg.run()


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; font-family: Fira Sans, sans-serif;'>
    <p>Decode My Brain &mdash; Built with Streamlit &amp; Plotly</p>
    <p>An interactive tool for exploring neural coding and decoding concepts</p>
</div>
""", unsafe_allow_html=True)
