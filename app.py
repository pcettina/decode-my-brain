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

import streamlit as st
import numpy as np
import pandas as pd
from typing import Optional

# Local imports
from simulation import (
    generate_neuron_population,
    simulate_trial,
    simulate_raster,
    simulate_random_trials,
    compute_firing_rate_stats,
    create_lesioned_population,
    NeuronPopulation,
    # Temporal dynamics
    TemporalParams,
    simulate_temporal_spikes,
    simulate_continuous_activity,
    spike_times_to_binned,
    # Multi-area hierarchy
    BrainArea,
    HierarchicalNetwork,
    create_hierarchical_network,
    simulate_hierarchical_trial
)
from decoders import (
    PopulationVectorDecoder,
    MaximumLikelihoodDecoder,
    KalmanFilterDecoder,
    evaluate_decoder,
    compare_decoders
)
from challenges import (
    ChallengeMode,
    ChallengeConfig,
    ChallengeManager,
    AchievementManager,
    CHALLENGE_CONFIGS
)
from visualization import (
    plot_tuning_curves,
    plot_raster_heatmap,
    plot_population_bar,
    plot_polar_comparison,
    plot_decoder_performance_vs_noise,
    plot_condition_comparison,
    plot_likelihood_curve,
    create_scoreboard_table,
    # New visualization functions
    create_spike_raster_snapshot,
    create_bci_canvas,
    create_bci_metrics_display,
    create_pv_decoder_step,
    create_ml_decoder_step,
    create_vector_animation_polar,
    # Advanced visualizations
    compute_neural_manifold,
    plot_neural_manifold_3d,
    plot_neural_manifold_2d,
    plot_variance_explained,
    plot_brain_connectivity,
    plot_area_comparison,
    plot_challenge_progress,
    plot_leaderboard,
    plot_achievement_badges
)
from utils import (
    radians_to_degrees,
    degrees_to_radians,
    angular_error,
    angular_error_degrees,
    export_to_npz,
    export_to_csv,
    wrap_angle
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Decode My Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .winner-user {
        color: #2ecc71;
        font-weight: bold;
    }
    .winner-model {
        color: #e74c3c;
        font-weight: bold;
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
    
    # BCI Simulator state
    if 'bci_cursor_pos' not in st.session_state:
        st.session_state.bci_cursor_pos = (0.0, 0.0)
    if 'bci_target_pos' not in st.session_state:
        st.session_state.bci_target_pos = None
    if 'bci_trail' not in st.session_state:
        st.session_state.bci_trail = []
    if 'bci_running' not in st.session_state:
        st.session_state.bci_running = False
    if 'bci_hits' not in st.session_state:
        st.session_state.bci_hits = 0
    if 'bci_attempts' not in st.session_state:
        st.session_state.bci_attempts = 0
    if 'bci_start_time' not in st.session_state:
        st.session_state.bci_start_time = None
    
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


init_session_state()


# =============================================================================
# Sidebar - About & Quick Controls
# =============================================================================

with st.sidebar:
    st.title("🧠 Decode My Brain")
    st.markdown("---")
    
    st.markdown("""
    ### About This App
    
    This interactive tool demonstrates how the brain might encode 
    and decode movement direction using populations of neurons.
    
    **Key concepts:**
    - **Tuning curves**: Each neuron responds most strongly to 
      a preferred direction
    - **Population coding**: Direction is represented by the 
      combined activity of many neurons
    - **Decoding**: We can infer direction from neural activity
    
    ---
    
    ### How to Use
    
    1. **Simulate**: Configure neurons and generate trials
    2. **Visualize**: Explore tuning curves and spike patterns
    3. **Decode**: Play the game and compete with the model!
    4. **Analyze**: Investigate noise effects and lesions
    
    ---
    
    *Built for teaching neural coding concepts*
    """)


# =============================================================================
# Main Content - Tabs
# =============================================================================

st.title("🧠 Decode My Brain")
st.markdown("*Explore neural coding and decoding of movement direction*")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Simulation",
    "👁️ Visualize", 
    "🎮 Game",
    "⚡ Live Activity",
    "🎯 BCI Simulator",
    "📚 Learn Decoding",
    "🧠 Brain Areas",
    "🌀 Neural Manifold",
    "🏆 Challenges",
    "🔬 Analysis"
])


# =============================================================================
# Tab 1: Simulation Controls
# =============================================================================

with tab1:
    st.header("Simulation Controls")
    st.markdown("""
    Configure the neural population and simulate trials. Each neuron has a 
    **preferred direction** and responds according to a cosine tuning curve:
    
    > λ(θ) = r₀ + k · cos(θ − μ)
    
    where r₀ is the baseline rate, k is the modulation depth, and μ is the 
    preferred direction.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Population Parameters")
        n_neurons = st.slider(
            "Number of Neurons (N)",
            min_value=10, max_value=200, value=st.session_state.n_neurons,
            help="More neurons = better direction representation"
        )
        baseline_rate = st.slider(
            "Baseline Rate r₀ (Hz)",
            min_value=1.0, max_value=20.0, value=st.session_state.baseline_rate, step=0.5,
            help="Firing rate when stimulus is opposite to preferred direction"
        )
        modulation_depth = st.slider(
            "Modulation Depth k (Hz)",
            min_value=5.0, max_value=40.0, value=st.session_state.modulation_depth, step=1.0,
            help="How much firing rate increases at preferred direction"
        )
    
    with col2:
        st.subheader("Trial Parameters")
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
    
    with col3:
        st.subheader("Actions")
        st.markdown("")
        st.markdown("")
        
        if st.button("🚀 Simulate Trials", type="primary", use_container_width=True):
            # Update session state
            st.session_state.n_neurons = n_neurons
            st.session_state.duration_ms = duration_ms
            st.session_state.baseline_rate = baseline_rate
            st.session_state.modulation_depth = modulation_depth
            st.session_state.variance_scale = variance_scale
            st.session_state.n_trials = n_trials
            
            # Generate neuron population
            with st.spinner("Generating neural population..."):
                neurons = generate_neuron_population(
                    n_neurons=n_neurons,
                    baseline_rate=baseline_rate,
                    modulation_depth=modulation_depth
                )
                st.session_state.neurons = neurons
            
            # Simulate trials
            with st.spinner(f"Simulating {n_trials} trials..."):
                spike_counts, true_directions = simulate_random_trials(
                    n_trials=n_trials,
                    neurons=neurons,
                    duration_ms=duration_ms,
                    variance_scale=variance_scale
                )
                st.session_state.spike_counts = spike_counts
                st.session_state.true_directions = true_directions
                st.session_state.simulated = True
            
            st.success(f"✅ Simulated {n_trials} trials with {n_neurons} neurons!")
            st.rerun()
    
    # Show simulation summary
    if st.session_state.simulated:
        st.markdown("---")
        st.subheader("Simulation Summary")
        
        stats = compute_firing_rate_stats(
            st.session_state.spike_counts,
            st.session_state.duration_ms
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Neurons", st.session_state.n_neurons)
        with col2:
            st.metric("Trials", st.session_state.n_trials)
        with col3:
            st.metric("Mean Firing Rate", f"{stats['mean_rate']:.1f} Hz")
        with col4:
            st.metric("Std Firing Rate", f"{stats['std_rate']:.1f} Hz")
        
        st.info(f"""
        **Population**: {st.session_state.n_neurons} neurons with uniformly spaced 
        preferred directions (0° to 360°)
        
        **Tuning**: Baseline = {st.session_state.baseline_rate} Hz, 
        Modulation = {st.session_state.modulation_depth} Hz
        
        **Trials**: {st.session_state.n_trials} trials × {st.session_state.duration_ms} ms each
        """)


# =============================================================================
# Tab 2: Visualize Neural Code
# =============================================================================

with tab2:
    st.header("Visualize Neural Code")
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        spike_counts = st.session_state.spike_counts
        true_directions = st.session_state.true_directions
        
        # Tuning curves section
        st.subheader("Tuning Curves")
        st.markdown("""
        Each colored line shows how one neuron's firing rate varies with movement 
        direction. The peak of each curve indicates that neuron's **preferred direction**.
        """)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            opacity = st.slider("Curve Opacity", 0.1, 1.0, 0.4)
            show_trial_dir = st.checkbox("Show trial direction", value=True)
            if show_trial_dir and len(true_directions) > 0:
                trial_for_highlight = st.selectbox(
                    "Highlight trial",
                    range(len(true_directions)),
                    format_func=lambda x: f"Trial {x+1}: {radians_to_degrees(true_directions[x]):.0f}°"
                )
                highlight_theta = true_directions[trial_for_highlight]
            else:
                highlight_theta = None
        
        with col1:
            fig_tuning = plot_tuning_curves(
                neurons,
                highlight_theta=highlight_theta if show_trial_dir else None,
                opacity=opacity,
                show_highlight=show_trial_dir
            )
            st.plotly_chart(fig_tuning, use_container_width=True)
        
        st.markdown("---")
        
        # Single trial visualization
        st.subheader("Single Trial Activity")
        
        trial_idx = st.slider(
            f"Select Trial (1 to {len(true_directions)})",
            0, len(true_directions) - 1, 0
        )
        
        true_dir_deg = radians_to_degrees(true_directions[trial_idx])
        st.markdown(f"**True movement direction: {true_dir_deg:.0f}°**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Spike Count Heatmap")
            fig_heatmap = plot_raster_heatmap(
                spike_counts[trial_idx],
                neurons,
                true_theta=true_directions[trial_idx]
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.markdown("##### Population Activity")
            plot_type = st.radio("Plot type", ["Bar", "Polar"], horizontal=True)
            fig_pop = plot_population_bar(
                spike_counts[trial_idx],
                neurons,
                true_theta=true_directions[trial_idx],
                as_polar=(plot_type == "Polar")
            )
            st.plotly_chart(fig_pop, use_container_width=True)
        
        # Optional: Show raster with time bins
        st.markdown("---")
        with st.expander("🔍 View Time-Binned Raster (click to expand)"):
            st.markdown("""
            This shows spike activity across time bins within a single trial.
            Neurons are sorted by their preferred direction.
            """)
            
            bin_size = st.slider("Time bin size (ms)", 10, 100, 25, step=5)
            
            raster_data = simulate_raster(
                true_directions[trial_idx],
                neurons,
                duration_ms=st.session_state.duration_ms,
                bin_size_ms=bin_size,
                variance_scale=st.session_state.variance_scale
            )
            
            fig_raster = plot_raster_heatmap(
                raster_data,
                neurons,
                true_theta=true_directions[trial_idx]
            )
            fig_raster.update_layout(title="Time-Binned Spike Raster")
            st.plotly_chart(fig_raster, use_container_width=True)


# =============================================================================
# Tab 3: Decode My Brain (Game Mode)
# =============================================================================

with tab3:
    st.header("🎮 Decode My Brain")
    st.markdown("""
    **Can you decode movement direction from neural activity?**
    
    Look at the spike pattern and try to guess the hidden movement direction.
    Compare your accuracy to the model decoder!
    """)
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        
        # Initialize decoders
        pv_decoder = PopulationVectorDecoder()
        ml_decoder = MaximumLikelihoodDecoder()
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Game Controls")
            
            decoder_choice = st.radio(
                "Model decoder",
                ["Population Vector", "Maximum Likelihood"],
                help="Which decoder algorithm the model uses"
            )
            selected_decoder = pv_decoder if decoder_choice == "Population Vector" else ml_decoder
            
            if st.button("🎲 New Round", type="primary", use_container_width=True):
                # Generate new random direction
                st.session_state.game_theta = np.random.uniform(0, 2 * np.pi)
                st.session_state.game_spikes = simulate_trial(
                    st.session_state.game_theta,
                    neurons,
                    duration_ms=st.session_state.duration_ms,
                    variance_scale=st.session_state.variance_scale
                )
                st.session_state.game_active = True
                st.session_state.round_submitted = False
                st.session_state.last_result = None
                st.rerun()
            
            if st.button("🔄 Reset Game", use_container_width=True):
                st.session_state.game_rounds = []
                st.session_state.game_active = False
                st.session_state.game_theta = None
                st.session_state.game_spikes = None
                st.session_state.round_submitted = False
                st.session_state.last_result = None
                st.rerun()
            
            # Scoreboard
            if st.session_state.game_rounds:
                st.markdown("---")
                st.subheader("📊 Scoreboard")
                
                user_errors = [r['user_error'] for r in st.session_state.game_rounds]
                model_errors = [r['model_error'] for r in st.session_state.game_rounds]
                user_wins = sum(1 for r in st.session_state.game_rounds if r['winner'] == 'User')
                model_wins = sum(1 for r in st.session_state.game_rounds if r['winner'] == 'Model')
                
                st.metric("Rounds Played", len(st.session_state.game_rounds))
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Your Mean Error", f"{np.mean(user_errors):.1f}°")
                    st.metric("Your Wins", user_wins)
                with c2:
                    st.metric("Model Mean Error", f"{np.mean(model_errors):.1f}°")
                    st.metric("Model Wins", model_wins)
        
        with col1:
            if st.session_state.game_active and st.session_state.game_spikes is not None:
                st.subheader("🧩 What direction is this?")
                st.markdown("*Study the spike pattern below and make your guess!*")
                
                # Show spike pattern (without revealing true direction)
                fig_game = plot_population_bar(
                    st.session_state.game_spikes,
                    neurons,
                    true_theta=None,  # Don't reveal!
                    as_polar=True
                )
                fig_game.update_layout(title="Population Activity (Hidden Direction)")
                st.plotly_chart(fig_game, use_container_width=True)
                
                # User input
                if not st.session_state.round_submitted:
                    user_guess_deg = st.slider(
                        "Your guess (degrees)",
                        0, 359, 180,
                        help="Slide to select the direction you think this activity represents"
                    )
                    
                    if st.button("✅ Submit Guess", type="primary"):
                        # Compute results
                        user_theta = degrees_to_radians(user_guess_deg)
                        true_theta = st.session_state.game_theta
                        
                        # Decode using model
                        if decoder_choice == "Maximum Likelihood":
                            model_theta = ml_decoder.decode(
                                st.session_state.game_spikes,
                                neurons,
                                duration_s=st.session_state.duration_ms / 1000
                            )
                        else:
                            model_theta = pv_decoder.decode(
                                st.session_state.game_spikes,
                                neurons
                            )
                        
                        # Calculate errors
                        user_error = angular_error_degrees(user_guess_deg, radians_to_degrees(true_theta))
                        model_error = angular_error_degrees(
                            radians_to_degrees(model_theta),
                            radians_to_degrees(true_theta)
                        )
                        
                        # Determine winner
                        if user_error < model_error:
                            winner = "User"
                        elif model_error < user_error:
                            winner = "Model"
                        else:
                            winner = "Tie"
                        
                        # Store result
                        result = {
                            'true_rad': true_theta,
                            'true_deg': radians_to_degrees(true_theta),
                            'user_rad': user_theta,
                            'user_deg': user_guess_deg,
                            'model_rad': model_theta,
                            'model_deg': radians_to_degrees(model_theta),
                            'user_error': user_error,
                            'model_error': model_error,
                            'winner': winner
                        }
                        
                        st.session_state.game_rounds.append(result)
                        st.session_state.last_result = result
                        st.session_state.round_submitted = True
                        st.rerun()
                
                else:
                    # Show results
                    result = st.session_state.last_result
                    
                    st.markdown("### 🎯 Results")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("True Direction", f"{result['true_deg']:.0f}°")
                    with c2:
                        st.metric("Your Guess", f"{result['user_deg']:.0f}°", 
                                  delta=f"{result['user_error']:.1f}° error")
                    with c3:
                        st.metric("Model Decode", f"{result['model_deg']:.0f}°",
                                  delta=f"{result['model_error']:.1f}° error")
                    
                    # Winner announcement
                    if result['winner'] == "User":
                        st.success("🎉 **You win this round!** You beat the model decoder!")
                    elif result['winner'] == "Model":
                        st.error("🤖 **Model wins!** The decoder was more accurate.")
                    else:
                        st.info("🤝 **It's a tie!** You matched the model's accuracy.")
                    
                    # Polar comparison plot
                    fig_result = plot_polar_comparison(
                        result['true_rad'],
                        result['user_rad'],
                        result['model_rad']
                    )
                    st.plotly_chart(fig_result, use_container_width=True)
            
            else:
                st.info("👆 Click **New Round** to start playing!")
        
        # Game history table
        if st.session_state.game_rounds:
            st.markdown("---")
            st.subheader("📜 Round History")
            fig_table = create_scoreboard_table(st.session_state.game_rounds)
            st.plotly_chart(fig_table, use_container_width=True)


# =============================================================================
# Tab 4: Live Activity (Animated Spike Raster)
# =============================================================================

with tab4:
    st.header("⚡ Live Neural Activity")
    st.markdown("""
    Watch neural activity unfold in real-time! This visualization shows spikes 
    appearing as the neurons fire, with **temporal dynamics** including adaptation,
    refractory periods, and bursting.
    """)
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("⚙️ Temporal Dynamics")
            
            adaptation = st.slider(
                "Adaptation Strength",
                0.0, 0.8, 0.3,
                help="How much firing rate decreases with sustained activity"
            )
            refractory = st.slider(
                "Refractory Period (ms)",
                1.0, 10.0, 2.0,
                help="Minimum time between spikes"
            )
            burst_prob = st.slider(
                "Burst Probability",
                0.0, 0.5, 0.2,
                help="Probability a neuron fires in bursts"
            )
            
            st.markdown("---")
            st.subheader("🎬 Simulation")
            
            live_direction = st.slider(
                "Movement Direction (°)",
                0, 359, 90,
                help="Direction of simulated movement"
            )
            live_duration = st.slider(
                "Duration (ms)",
                500, 3000, 1500, step=100
            )
            
            if st.button("▶️ Generate Activity", type="primary", use_container_width=True):
                temporal_params = TemporalParams(
                    adaptation_strength=adaptation,
                    refractory_abs_ms=refractory,
                    refractory_rel_ms=refractory * 2,
                    burst_probability=burst_prob
                )
                
                theta = degrees_to_radians(live_direction)
                
                with st.spinner("Simulating temporal dynamics..."):
                    spike_times, spike_counts = simulate_temporal_spikes(
                        theta=theta,
                        neurons=neurons,
                        duration_ms=live_duration,
                        temporal_params=temporal_params
                    )
                    st.session_state.live_spike_times = spike_times
                    st.session_state.live_theta = theta
                    st.session_state.live_time_ms = live_duration
                
                st.success(f"Generated {sum(len(s) for s in spike_times)} spikes!")
                st.rerun()
        
        with col1:
            if st.session_state.live_spike_times is not None:
                # Time slider for viewing
                view_time = st.slider(
                    "View Time (ms)",
                    100, int(st.session_state.live_time_ms),
                    int(st.session_state.live_time_ms),
                    step=50,
                    key="live_view_time"
                )
                
                # Create raster snapshot
                fig_raster = create_spike_raster_snapshot(
                    st.session_state.live_spike_times,
                    neurons,
                    current_time_ms=view_time,
                    window_ms=min(500, view_time),
                    true_theta=st.session_state.live_theta
                )
                st.plotly_chart(fig_raster, use_container_width=True)
                
                # Show spike statistics
                total_spikes = sum(len(s) for s in st.session_state.live_spike_times)
                duration_s = st.session_state.live_time_ms / 1000
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Spikes", total_spikes)
                with c2:
                    st.metric("Mean Rate", f"{total_spikes / neurons.n_neurons / duration_s:.1f} Hz")
                with c3:
                    active_neurons = sum(1 for s in st.session_state.live_spike_times if len(s) > 0)
                    st.metric("Active Neurons", f"{active_neurons}/{neurons.n_neurons}")
            else:
                st.info("👆 Configure parameters and click **Generate Activity** to see live neural firing!")


# =============================================================================
# Tab 5: BCI Simulator
# =============================================================================

with tab5:
    st.header("🎯 BCI Cursor Control Simulator")
    st.markdown("""
    Experience brain-computer interface (BCI) control! Neural activity is decoded
    in real-time to move a cursor toward targets. See how decoding accuracy 
    affects control quality.
    """)
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        pv_decoder = PopulationVectorDecoder()
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("🎮 BCI Controls")
            
            cursor_speed = st.slider(
                "Cursor Speed",
                1.0, 10.0, 5.0,
                help="How fast the cursor moves based on decoded direction"
            )
            
            noise_level = st.slider(
                "Neural Noise",
                0.5, 3.0, 1.0,
                help="Variability in neural signals (affects control accuracy)"
            )
            
            st.markdown("---")
            
            if st.button("🎯 New Target", type="primary", use_container_width=True):
                # Generate random target position
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(50, 80)
                target_x = distance * np.cos(angle)
                target_y = distance * np.sin(angle)
                
                st.session_state.bci_target_pos = (target_x, target_y)
                st.session_state.bci_cursor_pos = (0.0, 0.0)
                st.session_state.bci_trail = [(0.0, 0.0)]
                st.session_state.bci_attempts += 1
                st.session_state.bci_start_time = 0
                st.rerun()
            
            if st.button("➡️ Move Cursor", use_container_width=True):
                if st.session_state.bci_target_pos is not None:
                    # Calculate direction to target
                    cx, cy = st.session_state.bci_cursor_pos
                    tx, ty = st.session_state.bci_target_pos
                    
                    true_angle = np.arctan2(ty - cy, tx - cx)
                    
                    # Simulate neural activity for this direction
                    spikes = simulate_trial(
                        true_angle,
                        neurons,
                        duration_ms=100,
                        variance_scale=noise_level
                    )
                    
                    # Decode direction
                    decoded_angle = pv_decoder.decode(spikes, neurons)
                    
                    # Move cursor based on decoded direction
                    step = cursor_speed
                    new_x = cx + step * np.cos(decoded_angle)
                    new_y = cy + step * np.sin(decoded_angle)
                    
                    st.session_state.bci_cursor_pos = (new_x, new_y)
                    st.session_state.bci_trail.append((new_x, new_y))
                    
                    # Check if target reached
                    dist_to_target = np.sqrt((new_x - tx)**2 + (new_y - ty)**2)
                    if dist_to_target < 20:
                        st.session_state.bci_hits += 1
                        st.balloons()
                    
                    st.rerun()
            
            if st.button("🔄 Reset Stats", use_container_width=True):
                st.session_state.bci_hits = 0
                st.session_state.bci_attempts = 0
                st.session_state.bci_trail = []
                st.session_state.bci_target_pos = None
                st.session_state.bci_cursor_pos = (0.0, 0.0)
                st.rerun()
            
            st.markdown("---")
            st.subheader("📊 Performance")
            
            st.metric("Targets Hit", st.session_state.bci_hits)
            st.metric("Attempts", st.session_state.bci_attempts)
            if st.session_state.bci_attempts > 0:
                hit_rate = st.session_state.bci_hits / st.session_state.bci_attempts * 100
                st.metric("Hit Rate", f"{hit_rate:.0f}%")
        
        with col1:
            # Calculate decoded direction for display
            decoded_dir = None
            if st.session_state.bci_target_pos is not None:
                cx, cy = st.session_state.bci_cursor_pos
                tx, ty = st.session_state.bci_target_pos
                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                
                # Show most recent decoded direction
                if len(st.session_state.bci_trail) > 1:
                    prev = st.session_state.bci_trail[-2]
                    curr = st.session_state.bci_trail[-1]
                    if prev != curr:
                        decoded_dir = np.arctan2(curr[1] - prev[1], curr[0] - prev[0])
            
            fig_bci = create_bci_canvas(
                cursor_pos=st.session_state.bci_cursor_pos,
                target_pos=st.session_state.bci_target_pos if st.session_state.bci_target_pos else (50, 50),
                decoded_direction=decoded_dir,
                cursor_trail=st.session_state.bci_trail if len(st.session_state.bci_trail) > 1 else None,
                canvas_size=200.0
            )
            
            if st.session_state.bci_target_pos is None:
                fig_bci.update_layout(title="BCI Canvas - Click 'New Target' to start!")
            
            st.plotly_chart(fig_bci, use_container_width=True)
            
            # Instructions
            st.info("""
            **How to play:**
            1. Click **New Target** to place a target (green circle)
            2. Click **Move Cursor** repeatedly to move toward the target
            3. Your neural population's decoded direction controls cursor movement
            4. Higher noise = less accurate control!
            """)


# =============================================================================
# Tab 6: Learn Decoding (Step-by-Step Walkthrough)
# =============================================================================

with tab6:
    st.header("📚 Learn How Decoding Works")
    st.markdown("""
    Walk through the decoding process step-by-step! See exactly how neural activity
    is transformed into a direction estimate.
    """)
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("📖 Tutorial Controls")
            
            decoder_type = st.radio(
                "Select Decoder",
                ["Population Vector", "Maximum Likelihood"],
                key="walkthrough_decoder"
            )
            
            st.markdown("---")
            
            if st.button("🎲 New Example", type="primary", use_container_width=True):
                # Generate new random trial
                theta = np.random.uniform(0, 2 * np.pi)
                spikes = simulate_trial(
                    theta,
                    neurons,
                    duration_ms=st.session_state.duration_ms,
                    variance_scale=st.session_state.variance_scale
                )
                st.session_state.walkthrough_theta = theta
                st.session_state.walkthrough_spikes = spikes
                st.session_state.walkthrough_step = 0
                st.rerun()
            
            st.markdown("---")
            
            if st.session_state.walkthrough_spikes is not None:
                max_step = 3
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("⬅️ Back", disabled=st.session_state.walkthrough_step <= 0):
                        st.session_state.walkthrough_step -= 1
                        st.rerun()
                with c2:
                    st.write(f"Step {st.session_state.walkthrough_step + 1}/4")
                with c3:
                    if st.button("Next ➡️", disabled=st.session_state.walkthrough_step >= max_step):
                        st.session_state.walkthrough_step += 1
                        st.rerun()
                
                # Progress bar
                st.progress((st.session_state.walkthrough_step + 1) / 4)
                
                # Step descriptions
                if decoder_type == "Population Vector":
                    step_desc = [
                        "**Step 1:** Observe the spike counts from each neuron. Neurons tuned to the true direction fire more!",
                        "**Step 2:** Each neuron contributes a vector pointing in its preferred direction, scaled by spike count.",
                        "**Step 3:** Sum all the individual vectors to get one population vector.",
                        "**Step 4:** The direction of the summed vector is our decoded estimate!"
                    ]
                else:
                    step_desc = [
                        "**Step 1:** Observe the spike counts from the neural population.",
                        "**Step 2:** For each possible direction, calculate how likely it is to produce this pattern.",
                        "**Step 3:** The likelihood function shows probability for each direction.",
                        "**Step 4:** The peak of the likelihood is our decoded estimate!"
                    ]
                
                st.markdown(step_desc[st.session_state.walkthrough_step])
        
        with col1:
            if st.session_state.walkthrough_spikes is not None:
                if decoder_type == "Population Vector":
                    # Show polar vector animation
                    if st.session_state.walkthrough_step >= 1:
                        n_vectors = min(
                            neurons.n_neurons,
                            (st.session_state.walkthrough_step) * neurons.n_neurons // 3
                        )
                        if st.session_state.walkthrough_step == 3:
                            n_vectors = neurons.n_neurons
                        
                        fig = create_vector_animation_polar(
                            st.session_state.walkthrough_spikes,
                            neurons,
                            n_vectors_shown=n_vectors,
                            true_theta=st.session_state.walkthrough_theta
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Also show bar chart
                    fig_step = create_pv_decoder_step(
                        st.session_state.walkthrough_spikes,
                        neurons,
                        step=st.session_state.walkthrough_step,
                        true_theta=st.session_state.walkthrough_theta
                    )
                    st.plotly_chart(fig_step, use_container_width=True)
                    
                else:
                    # Maximum likelihood walkthrough
                    fig_ml = create_ml_decoder_step(
                        st.session_state.walkthrough_spikes,
                        neurons,
                        step=st.session_state.walkthrough_step,
                        duration_s=st.session_state.duration_ms / 1000,
                        true_theta=st.session_state.walkthrough_theta
                    )
                    st.plotly_chart(fig_ml, use_container_width=True)
                
                # Show result summary at final step
                if st.session_state.walkthrough_step == 3:
                    true_deg = radians_to_degrees(st.session_state.walkthrough_theta)
                    
                    if decoder_type == "Population Vector":
                        decoder = PopulationVectorDecoder()
                        decoded = decoder.decode(st.session_state.walkthrough_spikes, neurons)
                    else:
                        decoder = MaximumLikelihoodDecoder()
                        decoded = decoder.decode(
                            st.session_state.walkthrough_spikes, neurons,
                            duration_s=st.session_state.duration_ms / 1000
                        )
                    
                    decoded_deg = radians_to_degrees(decoded)
                    error = angular_error_degrees(true_deg, decoded_deg)
                    
                    st.success(f"""
                    **Decoding Complete!**
                    - True Direction: **{true_deg:.0f}°**
                    - Decoded Direction: **{decoded_deg:.0f}°**
                    - Error: **{error:.1f}°**
                    """)
            else:
                st.info("👆 Click **New Example** to generate a trial and start the walkthrough!")
                
                # Show explanation
                st.markdown("""
                ### What You'll Learn
                
                This interactive tutorial walks you through how neural decoders work:
                
                **Population Vector Decoder:**
                - Each neuron "votes" for its preferred direction
                - Votes are weighted by spike count
                - The sum of all votes gives the decoded direction
                
                **Maximum Likelihood Decoder:**
                - For each possible direction, ask: "How likely is this spike pattern?"
                - The direction with highest likelihood wins
                - Mathematically optimal for Poisson neurons!
                """)


# =============================================================================
# Tab 7: Brain Areas
# =============================================================================

with tab7:
    st.header("🧠 Multi-Brain-Area Simulation")
    st.markdown("""
    Explore how different brain areas encode movement direction! Each area has unique
    properties - from sharp motor cortex tuning to broader parietal representations.
    """)
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("⚙️ Network Settings")
            
            n_per_area = st.slider(
                "Neurons per Area",
                20, 100, 50,
                help="Number of neurons in each brain area"
            )
            
            if st.button("🧠 Create Network", type="primary", use_container_width=True):
                with st.spinner("Creating hierarchical network..."):
                    hierarchy = create_hierarchical_network(
                        n_neurons_per_area=n_per_area,
                        baseline_rate=st.session_state.baseline_rate,
                        modulation_depth=st.session_state.modulation_depth
                    )
                    st.session_state.hierarchy = hierarchy
                st.success("Network created!")
                st.rerun()
            
            if st.session_state.hierarchy is not None:
                st.markdown("---")
                st.subheader("🎯 Simulate Direction")
                
                sim_direction = st.slider(
                    "Movement Direction (°)",
                    0, 359, 45,
                    key="hierarchy_direction"
                )
                
                if st.button("▶️ Simulate All Areas", use_container_width=True):
                    theta = degrees_to_radians(sim_direction)
                    with st.spinner("Simulating hierarchy..."):
                        hierarchy_data = st.session_state.hierarchy.simulate_hierarchy(
                            theta,
                            duration_ms=st.session_state.duration_ms,
                            variance_scale=st.session_state.variance_scale
                        )
                        st.session_state.hierarchy_data = hierarchy_data
                        st.session_state.hierarchy_theta = theta
                    st.rerun()
        
        with col1:
            if st.session_state.hierarchy is not None:
                hierarchy = st.session_state.hierarchy
                
                # Show connectivity matrix
                st.subheader("Brain Area Connectivity")
                conn_matrix, area_names = hierarchy.get_connectivity_matrix()
                fig_conn = plot_brain_connectivity(conn_matrix, area_names)
                st.plotly_chart(fig_conn, use_container_width=True)
                
                # Show area descriptions
                st.subheader("Brain Areas")
                for name in hierarchy.get_hierarchy_order():
                    area = hierarchy.get_area(name)
                    with st.expander(f"**{name}** - {area.description}"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Neurons", area.neurons.n_neurons)
                        with c2:
                            st.metric("Tuning Sharpness", f"{area.tuning_sharpness:.1f}")
                        with c3:
                            st.metric("Delay", f"{area.delay_ms:.0f} ms")
                
                # Show simulated activity if available
                if st.session_state.hierarchy_data is not None:
                    st.markdown("---")
                    st.subheader("Simulated Activity Across Areas")
                    
                    # Create neurons dict for visualization
                    neurons_dict = {name: hierarchy.get_area(name).neurons 
                                   for name in hierarchy.get_area_names()}
                    
                    fig_areas = plot_area_comparison(
                        st.session_state.hierarchy_data,
                        neurons_dict,
                        st.session_state.hierarchy_theta
                    )
                    st.plotly_chart(fig_areas, use_container_width=True)
                    
                    # Decode from each area
                    st.subheader("Decoding from Each Area")
                    decoder = PopulationVectorDecoder()
                    
                    area_results = []
                    for name in hierarchy.get_area_names():
                        area = hierarchy.get_area(name)
                        spikes = st.session_state.hierarchy_data[name]
                        decoded = decoder.decode(spikes, area.neurons)
                        error = angular_error_degrees(
                            radians_to_degrees(st.session_state.hierarchy_theta),
                            radians_to_degrees(decoded)
                        )
                        area_results.append({
                            'Area': name,
                            'Decoded (°)': f"{radians_to_degrees(decoded):.0f}",
                            'Error (°)': f"{error:.1f}"
                        })
                    
                    st.table(area_results)
            else:
                st.info("👆 Click **Create Network** to build a hierarchical brain model!")


# =============================================================================
# Tab 8: Neural Manifold
# =============================================================================

with tab8:
    st.header("🌀 Neural Manifold Visualization")
    st.markdown("""
    See how high-dimensional neural activity lives on a low-dimensional manifold!
    PCA reveals the underlying structure of population activity.
    """)
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        spike_counts = st.session_state.spike_counts
        true_directions = st.session_state.true_directions
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("⚙️ Settings")
            
            n_components = st.slider(
                "PCA Components",
                2, min(10, neurons.n_neurons),
                3,
                help="Number of principal components to extract"
            )
            
            if st.button("🔄 Compute Manifold", type="primary", use_container_width=True):
                with st.spinner("Computing PCA..."):
                    manifold_data, model, explained_var = compute_neural_manifold(
                        spike_counts,
                        n_components=n_components
                    )
                    st.session_state.manifold_data = manifold_data
                    st.session_state.manifold_model = model
                    st.session_state.explained_variance = explained_var
                st.success("Manifold computed!")
                st.rerun()
            
            if st.session_state.manifold_data is not None:
                st.markdown("---")
                st.subheader("📊 View Options")
                
                show_3d = st.checkbox("Show 3D Plot", value=True)
                show_trajectories = st.checkbox("Show Trajectories", value=False)
                
                if not show_3d:
                    pc_x = st.selectbox("X-axis PC", range(n_components), index=0)
                    pc_y = st.selectbox("Y-axis PC", range(n_components), index=1)
        
        with col1:
            if st.session_state.manifold_data is not None:
                # Variance explained plot
                st.subheader("Variance Explained")
                fig_var = plot_variance_explained(
                    st.session_state.explained_variance,
                    n_components=len(st.session_state.explained_variance)
                )
                st.plotly_chart(fig_var, use_container_width=True)
                
                # Manifold visualization
                st.subheader("Neural State Space")
                
                if show_3d and st.session_state.manifold_data.shape[1] >= 3:
                    fig_manifold = plot_neural_manifold_3d(
                        st.session_state.manifold_data,
                        true_directions,
                        show_trajectories=show_trajectories
                    )
                else:
                    fig_manifold = plot_neural_manifold_2d(
                        st.session_state.manifold_data,
                        true_directions,
                        pc_x=pc_x if 'pc_x' in dir() else 0,
                        pc_y=pc_y if 'pc_y' in dir() else 1
                    )
                
                st.plotly_chart(fig_manifold, use_container_width=True)
                
                # Summary stats
                total_var = np.sum(st.session_state.explained_variance) * 100
                st.info(f"**{total_var:.1f}%** of variance explained by {len(st.session_state.explained_variance)} PCs")
            else:
                st.info("👆 Click **Compute Manifold** to visualize the neural state space!")
                
                st.markdown("""
                ### What is a Neural Manifold?
                
                Even though we record from many neurons, their activity is often 
                correlated. PCA finds the main "directions" of variation in the 
                population, revealing a low-dimensional manifold.
                
                **Colors represent movement direction** - if the manifold is 
                well-organized, similar directions should cluster together!
                """)


# =============================================================================
# Tab 9: Challenge Modes
# =============================================================================

with tab9:
    st.header("🏆 Challenge Modes")
    st.markdown("""
    Test your neural decoding skills in competitive challenge modes!
    Earn achievements and climb the leaderboard.
    """)
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        challenge_mgr = st.session_state.challenge_manager
        achievement_mgr = st.session_state.achievement_manager
        
        # Check if challenge is active
        if not st.session_state.challenge_active:
            # Mode selection
            st.subheader("Select Challenge Mode")
            
            col1, col2 = st.columns(2)
            
            with col1:
                mode_options = {
                    "Speed Trial ⚡": ChallengeMode.SPEED_TRIAL,
                    "Precision 🎯": ChallengeMode.PRECISION,
                    "Noise Gauntlet 📡": ChallengeMode.NOISE_GAUNTLET,
                    "Streak Challenge 🔥": ChallengeMode.STREAK
                }
                
                selected_mode_name = st.radio(
                    "Choose your challenge:",
                    list(mode_options.keys())
                )
                selected_mode = mode_options[selected_mode_name]
                
                config = CHALLENGE_CONFIGS[selected_mode]
                
                st.markdown(f"**{config.name}**")
                st.markdown(config.description)
                st.markdown(f"*Difficulty: {config.difficulty.upper()}*")
                
                player_name = st.text_input("Your Name", value="Player")
                
                if st.button("🚀 Start Challenge!", type="primary"):
                    challenge_mgr.start_challenge(selected_mode)
                    st.session_state.challenge_active = True
                    # Generate first trial
                    theta = np.random.uniform(0, 2 * np.pi)
                    spikes = simulate_trial(
                        theta, neurons,
                        duration_ms=st.session_state.duration_ms,
                        variance_scale=st.session_state.variance_scale * challenge_mgr.current_noise_level
                    )
                    st.session_state.challenge_theta = theta
                    st.session_state.challenge_spikes = spikes
                    st.session_state.player_name = player_name
                    st.rerun()
            
            with col2:
                # Show leaderboard
                st.subheader("🏅 Leaderboard")
                leaderboard = challenge_mgr.get_leaderboard(selected_mode)
                fig_lb = plot_leaderboard(leaderboard)
                st.plotly_chart(fig_lb, use_container_width=True)
                
                # Show achievements
                st.subheader("🏆 Your Achievements")
                earned, total = achievement_mgr.get_earned_count()
                st.progress(earned / total if total > 0 else 0)
                st.caption(f"{earned}/{total} achievements earned")
        
        else:
            # Active challenge
            state = challenge_mgr.get_state()
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader(f"📊 {state['mode'].replace('_', ' ').title()}")
                
                # Progress metrics
                st.metric("Trials", state['trials_completed'])
                st.metric("Mean Error", f"{state['mean_error']:.1f}°")
                
                if state['time_remaining'] is not None:
                    st.metric("Time Left", f"{state['time_remaining']:.0f}s")
                
                if state['trials_remaining'] is not None:
                    st.metric("Trials Left", state['trials_remaining'])
                
                if state['mode'] == 'streak':
                    st.metric("Current Streak", state['current_streak'])
                    st.metric("Best Streak", state['best_streak'])
                
                if state['mode'] == 'noise_gauntlet':
                    st.metric("Noise Level", f"{state['current_noise_level']:.1f}x")
                
                st.markdown("---")
                
                if st.button("❌ End Challenge", use_container_width=True):
                    result = challenge_mgr.finish_challenge(st.session_state.player_name)
                    newly_earned = achievement_mgr.check_achievements(result)
                    
                    st.session_state.challenge_active = False
                    st.session_state.last_challenge_result = result
                    st.session_state.new_achievements = newly_earned
                    st.rerun()
            
            with col1:
                # Check if challenge is over
                if challenge_mgr.is_challenge_over():
                    result = challenge_mgr.finish_challenge(st.session_state.player_name)
                    newly_earned = achievement_mgr.check_achievements(result)
                    
                    st.success(f"🎉 Challenge Complete! Score: **{result.score:.1f}**")
                    st.metric("Final Mean Error", f"{result.mean_error:.1f}°")
                    st.metric("Trials Completed", result.trials_completed)
                    
                    if newly_earned:
                        st.balloons()
                        for achievement in newly_earned:
                            st.success(f"🏆 Achievement Unlocked: **{achievement.name}**!")
                    
                    st.session_state.challenge_active = False
                    st.rerun()
                else:
                    # Show current trial
                    st.subheader("🧩 Decode This Pattern!")
                    
                    fig_pattern = plot_population_bar(
                        st.session_state.challenge_spikes,
                        neurons,
                        true_theta=None,
                        as_polar=True
                    )
                    st.plotly_chart(fig_pattern, use_container_width=True)
                    
                    # User input
                    user_guess = st.slider("Your Guess (°)", 0, 359, 180, key="challenge_guess")
                    
                    if st.button("✅ Submit", type="primary"):
                        true_theta = st.session_state.challenge_theta
                        error = angular_error_degrees(user_guess, radians_to_degrees(true_theta))
                        
                        # Record trial
                        challenge_mgr.record_trial(error)
                        
                        # Show result briefly
                        if error < 20:
                            st.success(f"Great! Error: {error:.1f}°")
                        elif error < 40:
                            st.info(f"Good! Error: {error:.1f}°")
                        else:
                            st.warning(f"Error: {error:.1f}°")
                        
                        # Generate next trial
                        theta = np.random.uniform(0, 2 * np.pi)
                        spikes = simulate_trial(
                            theta, neurons,
                            duration_ms=st.session_state.duration_ms,
                            variance_scale=st.session_state.variance_scale * challenge_mgr.current_noise_level
                        )
                        st.session_state.challenge_theta = theta
                        st.session_state.challenge_spikes = spikes
                        st.rerun()


# =============================================================================
# Tab 10: Analysis (Stretch Features)
# =============================================================================

with tab10:
    st.header("🔬 Analysis")
    
    if not st.session_state.simulated:
        st.warning("⚠️ Please run a simulation first (Tab 1)")
    else:
        neurons = st.session_state.neurons
        
        analysis_type = st.selectbox(
            "Select Analysis",
            ["Noise vs Performance", "Normal vs Lesioned", "Export Data"]
        )
        
        if analysis_type == "Noise vs Performance":
            st.subheader("Decoder Performance vs. Noise Level")
            st.markdown("""
            How does decoder accuracy change as neural variability increases?
            Higher variance scale means more noisy/variable spike counts.
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                n_test_trials = st.slider("Trials per noise level", 20, 100, 50)
                noise_levels = st.multiselect(
                    "Noise levels to test",
                    [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                )
                
                decoder_type = st.radio(
                    "Decoder",
                    ["Population Vector", "Maximum Likelihood"]
                )
                
                run_analysis = st.button("🔬 Run Analysis", type="primary")
            
            with col1:
                if run_analysis and noise_levels:
                    decoder = PopulationVectorDecoder() if decoder_type == "Population Vector" else MaximumLikelihoodDecoder()
                    
                    mean_errors = []
                    std_errors = []
                    
                    progress = st.progress(0)
                    
                    for i, noise in enumerate(sorted(noise_levels)):
                        # Simulate trials at this noise level
                        spike_counts, true_dirs = simulate_random_trials(
                            n_trials=n_test_trials,
                            neurons=neurons,
                            duration_ms=st.session_state.duration_ms,
                            variance_scale=noise
                        )
                        
                        # Evaluate decoder
                        results = evaluate_decoder(
                            decoder,
                            spike_counts,
                            true_dirs,
                            neurons,
                            duration_s=st.session_state.duration_ms / 1000
                        )
                        
                        mean_errors.append(results['mean_error_degrees'])
                        std_errors.append(results['std_error_degrees'])
                        progress.progress((i + 1) / len(noise_levels))
                    
                    # Plot results
                    fig = plot_decoder_performance_vs_noise(
                        np.array(sorted(noise_levels)),
                        np.array(mean_errors),
                        np.array(std_errors),
                        decoder_name=decoder_type
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("Analysis complete!")
                else:
                    st.info("Configure the analysis parameters and click **Run Analysis**")
        
        elif analysis_type == "Normal vs Lesioned":
            st.subheader("Normal vs. Lesioned Population")
            st.markdown("""
            Compare decoder performance between a normal population and a 
            "lesioned" population with reduced tuning strength.
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                lesion_type = st.radio(
                    "Lesion type",
                    ["Reduce modulation depth", "Reduce baseline rate"]
                )
                lesion_factor = st.slider(
                    "Lesion severity (multiplier)",
                    0.1, 0.9, 0.5,
                    help="How much to reduce the parameter (0.5 = 50% reduction)"
                )
                n_test_trials = st.slider("Trials to simulate", 20, 100, 50, key="lesion_trials")
                
                run_lesion = st.button("🔬 Run Comparison", type="primary")
            
            with col1:
                if run_lesion:
                    # Create lesioned population
                    lesioned = create_lesioned_population(
                        neurons,
                        lesion_factor=lesion_factor,
                        lesion_type='modulation' if 'modulation' in lesion_type else 'baseline'
                    )
                    
                    decoder = PopulationVectorDecoder()
                    
                    # Simulate normal
                    with st.spinner("Simulating normal condition..."):
                        normal_counts, normal_dirs = simulate_random_trials(
                            n_test_trials, neurons, st.session_state.duration_ms,
                            st.session_state.variance_scale
                        )
                        normal_results = evaluate_decoder(
                            decoder, normal_counts, normal_dirs, neurons,
                            st.session_state.duration_ms / 1000
                        )
                    
                    # Simulate lesioned
                    with st.spinner("Simulating lesioned condition..."):
                        lesioned_counts, lesioned_dirs = simulate_random_trials(
                            n_test_trials, lesioned, st.session_state.duration_ms,
                            st.session_state.variance_scale
                        )
                        lesioned_results = evaluate_decoder(
                            decoder, lesioned_counts, lesioned_dirs, lesioned,
                            st.session_state.duration_ms / 1000
                        )
                    
                    # Show tuning comparison
                    st.markdown("#### Tuning Curve Comparison")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_normal = plot_tuning_curves(neurons, opacity=0.3)
                        fig_normal.update_layout(title="Normal Population")
                        st.plotly_chart(fig_normal, use_container_width=True)
                    with c2:
                        fig_lesion = plot_tuning_curves(lesioned, opacity=0.3)
                        fig_lesion.update_layout(title="Lesioned Population")
                        st.plotly_chart(fig_lesion, use_container_width=True)
                    
                    # Show performance comparison
                    st.markdown("#### Decoder Performance Comparison")
                    fig_compare = plot_condition_comparison(
                        radians_to_degrees(normal_results['errors']),
                        radians_to_degrees(lesioned_results['errors'])
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Summary stats
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(
                            "Normal Mean Error",
                            f"{normal_results['mean_error_degrees']:.1f}°"
                        )
                    with c2:
                        st.metric(
                            "Lesioned Mean Error",
                            f"{lesioned_results['mean_error_degrees']:.1f}°",
                            delta=f"+{lesioned_results['mean_error_degrees'] - normal_results['mean_error_degrees']:.1f}°"
                        )
                else:
                    st.info("Configure the lesion parameters and click **Run Comparison**")
        
        else:  # Export Data
            st.subheader("Export Simulation Data")
            st.markdown("""
            Download the current simulation data for further analysis.
            Includes neuron parameters, spike counts, and true directions.
            """)
            
            export_format = st.radio("Export format", ["NPZ (NumPy)", "CSV"])
            
            if st.button("📥 Prepare Export", type="primary"):
                # Prepare export data
                export_data = {
                    'preferred_directions': neurons.preferred_directions,
                    'baseline_rate': neurons.baseline_rate,
                    'modulation_depth': neurons.modulation_depth,
                    'n_neurons': neurons.n_neurons,
                    'spike_counts': st.session_state.spike_counts,
                    'true_directions': st.session_state.true_directions,
                    'duration_ms': st.session_state.duration_ms,
                    'variance_scale': st.session_state.variance_scale
                }
                
                if export_format == "NPZ (NumPy)":
                    data_bytes = export_to_npz(export_data, "simulation")
                    st.download_button(
                        label="⬇️ Download NPZ",
                        data=data_bytes,
                        file_name="decode_my_brain_simulation.npz",
                        mime="application/octet-stream"
                    )
                else:
                    csv_str = export_to_csv(export_data, "simulation")
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_str,
                        file_name="decode_my_brain_simulation.csv",
                        mime="text/csv"
                    )
                
                st.success("Export ready! Click the download button above.")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Decode My Brain | Built with Streamlit & Plotly</p>
    <p>An interactive tool for exploring neural coding and decoding concepts</p>
</div>
""", unsafe_allow_html=True)

