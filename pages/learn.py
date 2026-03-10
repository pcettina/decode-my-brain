"""Learn page - Visualize neural code and learn decoding step-by-step.

Contains two tabs:
  - **Visualize Neural Code**: tuning curves with opacity/trial highlight
    controls, single-trial spike heatmaps, population bar/polar plots,
    and an expandable time-binned raster.
  - **Learn Decoding**: a 4-step interactive walkthrough for both
    Population Vector and Maximum Likelihood decoders.  Shows how spike
    counts are converted to direction estimates.
"""

import numpy as np
import streamlit as st

from simulation import simulate_trial, simulate_raster
from decoders import PopulationVectorDecoder, MaximumLikelihoodDecoder
from visualization import (
    plot_tuning_curves,
    plot_raster_heatmap,
    plot_population_bar,
    create_pv_decoder_step,
    create_ml_decoder_step,
    create_vector_animation_polar,
)
from utils import radians_to_degrees, degrees_to_radians, angular_error_degrees


if not st.session_state.get('simulated', False):
    st.warning("Please run a simulation using the sidebar controls.")
    st.stop()

neurons = st.session_state.neurons
spike_counts = st.session_state.spike_counts
true_directions = st.session_state.true_directions

tab_vis, tab_learn = st.tabs(["Visualize Neural Code", "Learn Decoding"])


# =========================================================================
# Visualize Neural Code
# =========================================================================

with tab_vis:
    st.header("Visualize Neural Code")

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
    with st.expander("View Time-Binned Raster (click to expand)"):
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


# =========================================================================
# Learn Decoding (Step-by-Step Walkthrough)
# =========================================================================

with tab_learn:
    st.header("Learn How Decoding Works")
    st.markdown("""
    Walk through the decoding process step-by-step! See exactly how neural activity
    is transformed into a direction estimate.
    """)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Tutorial Controls")

        decoder_type = st.radio(
            "Select Decoder",
            ["Population Vector", "Maximum Likelihood"],
            key="walkthrough_decoder"
        )

        st.markdown("---")

        if st.button("New Example", type="primary", use_container_width=True):
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
                if st.button("Back", disabled=st.session_state.walkthrough_step <= 0):
                    st.session_state.walkthrough_step -= 1
                    st.rerun()
            with c2:
                st.write(f"Step {st.session_state.walkthrough_step + 1}/4")
            with c3:
                if st.button("Next", disabled=st.session_state.walkthrough_step >= max_step):
                    st.session_state.walkthrough_step += 1
                    st.rerun()

            st.progress((st.session_state.walkthrough_step + 1) / 4)

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

                fig_step = create_pv_decoder_step(
                    st.session_state.walkthrough_spikes,
                    neurons,
                    step=st.session_state.walkthrough_step,
                    true_theta=st.session_state.walkthrough_theta
                )
                st.plotly_chart(fig_step, use_container_width=True)

            else:
                fig_ml = create_ml_decoder_step(
                    st.session_state.walkthrough_spikes,
                    neurons,
                    step=st.session_state.walkthrough_step,
                    duration_s=st.session_state.duration_ms / 1000,
                    true_theta=st.session_state.walkthrough_theta
                )
                st.plotly_chart(fig_ml, use_container_width=True)

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
            st.info("Click **New Example** to generate a trial and start the walkthrough!")

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
