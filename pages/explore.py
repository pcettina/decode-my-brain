"""Explore page - Live activity, BCI, brain areas, and neural manifold.

Contains four tabs:
  - **Live Activity**: temporal spike simulation with adaptation, refractory
    periods, and burst firing.  View raster snapshots at any time point.
  - **BCI Simulator**: cursor control using population-vector decoding.
    Adjust speed and noise to see how decoding accuracy affects BCI control.
  - **Brain Areas**: hierarchical network (M1, PMd, PPC, SMA) with
    inter-area connectivity, per-area decoding comparison.
  - **Neural Manifold**: PCA dimensionality reduction of population
    activity with 2D/3D visualization and variance-explained plots.
"""

import logging

import numpy as np
import streamlit as st

from simulation import (
    TemporalParams,
    simulate_temporal_spikes,
    create_hierarchical_network,
)
from decoders import PopulationVectorDecoder
from visualization import (
    create_spike_raster_snapshot,
    create_bci_canvas,
    plot_brain_connectivity,
    plot_area_comparison,
    compute_neural_manifold,
    plot_neural_manifold_3d,
    plot_neural_manifold_2d,
    plot_variance_explained,
)
from utils import radians_to_degrees, degrees_to_radians, angular_error_degrees

logger = logging.getLogger(__name__)

st.session_state.visited_explore = True

if not st.session_state.get('simulated', False):
    st.warning("Please run a simulation using the sidebar controls.")
    st.stop()

neurons = st.session_state.neurons

tab_live, tab_bci, tab_brain, tab_manifold = st.tabs([
    "Live Activity", "BCI Simulator", "Brain Areas", "Neural Manifold"
])


# =========================================================================
# Live Activity
# =========================================================================

with tab_live:
    st.header("Live Neural Activity")
    st.markdown("""
    Watch neural activity unfold in real-time! This visualization shows spikes
    appearing as the neurons fire, with **temporal dynamics** including adaptation,
    refractory periods, and bursting.
    """)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Temporal Dynamics")

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
        st.subheader("Simulation")

        live_direction = st.slider(
            "Movement Direction (°)",
            0, 359, 90,
            help="Direction of simulated movement"
        )
        live_duration = st.slider(
            "Duration (ms)",
            500, 3000, 1500, step=100
        )

        if st.button("Generate Activity", type="primary", use_container_width=True):
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
            view_time = st.slider(
                "View Time (ms)",
                100, int(st.session_state.live_time_ms),
                int(st.session_state.live_time_ms),
                step=50,
                key="live_view_time"
            )

            fig_raster = create_spike_raster_snapshot(
                st.session_state.live_spike_times,
                neurons,
                current_time_ms=view_time,
                window_ms=min(500, view_time),
                true_theta=st.session_state.live_theta
            )
            st.plotly_chart(fig_raster, use_container_width=True)

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
            st.info("Configure parameters and click **Generate Activity** to see live neural firing!")


# =========================================================================
# BCI Simulator
# =========================================================================

with tab_bci:
    st.header("BCI Cursor Control Simulator")
    st.markdown("""
    Experience brain-computer interface (BCI) control! Neural activity is decoded
    in real-time to move a cursor toward targets. See how decoding accuracy
    affects control quality.
    """)

    pv_decoder = PopulationVectorDecoder()
    bci = st.session_state.bci_sim

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("BCI Controls")

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

        if st.button("New Target", type="primary", use_container_width=True):
            bci.new_target()
            st.rerun()

        if st.button("Move Cursor", use_container_width=True):
            if bci.target_pos is not None:
                _, _, hit = bci.move_cursor(
                    neurons, pv_decoder,
                    cursor_speed=cursor_speed,
                    noise_level=noise_level,
                )
                if hit:
                    st.balloons()
                st.rerun()

        if st.button("Reset Stats", use_container_width=True):
            bci.reset_stats()
            st.rerun()

        st.markdown("---")
        st.subheader("Performance")

        st.metric("Targets Hit", bci.hits)
        st.metric("Attempts", bci.attempts)
        if bci.attempts > 0:
            hit_rate = bci.hits / bci.attempts * 100
            st.metric("Hit Rate", f"{hit_rate:.0f}%")

    with col1:
        decoded_dir = None
        if bci.target_pos is not None and len(bci.trail) > 1:
            prev = bci.trail[-2]
            curr = bci.trail[-1]
            if prev != curr:
                decoded_dir = np.arctan2(curr[1] - prev[1], curr[0] - prev[0])

        fig_bci = create_bci_canvas(
            cursor_pos=bci.cursor_pos,
            target_pos=bci.target_pos if bci.target_pos else (50, 50),
            decoded_direction=decoded_dir,
            cursor_trail=bci.trail if len(bci.trail) > 1 else None,
            canvas_size=200.0,
        )

        if bci.target_pos is None:
            fig_bci.update_layout(title="BCI Canvas - Click 'New Target' to start!")

        st.plotly_chart(fig_bci, use_container_width=True)

        st.info("""
        **How to play:**
        1. Click **New Target** to place a target (green circle)
        2. Click **Move Cursor** repeatedly to move toward the target
        3. Your neural population's decoded direction controls cursor movement
        4. Higher noise = less accurate control!
        """)


# =========================================================================
# Brain Areas
# =========================================================================

with tab_brain:
    st.header("Multi-Brain-Area Simulation")
    st.markdown("""
    Explore how different brain areas encode movement direction! Each area has unique
    properties - from sharp motor cortex tuning to broader parietal representations.
    """)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Network Settings")

        n_per_area = st.slider(
            "Neurons per Area",
            20, 100, 50,
            help="Number of neurons in each brain area"
        )

        if st.button("Create Network", type="primary", use_container_width=True):
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
            st.subheader("Simulate Direction")

            sim_direction = st.slider(
                "Movement Direction (°)",
                0, 359, 45,
                key="hierarchy_direction"
            )

            if st.button("Simulate All Areas", use_container_width=True):
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

            st.subheader("Brain Area Connectivity")
            conn_matrix, area_names = hierarchy.get_connectivity_matrix()
            fig_conn = plot_brain_connectivity(conn_matrix, area_names)
            st.plotly_chart(fig_conn, use_container_width=True)

            st.subheader("Brain Areas")
            for name in hierarchy.get_hierarchy_order():
                area = hierarchy.get_area(name)
                with st.expander(f"**{name}** - {area.description}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Neurons", area.neurons.n_neurons)
                    with c2:
                        st.metric("Modulation Gain", f"{area.modulation_gain:.1f}")
                    with c3:
                        st.metric("Delay", f"{area.delay_ms:.0f} ms")

            if st.session_state.hierarchy_data is not None:
                st.markdown("---")
                st.subheader("Simulated Activity Across Areas")

                neurons_dict = {name: hierarchy.get_area(name).neurons
                               for name in hierarchy.get_area_names()}

                fig_areas = plot_area_comparison(
                    st.session_state.hierarchy_data,
                    neurons_dict,
                    st.session_state.hierarchy_theta
                )
                st.plotly_chart(fig_areas, use_container_width=True)

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
            st.info("Click **Create Network** to build a hierarchical brain model!")


# =========================================================================
# Neural Manifold
# =========================================================================

with tab_manifold:
    st.header("Neural Manifold Visualization")
    st.markdown("""
    See how high-dimensional neural activity lives on a low-dimensional manifold!
    PCA reveals the underlying structure of population activity.
    """)

    spike_counts = st.session_state.spike_counts
    true_directions = st.session_state.true_directions

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Settings")

        n_components = st.slider(
            "PCA Components",
            2, min(10, neurons.n_neurons),
            3,
            help="Number of principal components to extract"
        )

        if st.button("Compute Manifold", type="primary", use_container_width=True):
            try:
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
            except Exception:
                logger.error("Manifold computation failed", exc_info=True)
                st.error("Manifold computation failed. Try different parameters.")

        if st.session_state.manifold_data is not None:
            st.markdown("---")
            st.subheader("View Options")

            show_3d = st.checkbox("Show 3D Plot", value=True)
            show_trajectories = st.checkbox("Show Trajectories", value=False)

            if not show_3d:
                pc_x = st.selectbox("X-axis PC", range(n_components), index=0)
                pc_y = st.selectbox("Y-axis PC", range(n_components), index=1)

    with col1:
        if st.session_state.manifold_data is not None:
            st.subheader("Variance Explained")
            fig_var = plot_variance_explained(
                st.session_state.explained_variance,
                n_components=len(st.session_state.explained_variance)
            )
            st.plotly_chart(fig_var, use_container_width=True)

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

            total_var = np.sum(st.session_state.explained_variance) * 100
            st.info(f"**{total_var:.1f}%** of variance explained by {len(st.session_state.explained_variance)} PCs")
        else:
            st.info("Click **Compute Manifold** to visualize the neural state space!")

            st.markdown("""
            ### What is a Neural Manifold?

            Even though we record from many neurons, their activity is often
            correlated. PCA finds the main "directions" of variation in the
            population, revealing a low-dimensional manifold.

            **Colors represent movement direction** - if the manifold is
            well-organized, similar directions should cluster together!
            """)
