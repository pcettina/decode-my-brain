"""Analyze page - Noise analysis, lesion comparison, and data export.

Provides three analysis modes selected via a dropdown:
  - **Noise vs Performance**: sweeps variance scale and plots decoder
    error vs. noise level for PV or ML decoders.
  - **Normal vs Lesioned**: compares decoder accuracy between intact
    and lesioned populations (reduced modulation or baseline).
  - **Export Data**: download spike counts, directions, and neuron
    parameters as NPZ or CSV.
"""

import logging

import numpy as np
import streamlit as st

from simulation import (
    simulate_random_trials,
    create_lesioned_population,
)
from decoders import PopulationVectorDecoder, MaximumLikelihoodDecoder, evaluate_decoder
from visualization import (
    plot_tuning_curves,
    plot_decoder_performance_vs_noise,
    plot_condition_comparison,
)
from utils import radians_to_degrees, export_to_npz, export_to_csv

logger = logging.getLogger(__name__)

if not st.session_state.get('simulated', False):
    st.warning("Please run a simulation using the sidebar controls.")
    st.stop()

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

        run_analysis = st.button("Run Analysis", type="primary")

    with col1:
        if run_analysis and noise_levels:
          try:
            decoder = PopulationVectorDecoder() if decoder_type == "Population Vector" else MaximumLikelihoodDecoder()

            mean_errors = []
            std_errors = []

            progress = st.progress(0)

            for i, noise in enumerate(sorted(noise_levels)):
                spike_counts, true_dirs = simulate_random_trials(
                    n_trials=n_test_trials,
                    neurons=neurons,
                    duration_ms=st.session_state.duration_ms,
                    variance_scale=noise
                )

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

            fig = plot_decoder_performance_vs_noise(
                np.array(sorted(noise_levels)),
                np.array(mean_errors),
                np.array(std_errors),
                decoder_name=decoder_type
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success("Analysis complete!")
          except Exception:
            logger.error("Analysis failed", exc_info=True)
            st.error("Analysis failed. Try different parameters.")
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

        run_lesion = st.button("Run Comparison", type="primary")

    with col1:
        if run_lesion:
            lesioned = create_lesioned_population(
                neurons,
                lesion_factor=lesion_factor,
                lesion_type='modulation' if 'modulation' in lesion_type else 'baseline'
            )

            decoder = PopulationVectorDecoder()

            with st.spinner("Simulating normal condition..."):
                normal_counts, normal_dirs = simulate_random_trials(
                    n_test_trials, neurons, st.session_state.duration_ms,
                    st.session_state.variance_scale
                )
                normal_results = evaluate_decoder(
                    decoder, normal_counts, normal_dirs, neurons,
                    st.session_state.duration_ms / 1000
                )

            with st.spinner("Simulating lesioned condition..."):
                lesioned_counts, lesioned_dirs = simulate_random_trials(
                    n_test_trials, lesioned, st.session_state.duration_ms,
                    st.session_state.variance_scale
                )
                lesioned_results = evaluate_decoder(
                    decoder, lesioned_counts, lesioned_dirs, lesioned,
                    st.session_state.duration_ms / 1000
                )

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

            st.markdown("#### Decoder Performance Comparison")
            fig_compare = plot_condition_comparison(
                radians_to_degrees(normal_results['errors']),
                radians_to_degrees(lesioned_results['errors'])
            )
            st.plotly_chart(fig_compare, use_container_width=True)

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

    if st.button("Prepare Export", type="primary"):
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
                label="Download NPZ",
                data=data_bytes,
                file_name="decode_my_brain_simulation.npz",
                mime="application/octet-stream"
            )
        else:
            csv_str = export_to_csv(export_data, "simulation")
            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name="decode_my_brain_simulation.csv",
                mime="text/csv"
            )

        st.success("Export ready! Click the download button above.")
