"""Setup page - Simulation overview and configuration summary.

Displays population statistics (neuron count, trial count, mean/std firing
rate) computed from the current simulation stored in session state.
Simulation parameters are configured via the sidebar in app.py.
"""

import streamlit as st

from simulation import compute_firing_rate_stats


st.header("Simulation Overview")
st.markdown("""
Configure the neural population using the **sidebar controls**, then explore
results here. Each neuron has a **preferred direction** and responds according
to a cosine tuning curve:

> lambda(theta) = r0 + k * cos(theta - mu)

where r0 is the baseline rate, k is the modulation depth, and mu is the
preferred direction.
""")

if not st.session_state.get('simulated', False):
    st.warning("Please run a simulation using the sidebar controls.")
    st.stop()

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
