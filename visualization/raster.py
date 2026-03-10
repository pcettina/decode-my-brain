"""Raster and heatmap spike visualizations."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, List

from simulation.core import NeuronPopulation
from utils import radians_to_degrees, wrap_angle
from visualization.colors import TRUE_COLOR, get_direction_color


@st.cache_data
def plot_raster_heatmap(
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    true_theta: Optional[float] = None,
    sort_by_preferred: bool = True
) -> go.Figure:
    """
    Plot spike counts as a heatmap.

    Args:
        spike_counts: Array of spike counts for each neuron (1D or 2D for raster)
        neurons: NeuronPopulation object
        true_theta: True movement direction (for annotation)
        sort_by_preferred: If True, sort neurons by preferred direction

    Returns:
        Plotly Figure
    """
    # Handle 1D spike counts (single time bin)
    if spike_counts.ndim == 1:
        spike_counts = spike_counts.reshape(-1, 1)

    n_neurons, n_bins = spike_counts.shape

    # Sort by preferred direction if requested
    if sort_by_preferred:
        sort_idx = np.argsort(neurons.preferred_directions)
        spike_counts = spike_counts[sort_idx]
        pref_dirs = neurons.preferred_directions[sort_idx]
    else:
        sort_idx = np.arange(n_neurons)
        pref_dirs = neurons.preferred_directions

    # Create y-axis labels
    y_labels = [f'{radians_to_degrees(pref_dirs[i]):.0f}°' for i in range(n_neurons)]

    # Create x-axis labels (time bins or single "Count")
    if n_bins == 1:
        x_labels = ['Spike Count']
    else:
        x_labels = [f'{i*10}ms' for i in range(n_bins)]  # Assuming 10ms bins

    fig = go.Figure(data=go.Heatmap(
        z=spike_counts,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        colorbar=dict(title='Spikes'),
        hovertemplate=(
            'Preferred Dir: %{y}<br>'
            '%{x}: %{z} spikes<extra></extra>'
        )
    ))

    # Add marker for true direction neuron
    if true_theta is not None:
        # Find the neuron closest to true direction
        diffs = np.abs(pref_dirs - wrap_angle(true_theta))
        diffs = np.minimum(diffs, 2*np.pi - diffs)
        closest_idx = np.argmin(diffs)

        fig.add_annotation(
            x=x_labels[-1] if n_bins > 1 else x_labels[0],
            y=y_labels[closest_idx],
            text=f"← True: {radians_to_degrees(true_theta):.0f}°",
            showarrow=True,
            arrowhead=2,
            arrowcolor=TRUE_COLOR,
            font=dict(color=TRUE_COLOR, size=12),
            ax=50,
            ay=0
        )

    fig.update_layout(
        title="Spike Count Heatmap (sorted by preferred direction)",
        xaxis_title="Time Bin" if n_bins > 1 else "",
        yaxis_title="Preferred Direction",
        height=max(400, n_neurons * 8),
        template='plotly_white'
    )

    return fig


def create_spike_raster_snapshot(
    spike_times: List[List[float]],
    neurons: NeuronPopulation,
    current_time_ms: float,
    window_ms: float = 500.0,
    true_theta: Optional[float] = None
) -> go.Figure:
    """
    Create a spike raster plot showing spikes in a time window.

    Args:
        spike_times: List of spike time lists per neuron
        neurons: NeuronPopulation object
        current_time_ms: Current time (right edge of window)
        window_ms: Width of time window to show
        true_theta: True movement direction for highlighting

    Returns:
        Plotly Figure with spike raster
    """
    fig = go.Figure()

    n_neurons = len(spike_times)
    start_time = max(0, current_time_ms - window_ms)

    # Sort neurons by preferred direction
    sort_idx = np.argsort(neurons.preferred_directions)

    # Collect all spikes into single arrays for one scatter trace
    all_spike_x: list = []
    all_spike_y: list = []
    all_spike_colors: list = []
    all_hover: list = []

    for plot_idx, neuron_idx in enumerate(sort_idx):
        times = spike_times[neuron_idx]
        pref_dir = neurons.preferred_directions[neuron_idx]
        color = get_direction_color(pref_dir)

        # Filter spikes in current window
        visible_spikes = [t for t in times if start_time <= t <= current_time_ms]

        if visible_spikes:
            all_spike_x.extend(visible_spikes)
            all_spike_y.extend([plot_idx] * len(visible_spikes))
            all_spike_colors.extend([color] * len(visible_spikes))
            all_hover.extend([
                f'Neuron {neuron_idx+1}<br>'
                f'Pref: {radians_to_degrees(pref_dir):.0f}°<br>'
                f'Time: {t:.1f} ms'
                for t in visible_spikes
            ])

    if all_spike_x:
        fig.add_trace(go.Scatter(
            x=all_spike_x,
            y=all_spike_y,
            mode='markers',
            marker=dict(
                symbol='line-ns',
                size=12,
                color=all_spike_colors,
                line=dict(width=2, color=all_spike_colors),
            ),
            showlegend=False,
            text=all_hover,
            hoverinfo='text',
        ))

    # Highlight neurons near true direction
    if true_theta is not None:
        for plot_idx, neuron_idx in enumerate(sort_idx):
            pref_dir = neurons.preferred_directions[neuron_idx]
            diff = abs(pref_dir - true_theta)
            diff = min(diff, 2*np.pi - diff)
            if diff < np.pi/8:  # Within ~22.5 degrees
                fig.add_hrect(
                    y0=plot_idx - 0.4, y1=plot_idx + 0.4,
                    fillcolor=TRUE_COLOR, opacity=0.1,
                    line_width=0
                )

    # Create y-axis labels (preferred directions)
    y_labels = [f'{radians_to_degrees(neurons.preferred_directions[i]):.0f}°'
                for i in sort_idx]

    fig.update_layout(
        title=f"Live Spike Raster (t = {current_time_ms:.0f} ms)",
        xaxis_title="Time (ms)",
        yaxis_title="Neuron (by preferred direction)",
        xaxis=dict(range=[start_time, current_time_ms + 10]),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, n_neurons, max(1, n_neurons//10))),
            ticktext=[y_labels[i] for i in range(0, n_neurons, max(1, n_neurons//10))],
            range=[-0.5, n_neurons - 0.5]
        ),
        template='plotly_white',
        height=500
    )

    return fig
