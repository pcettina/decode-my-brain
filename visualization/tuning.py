"""Tuning curve and population activity plots."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Optional

from simulation.core import NeuronPopulation
from utils import radians_to_degrees, wrap_angle
from visualization.colors import TRUE_COLOR, USER_COLOR, MODEL_COLOR, get_direction_color


@st.cache_data
def plot_tuning_curves(
    neurons: NeuronPopulation,
    highlight_theta: Optional[float] = None,
    opacity: float = 0.5,
    show_highlight: bool = True
) -> go.Figure:
    """Plot tuning curves for all neurons (grouped by direction bins)."""
    theta_rad = np.linspace(0, 2 * np.pi, 361)
    theta_deg = radians_to_degrees(theta_rad)

    fig = go.Figure()

    # Group neurons into 8 direction bins to reduce trace count
    n_bins = 8
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        indices = [i for i in range(neurons.n_neurons)
                   if lo <= neurons.preferred_directions[i] < hi]
        if not indices:
            continue

        color = get_direction_color(bin_centers[b])
        seg_x: list = []
        seg_y: list = []
        for i in indices:
            rates = neurons.get_tuning_curve(theta_rad, i)
            seg_x.extend(list(theta_deg) + [None])
            seg_y.extend(list(rates) + [None])

        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode='lines',
            line=dict(color=color, width=1.5),
            opacity=opacity, showlegend=False,
            hovertemplate='Direction: %{x:.0f}°<br>Rate: %{y:.1f} Hz<extra></extra>',
        ))

    if show_highlight and highlight_theta is not None:
        theta_deg_highlight = radians_to_degrees(wrap_angle(highlight_theta))
        fig.add_vline(
            x=theta_deg_highlight, line_dash="dash",
            line_color=TRUE_COLOR, line_width=2,
            annotation_text=f"True: {theta_deg_highlight:.0f}°",
            annotation_position="top"
        )

    fig.update_layout(
        title="Neural Tuning Curves",
        xaxis_title="Movement Direction (degrees)",
        yaxis_title="Firing Rate (Hz)",
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 90, 180, 270, 360],
            ticktext=['0° (E)', '90° (N)', '180° (W)', '270° (S)', '360° (E)'],
            range=[0, 360]
        ),
        yaxis=dict(rangemode='tozero'),
        hovermode='closest', template='plotly_white', height=400
    )
    return fig


@st.cache_data
def plot_population_bar(
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    true_theta: Optional[float] = None,
    as_polar: bool = False
) -> go.Figure:
    """Plot spike counts vs neuron preferred direction."""
    sort_idx = np.argsort(neurons.preferred_directions)
    sorted_counts = spike_counts[sort_idx]
    sorted_pref = neurons.preferred_directions[sort_idx]
    sorted_pref_deg = radians_to_degrees(sorted_pref)
    colors = [get_direction_color(p) for p in sorted_pref]

    if as_polar:
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=sorted_counts, theta=sorted_pref_deg,
            width=360 / len(sorted_pref) * 0.8,
            marker_color=colors, opacity=0.8,
            hovertemplate='Preferred: %{theta:.0f}°<br>Spikes: %{r}<extra></extra>'
        ))
        if true_theta is not None:
            max_count = max(sorted_counts) if len(sorted_counts) > 0 else 1
            true_deg = radians_to_degrees(wrap_angle(true_theta))
            fig.add_trace(go.Scatterpolar(
                r=[0, max_count * 1.2], theta=[true_deg, true_deg],
                mode='lines',
                line=dict(color=TRUE_COLOR, width=3, dash='dash'),
                name=f'True Direction ({true_deg:.0f}°)'
            ))
        fig.update_layout(
            title="Population Activity (Polar)",
            polar=dict(
                radialaxis=dict(title='Spike Count'),
                angularaxis=dict(
                    direction='counterclockwise',
                    tickmode='array',
                    tickvals=[0, 90, 180, 270],
                    ticktext=['0° (E)', '90° (N)', '180° (W)', '270° (S)']
                )
            ),
            template='plotly_white', height=500,
            showlegend=true_theta is not None
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_pref_deg, y=sorted_counts,
            marker_color=colors, opacity=0.8,
            hovertemplate='Preferred: %{x:.0f}°<br>Spikes: %{y}<extra></extra>',
            showlegend=False
        ))
        if true_theta is not None:
            true_deg = radians_to_degrees(wrap_angle(true_theta))
            fig.add_vline(
                x=true_deg, line_dash="dash",
                line_color=TRUE_COLOR, line_width=3,
                annotation_text=f"True: {true_deg:.0f}°",
                annotation_position="top"
            )
        fig.update_layout(
            title="Population Activity",
            xaxis_title="Preferred Direction (degrees)",
            yaxis_title="Spike Count",
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 90, 180, 270, 360],
                ticktext=['0° (E)', '90° (N)', '180° (W)', '270° (S)', '360° (E)']
            ),
            template='plotly_white', height=400
        )
    return fig


@st.cache_data
def plot_polar_comparison(
    true_theta: float,
    user_theta: Optional[float] = None,
    model_theta: Optional[float] = None,
    show_legend: bool = True
) -> go.Figure:
    """Create polar plot showing true, user, and model directions as arrows."""
    fig = go.Figure()
    arrow_length = 1.0

    true_deg = radians_to_degrees(wrap_angle(true_theta))
    fig.add_trace(go.Scatterpolar(
        r=[0, arrow_length], theta=[true_deg, true_deg],
        mode='lines+markers',
        line=dict(color=TRUE_COLOR, width=4),
        marker=dict(size=[0, 15], symbol=['circle', 'triangle-up']),
        name=f'True ({true_deg:.0f}°)'
    ))

    if user_theta is not None:
        user_deg = radians_to_degrees(wrap_angle(user_theta))
        fig.add_trace(go.Scatterpolar(
            r=[0, arrow_length * 0.95], theta=[user_deg, user_deg],
            mode='lines+markers',
            line=dict(color=USER_COLOR, width=4),
            marker=dict(size=[0, 15], symbol=['circle', 'triangle-up']),
            name=f'Your Guess ({user_deg:.0f}°)'
        ))

    if model_theta is not None:
        model_deg = radians_to_degrees(wrap_angle(model_theta))
        fig.add_trace(go.Scatterpolar(
            r=[0, arrow_length * 0.9], theta=[model_deg, model_deg],
            mode='lines+markers',
            line=dict(color=MODEL_COLOR, width=4),
            marker=dict(size=[0, 15], symbol=['circle', 'triangle-up']),
            name=f'Model ({model_deg:.0f}°)'
        ))

    fig.update_layout(
        title="Direction Comparison",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.2]),
            angularaxis=dict(
                direction='counterclockwise',
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
            )
        ),
        template='plotly_white', height=400,
        showlegend=show_legend,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05)
    )
    return fig
