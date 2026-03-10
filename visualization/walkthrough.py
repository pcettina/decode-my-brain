"""Decoder walkthrough step-by-step visualizations."""

import numpy as np
import plotly.graph_objects as go
from typing import Optional

from simulation.core import NeuronPopulation
from utils import radians_to_degrees, wrap_angle
from visualization.colors import TRUE_COLOR, MODEL_COLOR, get_direction_color


def create_pv_decoder_step(
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    step: int,
    true_theta: Optional[float] = None
) -> go.Figure:
    """
    Create a visualization for one step of population vector decoding.

    Steps:
        0: Show spike counts as bars
        1: Show individual neuron vectors
        2: Show vector sum building up
        3: Show final decoded direction

    Args:
        spike_counts: Spike counts per neuron
        neurons: NeuronPopulation
        step: Which step to visualize (0-3)
        true_theta: True direction for comparison

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Sort by preferred direction
    sort_idx = np.argsort(neurons.preferred_directions)
    sorted_counts = spike_counts[sort_idx]
    sorted_pref = neurons.preferred_directions[sort_idx]

    if step >= 0:
        # Step 0: Show spike counts
        colors = [get_direction_color(p) for p in sorted_pref]
        fig.add_trace(go.Bar(
            x=radians_to_degrees(sorted_pref),
            y=sorted_counts,
            marker_color=colors,
            opacity=0.7,
            name='Spike Counts',
            showlegend=True
        ))

    if step >= 1:
        # Step 1: Show individual contribution vectors
        max_count = max(sorted_counts) if max(sorted_counts) > 0 else 1
        n_show = min(10, len(sorted_counts))  # Show top 10 contributors
        top_indices = np.argsort(sorted_counts)[-n_show:]

        for idx in top_indices:
            if sorted_counts[idx] > 0:
                pref = sorted_pref[idx]
                magnitude = sorted_counts[idx] / max_count * 50

                # Draw vector from origin
                fig.add_annotation(
                    x=magnitude * np.cos(pref) + 180,
                    y=magnitude * np.sin(pref) + max_count/2,
                    ax=180, ay=max_count/2,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=get_direction_color(pref),
                    opacity=0.6
                )

    if step >= 2:
        # Step 2: Show cumulative sum vector
        sum_x = np.sum(sorted_counts * np.cos(sorted_pref))
        sum_y = np.sum(sorted_counts * np.sin(sorted_pref))

        # Normalize for display
        magnitude = np.sqrt(sum_x**2 + sum_y**2)
        if magnitude > 0:
            scale = 50 / max(sorted_counts) if max(sorted_counts) > 0 else 1
            fig.add_annotation(
                x=sum_x * scale + 180,
                y=sum_y * scale + max(sorted_counts)/2,
                ax=180, ay=max(sorted_counts)/2,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=4,
                arrowcolor='#9b59b6'
            )

    if step >= 3:
        # Step 3: Show decoded direction
        sum_x = np.sum(sorted_counts * np.cos(sorted_pref))
        sum_y = np.sum(sorted_counts * np.sin(sorted_pref))
        decoded = np.arctan2(sum_y, sum_x)
        decoded_deg = radians_to_degrees(wrap_angle(decoded))

        fig.add_vline(
            x=decoded_deg,
            line_dash="solid",
            line_color='#9b59b6',
            line_width=3,
            annotation_text=f"Decoded: {decoded_deg:.0f}°",
            annotation_position="top"
        )

        if true_theta is not None:
            true_deg = radians_to_degrees(wrap_angle(true_theta))
            fig.add_vline(
                x=true_deg,
                line_dash="dash",
                line_color=TRUE_COLOR,
                line_width=3,
                annotation_text=f"True: {true_deg:.0f}°",
                annotation_position="bottom"
            )

    step_titles = [
        "Step 1: Observe Spike Counts",
        "Step 2: Each Neuron Contributes a Vector",
        "Step 3: Sum All Vectors",
        "Step 4: Decoded Direction!"
    ]

    fig.update_layout(
        title=step_titles[min(step, 3)],
        xaxis_title="Preferred Direction (degrees)",
        yaxis_title="Spike Count",
        xaxis=dict(range=[0, 360]),
        template='plotly_white',
        height=400
    )

    return fig


def create_ml_decoder_step(
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    step: int,
    duration_s: float = 0.5,
    true_theta: Optional[float] = None
) -> go.Figure:
    """
    Create visualization for maximum likelihood decoder steps.

    Steps:
        0: Show spike counts
        1: Show likelihood being computed (partial)
        2: Show full likelihood curve
        3: Show peak = decoded direction

    Args:
        spike_counts: Spike counts per neuron
        neurons: NeuronPopulation
        step: Which step (0-3)
        duration_s: Trial duration
        true_theta: True direction

    Returns:
        Plotly Figure
    """
    from decoders import MaximumLikelihoodDecoder

    ml = MaximumLikelihoodDecoder(n_grid_points=360)
    theta_grid, likelihoods = ml.get_likelihood_curve(spike_counts, neurons, duration_s)

    fig = go.Figure()
    theta_deg = radians_to_degrees(theta_grid)

    if step == 0:
        # Just show spike counts
        sort_idx = np.argsort(neurons.preferred_directions)
        sorted_counts = spike_counts[sort_idx]
        sorted_pref_deg = radians_to_degrees(neurons.preferred_directions[sort_idx])
        colors = [get_direction_color(neurons.preferred_directions[i]) for i in sort_idx]

        fig.add_trace(go.Bar(
            x=sorted_pref_deg,
            y=sorted_counts,
            marker_color=colors,
            name='Spike Counts'
        ))
        title = "Step 1: Observe the Spike Pattern"

    elif step == 1:
        # Show partial likelihood (first half)
        n_show = len(theta_deg) // 2
        fig.add_trace(go.Scatter(
            x=theta_deg[:n_show],
            y=likelihoods[:n_show],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.3)',
            line=dict(color='#9b59b6', width=2),
            name='Likelihood (computing...)'
        ))
        title = "Step 2: Computing Likelihood for Each Direction..."

    elif step == 2:
        # Show full likelihood
        fig.add_trace(go.Scatter(
            x=theta_deg,
            y=likelihoods,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.3)',
            line=dict(color='#9b59b6', width=2),
            name='Likelihood'
        ))
        title = "Step 3: Full Likelihood Curve"

    else:  # step >= 3
        # Show full likelihood with peak
        fig.add_trace(go.Scatter(
            x=theta_deg,
            y=likelihoods,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.3)',
            line=dict(color='#9b59b6', width=2),
            name='Likelihood'
        ))

        # Mark peak
        peak_idx = np.argmax(likelihoods)
        decoded_deg = theta_deg[peak_idx]

        fig.add_trace(go.Scatter(
            x=[decoded_deg],
            y=[likelihoods[peak_idx]],
            mode='markers',
            marker=dict(size=15, color=MODEL_COLOR, symbol='star'),
            name=f'Peak: {decoded_deg:.0f}°'
        ))

        fig.add_vline(
            x=decoded_deg,
            line_dash="solid",
            line_color=MODEL_COLOR,
            line_width=2
        )

        if true_theta is not None:
            true_deg = radians_to_degrees(wrap_angle(true_theta))
            fig.add_vline(
                x=true_deg,
                line_dash="dash",
                line_color=TRUE_COLOR,
                line_width=2,
                annotation_text=f"True: {true_deg:.0f}°",
                annotation_position="top"
            )

        title = f"Step 4: Maximum Likelihood = {decoded_deg:.0f}°"

    fig.update_layout(
        title=title,
        xaxis_title="Direction (degrees)",
        yaxis_title="Likelihood" if step > 0 else "Spike Count",
        xaxis=dict(range=[0, 360]),
        template='plotly_white',
        height=400
    )

    return fig


def create_vector_animation_polar(
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    n_vectors_shown: int,
    true_theta: Optional[float] = None
) -> go.Figure:
    """
    Create a polar plot showing vectors being summed progressively.

    Args:
        spike_counts: Spike counts per neuron
        neurons: NeuronPopulation
        n_vectors_shown: How many neuron vectors to show (for animation)
        true_theta: True direction

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Sort by spike count (show highest contributors first)
    sort_idx = np.argsort(spike_counts)[::-1]

    max_count = max(spike_counts) if max(spike_counts) > 0 else 1

    # Running sum
    sum_x, sum_y = 0, 0

    for i in range(min(n_vectors_shown, len(spike_counts))):
        idx = sort_idx[i]
        count = spike_counts[idx]
        if count == 0:
            continue

        pref = neurons.preferred_directions[idx]
        pref_deg = radians_to_degrees(pref)

        # Normalized magnitude
        magnitude = count / max_count

        # Add individual vector
        fig.add_trace(go.Scatterpolar(
            r=[0, magnitude],
            theta=[pref_deg, pref_deg],
            mode='lines',
            line=dict(color=get_direction_color(pref), width=2),
            opacity=0.5,
            showlegend=False
        ))

        sum_x += count * np.cos(pref)
        sum_y += count * np.sin(pref)

    # Add sum vector
    if n_vectors_shown > 0:
        sum_magnitude = np.sqrt(sum_x**2 + sum_y**2) / (max_count * n_vectors_shown) * 2
        sum_theta = np.arctan2(sum_y, sum_x)
        sum_theta_deg = radians_to_degrees(wrap_angle(sum_theta))

        fig.add_trace(go.Scatterpolar(
            r=[0, min(sum_magnitude, 1.5)],
            theta=[sum_theta_deg, sum_theta_deg],
            mode='lines+markers',
            line=dict(color='#9b59b6', width=4),
            marker=dict(size=[0, 12], symbol=['circle', 'triangle-up']),
            name=f'Sum Vector ({sum_theta_deg:.0f}°)'
        ))

    # Add true direction
    if true_theta is not None:
        true_deg = radians_to_degrees(wrap_angle(true_theta))
        fig.add_trace(go.Scatterpolar(
            r=[0, 1.2],
            theta=[true_deg, true_deg],
            mode='lines',
            line=dict(color=TRUE_COLOR, width=3, dash='dash'),
            name=f'True ({true_deg:.0f}°)'
        ))

    fig.update_layout(
        title=f"Population Vector: {n_vectors_shown} neurons summed",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.5]),
            angularaxis=dict(direction='counterclockwise')
        ),
        template='plotly_white',
        height=450,
        showlegend=True
    )

    return fig
