"""Decoder performance analysis and scoreboard plots."""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List

from utils import radians_to_degrees, wrap_angle
from visualization.colors import TRUE_COLOR, USER_COLOR, MODEL_COLOR


def plot_decoder_performance_vs_noise(
    noise_levels: np.ndarray,
    mean_errors: np.ndarray,
    std_errors: Optional[np.ndarray] = None,
    decoder_name: str = "Decoder"
) -> go.Figure:
    """
    Plot decoder performance as a function of noise level.

    Args:
        noise_levels: Array of noise/variance scale values
        mean_errors: Mean angular errors (in degrees)
        std_errors: Standard deviation of errors (optional)
        decoder_name: Name for legend

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=noise_levels,
        y=mean_errors,
        mode='lines+markers',
        name=decoder_name,
        line=dict(color=MODEL_COLOR, width=2),
        marker=dict(size=8),
        error_y=dict(
            type='data',
            array=std_errors if std_errors is not None else None,
            visible=std_errors is not None
        )
    ))

    # Add reference line for chance performance (90°)
    fig.add_hline(
        y=90,
        line_dash="dot",
        line_color="gray",
        annotation_text="Chance (90°)",
        annotation_position="right"
    )

    fig.update_layout(
        title="Decoder Performance vs. Noise Level",
        xaxis_title="Variance Scale",
        yaxis_title="Mean Angular Error (degrees)",
        yaxis=dict(range=[0, 100]),
        template='plotly_white',
        height=400,
        showlegend=True
    )

    return fig


def plot_condition_comparison(
    normal_errors: np.ndarray,
    lesioned_errors: np.ndarray
) -> go.Figure:
    """
    Compare decoder performance between normal and lesioned conditions.

    Args:
        normal_errors: Errors for normal condition (degrees)
        lesioned_errors: Errors for lesioned condition (degrees)

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Create box plots
    fig.add_trace(go.Box(
        y=normal_errors,
        name='Normal',
        marker_color='#2ecc71',
        boxmean='sd'
    ))

    fig.add_trace(go.Box(
        y=lesioned_errors,
        name='Lesioned',
        marker_color='#e74c3c',
        boxmean='sd'
    ))

    fig.update_layout(
        title="Decoder Performance: Normal vs. Lesioned",
        yaxis_title="Angular Error (degrees)",
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def plot_likelihood_curve(
    theta_grid: np.ndarray,
    likelihoods: np.ndarray,
    true_theta: Optional[float] = None,
    decoded_theta: Optional[float] = None
) -> go.Figure:
    """
    Plot the likelihood curve from ML decoder.

    Args:
        theta_grid: Array of candidate directions (radians)
        likelihoods: Normalized likelihood values
        true_theta: True direction (optional)
        decoded_theta: Decoded direction (optional)

    Returns:
        Plotly Figure
    """
    theta_deg = radians_to_degrees(theta_grid)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=theta_deg,
        y=likelihoods,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#9b59b6', width=2),
        fillcolor='rgba(155, 89, 182, 0.3)',
        name='Likelihood'
    ))

    if true_theta is not None:
        true_deg = radians_to_degrees(wrap_angle(true_theta))
        fig.add_vline(
            x=true_deg,
            line_dash="dash",
            line_color=TRUE_COLOR,
            line_width=2,
            annotation_text=f"True: {true_deg:.0f}°"
        )

    if decoded_theta is not None:
        decoded_deg = radians_to_degrees(wrap_angle(decoded_theta))
        fig.add_vline(
            x=decoded_deg,
            line_dash="dash",
            line_color=MODEL_COLOR,
            line_width=2,
            annotation_text=f"Decoded: {decoded_deg:.0f}°"
        )

    fig.update_layout(
        title="Likelihood Function",
        xaxis_title="Direction (degrees)",
        yaxis_title="Normalized Likelihood",
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 90, 180, 270, 360],
            range=[0, 360]
        ),
        template='plotly_white',
        height=300
    )

    return fig


def create_scoreboard_table(
    rounds: List[dict]
) -> go.Figure:
    """
    Create a table showing game round history.

    Args:
        rounds: List of round result dictionaries

    Returns:
        Plotly Figure with table
    """
    if not rounds:
        return go.Figure()

    # Extract data
    round_nums = [i + 1 for i in range(len(rounds))]
    true_dirs = [f"{r['true_deg']:.0f}°" for r in rounds]
    user_guesses = [f"{r['user_deg']:.0f}°" for r in rounds]
    model_decodes = [f"{r['model_deg']:.0f}°" for r in rounds]
    user_errors = [f"{r['user_error']:.1f}°" for r in rounds]
    model_errors = [f"{r['model_error']:.1f}°" for r in rounds]
    winners = [r['winner'] for r in rounds]

    # Color cells based on winner
    user_colors = ['#d4edda' if w == 'User' else '#ffffff' for w in winners]
    model_colors = ['#f8d7da' if w == 'Model' else '#ffffff' for w in winners]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Round', 'True', 'Your Guess', 'Model', 'Your Error', 'Model Error', 'Winner'],
            fill_color='#34495e',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[round_nums, true_dirs, user_guesses, model_decodes,
                    user_errors, model_errors, winners],
            fill_color=[
                ['white'] * len(rounds),
                ['white'] * len(rounds),
                user_colors,
                model_colors,
                user_colors,
                model_colors,
                ['#d4edda' if w == 'User' else '#f8d7da' if w == 'Model' else '#fff3cd'
                 for w in winners]
            ],
            align='center',
            height=25
        )
    )])

    fig.update_layout(
        title="Game History",
        height=min(400, 100 + len(rounds) * 30),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig
