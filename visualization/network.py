"""Brain network connectivity and leaderboard visualizations."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional, List

from utils import radians_to_degrees
from visualization.colors import get_direction_color


@st.cache_data
def plot_brain_connectivity(
    connectivity_matrix: np.ndarray,
    area_names: List[str]
) -> go.Figure:
    """
    Create a heatmap of connectivity between brain areas.

    Args:
        connectivity_matrix: Matrix of connection strengths
        area_names: Names of brain areas

    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=connectivity_matrix,
        x=area_names,
        y=area_names,
        colorscale='Blues',
        colorbar=dict(title='Connection<br>Strength'),
        hovertemplate='%{y} → %{x}<br>Strength: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title="Brain Area Connectivity",
        xaxis_title="Target Area",
        yaxis_title="Source Area",
        template='plotly_white',
        height=400,
        width=500
    )

    return fig


@st.cache_data
def plot_area_comparison(
    area_data: dict,
    neurons_dict: dict,
    true_theta: float
) -> go.Figure:
    """
    Compare activity across brain areas for a single trial.

    Args:
        area_data: Dict mapping area name to spike counts
        neurons_dict: Dict mapping area name to NeuronPopulation
        true_theta: True movement direction

    Returns:
        Plotly Figure with subplots
    """
    n_areas = len(area_data)

    fig = make_subplots(
        rows=1, cols=n_areas,
        subplot_titles=list(area_data.keys()),
        specs=[[{'type': 'barpolar'}] * n_areas]
    )

    for i, (area_name, spikes) in enumerate(area_data.items()):
        neurons = neurons_dict[area_name]
        sort_idx = np.argsort(neurons.preferred_directions)
        sorted_spikes = spikes[sort_idx]
        sorted_pref = neurons.preferred_directions[sort_idx]

        colors = [get_direction_color(p) for p in sorted_pref]

        fig.add_trace(
            go.Barpolar(
                r=sorted_spikes,
                theta=radians_to_degrees(sorted_pref),
                marker_color=colors,
                opacity=0.7,
                name=area_name,
                showlegend=False
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        title=f"Population Activity Across Areas (True Direction: {radians_to_degrees(true_theta):.0f}°)",
        template='plotly_white',
        height=350
    )

    return fig


def plot_leaderboard(
    scores: List[dict],
    current_score: Optional[float] = None
) -> go.Figure:
    """
    Create a leaderboard table display.

    Args:
        scores: List of score dictionaries with keys: rank, name, score, trials, date
        current_score: Current user's score to highlight

    Returns:
        Plotly Figure with table
    """
    if not scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No scores yet! Be the first to compete!",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=200)
        return fig

    # Prepare data
    ranks = [s.get('rank', i+1) for i, s in enumerate(scores)]
    names = [s.get('name', f'Player {i+1}') for i, s in enumerate(scores)]
    score_vals = [f"{s.get('score', 0):.1f}°" for s in scores]
    trials = [s.get('trials', 0) for s in scores]

    # Highlight current score
    colors = ['#d4edda' if current_score and abs(s.get('score', 0) - current_score) < 0.1
              else 'white' for s in scores]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'Player', 'Score', 'Trials'],
            fill_color='#2c3e50',
            font=dict(color='white', size=14),
            align='center'
        ),
        cells=dict(
            values=[ranks, names, score_vals, trials],
            fill_color=[colors],
            align='center',
            height=30
        )
    )])

    fig.update_layout(
        title="Leaderboard",
        height=min(400, 100 + len(scores) * 35),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig
