"""BCI cursor control and metrics visualizations."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple

from visualization.colors import TRUE_COLOR, USER_COLOR, MODEL_COLOR


def create_bci_canvas(
    cursor_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    decoded_direction: Optional[float] = None,
    cursor_trail: Optional[List[Tuple[float, float]]] = None,
    canvas_size: float = 200.0
) -> go.Figure:
    """
    Create a BCI cursor control canvas.

    Args:
        cursor_pos: Current cursor position (x, y)
        target_pos: Target position (x, y)
        decoded_direction: Decoded movement direction (radians)
        cursor_trail: List of previous cursor positions
        canvas_size: Size of the canvas (square)

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Add target zone
    target_radius = 20
    theta_circle = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter(
        x=target_pos[0] + target_radius * np.cos(theta_circle),
        y=target_pos[1] + target_radius * np.sin(theta_circle),
        mode='lines',
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.3)',
        line=dict(color=TRUE_COLOR, width=2),
        name='Target',
        showlegend=True
    ))

    # Add cursor trail
    if cursor_trail and len(cursor_trail) > 1:
        trail_x = [p[0] for p in cursor_trail]
        trail_y = [p[1] for p in cursor_trail]

        # Color gradient for trail (older = more transparent)
        n_points = len(cursor_trail)
        for i in range(n_points - 1):
            alpha = 0.2 + 0.6 * (i / n_points)
            fig.add_trace(go.Scatter(
                x=[trail_x[i], trail_x[i+1]],
                y=[trail_y[i], trail_y[i+1]],
                mode='lines',
                line=dict(color=f'rgba(52, 152, 219, {alpha})', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add cursor
    fig.add_trace(go.Scatter(
        x=[cursor_pos[0]],
        y=[cursor_pos[1]],
        mode='markers',
        marker=dict(size=20, color=USER_COLOR, symbol='circle'),
        name='Cursor'
    ))

    # Add decoded direction arrow
    if decoded_direction is not None:
        arrow_length = 30
        arrow_x = cursor_pos[0] + arrow_length * np.cos(decoded_direction)
        arrow_y = cursor_pos[1] + arrow_length * np.sin(decoded_direction)

        fig.add_annotation(
            x=arrow_x, y=arrow_y,
            ax=cursor_pos[0], ay=cursor_pos[1],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=MODEL_COLOR
        )

    # Add center marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=10, color='gray', symbol='x'),
        name='Center',
        showlegend=False
    ))

    fig.update_layout(
        title="BCI Cursor Control",
        xaxis=dict(
            range=[-canvas_size/2, canvas_size/2],
            showgrid=True,
            zeroline=True,
            scaleanchor='y'
        ),
        yaxis=dict(
            range=[-canvas_size/2, canvas_size/2],
            showgrid=True,
            zeroline=True
        ),
        template='plotly_white',
        height=500,
        width=500,
        showlegend=True,
        legend=dict(x=1.02, y=1)
    )

    return fig


def create_bci_metrics_display(
    time_elapsed: float,
    distance_to_target: float,
    path_length: float,
    n_targets_hit: int,
    n_attempts: int
) -> go.Figure:
    """
    Create a metrics display for BCI performance.

    Args:
        time_elapsed: Time since trial start (seconds)
        distance_to_target: Current distance to target
        path_length: Total path traveled
        n_targets_hit: Number of successful acquisitions
        n_attempts: Total attempts

    Returns:
        Plotly Figure with indicator gauges
    """
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'indicator'}] * 4]
    )

    fig.add_trace(go.Indicator(
        mode='number',
        value=time_elapsed,
        title={'text': 'Time (s)'},
        number={'suffix': 's', 'font': {'size': 30}}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode='number',
        value=distance_to_target,
        title={'text': 'Distance'},
        number={'font': {'size': 30}}
    ), row=1, col=2)

    efficiency = (np.sqrt(2) * 100 / path_length * 100) if path_length > 0 else 100
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=min(100, efficiency),
        title={'text': 'Efficiency'},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': TRUE_COLOR if efficiency > 50 else MODEL_COLOR}}
    ), row=1, col=3)

    fig.add_trace(go.Indicator(
        mode='number',
        value=n_targets_hit,
        title={'text': 'Hits'},
        number={'suffix': f'/{n_attempts}', 'font': {'size': 30}}
    ), row=1, col=4)

    fig.update_layout(height=150, margin=dict(t=50, b=10))

    return fig
