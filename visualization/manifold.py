"""Neural manifold / PCA dimensionality reduction visualizations."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional, Tuple

from utils import radians_to_degrees
from visualization.colors import get_direction_color


@st.cache_data
def compute_neural_manifold(
    spike_data: np.ndarray,
    n_components: int = 3,
    method: str = 'pca'
) -> Tuple[np.ndarray, object, np.ndarray]:
    """
    Reduce population activity to low-dimensional representation.

    Args:
        spike_data: Array of shape (n_trials, n_neurons)
        n_components: Number of dimensions to reduce to
        method: 'pca' or 'tsne'

    Returns:
        Tuple of (manifold_data, model, explained_variance)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(spike_data)

    if method == 'pca':
        model = PCA(n_components=n_components)
        manifold_data = model.fit_transform(scaled_data)
        explained_var = model.explained_variance_ratio_
    else:
        # Default to PCA
        model = PCA(n_components=n_components)
        manifold_data = model.fit_transform(scaled_data)
        explained_var = model.explained_variance_ratio_

    return manifold_data, model, explained_var


@st.cache_data
def plot_neural_manifold_3d(
    manifold_data: np.ndarray,
    true_directions: np.ndarray,
    decoded_directions: Optional[np.ndarray] = None,
    show_trajectories: bool = False
) -> go.Figure:
    """
    Create an interactive 3D scatter plot of neural manifold.

    Args:
        manifold_data: Array of shape (n_trials, 3) from PCA
        true_directions: True movement directions (radians)
        decoded_directions: Optional decoded directions for comparison
        show_trajectories: Whether to connect sequential points

    Returns:
        Plotly Figure with 3D scatter
    """
    # Color by direction
    colors = [get_direction_color(theta) for theta in true_directions]

    # Create hover text
    hover_text = [
        f"Trial {i+1}<br>Direction: {radians_to_degrees(true_directions[i]):.0f}°"
        for i in range(len(true_directions))
    ]

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=manifold_data[:, 0],
        y=manifold_data[:, 1],
        z=manifold_data[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=[radians_to_degrees(d) for d in true_directions],
            colorscale='hsv',
            colorbar=dict(title='Direction (°)'),
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text',
        name='Trials'
    ))

    # Add trajectories if requested
    if show_trajectories and len(manifold_data) > 1:
        fig.add_trace(go.Scatter3d(
            x=manifold_data[:, 0],
            y=manifold_data[:, 1],
            z=manifold_data[:, 2],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.3)', width=1),
            name='Trajectory',
            showlegend=True
        ))

    fig.update_layout(
        title="Neural Manifold (3D PCA)",
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        template='plotly_white',
        height=600
    )

    return fig


@st.cache_data
def plot_neural_manifold_2d(
    manifold_data: np.ndarray,
    true_directions: np.ndarray,
    pc_x: int = 0,
    pc_y: int = 1
) -> go.Figure:
    """
    Create a 2D projection of the neural manifold.

    Args:
        manifold_data: Array from PCA
        true_directions: True movement directions
        pc_x: Which PC for x-axis
        pc_y: Which PC for y-axis

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=manifold_data[:, pc_x],
        y=manifold_data[:, pc_y],
        mode='markers',
        marker=dict(
            size=10,
            color=[radians_to_degrees(d) for d in true_directions],
            colorscale='hsv',
            colorbar=dict(title='Direction (°)'),
            opacity=0.8
        ),
        hovertemplate=(
            f'PC{pc_x+1}: %{{x:.2f}}<br>'
            f'PC{pc_y+1}: %{{y:.2f}}<br>'
            'Direction: %{marker.color:.0f}°<extra></extra>'
        )
    ))

    fig.update_layout(
        title=f"Neural Manifold: PC{pc_x+1} vs PC{pc_y+1}",
        xaxis_title=f'PC{pc_x+1}',
        yaxis_title=f'PC{pc_y+1}',
        template='plotly_white',
        height=500
    )

    return fig


@st.cache_data
def plot_variance_explained(
    explained_variance: np.ndarray,
    n_components: int = 10
) -> go.Figure:
    """
    Create a scree plot showing variance explained by each PC.

    Args:
        explained_variance: Array of variance ratios
        n_components: Number of components to show

    Returns:
        Plotly Figure
    """
    n_show = min(n_components, len(explained_variance))
    cumulative = np.cumsum(explained_variance[:n_show])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Individual variance
    fig.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(n_show)],
            y=explained_variance[:n_show] * 100,
            name='Individual',
            marker_color='#3498db'
        ),
        secondary_y=False
    )

    # Cumulative variance
    fig.add_trace(
        go.Scatter(
            x=[f'PC{i+1}' for i in range(n_show)],
            y=cumulative * 100,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Variance Explained by Principal Components",
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(x=0.7, y=0.3)
    )

    fig.update_yaxes(title_text="Individual Variance (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True, range=[0, 105])

    return fig


def plot_manifold_by_area(
    area_manifolds: dict,
    true_directions: np.ndarray
) -> go.Figure:
    """
    Compare neural manifolds across different brain areas.

    Args:
        area_manifolds: Dict mapping area name to manifold data
        true_directions: True movement directions

    Returns:
        Plotly Figure with subplots for each area
    """
    n_areas = len(area_manifolds)

    fig = make_subplots(
        rows=1, cols=n_areas,
        subplot_titles=list(area_manifolds.keys()),
        specs=[[{'type': 'scatter'}] * n_areas]
    )

    for i, (area_name, manifold) in enumerate(area_manifolds.items()):
        fig.add_trace(
            go.Scatter(
                x=manifold[:, 0],
                y=manifold[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=[radians_to_degrees(d) for d in true_directions],
                    colorscale='hsv',
                    showscale=(i == n_areas - 1)
                ),
                name=area_name,
                showlegend=False
            ),
            row=1, col=i+1
        )

        fig.update_xaxes(title_text='PC1', row=1, col=i+1)
        fig.update_yaxes(title_text='PC2', row=1, col=i+1)

    fig.update_layout(
        title="Neural Manifolds Across Brain Areas",
        template='plotly_white',
        height=400
    )

    return fig
