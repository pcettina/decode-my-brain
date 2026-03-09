"""
Visualization helpers for the Decode My Brain app.

Uses Plotly for interactive plots.
"""

import logging
import numpy as np
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
from simulation import NeuronPopulation
from utils import radians_to_degrees, wrap_angle


# Color schemes
NEURON_COLORSCALE = 'HSL'  # For coloring by preferred direction
TRUE_COLOR = '#3498db'      # Blue for true direction
USER_COLOR = '#2ecc71'      # Green for user guess
MODEL_COLOR = '#e67e22'     # Orange for model decode


def get_direction_color(theta: float) -> str:
    """
    Get a color based on direction (using HSL color wheel).
    
    Args:
        theta: Direction in radians
        
    Returns:
        HSL color string
    """
    hue = (radians_to_degrees(theta) % 360)
    return f'hsl({hue}, 70%, 50%)'


@st.cache_data
def plot_tuning_curves(
    neurons: NeuronPopulation,
    highlight_theta: Optional[float] = None,
    opacity: float = 0.5,
    show_highlight: bool = True
) -> go.Figure:
    """
    Plot tuning curves for all neurons.
    
    Args:
        neurons: NeuronPopulation object
        highlight_theta: Optional direction to highlight with vertical line
        opacity: Opacity of individual curves
        show_highlight: Whether to show the highlighted direction
        
    Returns:
        Plotly Figure
    """
    # Create direction array for x-axis (in degrees for readability)
    theta_rad = np.linspace(0, 2 * np.pi, 361)
    theta_deg = radians_to_degrees(theta_rad)
    
    fig = go.Figure()
    
    # Plot each neuron's tuning curve
    for i in range(neurons.n_neurons):
        mu = neurons.preferred_directions[i]
        rates = neurons.get_tuning_curve(theta_rad, i)
        color = get_direction_color(mu)
        
        fig.add_trace(go.Scatter(
            x=theta_deg,
            y=rates,
            mode='lines',
            line=dict(color=color, width=1.5),
            opacity=opacity,
            name=f'Neuron {i+1} (μ={radians_to_degrees(mu):.0f}°)',
            hovertemplate=(
                f'Neuron {i+1}<br>'
                f'Preferred: {radians_to_degrees(mu):.0f}°<br>'
                'Direction: %{x:.0f}°<br>'
                'Rate: %{y:.1f} Hz<extra></extra>'
            ),
            showlegend=False
        ))
    
    # Add vertical line for highlighted direction
    if show_highlight and highlight_theta is not None:
        theta_deg_highlight = radians_to_degrees(wrap_angle(highlight_theta))
        fig.add_vline(
            x=theta_deg_highlight,
            line_dash="dash",
            line_color=TRUE_COLOR,
            line_width=2,
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
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return fig


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


@st.cache_data
def plot_population_bar(
    spike_counts: np.ndarray,
    neurons: NeuronPopulation,
    true_theta: Optional[float] = None,
    as_polar: bool = False
) -> go.Figure:
    """
    Plot spike counts vs neuron preferred direction.
    
    Args:
        spike_counts: Array of spike counts for each neuron
        neurons: NeuronPopulation object
        true_theta: True movement direction (optional)
        as_polar: If True, create polar bar plot
        
    Returns:
        Plotly Figure
    """
    # Sort by preferred direction
    sort_idx = np.argsort(neurons.preferred_directions)
    sorted_counts = spike_counts[sort_idx]
    sorted_pref = neurons.preferred_directions[sort_idx]
    sorted_pref_deg = radians_to_degrees(sorted_pref)
    
    # Get colors based on preferred direction
    colors = [get_direction_color(p) for p in sorted_pref]
    
    if as_polar:
        fig = go.Figure()
        
        # Add bars in polar coordinates
        fig.add_trace(go.Barpolar(
            r=sorted_counts,
            theta=sorted_pref_deg,
            width=360/len(sorted_pref) * 0.8,
            marker_color=colors,
            opacity=0.8,
            hovertemplate=(
                'Preferred: %{theta:.0f}°<br>'
                'Spikes: %{r}<extra></extra>'
            )
        ))
        
        # Add true direction indicator
        if true_theta is not None:
            max_count = max(sorted_counts) if len(sorted_counts) > 0 else 1
            true_deg = radians_to_degrees(wrap_angle(true_theta))
            fig.add_trace(go.Scatterpolar(
                r=[0, max_count * 1.2],
                theta=[true_deg, true_deg],
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
            template='plotly_white',
            height=500,
            showlegend=true_theta is not None
        )
    else:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_pref_deg,
            y=sorted_counts,
            marker_color=colors,
            opacity=0.8,
            hovertemplate=(
                'Preferred: %{x:.0f}°<br>'
                'Spikes: %{y}<extra></extra>'
            ),
            showlegend=False
        ))
        
        # Add vertical line for true direction
        if true_theta is not None:
            true_deg = radians_to_degrees(wrap_angle(true_theta))
            fig.add_vline(
                x=true_deg,
                line_dash="dash",
                line_color=TRUE_COLOR,
                line_width=3,
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
            template='plotly_white',
            height=400
        )
    
    return fig


@st.cache_data
def plot_polar_comparison(
    true_theta: float,
    user_theta: Optional[float] = None,
    model_theta: Optional[float] = None,
    show_legend: bool = True
) -> go.Figure:
    """
    Create polar plot showing true, user, and model directions as arrows.
    
    Args:
        true_theta: True direction in radians
        user_theta: User's guess in radians (optional)
        model_theta: Model's decode in radians (optional)
        show_legend: Whether to show legend
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    arrow_length = 1.0
    
    # True direction (always shown)
    true_deg = radians_to_degrees(wrap_angle(true_theta))
    fig.add_trace(go.Scatterpolar(
        r=[0, arrow_length],
        theta=[true_deg, true_deg],
        mode='lines+markers',
        line=dict(color=TRUE_COLOR, width=4),
        marker=dict(size=[0, 15], symbol=['circle', 'triangle-up']),
        name=f'True ({true_deg:.0f}°)'
    ))
    
    # User guess
    if user_theta is not None:
        user_deg = radians_to_degrees(wrap_angle(user_theta))
        fig.add_trace(go.Scatterpolar(
            r=[0, arrow_length * 0.95],
            theta=[user_deg, user_deg],
            mode='lines+markers',
            line=dict(color=USER_COLOR, width=4),
            marker=dict(size=[0, 15], symbol=['circle', 'triangle-up']),
            name=f'Your Guess ({user_deg:.0f}°)'
        ))
    
    # Model decode
    if model_theta is not None:
        model_deg = radians_to_degrees(wrap_angle(model_theta))
        fig.add_trace(go.Scatterpolar(
            r=[0, arrow_length * 0.9],
            theta=[model_deg, model_deg],
            mode='lines+markers',
            line=dict(color=MODEL_COLOR, width=4),
            marker=dict(size=[0, 15], symbol=['circle', 'triangle-up']),
            name=f'Model ({model_deg:.0f}°)'
        ))
    
    fig.update_layout(
        title="Direction Comparison",
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 1.2]
            ),
            angularaxis=dict(
                direction='counterclockwise',
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
            )
        ),
        template='plotly_white',
        height=400,
        showlegend=show_legend,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig


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


# =============================================================================
# Animated Spike Raster Visualization
# =============================================================================

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
    
    # Plot spikes for each neuron
    for plot_idx, neuron_idx in enumerate(sort_idx):
        times = spike_times[neuron_idx]
        pref_dir = neurons.preferred_directions[neuron_idx]
        color = get_direction_color(pref_dir)
        
        # Filter spikes in current window
        visible_spikes = [t for t in times if start_time <= t <= current_time_ms]
        
        if visible_spikes:
            # Add spikes as scatter points
            fig.add_trace(go.Scatter(
                x=visible_spikes,
                y=[plot_idx] * len(visible_spikes),
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=12,
                    line=dict(width=2, color=color)
                ),
                name=f'Neuron {neuron_idx}',
                showlegend=False,
                hovertemplate=(
                    f'Neuron {neuron_idx+1}<br>'
                    f'Pref: {radians_to_degrees(pref_dir):.0f}°<br>'
                    'Time: %{x:.1f} ms<extra></extra>'
                )
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



# =============================================================================
# BCI Cursor Control Visualization
# =============================================================================

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


# =============================================================================
# Decoder Walkthrough Visualization
# =============================================================================

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


# =============================================================================
# Neural Manifold / PCA Visualization
# =============================================================================

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


# =============================================================================
# Brain Area Connectivity Visualization
# =============================================================================

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


# =============================================================================
# Challenge Mode Visualizations
# =============================================================================


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



