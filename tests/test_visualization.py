"""Tests for visualization.py — basic smoke tests."""

import plotly.graph_objects as go


def test_plot_tuning_curves_returns_figure(small_population):
    """plot_tuning_curves should return a Plotly Figure with traces."""
    # Import here to avoid Streamlit init at module level
    from visualization import plot_tuning_curves
    fig = plot_tuning_curves(small_population)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
