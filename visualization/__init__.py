"""Visualization subpackage for the Decode My Brain app."""

from visualization.colors import (
    NEURON_COLORSCALE,
    TRUE_COLOR,
    USER_COLOR,
    MODEL_COLOR,
    get_direction_color,
)

from visualization.tuning import (
    plot_tuning_curves,
    plot_population_bar,
    plot_polar_comparison,
)

from visualization.raster import (
    plot_raster_heatmap,
    create_spike_raster_snapshot,
)

from visualization.analysis import (
    plot_decoder_performance_vs_noise,
    plot_condition_comparison,
    plot_likelihood_curve,
    create_scoreboard_table,
)

from visualization.bci import (
    create_bci_canvas,
    create_bci_metrics_display,
)

from visualization.walkthrough import (
    create_pv_decoder_step,
    create_ml_decoder_step,
    create_vector_animation_polar,
)

from visualization.manifold import (
    compute_neural_manifold,
    plot_neural_manifold_3d,
    plot_neural_manifold_2d,
    plot_variance_explained,
    plot_manifold_by_area,
)

from visualization.network import (
    plot_brain_connectivity,
    plot_area_comparison,
    plot_leaderboard,
)
