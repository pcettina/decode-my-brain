"""Extended Streamlit AppTest UI interaction tests.

Tests widget interactions, state changes, and multi-page navigation
using the Streamlit AppTest framework. Goes beyond smoke tests to
exercise actual widget manipulation and cross-page state persistence.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app():
    """Create an AppTest instance with demo data auto-loaded."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    return at


# =====================================================================
# Setup Page — Metric Verification
# =====================================================================


class TestSetupPageInteractions:
    """Test setup page reflects simulation parameters."""

    def test_metrics_show_correct_neuron_count(self, app):
        """Default demo has 50 neurons."""
        neuron_metrics = [m for m in app.metric if m.label == "Neurons"]
        assert len(neuron_metrics) == 1
        assert neuron_metrics[0].value == "50"

    def test_metrics_show_correct_trial_count(self, app):
        """Default demo has 20 trials."""
        trial_metrics = [m for m in app.metric if m.label == "Trials"]
        assert len(trial_metrics) == 1
        assert trial_metrics[0].value == "20"


# =====================================================================
# Sidebar Simulation Controls
# =====================================================================


class TestSidebarControls:
    """Test sidebar slider interactions and re-simulation."""

    def test_sidebar_has_sliders(self, app):
        """Sidebar should have sliders for simulation parameters."""
        assert len(app.sidebar.slider) >= 5

    def test_simulate_button_exists(self, app):
        """Sidebar should have a Simulate Trials button."""
        sim_buttons = [b for b in app.sidebar.button if "Simulate" in b.label]
        assert len(sim_buttons) >= 1

    @pytest.mark.slow
    def test_resimulate_updates_state(self, app):
        """Changing neuron count and clicking Simulate updates metrics."""
        # First sidebar slider is n_neurons
        app.sidebar.slider[0].set_value(100)
        # Click Simulate Trials
        sim_buttons = [b for b in app.sidebar.button if "Simulate" in b.label]
        if sim_buttons:
            sim_buttons[0].click()
        app.run()
        assert not app.exception
        neuron_metrics = [m for m in app.metric if m.label == "Neurons"]
        if neuron_metrics:
            assert neuron_metrics[0].value == "100"


# =====================================================================
# Learn Page Interactions
# =====================================================================


class TestLearnPageInteractions:
    """Test learn page widget interactions."""

    def test_learn_page_loads_with_widgets(self, app):
        """Learn page should have sliders and radios."""
        app.switch_page("pages/learn.py")
        app.run()
        assert not app.exception
        assert len(app.slider) > 0

    def test_plot_type_radio(self, app):
        """Switching between Bar and Polar plot type should not crash."""
        app.switch_page("pages/learn.py")
        app.run()
        plot_radios = [r for r in app.radio if "Plot type" in str(r.label)]
        if plot_radios:
            plot_radios[0].set_value("Polar")
            app.run()
            assert not app.exception


# =====================================================================
# Play Page — Game Mode
# =====================================================================


class TestPlayPageGameInteractions:
    """Test game mode widget interactions on the play page."""

    def test_play_page_has_game_buttons(self, app):
        """Play page should have New Round and Reset Game buttons."""
        app.switch_page("pages/play.py")
        app.run()
        assert not app.exception
        button_labels = [b.label for b in app.button]
        assert any("New Round" in lbl for lbl in button_labels)

    def test_decoder_radio_selection(self, app):
        """Switching decoder radio should not crash."""
        app.switch_page("pages/play.py")
        app.run()
        decoder_radios = [r for r in app.radio if "Model decoder" in str(r.label)]
        if decoder_radios:
            decoder_radios[0].set_value("Maximum Likelihood")
            app.run()
            assert not app.exception

    def test_reset_game_clears_state(self, app):
        """Reset Game button should clear game_rounds."""
        app.switch_page("pages/play.py")
        app.run()
        reset_buttons = [b for b in app.button if "Reset" in b.label]
        if reset_buttons:
            reset_buttons[0].click()
            app.run()
            assert app.session_state.game_rounds == []


# =====================================================================
# Play Page — Challenge Mode
# =====================================================================


class TestPlayPageChallengeInteractions:
    """Test challenge mode interactions."""

    def test_player_name_input(self, app):
        """Player name text input should accept values."""
        app.switch_page("pages/play.py")
        app.run()
        name_inputs = [t for t in app.text_input if "Name" in str(t.label)]
        if name_inputs:
            name_inputs[0].set_value("TestPlayer")
            app.run()
            assert not app.exception


# =====================================================================
# Explore Page Interactions
# =====================================================================


class TestExplorePageInteractions:
    """Test explore page widget interactions."""

    def test_explore_page_loads(self, app):
        """Explore page should load without errors."""
        app.switch_page("pages/explore.py")
        app.run()
        assert not app.exception

    def test_temporal_sliders_exist(self, app):
        """Explore page should have temporal dynamics sliders."""
        app.switch_page("pages/explore.py")
        app.run()
        assert len(app.slider) >= 3


# =====================================================================
# Analyze Page Interactions
# =====================================================================


class TestAnalyzePageInteractions:
    """Test analyze page widget interactions."""

    def test_analysis_selectbox_exists(self, app):
        """Analysis type selectbox should exist."""
        app.switch_page("pages/analyze.py")
        app.run()
        assert len(app.selectbox) >= 1

    def test_switch_to_export(self, app):
        """Switching to Export Data mode should not crash."""
        app.switch_page("pages/analyze.py")
        app.run()
        if app.selectbox:
            app.selectbox[0].set_value("Export Data")
            app.run()
            assert not app.exception


# =====================================================================
# Cross-Page Navigation with State
# =====================================================================


class TestCrossPageNavigation:
    """Test that session state persists across page navigation."""

    def test_state_persists_setup_to_learn(self, app):
        """Session state should persist from setup to learn."""
        assert app.session_state.simulated is True
        app.switch_page("pages/learn.py")
        app.run()
        assert not app.exception
        assert app.session_state.simulated is True
        assert app.session_state.neurons is not None

    def test_state_persists_through_all_pages(self, app):
        """Navigate all pages verifying state survives."""
        pages = [
            "pages/setup.py",
            "pages/learn.py",
            "pages/play.py",
            "pages/explore.py",
            "pages/analyze.py",
        ]
        for page_path in pages:
            app.switch_page(page_path)
            app.run()
            assert not app.exception, f"{page_path} raised: {app.exception}"
            assert app.session_state.simulated is True
            assert app.session_state.spike_counts is not None
