"""Playwright E2E browser tests for Decode My Brain.

Tests run against a live Streamlit server and verify:
- Pages load correctly in a real browser
- Interactive elements are visible and clickable
- Navigation between pages works
- Visual regression via screenshots
"""

import pytest

pytestmark = pytest.mark.e2e


class TestPageLoading:
    """Verify all pages load successfully in a real browser."""

    def test_app_title_visible(self, app_page):
        """App title should be visible in the sidebar."""
        sidebar = app_page.locator("[data-testid='stSidebar']")
        title = sidebar.get_by_text("Decode My Brain")
        assert title.is_visible()

    def test_setup_page_has_metrics(self, app_page):
        """Setup page should display metric cards from auto-generated demo."""
        metrics = app_page.locator("[data-testid='stMetric']")
        metrics.first.wait_for(timeout=10000)
        assert metrics.count() >= 4

    def test_setup_page_screenshot(self, app_page):
        """Capture setup page screenshot for visual regression."""
        app_page.wait_for_timeout(2000)
        app_page.screenshot(
            path="tests/e2e/screenshots/setup_page.png", full_page=True
        )


class TestNavigation:
    """Test page navigation via sidebar links."""

    def test_navigate_to_learn(self, app_page):
        app_page.get_by_role("link", name="Learn").click()
        app_page.wait_for_selector("text=Visualize Neural Code", timeout=10000)
        assert app_page.get_by_text("Visualize Neural Code").is_visible()

    def test_navigate_to_play(self, app_page):
        app_page.get_by_role("link", name="Play").click()
        app_page.wait_for_timeout(3000)
        assert app_page.get_by_text("New Round").is_visible()

    def test_navigate_to_explore(self, app_page):
        app_page.get_by_role("link", name="Explore").click()
        app_page.wait_for_selector("text=Live Neural Activity", timeout=10000)
        assert app_page.get_by_text("Live Neural Activity").is_visible()

    def test_navigate_to_analyze(self, app_page):
        app_page.get_by_role("link", name="Analyze").click()
        app_page.wait_for_selector("text=Select Analysis", timeout=10000)

    def test_full_navigation_cycle(self, app_page):
        """Navigate through all pages sequentially without errors."""
        for name in ["Learn", "Play", "Explore", "Analyze", "Setup"]:
            app_page.get_by_role("link", name=name).click()
            app_page.wait_for_timeout(2000)
            assert app_page.title() != ""


class TestInteractiveElements:
    """Test that interactive widgets respond to user input."""

    def test_sidebar_slider_visible(self, app_page):
        sidebar = app_page.locator("[data-testid='stSidebar']")
        slider = sidebar.locator("[data-testid='stSlider']").first
        assert slider.is_visible()

    def test_simulate_button_click(self, app_page):
        """Clicking Simulate Trials triggers re-simulation."""
        sidebar = app_page.locator("[data-testid='stSidebar']")
        simulate_btn = sidebar.get_by_text("Simulate Trials")
        assert simulate_btn.is_visible()
        simulate_btn.click()
        app_page.wait_for_timeout(5000)
        metrics = app_page.locator("[data-testid='stMetric']")
        assert metrics.count() >= 4

    def test_play_page_new_round(self, app_page):
        """New Round button shows the game interface."""
        app_page.get_by_role("link", name="Play").click()
        app_page.wait_for_timeout(2000)
        app_page.get_by_text("New Round").click()
        app_page.wait_for_timeout(3000)
        guess_input = app_page.locator("[data-testid='stNumberInput']")
        assert guess_input.count() >= 1

    def test_learn_page_walkthrough(self, app_page):
        """Learn page walkthrough buttons are interactive."""
        app_page.get_by_role("link", name="Learn").click()
        app_page.wait_for_timeout(2000)
        learn_tab = app_page.get_by_text("Learn Decoding")
        if learn_tab.is_visible():
            learn_tab.click()
            app_page.wait_for_timeout(1000)
            new_example = app_page.get_by_text("New Example")
            if new_example.is_visible():
                new_example.click()
                app_page.wait_for_timeout(3000)
                next_btn = app_page.get_by_text("Next")
                assert next_btn.is_visible()


class TestVisualRegression:
    """Capture screenshots of all pages for baseline comparison."""

    def test_capture_all_pages(self, app_page):
        for name in ["Setup", "Learn", "Play", "Explore", "Analyze"]:
            app_page.get_by_role("link", name=name).click()
            app_page.wait_for_timeout(3000)
            app_page.screenshot(
                path=f"tests/e2e/screenshots/{name.lower()}_baseline.png",
                full_page=True,
            )
