"""Playwright E2E test configuration.

Starts a Streamlit server as a subprocess, waits for it to be ready,
and tears it down after the test session.
"""

import subprocess
import time

import pytest
import requests


@pytest.fixture(scope="session")
def streamlit_server():
    """Start Streamlit server for E2E tests on port 8599."""
    proc = subprocess.Popen(
        [
            "streamlit", "run", "app.py",
            "--server.port=8599",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base_url = "http://localhost:8599"

    # Wait up to 30 seconds for server readiness
    for _ in range(60):
        try:
            resp = requests.get(f"{base_url}/_stcore/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("Streamlit server did not start within 30 seconds")

    yield base_url

    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def app_page(page, streamlit_server):
    """Navigate to the Streamlit app and wait for initial load."""
    page.goto(streamlit_server)
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=15000)
    return page
