import pytest
from qcportal import FractalClient

from openff.qcsubmit.results.caching import clear_results_caches


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return FractalClient()


@pytest.fixture(scope="session", autouse=True)
def clear_results_caches_before_tests():
    clear_results_caches()
