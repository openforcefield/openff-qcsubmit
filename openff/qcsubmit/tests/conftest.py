import pytest
from openff.toolkit.topology import Molecule
from qcportal import PortalClient

from openff.qcsubmit.results.caching import clear_results_caches
from openff.qcsubmit.utils import get_data


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return PortalClient()


@pytest.fixture(scope="function", autouse=True)
def clear_results_caches_before_tests():
    clear_results_caches()


@pytest.fixture()
def imatinib_mesylate() -> Molecule:
    return Molecule.from_file(get_data("imatinib_mesylate.sdf"))
