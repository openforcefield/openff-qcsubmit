import pytest
from openff.toolkit.topology import Molecule
from qcportal import PortalClient

from openff.qcsubmit.utils import get_data


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return PortalClient("https://api.qcarchive.molssi.org:443/")


@pytest.fixture()
def imatinib_mesylate() -> Molecule:
    return Molecule.from_file(get_data("imatinib_mesylate.sdf"))
