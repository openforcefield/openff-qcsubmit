import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit
from qcportal import PortalClient

from openff.qcsubmit.utils import get_data


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return PortalClient("https://api.qcarchive.molssi.org:443/")


@pytest.fixture()
def imatinib_mesylate() -> Molecule:
    return Molecule.from_file(get_data("imatinib_mesylate.sdf"))


@pytest.fixture()
def water() -> Molecule:
    """Water with 2 conformers"""
    water = Molecule.from_smiles("O")
    water.generate_conformers(n_conformers=1)
    return water


@pytest.fixture()
def conformer_water(water) -> Molecule:
    """Water with a translated conformer"""
    conf = water.conformers[0]
    # translate by 3 angstroms
    conf2 = conf + 3 * unit.angstrom
    water.add_conformer(conf2)
    return water
