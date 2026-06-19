import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit
from qcportal import PortalClient

from openff.qcsubmit.utils import get_data


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return PortalClient("https://api.qcarchive.molssi.org:443/")


@pytest.fixture(scope="function")
def fulltest_client():
    """A portal client backed by a local FractalSnowflake for full end-to-end tests.

    This shadows the ``fulltest_client`` fixture provided by the ``qcarchivetesting``
    pytest plugin. As of qcarchivetesting 0.65 that fixture reads the ``--fractal-uri``
    option, but the plugin that registered the option was removed, so the upstream
    fixture raises ``ValueError: no option named '--fractal-uri'``. We only ever use
    the default (a local snowflake), so build it directly here.
    """
    from qcfractal.snowflake import FractalSnowflake

    snowflake = FractalSnowflake()
    try:
        yield snowflake.client()
    finally:
        snowflake.stop()


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
