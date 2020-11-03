"""
Test the PCM settings validators.
"""

import pytest
from qcelemental import constants

from qcsubmit.common_structures import PCMSettings, QCSpec
from qcsubmit.exceptions import PCMSettingError, QCSpecificationError


@pytest.mark.parametrize("data", [
    pytest.param(("ANGSTROM", None), id="angstrom"),
    pytest.param(("bohr", PCMSettingError), id="Bhor not supported")
])
def test_pcm_units(data):
    """
    Make sure proper units are validated.
    """
    unit, error = data
    if error is not None:
        with pytest.raises(error):
            _ = PCMSettings(units=unit, solvent="Water")

    else:
        pcm = PCMSettings(units=unit, solvent="Water")
        assert pcm.medium_Solvent == "H2O"


@pytest.mark.parametrize("data", [
    pytest.param((1998, None), id="1998"),
    pytest.param((2020, PCMSettingError), id="2020 not valid.")
])
def test_pcm_codata(data):
    """
    Make sure an accptable codata value is passed and an error is raised if not.
    """
    codata, error = data
    if error is not None:
        with pytest.raises(error):
            _ = PCMSettings(units="AU", solvent="water", codata=codata)

    else:
        pcm = PCMSettings(units="AU", solvent="water", codata=codata)
        assert pcm.codata == codata


def test_pcm_cavity():
    """
    Make sure only the GePol cavity can be set.
    """
    # try and change from GePol
    with pytest.raises(PCMSettingError):
        _ = PCMSettings(units="au", solvent="Water", cavity_Type="isosurface")

    # make sure gepol is the default
    pcm = PCMSettings(units="au", solvent="Water", cavity_Type="gepol")
    assert pcm.cavity_Type == "GePol"


@pytest.mark.parametrize("data", [
    pytest.param(("uff", None), id="UFF"),
    pytest.param(("openff", PCMSettingError), id="Openff error")
])
def test_pcm_radiisets(data):
    """
    Make sure only valid radii are allowed
    """
    radii, error = data
    if error is not None:
        with pytest.raises(error):
            _ = PCMSettings(units="au", solvent="Water", cavity_RadiiSet=radii)

    else:
        pcm = PCMSettings(units="au", solvent="Water", cavity_RadiiSet=radii)
        assert pcm.cavity_RadiiSet == radii


def test_pcm_cavity_mode():
    """
    Make sure only the implicit mode is allowed for collection computing.
    """
    # try and change to explicit
    with pytest.raises(PCMSettingError):
        _ = PCMSettings(units="au", solvent="water", cavity_Mode="Explicit")

    # make sure the default is implicit
    pcm = PCMSettings(units="au", solvent="water", cavity_Mode="implicit")
    assert pcm.cavity_Mode == "Implicit"


@pytest.mark.parametrize("data", [
    pytest.param(("IEFPCM", None), id="IEFPCM"),
    pytest.param(("BadSolver", PCMSettingError), id="BadSolver error")
])
def test_pcm_solver(data):
    """
    Make sure only IEFPCM and CPCM solvers are allowed.
    """
    solver, error = data
    if error is not None:
        with pytest.raises(error):
            _ = PCMSettings(units="au", solvent="water", medium_SolverType=solver)
    else:
        pcm = PCMSettings(units="au", solvent="water", medium_SolverType=solver)
        assert pcm.medium_SolverType == solver


@pytest.mark.parametrize("solvent_data", [
    pytest.param(("Water", "H2O", None), id="Water"),
    pytest.param(("DMSO", "DMSO", None), id="DMSO"),
    pytest.param(("1,2-dichloroethane", "C2H4CL2", None), id="1,2-dichloroethane"),
    pytest.param(("THF", "THF", None), id="THF"),
    pytest.param(("explicit", "explicit", PCMSettingError), id="Bad solvent")
])
def test_pcm_solvent(solvent_data):
    """
    Make sure solvents can be accepted as either names or chemical formula but are always converted to formula.
    """
    solvent, formula, error = solvent_data
    if error is not None:
        with pytest.raises(error):
            _ = PCMSettings(units="au", solvent=solvent)

    else:
        pcm = PCMSettings(units="au", solvent=solvent)
        assert pcm.medium_Solvent == formula


def test_pcm_unit_conversion_defaults():
    """
    Make sure the the default settings are converted to the correct units.
    """
    # make sure the au are kept as default
    pcm = PCMSettings(units="au", solvent="water")
    assert pcm.medium_ProbeRadius == 1.0
    assert pcm.cavity_Area == 0.3
    assert pcm.cavity_MinRadius == 100

    pcm2 = PCMSettings(units="angstrom", solvent="water")
    assert pcm2.medium_ProbeRadius == pcm.medium_ProbeRadius * constants.bohr2angstroms
    assert pcm2.cavity_Area == pcm.cavity_Area * constants.bohr2angstroms ** 2
    assert pcm2.cavity_MinRadius == pcm.cavity_MinRadius * constants.bohr2angstroms


def test_pcm_unit_conversion():
    """
    Make sure only defaults are converted and given options are kept constant.
    """
    # set the probe radius to 2 angstroms
    pcm = PCMSettings(units="angstrom", solvent="water", medium_ProbeRadius=2)
    assert pcm.medium_ProbeRadius == 2
    # make sure this has been converted
    assert pcm.cavity_Area != 0.3


def test_pcm_default_string():
    """
    Make sure the default string is correctly formatted.
    """

    pcm = PCMSettings(units="au", solvent="Water")

    assert pcm.to_string() == '\n        Units = au\n        CODATA = 2010\n        Medium {\n     SolverType = IEFPCM\n     Nonequilibrium = False\n     Solvent = H2O\n     MatrixSymm = True\n     Correction = 0.0\n     DiagonalScaling = 1.07\n     ProbeRadius = 1.0}\n        Cavity {\n     Type = GePol\n     Area = 0.3\n     Scaling = True\n     RadiiSet = Bondi\n     MinRadius = 100\n     Mode = Implicit}'


def test_qcspec_with_solvent():
    """
    Make sure we only allow PCM to be used with PSI4.
    """

    # make sure an error is raised with any program that is not psi4
    with pytest.raises(QCSpecificationError):
        _ = QCSpec(method="ani2x", basis=None, program="torchani", spec_name="ani2x", spec_description="testing ani with solvent", implicit_solvent=PCMSettings(units="au", solvent="water"))

    # now try with PSI4
    qc_spec = QCSpec(implicit_solvent=PCMSettings(units="au", solvent="water"))
    assert qc_spec.implicit_solvent is not None
    assert qc_spec.implicit_solvent.medium_Solvent == "H2O"

