"""
Tests for each of the workflow components try to avoid specific openeye functions.
"""
from functools import lru_cache
from typing import Dict, List

import pytest

from openforcefield.topology import Molecule
from openforcefield.utils import get_data_file_path
from openforcefield.utils.toolkits import (
    OpenEyeToolkitWrapper,
    RDKitToolkitWrapper,
    UndefinedStereochemistryError,
)
from qcsubmit import workflow_components
from qcsubmit.datasets import ComponentResult
from qcsubmit.utils import get_data


@lru_cache()
def get_molecues(incude_undefined_stereo: bool = False, include_conformers: bool = True):
    """
    Return a list of molecules meeting the required spec from the minidrugbank list.
    """

    mols = Molecule.from_file(get_data_file_path("molecules/MiniDrugBank.sdf"), allow_undefined_stereo=True)

    if not incude_undefined_stereo:
        temp_mols = []
        # find moles with missing stereo and remove them
        for mol in mols:
            try:
                new_mol = Molecule.from_smiles(mol.to_smiles())
                temp_mols.append(new_mol)
            except UndefinedStereochemistryError:
                continue

        mols = temp_mols

    if not include_conformers:
        for mol in mols:
            mol._conformers = []

    return mols


@lru_cache()
def get_stereoisomers():
    """
    Get a set of molecules that all have some undefined stereochemistry.
    """
    mols = Molecule.from_file(get_data("stereoisomers.smi"), allow_undefined_stereo=True)

    return mols


@lru_cache()
def get_tautomers():
    """
    Get a set of molecules that all have tauomers
    """

    mols = Molecule.from_file(get_data("tautomers.smi"), allow_undefined_stereo=True)

    return mols


def test_custom_component():
    """
    Make sure users can not use custom components unless all method are implemented.
    """

    with pytest.raises(TypeError):
        test = workflow_components.CustomWorkflowComponent()

    class TestComponent(workflow_components.CustomWorkflowComponent):

        component_name = "Test component"
        component_description = "Test component"
        component_fail_message = "Test fail"

        def apply(self, molecules: List[Molecule]) -> ComponentResult:
            pass

        def provenance(self) -> Dict:
            return {"test": "version1"}

        @staticmethod
        def is_available() -> bool:
            return True

    test = TestComponent()
    assert test.component_name == "Test component"
    assert test.component_description == "Test component"
    assert test.component_fail_message == "Test fail"
    assert {"test": "version1"} == test.provenance()


@pytest.mark.parametrize(
    "toolkit",
    [
        pytest.param(("openeye", OpenEyeToolkitWrapper), id="openeye"),
        pytest.param(("rdkit", RDKitToolkitWrapper), id="rdkit"),
    ],
)
def test_toolkit_mixin(toolkit):
    """
    Make sure the pydantic ToolkitValidator mixin is working correctly, it should provide provenance and toolkit name
    validation.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        class TestClass(workflow_components.ToolkitValidator, workflow_components.CustomWorkflowComponent):
            """Should not need to implement the provenance."""

            component_name = "ToolkitValidator"
            component_description = "ToolkitValidator test class."
            component_fail_message = "Test fail"

            def apply(self, molecules: List[Molecule]) -> ComponentResult:
                pass

        test = TestClass()
        with pytest.raises(ValueError):
            test.toolkit = "ambertools"

        test.toolkit = toolkit_name
        prov = test.provenance()
        assert toolkit_name in prov
        assert "OpenforcefieldToolkit" in prov

    else:
        pytest.skip(f"Toolkit {toolkit_name} not avilable.")


@pytest.mark.parametrize(
    "toolkit",
    [
        pytest.param(("openeye", OpenEyeToolkitWrapper), id="openeye"),
        pytest.param(("rdkit", RDKitToolkitWrapper), id="rdkit"),
    ],
)
def test_standardconformer_generator_validators(toolkit):
    """
    Test the standard conformer generator which calls the OFFTK.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        conf_gen = workflow_components.StandardConformerGenerator()

        # make sure we are casting to ints
        conf_gen.max_conformers = 1.02
        assert conf_gen.max_conformers == 1

        # test the toolkit validator
        with pytest.raises(ValueError):
            conf_gen.toolkit = "ambertools"
        conf_gen.toolkit = toolkit_name
        assert toolkit_name in conf_gen.provenance()

        conf_gen.clear_existing = "no"
        assert conf_gen.clear_existing is False

        assert "OpenforcefieldToolkit" in conf_gen.provenance()

    else:
        pytest.skip(f"Toolkit {toolkit} not available.")


def test_element_filter_validators():
    """
    Make sure the element filter validators are working.
    """

    elem_filter = workflow_components.ElementFilter()

    with pytest.raises(KeyError):
        elem_filter.allowed_elements = ["carbon", "hydrogen"]

    elem_filter.allowed_elements = [1.02, 2.02, 3.03]

    assert elem_filter.allowed_elements == [1, 2, 3]

    assert "openmm_elements" in elem_filter.provenance()


def test_weight_filter_validator():
    """
    Make sure the weight filter simple validators work.
    """
    weight = workflow_components.MolecularWeightFilter()

    weight.minimum_weight = 0.0
    assert weight.minimum_weight == 0

    assert "openmm_units" in weight.provenance()
    assert "OpenforcefieldToolkit" in weight.provenance()


@pytest.mark.parametrize(
    "data",
    [
        pytest.param((workflow_components.ElementFilter, "allowed_elements", [1, 10, 100]), id="ElementFilter"),
        pytest.param((workflow_components.MolecularWeightFilter, "minimum_weight", 1), id="WeightFilter"),
        pytest.param((workflow_components.StandardConformerGenerator, "max_conformers", 1), id="StandardConformers"),
        pytest.param((workflow_components.EnumerateTautomers, "max_tautomers", 2), id="EnumerateTautomers"),
        pytest.param((workflow_components.EnumerateStereoisomers, "undefined_only", True), id="EnumerateStereoisomers"),
        pytest.param((workflow_components.RotorFilter, "maximum_rotors", 3), id="RotorFilter"),
        pytest.param((workflow_components.SmartsFilter, "allowed_substructures", ["[C:1]-[C:2]"]), id="SmartsFilter")
    ],
)
def test_to_from_object(data):
    """
    Test changing a attribute of the class and making another class by parsing the object, this replicates building
    the workflow from file which is also tested in test_factories.py
    """

    component, attribute, value = data

    active_comp = component()
    # set the attribute
    setattr(active_comp, attribute, value)
    # make a new comp from parsing
    copy_comp = component.parse_obj(active_comp)

    assert copy_comp == active_comp


@pytest.mark.parametrize(
    "toolkit",
    [
        pytest.param(("openeye", OpenEyeToolkitWrapper), id="openeye"),
        pytest.param(("rdkit", RDKitToolkitWrapper), id="rdkit"),
    ],
)
def test_conformer_apply(toolkit):
    """
    Test applying the standard conformer generator to a workflow.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        conf_gen = workflow_components.StandardConformerGenerator()
        conf_gen.toolkit = toolkit_name
        conf_gen.max_conformers = 1
        conf_gen.clear_existing = True

        mols = get_tautomers()
        # remove duplicates from the set
        molecule_container = ComponentResult(
            component_name="intial", component_description={"description": "initial filter"}, molecules=mols,
            component_provenance={"test": "test component"}
        )

        result = conf_gen.apply(molecule_container.molecules)

        assert result.component_name == conf_gen.component_name
        assert result.component_description == conf_gen.dict()
        # make sure each molecule has a conformer that passed
        for molecule in result.molecules:
            assert molecule.n_conformers == 1, print(molecule.conformers)

        for molecule in result.filtered:
            assert molecule.n_conformers == 0

    else:
        pytest.skip(f"Toolkit {toolkit_name} not available.")


def test_elementfilter_apply():
    """
    Test applying the element filter to a workflow.
    """

    elem_filter = workflow_components.ElementFilter()
    elem_filter.allowed_elements = [1, 6, 7, 8]

    mols = get_molecues(include_conformers=False)

    result = elem_filter.apply(mols)

    assert result.component_name == elem_filter.component_name
    assert result.component_description == elem_filter.dict()
    # make sure there are no unwanted elements in the pass set
    for molecue in result.molecules:
        for atom in molecue.atoms:
            assert atom.atomic_number in elem_filter.allowed_elements

    for molecue in result.filtered:
        elements = set([atom.atomic_number for atom in molecue.atoms])
        assert sorted(elements) != sorted(elem_filter.allowed_elements)


@pytest.mark.parametrize(
    "toolkit",
    [
        pytest.param(("openeye", OpenEyeToolkitWrapper), id="openeye"),
        pytest.param(("rdkit", RDKitToolkitWrapper), id="rdkit"),
    ],
)
def test_enumerating_stereoisomers_validator(toolkit):
    """
    Test the validators in enumerating stereoisomers.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        enumerate_stereo = workflow_components.EnumerateStereoisomers()
        with pytest.raises(ValueError):
            enumerate_stereo.toolkit = "ambertools"

        enumerate_stereo.toolkit = toolkit_name
        assert toolkit_name in enumerate_stereo.provenance()

        enumerate_stereo.undefined_only = "y"
        assert enumerate_stereo.undefined_only is True

        enumerate_stereo.max_isomers = 1.1
        assert enumerate_stereo.max_isomers == 1

    else:
        pytest.skip(f"Toolkit {toolkit_name} is not available.")


@pytest.mark.parametrize(
    "toolkit",
    [
        pytest.param(("openeye", OpenEyeToolkitWrapper), id="openeye"),
        pytest.param(("rdkit", RDKitToolkitWrapper), id="rdkit"),
    ],
)
def test_enumerating_stereoisomers_apply(toolkit):
    """
    Test the stereoisomer enumeration.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        enumerate_stereo = workflow_components.EnumerateStereoisomers()
        # set the options
        enumerate_stereo.toolkit = toolkit_name
        enumerate_stereo.include_input = False
        enumerate_stereo.undefined_only = True
        enumerate_stereo.rationalise = True

        mols = get_stereoisomers()

        result = enumerate_stereo.apply(mols)

        # make sure the input molecules are not present
        for mol in mols:
            result.filter_molecule(mol)

        # make sure no molecules have undefined stereo
        for molecule in result.molecules:
            assert Molecule.from_smiles(molecule.to_smiles()) == molecule
            assert molecule.n_conformers >= 1

    else:
        pytest.skip(f"Toolkit {toolkit_name} is not available.")


@pytest.mark.parametrize(
    "toolkit",
    [
        pytest.param(("openeye", OpenEyeToolkitWrapper), id="openeye"),
        pytest.param(("rdkit", RDKitToolkitWrapper), id="rdkit"),
    ],
)
def test_enumerating_tautomers_apply(toolkit):
    """
    Test enumerating tautomers.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        enumerate_tauts = workflow_components.EnumerateTautomers()
        enumerate_tauts.toolkit = toolkit_name
        enumerate_tauts.max_tautomers = 2

        mols = get_tautomers()

        result = enumerate_tauts.apply(mols)

        # remove the input molecules by filtering
        for mol in mols:
            result.filter_molecule(mol)

        assert len(result.molecules) > 0

    else:
        pytest.skip(f"Toolkit {toolkit_name} is not available.")


@pytest.mark.parametrize(
    "toolkit",
    [
        pytest.param(("openeye", OpenEyeToolkitWrapper), id="openeye"),
        pytest.param(("rdkit", RDKitToolkitWrapper), id="rdkit"),
    ],
)
def test_enumerating_tautomers_validator(toolkit):
    """
    Test the validators in enumerating tautomers.
    """

    toolkit_name, toolkit_class = toolkit

    if toolkit_class.is_available():

        enumerate_tautomers = workflow_components.EnumerateTautomers()

        with pytest.raises(ValueError):
            enumerate_tautomers.toolkit = "ambertools"

        enumerate_tautomers.toolkit = toolkit_name
        enumerate_tautomers.max_tautomers = 1.1
        assert enumerate_tautomers.max_tautomers == 1

        assert toolkit_name in enumerate_tautomers.provenance()

    else:
        pytest.skip(f"Toolkit {toolkit_name} not available.")


def test_coverage_filter():
    """
    Make sure the coverage filter removes the correct molecules.
    """

    coverage_filter = workflow_components.CoverageFilter()
    coverage_filter.allowed_ids = ["b83"]
    coverage_filter.filtered_ids = ["b87"]

    mols = get_tautomers()

    # we have to remove duplicated records
    # remove duplicates from the set
    molecule_container = ComponentResult(
        component_name="intial", component_description={"description": "initial filter"}, molecules=mols,
        component_provenance={"test": "test component"}
    )
    result = coverage_filter.apply(molecule_container.molecules)
    # we now need to check that the molecules passed contain only the allowed atoms
    # do this by running the component again
    result2 = coverage_filter.apply(result.molecules)
    assert result2.n_filtered == 0
    assert result.n_molecules == result.n_molecules


def test_rotor_filter():
    """
    Make sure the rotor filter removes the correct molecules.
    """

    rotor_filter = workflow_components.RotorFilter()
    rotor_filter.maximum_rotors = 3

    mols = get_tautomers()

    # we have to remove duplicated records
    # remove duplicates from the set
    molecule_container = ComponentResult(
        component_name="intial", component_description={"description": "initial filter"}, molecules=mols,
        component_provenance={"test": "test component"}
    )
    result = rotor_filter.apply(molecule_container.molecules)
    for molecule in result.molecules:
        assert len(molecule.find_rotatable_bonds()) <= rotor_filter.maximum_rotors


def test_environment_filter_validator():
    """
    Make sure the validator is checking the allowed and filtered fields have valid smirks strings.
    """

    from openforcefield.typing.chemistry import ChemicalEnvironment, SMIRKSParsingError

    filter = workflow_components.SmartsFilter()

    with pytest.raises(SMIRKSParsingError):
        filter.allowed_substructures = [1, 2, 3, 4]

    with pytest.raises(SMIRKSParsingError):
        filter.allowed_substructures = ["fkebfsjb"]

    with pytest.raises(SMIRKSParsingError):
        filter.allowed_substructures = ["[C:1]-[C:2]", "ksbfsb"]

    filter.allowed_substructures = [ChemicalEnvironment("[C:1]=[C:2]")]

    assert len(filter.allowed_substructures) == 1


def test_environment_filter_apply_parameters():
    """
    Make sure the environment filter is correctly identifying substructures.
    """

    filter = workflow_components.SmartsFilter()

    # this should only allow C, H, N, O containing molecules through similar to the element filter
    filter.allowed_substructures = ["[C:1]", "[c:1]", "[H:1]", "[O:1]", "[N:1]"]
    filter.filtered_substructures = ["[Cl:1]", "[F:1]", "[P:1]", "[Br:1]", "[S:1]", "[I:1]", "[B:1]"]
    allowed_elements = [1, 6, 7, 8]

    molecules = get_molecues(incude_undefined_stereo=False, include_conformers=False)

    result = filter.apply(molecules)

    assert result.component_name == filter.component_name
    assert result.component_description == filter.dict()
    # make sure there are no unwanted elements in the pass set
    for molecule in result.molecules:
        for atom in molecule.atoms:
            assert atom.atomic_number in allowed_elements, print(molecule)

    for molecule in result.filtered:
        elements = set([atom.atomic_number for atom in molecule.atoms])
        assert sorted(elements) != sorted(allowed_elements)


def test_environment_filter_apply_none():
    """
    Make sure we get the expected behaviour when we supply None as the filter list.
    """

    filter = workflow_components.SmartsFilter()

    filter.allowed_substructures = None

    molecules = get_molecues(incude_undefined_stereo=False, include_conformers=False)

    # this should allow all molecules through
    result = filter.apply(molecules=molecules)

    assert len(result.molecules) == len(molecules)

    # now filter the molecule set again removing aromatic carbons
    filter.filtered_substructures = ["[c:1]"]

    result2 = filter.apply(molecules=result.molecules)

    assert len(result2.molecules) != result.molecules
