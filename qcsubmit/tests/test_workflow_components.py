"""
Tests for each of the workflow components try to avoid specific openeye functions.
"""
from functools import lru_cache
from typing import Dict, List

import pytest
from openforcefield.topology import Molecule
from openforcefield.utils.toolkits import OpenEyeToolkitWrapper, RDKitToolkitWrapper

from qcsubmit import workflow_components
from qcsubmit.datasets import ComponentResult
from qcsubmit.utils import check_missing_stereo, get_data


def get_container(molecules: [Molecule]) -> ComponentResult:
    """
    Make and return a new molecule container.
    """

    return ComponentResult(
        component_name="intial", component_description={"description": "initial filter"}, molecules=molecules,
        component_provenance={"test": "test component"})


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

    mols = Molecule.from_file(get_data("tautomers_small.smi"), allow_undefined_stereo=True)

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


def test_weight_filter_apply():
    """
    Make sure the weight filter returns molecules within the limits.
    """

    weight = workflow_components.MolecularWeightFilter()
    weight.minimum_weight = 0
    weight.maximum_weight = 80

    molecules = get_tautomers()

    result = weight.apply(molecules)
    assert result.n_molecules == 14
    assert result.n_filtered == 36


@pytest.mark.parametrize(
    "data",
    [
        pytest.param((workflow_components.ElementFilter, "allowed_elements", [1, 10, 100]), id="ElementFilter"),
        pytest.param((workflow_components.MolecularWeightFilter, "minimum_weight", 1), id="WeightFilter"),
        pytest.param((workflow_components.StandardConformerGenerator, "max_conformers", 1), id="StandardConformers"),
        pytest.param((workflow_components.EnumerateTautomers, "max_tautomers", 2), id="EnumerateTautomers"),
        pytest.param((workflow_components.EnumerateStereoisomers, "undefined_only", True), id="EnumerateStereoisomers"),
        pytest.param((workflow_components.RotorFilter, "maximum_rotors", 3), id="RotorFilter"),
        pytest.param((workflow_components.SmartsFilter, "allowed_substructures", ["[C:1]-[C:2]"]), id="SmartsFilter"),
        pytest.param((workflow_components.WBOFragmenter, "heuristic", "wbo"), id="WBOFragmenter"),
        pytest.param((workflow_components.EnumerateProtomers, "max_states", 5), id="EnumerateProtomers")
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

        mols = get_stereoisomers()
        # remove duplicates from the set
        molecule_container = get_container(mols)

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

    mols = get_tautomers()

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
        enumerate_stereo.undefined_only = True
        enumerate_stereo.rationalise = True

        mols = get_stereoisomers()

        result = enumerate_stereo.apply(mols)

        # make sure no molecules have undefined stereo
        for molecule in result.molecules:
            assert Molecule.from_smiles(molecule.to_smiles(), toolkit_registry=RDKitToolkitWrapper()) == molecule
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
def test_enumerating_stereoisomers_poor_input(toolkit):
    """
    Test stereoisomer enumeration with an impossible stereochemistry.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():
        molecule = Molecule.from_smiles("C=CCn1c([C@@H]2C[C@@H]3CC[C@@H]2O3)nnc1N1CCN(c2ccccc2)CC1")
        enumerate_stereo = workflow_components.EnumerateStereoisomers()
        # the molecule should fail conformer generation
        enumerate_stereo.toolkit = toolkit_name
        enumerate_stereo.undefined_only = True
        enumerate_stereo.rationalise = True

        result = enumerate_stereo.apply(molecules=[molecule, ])
        assert result.n_molecules == 0
        assert result.n_filtered == 1

        # now turn of rationalise
        enumerate_stereo.rationalise = False
        result = enumerate_stereo.apply([molecule, ])
        assert molecule in result.molecules
        assert result.n_molecules == 1

        # now enumerate all stereo and rationalise
        enumerate_stereo.rationalise = True
        enumerate_stereo.undefined_only = False
        # make sure the input is missing and new isomers are found

        result = enumerate_stereo.apply([molecule, ])
        assert molecule not in result.molecules
        assert molecule in result.filtered

    else:
        pytest.skip(f"Toolkit {toolkit_name} is not available.")


@pytest.mark.parametrize("data", [
    pytest.param(("[H]C(=C([H])Cl)Cl", True), id="Molecule with missing stereo"),
    pytest.param(("[H]c1c(c(c(c(c1N([C@@]([H])(O[H])SC([H])([H])[C@]([H])(C(=O)N([H])C([H])([H])C(=O)O[H])N([H])C(=O)C([H])([H])C([H])([H])[C@@]([H])(C(=O)O[H])N([H])[H])O[H])[H])[H])I)[H]", False), id="Molecule with good stereo")
])
def test_check_missing_stereo(data):
    """
    Make sure that molecules with missing stereo chem are correctly identified.
    """
    smiles, result = data
    molecule = Molecule.from_smiles(smiles=smiles, allow_undefined_stereo=True)
    assert result is check_missing_stereo(molecule=molecule)


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


def test_enumerating_protomers_apply():
    """
    Test enumerating protomers which is only availabe in openeye.
    """

    enumerate_protomers = workflow_components.EnumerateProtomers(max_states=2)

    with pytest.raises(ValueError):
        # make sure rdkit is not allowed here
        enumerate_protomers.toolkit = "rdkit"

    mol = Molecule.from_smiles('Oc2ccc(c1ccncc1)cc2')
    result = enumerate_protomers.apply([mol, ])

    # this means that the parent molecule was included
    assert result.n_molecules == 3


def test_coverage_filter():
    """
    Make sure the coverage filter removes the correct molecules.
    """
    from openforcefield.typing.engines.smirnoff import ForceField
    from openforcefield.utils.structure import get_molecule_parameterIDs

    coverage_filter = workflow_components.CoverageFilter()
    coverage_filter.allowed_ids = ["b83"]
    coverage_filter.filtered_ids = ["b87"]

    mols = get_stereoisomers()

    # we have to remove duplicated records
    # remove duplicates from the set
    molecule_container = get_container(mols)
    result = coverage_filter.apply(molecule_container.molecules)

    forcefield = ForceField("openff_unconstrained-1.0.0.offxml")
    # now see if any molecules do not have b83
    parameters_by_molecule, parameters_by_ID = get_molecule_parameterIDs(
        result.molecules, forcefield
    )

    expected = parameters_by_ID["b83"]
    for molecule in result.molecules:
        assert molecule.to_smiles() in expected
        assert "dihedrals" not in molecule.properties

    # we now need to check that the molecules passed contain only the allowed atoms
    # do this by running the component again
    result2 = coverage_filter.apply(result.molecules)
    assert result2.n_filtered == 0
    assert result.n_molecules == result.n_molecules


def test_coverage_filter_tag_dihedrals():
    """
    Make sure the coverage filter tags dihedrals that we request.
    """

    coverage_filter = workflow_components.CoverageFilter()
    coverage_filter.allowed_ids = ["t1"]
    coverage_filter.tag_dihedrals = True

    mols = get_tautomers()

    # we have to remove duplicated records
    # remove duplicates from the set
    molecule_container = get_container(mols)

    result = coverage_filter.apply(molecule_container.molecules)

    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties
        torsion_indexer = molecule.properties["dihedrals"]
        assert torsion_indexer.n_torsions > 0, print(molecule)
        assert torsion_indexer.n_double_torsions == 0
        assert torsion_indexer.n_impropers == 0


def test_fragmentation_settings():
    """
    Make sure the settings are correctly handled.
    """

    fragmenter = workflow_components.WBOFragmenter()
    with pytest.raises(ValueError):
        fragmenter.functional_groups = get_data("functional_groups_error.yaml")

    fragmenter.functional_groups = get_data("functional_groups.yaml")

    assert fragmenter.functional_groups is not None


def test_fragmentation_apply():
    """
    Make sure that fragmentation is working.
    """

    fragmenter = workflow_components.WBOFragmenter()

    # check that a molecule with no rotatable bonds fails if we dont want the parent back
    benzene = Molecule.from_file(get_data("benzene.sdf"), "sdf")
    result = fragmenter.apply([benzene, ])
    assert result.n_molecules == 0

    # now try ethanol
    ethanol = Molecule.from_file(get_data("methanol.sdf"), "sdf")
    fragmenter.include_parent = True
    result = fragmenter.apply([ethanol, ])
    assert result.n_molecules == 1

    # now try a molecule which should give fragments
    diphenhydramine = Molecule.from_smiles("O(CCN(C)C)C(c1ccccc1)c2ccccc2")
    fragmenter.include_parent = False
    result = fragmenter.apply([diphenhydramine, ])
    assert result.n_molecules == 4
    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties


def test_rotor_filter_pass():
    """
    Make sure the rotor filter removes the correct molecules.
    """

    rotor_filter = workflow_components.RotorFilter()
    rotor_filter.maximum_rotors = 3

    mols = get_tautomers()

    # we have to remove duplicated records
    molecule_container = get_container(mols)
    result = rotor_filter.apply(molecule_container.molecules)
    for molecule in result.molecules:
        assert len(molecule.find_rotatable_bonds()) <= rotor_filter.maximum_rotors


def test_rotor_filter_fail():
    """
    Test filtering out molecules with too many rotatable bonds.
    """

    rotor_filter = workflow_components.RotorFilter()
    rotor_filter.maximum_rotors = 1

    mols = get_tautomers()

    molecule_container = get_container(mols)
    result = rotor_filter.apply(molecule_container.molecules)
    for molecule in result.molecules:
        assert len(molecule.find_rotatable_bonds()) <= rotor_filter.maximum_rotors
    for molecule in result.filtered:
        assert len(molecule.find_rotatable_bonds()) > rotor_filter.maximum_rotors


def test_environment_filter_validator():
    """
    Make sure the validator is checking the allowed and filtered fields have valid smirks strings.
    """

    from openforcefield.typing.chemistry import SMIRKSParsingError

    filter = workflow_components.SmartsFilter()

    with pytest.raises(SMIRKSParsingError):
        filter.allowed_substructures = [1, 2, 3, 4]

    with pytest.raises(SMIRKSParsingError):
        # bad string
        filter.allowed_substructures = ["fkebfsjb"]

    with pytest.raises(SMIRKSParsingError):
        # make sure each item is checked
        filter.allowed_substructures = ["[C:1]-[C:2]", "ksbfsb"]

    with pytest.raises(SMIRKSParsingError):
        # good smarts with no tagged atoms.
        filter.allowed_substructures = ["[C]=[C]"]

    # a good search string
    filter.allowed_substructures = ["[C:1]=[C:2]"]
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

    molecules = get_tautomers()

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

    molecules = get_tautomers()

    # this should allow all molecules through
    result = filter.apply(molecules=molecules)

    assert len(result.molecules) == len(molecules)

    # now filter the molecule set again removing aromatic carbons
    filter.filtered_substructures = ["[c:1]"]

    result2 = filter.apply(molecules=result.molecules)

    assert len(result2.molecules) != len(result.molecules)
