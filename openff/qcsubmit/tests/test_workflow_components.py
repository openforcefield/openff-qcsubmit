"""
Tests for each of the workflow components try to avoid specific openeye functions.
"""
from typing import Dict, List

import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.toolkits import OpenEyeToolkitWrapper, RDKitToolkitWrapper
from pydantic import ValidationError
from typing_extensions import Literal

from openff.qcsubmit import workflow_components
from openff.qcsubmit.common_structures import ComponentProperties
from openff.qcsubmit.exceptions import (
    ComponentRegisterError,
    InvalidWorkflowComponentError,
)
from openff.qcsubmit.utils import check_missing_stereo, get_data
from openff.qcsubmit.validators import check_torsion_connection
from openff.qcsubmit.workflow_components import (
    ComponentResult,
    CustomWorkflowComponent,
    ToolkitValidator,
    TorsionIndexer,
    deregister_component,
    get_component,
    list_components,
    register_component,
)


def get_container(molecules: [Molecule]) -> ComponentResult:
    """
    Make and return a new molecule container.
    """

    return ComponentResult(
        component_name="intial", component_description={"description": "initial filter"}, molecules=molecules,
        component_provenance={"test": "test component"})


def get_stereoisomers():
    """
    Get a set of molecules that all have some undefined stereochemistry.
    """
    mols = Molecule.from_file(get_data("stereoisomers.smi"), allow_undefined_stereo=True)

    return mols


def get_tautomers():
    """
    Get a set of molecules that all have tautomers
    """

    mols = Molecule.from_file(get_data("tautomers_small.smi"), allow_undefined_stereo=True)

    return mols


def test_list_components():
    """
    Make sure that all registered components are returned.
    """
    components = list_components()
    for component in components:
        assert component.__fields__["type"].default == component.__name__


def test_register_component_replace():
    """
    Test registering a component that is already registered with and without using replace.
    """
    # get the standard conformer generator
    gen = workflow_components.StandardConformerGenerator

    with pytest.raises(ComponentRegisterError):
        register_component(component=workflow_components.StandardConformerGenerator, replace=False)

    # now register using replace with a new default
    register_component(component=gen, replace=True)


def test_register_component_error():
    """
    Make sure an error is raised if we try and register a component that is not a sub class of CustomWorkflowComponent.
    """
    # fake component
    charge_filter = dict

    with pytest.raises(InvalidWorkflowComponentError):
        register_component(component=charge_filter)


@pytest.mark.parametrize("component", [
    pytest.param(workflow_components.SmartsFilter, id="Class instance"),
])
def test_deregister_component(component):
    """
    Make sure we can deregister components via name or class.
    """
    # deregister the component
    deregister_component(component=component)

    assert workflow_components.SmartsFilter not in list_components()

    # now add it back
    register_component(component=workflow_components.SmartsFilter)


def test_deregister_component_error():
    """
    Make sure an error is raised when we try to remove a component that was not registered first.
    """

    with pytest.raises(ComponentRegisterError):
        deregister_component(component="BadComponent")


def test_get_component():
    """
    Make sure the correct component is returned when requested.
    """
    gen = get_component("standardconformergenerator")

    assert gen == workflow_components.StandardConformerGenerator


def test_get_component_error():
    """
    Make sure an error is rasied when we try to get a component that was not registered.
    """
    with pytest.raises(ComponentRegisterError):
        get_component(component_name="ChargeFilter")


def test_custom_component():
    """
    Make sure users can not use custom components unless all method are implemented.
    """

    with pytest.raises(TypeError):
        test = CustomWorkflowComponent()

    class TestComponent(CustomWorkflowComponent):

        component_name: Literal["TestComponent"] = "TestComponent"

        @classmethod
        def description(cls) -> str:
            return "Test component"

        @classmethod
        def fail_reason(cls) -> str:
            return "Test fail"

        @classmethod
        def properties(cls) -> ComponentProperties:
            return ComponentProperties(process_parallel=True, produces_duplicates=True)

        def _apply(self, molecules: List[Molecule]) -> ComponentResult:
            pass

        def provenance(self) -> Dict:
            return {"test": "version1"}

        @classmethod
        def is_available(cls) -> bool:
            return True

    test = TestComponent()
    assert test.component_name == "TestComponent"
    assert test.description() == "Test component"
    assert test.fail_reason() == "Test fail"
    assert {"test": "version1"} == test.provenance()
    assert TestComponent.info() == {"name": "TestComponent", "description": "Test component", "fail_reason": "Test fail"}


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

        class TestClass(ToolkitValidator, CustomWorkflowComponent):
            """Should not need to implement the provenance."""

            component_name: Literal["TestClass"] = "TestClass"

            @classmethod
            def description(cls) -> str:
                return "ToolkitValidator test class."

            @classmethod
            def fail_reason(cls) -> str:
                return "Test fail"

            @classmethod
            def properties(cls) -> ComponentProperties:
                return ComponentProperties(process_parallel=True, produces_duplicates=True)

            def _apply(self, molecules: List[Molecule]) -> ComponentResult:
                pass

        test = TestClass()
        with pytest.raises(ValidationError):
            test.toolkit = "ambertools"

        test.toolkit = toolkit_name
        prov = test.provenance()
        assert toolkit_name in prov
        assert "openff-toolkit" in prov

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

        assert "openff-toolkit" in conf_gen.provenance()

    else:
        pytest.skip(f"Toolkit {toolkit} not available.")


def test_element_filter_validators():
    """
    Make sure the element filter validators are working.
    """

    elem_filter = workflow_components.ElementFilter()

    with pytest.raises(ValidationError):
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
    assert "openff-toolkit" in weight.provenance()


def test_weight_filter_apply():
    """
    Make sure the weight filter returns molecules within the limits.
    """

    weight = workflow_components.MolecularWeightFilter()
    weight.minimum_weight = 0
    weight.maximum_weight = 80

    molecules = get_tautomers()

    result = weight.apply(molecules, processors=1)
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
        pytest.param((workflow_components.WBOFragmenter, "threshold", 0.5), id="WBOFragmenter"),
        pytest.param((workflow_components.EnumerateProtomers, "max_states", 5), id="EnumerateProtomers"),
        pytest.param((workflow_components.RMSDCutoffConformerFilter, "cutoff", 1.2), id="RMSDCutoffConformerFilter")
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


def test_rmsd_filter():
    """
    Test the RMSD conformer filter method.
    """
    import copy

    from simtk import unit

    rmsd_filter = workflow_components.RMSDCutoffConformerFilter(cutoff=1)
    mol = Molecule.from_smiles("CCCC")
    # make a lot of conformers for the molecule
    mol.generate_conformers(n_conformers=1000, rms_cutoff=0.05 * unit.angstrom, toolkit_registry=RDKitToolkitWrapper())
    ref_mol = copy.deepcopy(mol)
    result = rmsd_filter.apply([mol, ], processors=1)
    # now make sure the number of conformers is different
    assert result.molecules[0].n_conformers != ref_mol.n_conformers


def test_rmsd_filter_no_conformers():
    """
    Make sure the molecule is failed when no conformers are present.
    """
    rmsd_filter = workflow_components.RMSDCutoffConformerFilter(cutoff=1)
    mol = Molecule.from_smiles("CCCC")
    result = rmsd_filter.apply([mol, ], processors=1)
    # now make sure the number of conformers is different
    assert result.n_molecules == 0
    assert result.n_filtered == 1


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

        result = conf_gen.apply(molecule_container.molecules, processors=1)

        assert result.component_name == conf_gen.type
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

    result = elem_filter.apply(mols, processors=1)

    assert result.component_name == elem_filter.type
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

        result = enumerate_stereo.apply(mols, processors=1)
        for mol in mols:
            assert mol in result.molecules

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

        result = enumerate_stereo.apply(molecules=[molecule, ], processors=1)
        assert result.n_molecules == 0
        assert result.n_filtered == 1

        # now turn of rationalise
        enumerate_stereo.rationalise = False
        result = enumerate_stereo.apply([molecule, ], processors=1)
        assert molecule in result.molecules
        assert result.n_molecules == 1

        # now enumerate all stereo and rationalise
        enumerate_stereo.rationalise = True
        enumerate_stereo.undefined_only = False
        # make sure the input is missing and new isomers are found

        result = enumerate_stereo.apply([molecule, ], processors=1)
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
    Test enumerating tautomers and make sue the input molecule is also returned.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        enumerate_tauts = workflow_components.EnumerateTautomers()
        enumerate_tauts.toolkit = toolkit_name
        enumerate_tauts.max_tautomers = 2

        mols = get_tautomers()

        result = enumerate_tauts.apply(mols, processors=1)

        # check the input molecule is present
        for mol in mols:
            assert mol in result.molecules

        assert result.n_molecules > len(mols)

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
    assert enumerate_protomers.is_available()

    with pytest.raises(ValueError):
        # make sure rdkit is not allowed here
        enumerate_protomers.toolkit = "rdkit"

    mol = Molecule.from_smiles('Oc2ccc(c1ccncc1)cc2')
    result = enumerate_protomers.apply([mol, ], processors=1)

    assert mol in result.molecules
    # this means that the parent molecule was included
    assert result.n_molecules == 3


def test_coverage_filter():
    """
    Make sure the coverage filter removes the correct molecules.
    """
    from openff.toolkit.typing.engines.smirnoff import ForceField

    coverage_filter = workflow_components.CoverageFilter()
    coverage_filter.allowed_ids = ["b83"]
    coverage_filter.filtered_ids = ["b87"]

    mols = get_stereoisomers()

    # we have to remove duplicated records
    # remove duplicates from the set
    molecule_container = get_container(mols)
    result = coverage_filter.apply(molecule_container.molecules, processors=1)

    forcefield = ForceField("openff_unconstrained-1.0.0.offxml")
    # now see if any molecules do not have b83
    parameters_by_id = {}
    for molecule in result.molecules:
        labels = forcefield.label_molecules(molecule.to_topology())[0]
        covered_types = set(
            [label.id for types in labels.values() for label in types.values()]
        )
        # now store the smiles under the ids
        for parameter in covered_types:
            parameters_by_id.setdefault(parameter, []).append(molecule.to_smiles())

    expected = parameters_by_id["b83"]
    for molecule in result.molecules:
        assert molecule.to_smiles() in expected
        assert "dihedrals" not in molecule.properties

    # we now need to check that the molecules passed contain only the allowed atoms
    # do this by running the component again
    result2 = coverage_filter.apply(result.molecules, processors=1)
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

    result = coverage_filter.apply(molecule_container.molecules, processors=1)

    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties
        torsion_indexer = molecule.properties["dihedrals"]
        assert torsion_indexer.n_torsions > 0, print(molecule)
        assert torsion_indexer.n_double_torsions == 0
        assert torsion_indexer.n_impropers == 0


def test_wbo_fragmentation_apply():
    """
    Make sure that wbo fragmentation is working.
    """
    fragmenter = workflow_components.WBOFragmenter()
    assert fragmenter.is_available()
    # check that a molecule with no rotatable bonds fails if we dont want the parent back
    benzene = Molecule.from_file(get_data("benzene.sdf"), "sdf")
    result = fragmenter.apply([benzene, ], processors=1)
    assert result.n_molecules == 0

    # now try a molecule which should give fragments
    diphenhydramine = Molecule.from_smiles("O(CCN(C)C)C(c1ccccc1)c2ccccc2")
    result = fragmenter.apply([diphenhydramine, ], processors=1)
    assert result.n_molecules == 4
    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties


def test_pfizer_fragmentation_apply():
    """Make sure that the pfizer fragmentation is working and correctly tagging bonds."""

    fragmenter = workflow_components.PfizerFragmenter()
    assert fragmenter.is_available()
    # now try a molecule which should give fragments
    diphenhydramine = Molecule.from_smiles("O(CCN(C)C)C(c1ccccc1)c2ccccc2")
    result = fragmenter.apply([diphenhydramine, ], processors=1)
    assert result.n_molecules == 4
    for molecule in result.molecules:
        assert "dihedrals" in molecule.properties


def test_rotor_filter_maximum():
    """
    Remove molecules with too many rotatable bonds.
    """

    rotor_filter = workflow_components.RotorFilter()
    rotor_filter.maximum_rotors = 3

    mols = get_tautomers()

    # we have to remove duplicated records
    molecule_container = get_container(mols)
    result = rotor_filter.apply(molecule_container.molecules, processors=1)
    for molecule in result.molecules:
        assert len(molecule.find_rotatable_bonds()) <= rotor_filter.maximum_rotors


def test_rotor_filter_minimum():
    """
    Remove molecules with too few rotatable bonds.
    """
    rotor_filter = workflow_components.RotorFilter()
    rotor_filter.minimum_rotors = 3
    # not capped
    rotor_filter.maximum_rotors = None

    mols = get_tautomers()
    mol_container = get_container(mols)
    result = rotor_filter.apply(mol_container.molecules, processors=1)
    for molecule in result.molecules:
        assert len(molecule.find_rotatable_bonds()) >= rotor_filter.minimum_rotors
    for molecule in result.filtered:
        assert len(molecule.find_rotatable_bonds()) < rotor_filter.minimum_rotors


def test_rotor_filter_validation():
    """
    Make sure that the maximum number of rotors is >= the minimum if defined.
    """

    rotor_filter = workflow_components.RotorFilter()
    rotor_filter.maximum_rotors = 4
    rotor_filter.minimum_rotors = 4
    mols = get_tautomers()
    mol_container = get_container(mols)
    rotor_filter.apply(mol_container.molecules, processors=1)
    rotor_filter.minimum_rotors = 5
    with pytest.raises(ValueError):
        rotor_filter.apply(mol_container.molecules, processors=1)


def test_rotor_filter_fail():
    """
    Test filtering out molecules with too many rotatable bonds.
    """

    rotor_filter = workflow_components.RotorFilter()
    rotor_filter.maximum_rotors = 1

    mols = get_tautomers()

    molecule_container = get_container(mols)
    result = rotor_filter.apply(molecule_container.molecules, processors=1)
    for molecule in result.molecules:
        assert len(molecule.find_rotatable_bonds()) <= rotor_filter.maximum_rotors
    for molecule in result.filtered:
        assert len(molecule.find_rotatable_bonds()) > rotor_filter.maximum_rotors


def test_smarts_filter_validator():
    """
    Make sure the validator is checking the allowed and filtered fields have valid smirks strings.
    """

    from openff.toolkit.typing.chemistry import SMIRKSParsingError

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


def test_smarts_filter_apply_parameters():
    """
    Make sure the environment filter is correctly identifying substructures.
    """

    filter = workflow_components.SmartsFilter()

    # this should only allow C, H, N, O containing molecules through similar to the element filter
    filter.allowed_substructures = ["[C:1]", "[c:1]", "[H:1]", "[O:1]", "[N:1]"]
    filter.filtered_substructures = ["[Cl:1]", "[F:1]", "[P:1]", "[Br:1]", "[S:1]", "[I:1]", "[B:1]"]
    allowed_elements = [1, 6, 7, 8]

    molecules = get_tautomers()

    result = filter.apply(molecules, processors=1)

    assert result.component_name == filter.type
    assert result.component_description == filter.dict()
    # make sure there are no unwanted elements in the pass set
    for molecule in result.molecules:
        for atom in molecule.atoms:
            assert atom.atomic_number in allowed_elements, print(molecule)

    for molecule in result.filtered:
        elements = set([atom.atomic_number for atom in molecule.atoms])
        assert sorted(elements) != sorted(allowed_elements)


def test_smarts_filter_apply_none():
    """
    Make sure we get the expected behaviour when we supply None as the filter list.
    """

    filter = workflow_components.SmartsFilter()

    filter.allowed_substructures = None

    molecules = get_tautomers()

    # this should allow all molecules through
    result = filter.apply(molecules=molecules, processors=1)

    assert len(result.molecules) == len(molecules)

    # now filter the molecule set again removing aromatic carbons
    filter.filtered_substructures = ["[c:1]"]

    result2 = filter.apply(molecules=result.molecules, processors=1)

    assert len(result2.molecules) != len(result.molecules)


@pytest.mark.parametrize("tag_dihedrals", [
    pytest.param(True, id="Tag dihedrals"),
    pytest.param(False, id="Do not tag Dihedrals")
])
def test_smarts_filter_apply_tag_torsions(tag_dihedrals):
    """
    Make sure that torsions in molecules are tagged if we supply a torsion smarts a pattern.
    """
    filter = workflow_components.SmartsFilter(tag_dihedrals=tag_dihedrals)

    # look for methyl torsions here
    filter.allowed_substructures = ["[*:1]-[*:2]-[#6H3:3]-[#1:4]"]

    molecules = get_tautomers()
    # this should filter and tag the dihedrals
    result = filter.apply(molecules, processors=1)
    for molecule in result.molecules:
        if tag_dihedrals:
            assert "dihedrals" in molecule.properties
            # make sure the torsion is connected
            torsions = molecule.properties["dihedrals"]
            for torsion in torsions.get_dihedrals:
                _ = check_torsion_connection(torsion=torsion.torsion1, molecule=molecule)
        else:
            assert "dihedrals" not in molecule.properties


def test_scan_filter_mutually_exclusive():
    """
    Make sure an error is raised when both options are passed.
    """
    with pytest.raises(ValidationError):
        workflow_components.ScanFilter(scans_to_include=["[*:1]~[*:2]:[*:3]~[*:4]"], scans_to_exclude=["[*:1]~[*:2]-[CH3:3]-[H:4]"])


def test_scan_filter_no_torsions():
    """
    Filter all molecules which do not have any torsions tagged for scanning.
    """

    molecule = Molecule.from_smiles("CC")

    filter = workflow_components.ScanFilter(scans_to_include=["[*:1]~[*:2]-[CH3:3]-[H:4]"])

    result = filter.apply(molecules=[molecule], processors=1)
    assert result.n_molecules == 0


def test_scan_filter_include():
    """
    Make sure only torsion scans we want to include are kept when more than one is defined.
    """
    ethanol = Molecule.from_file(get_data("ethanol.sdf"))
    t_indexer = TorsionIndexer()
    # C-C torsion
    t_indexer.add_torsion(torsion=(3, 0, 1, 2))
    # O-C torsion
    t_indexer.add_torsion(torsion=(0, 1, 2, 8))
    ethanol.properties["dihedrals"] = t_indexer

    # only keep the C-C scan
    filter = workflow_components.ScanFilter(scans_to_include=["[#1:1]~[#6:2]-[#6:3]-[#1:4]"])

    result = filter.apply(molecules=[ethanol], processors=1)

    assert result.n_molecules == 1
    assert (0, 1) in ethanol.properties["dihedrals"].torsions


def test_scan_filter_exclude():
    """
    Make sure only exclude torsion scans we do not want and keep any other defined scans.
    """
    ethanol = Molecule.from_file(get_data("ethanol.sdf"))
    t_indexer = TorsionIndexer()
    # C-C torsion
    t_indexer.add_torsion(torsion=(3, 0, 1, 2))
    # O-C torsion
    t_indexer.add_torsion(torsion=(0, 1, 2, 8))
    ethanol.properties["dihedrals"] = t_indexer

    # only keep the C-C scan
    filter = workflow_components.ScanFilter(scans_to_exclude=["[#1:1]~[#6:2]-[#6:3]-[#1:4]"])

    result = filter.apply(molecules=[ethanol], processors=1)

    assert result.n_molecules == 1
    assert (1, 2) in ethanol.properties["dihedrals"].torsions


def test_scan_enumerator_no_scans():
    """
    Make sure molecules are filtered if they have no scans assigned.
    """
    mol = Molecule.from_smiles("CC")

    scan_tagger = workflow_components.ScanEnumerator()
    scan_tagger.add_torsion_scan(smarts="[*:1]~[#8:1]-[#6:3]~[*:4]", scan_rage=(-40, 40), scan_increment=15)

    result = scan_tagger.apply([mol], processors=1)

    assert result.n_molecules == 0
    assert result.n_filtered == 1


def test_scan_enumerator_1d():
    """
    Make sure only one match is tagged per torsion.
    """
    mol = Molecule.from_smiles("CCC")

    scan_tagger = workflow_components.ScanEnumerator()
    scan_tagger.add_torsion_scan(smarts="[*:1]~[#6:2]-[#6:3]~[*:4]", scan_rage=(-60, 60), scan_increment=20)

    result = scan_tagger.apply([mol], processors=1)

    assert result.n_molecules == 1
    indexer = mol.properties["dihedrals"]
    assert indexer.n_torsions == 1
    assert indexer.torsions[(1, 2)].scan_range1 == (-60, 60)


def test_scan_enumerator_unique():
    """
    If the enumerator would hit multiple torsions in a molecule make sure they are unique by symmetry.
    """
    mol = Molecule.from_smiles("CCCC")

    scan_tagger = workflow_components.ScanEnumerator()
    scan_tagger.add_torsion_scan(smarts="[*:1]~[#6:2]-[#6:3]~[*:4]")

    result = scan_tagger.apply(molecules=[mol], processors=1)

    assert result.n_molecules == 1
    indexer = mol.properties["dihedrals"]
    assert indexer.n_torsions == 2


def test_scan_enumerator_2d():
    """
    Make sure one combination of the 2D scan is tagged.
    """

    mol = Molecule.from_smiles("COc1ccc(cc1)N")

    scan_tagger = workflow_components.ScanEnumerator()
    scan_tagger.add_double_torsion(smarts1="[*:1]-[#7X3+0:2]-[#6:3]@[#6,#7:4]", smarts2="[#7X3+0:1](-[*:3])(-[*:4])-[#6:2]@[#6,#7]",
                                   scan_range1=(-165, 180),
                                   scan_range2=(-60, 60),
                                   scan_increments=[15, 4])

    result = scan_tagger.apply([mol], processors=1)
    assert result.n_molecules == 1
    indexer = mol.properties["dihedrals"]
    assert indexer.n_double_torsions == 1
    assert indexer.double_torsions[((5, 8), (5, 17))].scan_range1 == (-165, 180)
    assert indexer.double_torsions[((5, 8), (5, 17))].scan_range2 == (-60, 60)


def test_improper_enumerator():
    """
    Make sure improper torsions are correctly tagged.
    """

    mol = Molecule.from_file(get_data("benzene.sdf"))

    scan_tagger = workflow_components.ScanEnumerator()
    # even though there is more than one improper make sure we only get one scan back
    scan_tagger.add_improper_torsion(smarts="[#6:1](-[#1:2])(:[#6:3]):[#6:4]", central_smarts="[#6:1]", scan_range=(-40, 40), scan_increment=4)

    result = scan_tagger.apply([mol], processors=1)

    assert result.n_molecules == 1
    indexer = mol.properties["dihedrals"]
    assert indexer.n_impropers == 1
    assert indexer.imporpers[0].scan_increment == [4]


def test_formal_charge_filter_exclusive():
    """
    Raise an error if both allowed and filtered charges are supplied
    """

    with pytest.raises(ValidationError):
        workflow_components.ChargeFilter(charges_to_include=[0, 1], charges_to_exclude=[-1])


def test_formal_charge_filter():
    """
    Make sure we can correctly filter by the molecules net formal charge.
    """

    molecule = Molecule.from_mapped_smiles("[N+:1](=[O:2])([O-:3])[O-:4]")

    # filter out the molecule
    charge_filter = workflow_components.ChargeFilter(charges_to_exclude=[-1, 0])
    result = charge_filter.apply([molecule], processors=1)
    assert result.n_molecules == 0
    assert result.n_filtered == 1

    # now allow it through
    charge_filter = workflow_components.ChargeFilter(charges_to_include=[-1])
    result = charge_filter.apply([molecule], processors=1)
    assert result.n_molecules == 1
