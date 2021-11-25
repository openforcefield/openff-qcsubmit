"""
Tests for building and running workflows, exporting and importing settings.
"""

import pytest
from openff.toolkit.topology import Molecule
from pydantic import ValidationError

from openff.qcsubmit import workflow_components
from openff.qcsubmit.datasets import BasicDataset, OptimizationDataset
from openff.qcsubmit.exceptions import DriverError, InvalidWorkflowComponentError
from openff.qcsubmit.factories import (
    BasicDatasetFactory,
    OptimizationDatasetFactory,
    TorsiondriveDatasetFactory,
)
from openff.qcsubmit.testing import temp_directory
from openff.qcsubmit.utils import get_data, get_torsion


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_adding_workflow_components(factory_type):
    """
    Test building workflows from a verity of workflow components.
    """

    factory = factory_type()

    # element filter
    efilter = workflow_components.ElementFilter()
    factory.add_workflow_components(efilter)

    assert len(factory.workflow) == 1

    # conformer generator
    conformer_gen = workflow_components.StandardConformerGenerator()
    conformer_gen.max_conformers = 200
    factory.add_workflow_components(conformer_gen)

    assert len(factory.workflow) == 2

    # add element filter again and make sure the component name has been incremented
    factory.add_workflow_components(efilter)
    assert len(factory.workflow) == 3
    assert efilter in factory.workflow

    # try to add a non component
    with pytest.raises(InvalidWorkflowComponentError):
        factory.add_workflow_components(3)

    with pytest.raises(ValidationError):
        factory.workflow = {"first component": 3}

    factory.workflow = [conformer_gen]

    assert len(factory.workflow) == 1


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_adding_multiple_workflow_components(factory_type):
    """
    Test adding a list of workflow components.
    """

    factory = factory_type()

    efilter = workflow_components.ElementFilter()
    weight = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    components = [efilter, weight, conformer]

    factory.add_workflow_components(*components)

    assert len(factory.workflow) == 3
    for component in components:
        assert component in factory.workflow


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_remove_workflow_component(factory_type):
    """
    Test removing a workflow component through the API.
    """

    factory = factory_type()
    efilter = workflow_components.ElementFilter()
    weight = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    components = [efilter, weight, conformer]

    factory.add_workflow_components(*components)

    assert len(factory.workflow) == 3

    for component in components:
        factory.remove_workflow_component(component.type)

    assert factory.workflow == []


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_get_wrokflow_component(factory_type):
    """
    Test retrieving a workflow component.
    """

    factory = factory_type()

    efilter = workflow_components.ElementFilter()
    weight = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    components = [efilter, weight, conformer]

    factory.add_workflow_components(*components)

    for component in components:
        assert factory.get_workflow_components(component.type) == [component]


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_clear_workflow(factory_type):
    """
    Test clearing out the workflow.
    """

    factory = factory_type()

    efilter = workflow_components.ElementFilter()
    weight = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    components = [efilter, weight, conformer]

    factory.add_workflow_components(*components)

    factory.clear_workflow()

    assert factory.workflow == []

    factory.add_workflow_components(*components)

    factory.workflow = []

    assert factory.workflow == []


@pytest.mark.parametrize("file_type", [pytest.param("json", id="json"), pytest.param("yaml", id="yaml")])
def test_factory_round_trip(file_type, tmpdir):
    """
    Test round tripping a factory to file with a workflow.
    """
    with tmpdir.as_cwd():
        factory = BasicDatasetFactory(driver="energy", maxiter=1)
        efilter = workflow_components.ElementFilter()
        weight = workflow_components.MolecularWeightFilter()
        conformer = workflow_components.StandardConformerGenerator()
        factory.add_workflow_components(efilter, weight, conformer)
        file_name = "test." + file_type
        factory.export(file_name)

        factory2 = BasicDatasetFactory.from_file(file_name)
        assert factory2.driver == factory.driver
        assert factory2.workflow == factory.workflow


@pytest.mark.parametrize("file_type", [pytest.param("json", id="json"), pytest.param("yaml", id="yaml")])
@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_exporting_settings_no_workflow(file_type, factory_type):
    """
    Test exporting the settings to different file types.
    """

    with temp_directory():
        factory = factory_type()

        changed_attrs = {"priority": "super_high", "compute_tag": "test tag"}
        for attr, value in changed_attrs.items():
            setattr(factory, attr, value)

        file_name = "test." + file_type

        factory.export_settings(file_name=file_name)

        with open(file_name) as f:
            data = f.read()
            for value in changed_attrs.values():
                assert str(value) in data


@pytest.mark.parametrize("file_type", [pytest.param("json", id="json"), pytest.param("yaml", id="yaml")])
@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_exporting_settings_workflow(file_type, factory_type):
    """
    Test exporting the settings and a workflow to the different file types.
    """

    with temp_directory():

        factory = factory_type()
        changed_attrs = {"priority": "super_high", "compute_tag": "test tag"}
        for attr, value in changed_attrs.items():
            setattr(factory, attr, value)

        conformer_gen = workflow_components.StandardConformerGenerator()
        conformer_gen.max_conformers = 100
        factory.add_workflow_components(conformer_gen)

        file_name = "test." + file_type

        factory.export_settings(file_name=file_name)

        with open(file_name) as f:
            data = f.read()
            assert conformer_gen.type in data
            assert str(conformer_gen.max_conformers) in data


@pytest.mark.parametrize("file_type", [pytest.param("json", id="json"), pytest.param("yaml", id="yaml")])
@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_importing_settings_no_workflow(file_type, factory_type):
    """
    Test importing the settings with no workflow components from the supported file types.
    """

    factory = factory_type()

    file_name = "settings." + file_type
    factory.import_settings(get_data(file_name))

    changed_attrs = {
        "priority": "super_high",
        "compute_tag": "loaded tag",
    }
    for attr, value in changed_attrs.items():
        assert getattr(factory, attr) == value


@pytest.mark.parametrize("file_type", [pytest.param("json", id="json"), pytest.param("yaml", id="yaml")])
@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_importing_settings_workflow(file_type, factory_type):
    """
    Test importing the settings and a workflow from the supported file types.
    """

    factory = factory_type()

    file_name = "settings_with_workflow." + file_type
    factory.import_settings(get_data(file_name))

    changed_attrs = {
        "priority": "super_high",
        "compute_tag": "loaded tag",
    }
    for attr, value in changed_attrs.items():
        assert getattr(factory, attr) == value

    assert len(factory.workflow) == 1
    component = factory.get_workflow_components("StandardConformerGenerator")[0]
    assert component.description() == "Generate conformations for the given molecules."
    assert isinstance(component, workflow_components.StandardConformerGenerator) is True


@pytest.mark.parametrize("file_type", [pytest.param("json", id="json"), pytest.param("yaml", id="yaml")])
@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_import_workflow_only(file_type, factory_type):
    """
    Test importing a workflow only from a workflow file.
    """

    factory = factory_type()

    factory2 = factory_type()

    file_name = "settings_with_workflow." + file_type

    factory.import_workflow(get_data(file_name))
    # make sure the settings have not changed from default
    assert factory.dict(exclude={"workflow"}) == factory2.dict(exclude={"workflow"})
    assert len(factory.workflow) == 1
    assert factory.workflow != factory2.workflow


@pytest.mark.parametrize("file_type", [pytest.param("json", id="json"), pytest.param("yaml", id="yaml")])
@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDatasetFactory")
])
def test_export_workflow_only(file_type, factory_type):
    """
    Test exporting the workflow only from the factory.
    """

    with temp_directory():
        factory = factory_type()

        conformer_gen = workflow_components.StandardConformerGenerator()
        conformer_gen.max_conformers = 100

        factory.add_workflow_components(conformer_gen)

        file_name = "workflow." + file_type
        factory.export_workflow(file_name)

        with open(file_name) as workflow:
            data = workflow.read()
            assert "method" not in data
            assert "basis" not in data
            assert "tag" not in data


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDatasetFactory"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDatasetFactory"),
])
def test_basic_opt_factory_index(factory_type):
    """
    Test the basic factories ability to make a molecule index this should be the canonical, isomeric smiles.
    """

    factory = factory_type()

    mol = Molecule.from_smiles("CC")

    index = factory.create_index(mol)

    assert index == mol.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False)


def test_torsiondrive_factory_index():
    """
    Test making an index with a torsiondrive factory the index should tag the torsion atoms.
    """

    factory = TorsiondriveDatasetFactory()

    mol = Molecule.from_smiles("CC")
    mol.properties["atom_map"] = {0: 0, 1: 1, 2: 2, 3: 3}
    index = factory.create_index(mol)
    assert index == mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)


def test_optimization_driver():
    """
    Test the optimization factory to make sure the driver can not be changed.
    """

    factory = OptimizationDatasetFactory()

    with pytest.raises(DriverError):
        factory.driver = "energy"

    assert factory.driver == "gradient"


def test_torsiondrive_index():
    """
    Test generating an index using torsiondrive, this should tag the atoms in the torsion.
    """

    mol = Molecule.from_file(get_data("methanol.sdf"))

    mol.properties["atom_map"] = {4: 0, 0: 1, 1: 2, 5: 3}

    factory = TorsiondriveDatasetFactory()

    index = factory.create_index(mol)

    tags = ["[C:2]", "[H:1]", "[O:3]", "[H:4]"]
    for tag in tags:
        assert tag in index


def test_torsiondrive_linear_torsion():
    """
    Test the torsiondrive factorys ability to find linear bonds which should not be driven.
    """

    factory = TorsiondriveDatasetFactory()
    molecules = Molecule.from_file(get_data("linear_molecules.sdf"), "sdf", allow_undefined_stereo=True)

    for molecule in molecules:
        assert bool(factory._detect_linear_torsions(molecule)) is True


def test_torsiondrive_torsion_string():
    """
    Test the torsiondrive factories ability to create a torsion string for a given bond.
    """

    methanol = Molecule.from_file(get_data("methanol.sdf"), "sdf")

    rotatable = methanol.find_rotatable_bonds()
    assert len(rotatable) == 1

    bond = rotatable[0]
    torsion = get_torsion(bond=bond)

    # now make sure this torsion is in the propers list
    reference_torsions = []
    for proper in methanol.propers:
        dihedral = []
        for atom in proper:
            dihedral.append(atom.molecule_atom_index)
        reference_torsions.append(tuple(dihedral))

    assert torsion in reference_torsions or tuple(reversed(torsion)) in reference_torsions


def test_create_dataset_missing_input():
    """
    Make sure an error is raised if the input can not be found.
    """
    factory = BasicDatasetFactory()
    with pytest.raises(FileNotFoundError, match="The input missing_file.smi could not be found."):
        _ = factory.create_dataset(
            dataset_name="test dataset",
            tagline="test dataset",
            description="test dataset",
            molecules="missing_file.smi"
        )


@pytest.mark.parametrize("factory_dataset_type", [
    pytest.param((BasicDatasetFactory, BasicDataset), id="BasicDatasetFactory"),
    pytest.param((OptimizationDatasetFactory, OptimizationDataset), id="OptimizationDatasetFactory"),
])
def test_create_dataset(factory_dataset_type):
    """
    Test making a the correct corresponding dataset type from a given factory type.
    """

    factory = factory_dataset_type[0]()
    element_filter = workflow_components.ElementFilter()
    element_filter.allowed_elements = [1, 6, 8, 7]
    factory.add_workflow_components(element_filter)
    conformer_generator = workflow_components.StandardConformerGenerator(max_conformers=1)
    factory.add_workflow_components(conformer_generator)

    mols = Molecule.from_file(get_data("tautomers_small.smi"), "smi", allow_undefined_stereo=True)

    # set some settings
    changed_attrs = {"compute_tag": "test tag",
                     "dataset_tags": ["openff", "test"]}
    for attr, value in changed_attrs.items():
        setattr(factory, attr, value)

    dataset = factory.create_dataset(dataset_name="test name", molecules=mols, description="Force field test",
                                     tagline="A test dataset")

    # check the attributes were changed
    for attr, value in changed_attrs.items():
        assert getattr(dataset, attr) == value

    assert dataset.dataset_name == "test name"

    assert isinstance(dataset, factory_dataset_type[1]) is True

    # make sure molecules we filtered and passed
    assert dataset.dataset != {}
    assert dataset.filtered != {}
    assert element_filter.type in dataset.filtered_molecules


def test_create_torsiondrive_dataset():
    """
    Make sure we can correclt make a dataset using the scan enumerator.
    """
    factory = TorsiondriveDatasetFactory()
    scan_filter = workflow_components.ScanEnumerator()
    scan_filter.add_torsion_scan(smarts="[*:1]~[*:2]-[#8:3]-[#1:4]", scan_rage=(-90, 90), scan_increment=10)
    factory.add_workflow_components(scan_filter)
    conformer_generator = workflow_components.StandardConformerGenerator(max_conformers=1)
    factory.add_workflow_components(conformer_generator)
    mols = Molecule.from_file(get_data("tautomers_small.smi"), "smi", allow_undefined_stereo=True)
    dataset = factory.create_dataset(dataset_name="test name", molecules=mols, description="Force field test",
                                     tagline="A test dataset", processors=1)

    assert dataset.n_molecules > 0
    assert dataset.n_records > 0
    for entry in dataset.dataset.values():
        assert entry.keywords.dihedral_ranges == [(-90, 90)]
        assert entry.keywords.grid_spacing == [10]


def test_create_dataset_atom_map():
    """Test creating a dataset with molecules with atom maps."""

    factory = OptimizationDatasetFactory()
    mol = Molecule.from_smiles("CCCC([O-])=O")
    mol.properties['atom_map'] = {1: 1, 2: 2, 3: 3, 4: 4}
    _ = factory.create_dataset(dataset_name="test name", molecules=mol, description="Force field test",
                               tagline="A test dataset")
