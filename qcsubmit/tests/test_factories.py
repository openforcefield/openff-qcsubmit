"""
Tests for building and running workflows, exporting and importing settings.
"""
import pytest

import tempfile
from qcsubmit.factories import BasicDatasetFactory, OptimizationDatasetFactory, TorsiondriveDatasetFactory
from qcsubmit.datasets import BasicDataSet
from qcsubmit import workflow_components
from qcsubmit.utils import get_data
from qcsubmit.exceptions import InvalidWorkflowComponentError, DriverError
from pydantic import ValidationError
from openforcefield.topology import Molecule
import os


def test_adding_workflow_components():
    """
    Test building workflows from a verity of workflow components.
    """

    factory = BasicDatasetFactory()

    # element filter
    efilter = workflow_components.ElementFilter()
    factory.add_workflow_component(efilter)

    assert len(factory.workflow) == 1

    # conformer generator
    conformer_gen = workflow_components.StandardConformerGenerator()
    conformer_gen.max_conformers = 200
    factory.add_workflow_component(conformer_gen)

    assert len(factory.workflow) == 2

    # add element filter again and make sure the component name has been incremented
    factory.add_workflow_component(efilter)
    assert len(factory.workflow) == 3
    assert efilter.component_name in factory.workflow

    # try to add a non component
    with pytest.raises(InvalidWorkflowComponentError):
        factory.add_workflow_component(3)

    with pytest.raises(ValidationError):
        factory.workflow = {'first compoent': 3}

    factory.workflow = {'testconformer': conformer_gen}

    assert len(factory.workflow) == 1


def test_adding_multipule_workflow_components():
    """
    Test adding a list of workflow components.
    """

    factory = BasicDatasetFactory()

    efilter = workflow_components.ElementFilter()
    wieght = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    componets = [efilter, wieght, conformer]

    factory.add_workflow_component(componets)

    assert len(factory.workflow) == 3
    for componet in componets:
        assert componet.component_name in factory.workflow


def test_remove_workflow_componet():
    """
    Test removing a workflow component through the API.
    """

    factory = BasicDatasetFactory()
    efilter = workflow_components.ElementFilter()
    wieght = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    componets = [efilter, wieght, conformer]

    factory.add_workflow_component(componets)

    assert len(factory.workflow) == 3

    for componet in componets:
        factory.remove_workflow_component(componet.component_name)

    assert factory.workflow == {}


def test_get_wrokflow_component():
    """
    Test retrieving a workflow component.
    """

    factory = BasicDatasetFactory()

    efilter = workflow_components.ElementFilter()
    wieght = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    componets = [efilter, wieght, conformer]

    factory.add_workflow_component(componets)

    for componet in componets:
        assert factory.get_workflow_component(componet.component_name) == componet


def test_clear_workflow():
    """
    Test clearing out the workflow.
    """

    factory = BasicDatasetFactory()

    efilter = workflow_components.ElementFilter()
    wieght = workflow_components.MolecularWeightFilter()
    conformer = workflow_components.StandardConformerGenerator()

    componets = [efilter, wieght, conformer]

    factory.add_workflow_component(componets)

    factory.clear_workflow()

    assert factory.workflow == {}

    factory.add_workflow_component(componets)

    factory.workflow = {}

    assert factory.workflow == {}


@pytest.mark.parametrize('file_type',
                         [
                             pytest.param('json', id='json'),
                             pytest.param('yaml', id='yaml')
                         ])
def test_exporting_settings_no_workflow(file_type):
    """
    Test exporting the settings to different file types.
    """

    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)
        factory = BasicDatasetFactory()

        changed_attrs = {'method': 'test method', 'basis': 'test basis', 'program': 'test program', 'tag': 'test tag'}
        for attr, value in changed_attrs.items():
            setattr(factory, attr, value)

        file_name = 'test.' + file_type

        factory.export_settings(file_name=file_name)

        with open(file_name) as f:
            data = f.read()
            for value in changed_attrs.values():
                assert value in data


@pytest.mark.parametrize('file_type',
                         [
                             pytest.param('json', id='json'),
                             pytest.param('yaml', id='yaml')
                         ])
def test_exporting_settings_workflow(file_type):
    """
    Test exporting the settings and a workflow to the different file types.
    """

    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)

        factory = BasicDatasetFactory()
        changed_attrs = {'method': 'test method', 'basis': 'test basis', 'program': 'test program', 'tag': 'test tag'}
        for attr, value in changed_attrs.items():
            setattr(factory, attr, value)

        conformer_gen = workflow_components.StandardConformerGenerator()
        conformer_gen.max_conformers = 100
        factory.add_workflow_component(conformer_gen)

        file_name = 'test.' + file_type

        factory.export_settings(file_name=file_name)

        with open(file_name) as f:
            data = f.read()
            assert conformer_gen.component_name in data
            assert str(conformer_gen.max_conformers) in data


@pytest.mark.parametrize('file_type',
                         [
                             pytest.param('json', id='json'),
                             pytest.param('yaml', id='yaml')
                         ])
def test_importing_settings_no_workflow(file_type):
    """
    Test importing the settings with no workflow components from the supported file types.
    """

    factory = BasicDatasetFactory()

    file_name = 'settings.' + file_type
    factory.import_settings(get_data(file_name))

    changed_attrs = {'method': 'loaded method', 'basis': 'loaded basis', 'program': 'loaded program', 'tag': 'loaded tag'}
    for attr, value in changed_attrs.items():
        assert getattr(factory, attr) == value


@pytest.mark.parametrize('file_type',
                         [
                             pytest.param('json', id='json'),
                             pytest.param('yaml', id='yaml')
                         ])
def test_importing_settings_workflow(file_type):
    """
    Test importing the settings and a workflow from the supported file types.
    """

    factory = BasicDatasetFactory()

    file_name = 'settings_with_workflow.' + file_type
    factory.import_settings(get_data(file_name))

    changed_attrs = {'method': 'loaded method', 'basis': 'loaded basis', 'program': 'loaded program',
                     'tag': 'loaded tag'}
    for attr, value in changed_attrs.items():
        assert getattr(factory, attr) == value

    assert len(factory.workflow) == 1
    assert 'StandardConformerGenerator' in factory.workflow
    component = factory.get_workflow_component('StandardConformerGenerator')
    assert component.component_description == 'loaded component'
    assert isinstance(component, workflow_components.StandardConformerGenerator) is True


@pytest.mark.parametrize('file_type',
                         [
                             pytest.param('json', id='json'),
                             pytest.param('yaml', id='yaml')
                         ])
def test_import_workflow_only(file_type):
    """
    Test importing a workflow only from a workflow file.
    """

    factory = BasicDatasetFactory()

    factory2 = BasicDatasetFactory()

    file_name = 'settings_with_workflow.' + file_type

    factory.import_workflow(get_data(file_name))
    # make sure the settings have not changed from default
    assert factory.dict(exclude={'workflow'}) == factory2.dict(exclude={'workflow'})
    assert len(factory.workflow) == 1


@pytest.mark.parametrize('file_type',
                         [
                             pytest.param('json', id='json'),
                             pytest.param('yaml', id='yaml')
                         ])
def test_export_workflow_only(file_type):
    """
    Test exporting the workflow only from the factory.
    """

    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)
        factory = BasicDatasetFactory()

        conformer_gen = workflow_components.StandardConformerGenerator()
        conformer_gen.max_conformers = 100

        factory.add_workflow_component(conformer_gen)

        file_name = 'workflow.' + file_type
        factory.export_workflow(file_name)

        with open(file_name) as workflow:
            data = workflow.read()
            assert "method" not in data
            assert "basis" not in data
            assert "tag" not in data


def test_basic_factory_index():
    """
    Test the basic factories ability to make a molecule index this should be the canonical, isomeric smiles.
    """

    factory = BasicDatasetFactory()

    mol = Molecule.from_smiles('CC')

    index = factory.create_index(mol)

    assert index == mol.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False)


def test_basic_factory_cmiles():
    """
    Test the basic factories ability to make cmiles attributes for the molecules.
    """

    factory = BasicDatasetFactory()
    mol = Molecule.from_smiles('CC')

    cmiles_factory = factory.create_cmiles_metadata(mol)

    # now make our own cmiles
    test_cmiles = {'canonical_smiles': mol.to_smiles(isomeric=False, explicit_hydrogens=False, mapped=False),
                   'canonical_isomeric_smiles': mol.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False),
                   'canonical_explicit_hydrogen_smiles': mol.to_smiles(isomeric=False, explicit_hydrogens=True,
                                                                       mapped=False),
                   'canonical_isomeric_explicit_hydrogen_smiles': mol.to_smiles(isomeric=True, explicit_hydrogens=True,
                                                                                mapped=False),
                   'canonical_isomeric_explicit_hydrogen_mapped_smiles': mol.to_smiles(isomeric=True,
                                                                                       explicit_hydrogens=True,
                                                                                       mapped=True),
                   'molecular_formula': mol.hill_formula,
                   'standard_inchi': mol.to_inchi(fixed_hydrogens=False),
                   'inchi_key': mol.to_inchikey(fixed_hydrogens=False)}
    assert test_cmiles == cmiles_factory


def test_optimization_driver():
    """
    Test the optimization factory to make sure the driver can not be changed.
    """

    factory = OptimizationDatasetFactory()

    with pytest.raises(DriverError):
        factory.driver = 'energy'

    assert factory.driver == 'gradient'


def test_torsiondrive_index():
    """
    Test generating an index using torsiondrive, this should tag the atoms in the torsion.
    """

    mol = Molecule.from_file(get_data('methanol.sdf'))

    mol.properties['atom_map'] = {4: 0, 0: 1, 1: 2, 5: 3}

    factory = TorsiondriveDatasetFactory()

    index = factory.create_index(mol)

    tags = ['[C:2]', '[H:1]', '[O:3]', '[H:4]']
    for tag in tags:
        assert tag in index


def test_torsiondrive_linear_torsion():
    """
    Test the torsiondrive factorys ability to find linear bonds which should not be driven.
    """

    factory = TorsiondriveDatasetFactory()
    molecules = Molecule.from_file(get_data('linear_molecules.sdf'), 'sdf', allow_undefined_stereo=True)

    for molecule in molecules:
        assert bool(factory._detect_linear_torsions(molecule)) is True


def test_torsiondrive_torsion_string():
    """
    Test the torsiondrive factories ability to create a torsion string for a given bond.
    """

    factory = TorsiondriveDatasetFactory()

    methanol = Molecule.from_file(get_data('methanol.sdf'), 'sdf')

    rotatable = methanol.find_rotatable_bonds()
    assert len(rotatable) == 1

    bond = (rotatable[0].atom1_index, rotatable[0].atom2_index)
    torsion = factory._get_torsion_string(molecule=methanol, bond=bond)

    # now make sure this torsion is in the propers list
    reference_torsions = []
    for proper in methanol.propers:
        dihedral = []
        for atom in proper:
            dihedral.append(atom.molecule_atom_index)
        reference_torsions.append(tuple(dihedral))

    assert torsion in reference_torsions or tuple(reversed(torsion)) in reference_torsions


def test_create_basic_dataset():
    """
    Test the basic datasets factory in making a basic dataset type with the correct settings.
    """

    factory = BasicDatasetFactory()
    element_filter = workflow_components.ElementFilter()
    element_filter.allowed_elements = [1, 6, 8, 7]
    factory.add_workflow_component(element_filter)

    mols = Molecule.from_file(get_data('tautomers.smi'), 'smi', allow_undefined_stereo=True)

    # set some settings
    changed_attrs = {'method': 'test method', 'basis': 'test basis', 'program': 'test program', 'tag': 'test tag'}
    for attr, value in changed_attrs.items():
        setattr(factory, attr, value)

    dataset = factory.create_dataset(dataset_name='test name', molecules=mols)

    # check the attributes were changed
    for attr, value in changed_attrs.items():
        assert getattr(dataset, attr) == value

    assert dataset.dataset_name == 'test name'

    assert isinstance(dataset, BasicDataSet) is True

    # make sure molecules we filtered and passed
    assert dataset.dataset != {}
    assert dataset.filtered != {}
    assert element_filter.component_name in dataset.filtered_molecules


