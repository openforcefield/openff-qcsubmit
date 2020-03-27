"""
Tests for each of the workflow components try to avoid specific openeye functions.
"""
from typing import List, Dict
import pytest
from qcsubmit import workflow_components
from qcsubmit.datasets import ComponentResult
from openforcefield.topology import Molecule
from openforcefield.utils import get_data_file_path
from openforcefield.utils.toolkits import UndefinedStereochemistryError
from functools import lru_cache


@lru_cache()
def molecues(incude_undefined_stereo: bool = False, include_conformers: bool = True):
    """
    Return a list of molecules meeting the required spec from the minidrugbank list.
    """

    mols = Molecule.from_file(get_data_file_path('molecules/MiniDrugBank.sdf'), allow_undefined_stereo=True)

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


def test_custom_component():
    """
    Make sure users can not use custom components unless all method are implemented.
    """

    with pytest.raises(TypeError):
        test = workflow_components.CustomWorkflowComponent()

    class TestComponent(workflow_components.CustomWorkflowComponent):

        component_name = 'Test component'
        component_description = 'Test component'
        component_fail_message = 'Test fail'

        def apply(self, molecules: List[Molecule]) -> ComponentResult:
            pass

        def provenance(self) -> Dict:
            pass

    test = TestComponent()
    assert test.component_name == 'Test component'
    assert test.component_description == 'Test component'
    assert test.component_fail_message == 'Test fail'


def test_standardconformer_generator_validators():
    """
    Test the standard conformer generator which calls the OFFTK.
    """

    conf_gen = workflow_components.StandardConformerGenerator()

    # make sure we are casting to ints
    conf_gen.max_conformers = 1.02
    assert conf_gen.max_conformers == 1

    # test the toolkit validator
    with pytest.raises(ValueError):
        conf_gen.toolkit = 'ambertools'
    conf_gen.toolkit = 'rdkit'
    assert 'rdkit' in conf_gen.provenance()
    assert 'openeye' not in conf_gen.provenance()

    conf_gen.toolkit = 'openeye'
    assert 'openeye' in conf_gen.provenance()
    assert 'rdkit' not in conf_gen.provenance()

    conf_gen.clear_exsiting = 'no'
    assert conf_gen.clear_exsiting is False

    assert 'OpenforcefieldToolkit' in conf_gen.provenance()


def test_element_filter_validators():
    """
    Make sure the element filter validators are working.
    """

    elem_filter = workflow_components.ElementFilter()

    with pytest.raises(KeyError):
        elem_filter.allowed_elements = ['carbon', 'hydrogen']

    elem_filter.allowed_elements = [1.02, 2.02, 3.03]

    assert elem_filter.allowed_elements == [1, 2, 3]

    assert 'openmm_elements' in elem_filter.provenance()


def test_weight_filter_validator():
    """
    Make sure the weight filter simple validators work.
    """
    weight = workflow_components.MolecularWeightFilter()

    weight.minimum_weight = 0.0
    assert weight.minimum_weight == 0

    assert 'openmm_units' in weight.provenance()
    assert 'OpenforcefieldToolkit' in weight.provenance()


@pytest.mark.parametrize(
    'data',
    [
        pytest.param((workflow_components.ElementFilter, 'allowed_elements', [1, 10, 100]), id='ElementFilter'),
        pytest.param((workflow_components.MolecularWeightFilter, 'minimum_weight', 1), id='WeightFilter'),
        pytest.param((workflow_components.StandardConformerGenerator, 'max_conformers', 1), id='StandardConformers')
    ]
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


def test_conformer_apply():
    """
    Test applying the standard conformer generator to a workflow.
    """

    conf_gen = workflow_components.StandardConformerGenerator()
    conf_gen.toolkit = 'rdkit'
    conf_gen.max_conformers = 1
    conf_gen.clear_exsiting = True

    mols = molecues(incude_undefined_stereo=False, include_conformers=False)

    result = conf_gen.apply(mols)

    assert result.component_name == conf_gen.component_name
    assert result.component_description == conf_gen.dict()
    # make sure each molecule has a conformer that passed
    for molecule in result.molecules:
        assert molecule.n_conformers == 1

    for molecule in result.filtered:
        assert molecule.n_conformers == 0

def test_elementfilter_apply():
    """
    Test applying the element filter to a workflow.
    """

    elem_filter = workflow_components.ElementFilter()
    elem_filter.allowed_elements = [1, 6, 7, 8]

    mols = molecues(include_conformers=False)

    result = elem_filter.apply(mols)

    assert result.component_name == elem_filter.component_name
    assert result.component_description == elem_filter.dict()
    # make sure there are no unwanted elements in the pass set
    for molecue in result.molecules:
        for atom in molecue.atoms:
            assert atom.atomic_number in elem_filter.allowed_elements

    for molecue in result.filtered:
        elements = set([atom.atomic_number for atom in molecue.atoms])
        assert sorted(elements) != elem_filter.allowed_elements

