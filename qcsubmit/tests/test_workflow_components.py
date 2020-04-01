"""
Tests for each of the workflow components try to avoid specific openeye functions.
"""
from typing import List, Dict
import pytest
from qcsubmit import workflow_components
from qcsubmit.datasets import ComponentResult
from openforcefield.topology import Molecule
from openforcefield.utils import get_data_file_path
from openforcefield.utils.toolkits import UndefinedStereochemistryError, RDKitToolkitWrapper, OpenEyeToolkitWrapper
from functools import lru_cache


@lru_cache()
def get_molecues(incude_undefined_stereo: bool = False, include_conformers: bool = True):
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


@pytest.mark.parametrize('toolkit',
                         [
                             pytest.param(('openeye', OpenEyeToolkitWrapper), id='openeye'),
                             pytest.param(('rdkit', RDKitToolkitWrapper), id='rdkit')
                         ])
def test_toolkit_mixin(toolkit):
    """
    Make sure the pydantic ToolkitValidator mixin is working correctly, it should provide provenance and toolkit name
    validation.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():
        class TestClass(workflow_components.ToolkitValidator, workflow_components.CustomWorkflowComponent):
            """Should not need to implement the provenance."""

            component_name = 'ToolkitValidator'
            component_description = 'ToolkitValidator test class.'
            component_fail_message = 'Test fail'

            def apply(self, molecules: List[Molecule]) -> ComponentResult:
                pass

        test = TestClass()
        with pytest.raises(ValueError):
            test.toolkit = 'ambertools'

        test.toolkit = toolkit_name
        prov = test.provenance()
        assert toolkit_name in prov
        assert 'OpenforcefieldToolkit' in prov

    else:
        pytest.skip(f'Toolkit {toolkit_name} not avilable.')


@pytest.mark.parametrize('toolkit',
                         [
                             pytest.param(('openeye', OpenEyeToolkitWrapper), id='openeye'),
                             pytest.param(('rdkit', RDKitToolkitWrapper), id='rdkit')
                         ])
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
            conf_gen.toolkit = 'ambertools'
        conf_gen.toolkit = toolkit_name
        assert toolkit_name in conf_gen.provenance()

        conf_gen.clear_existing = 'no'
        assert conf_gen.clear_existing is False

        assert 'OpenforcefieldToolkit' in conf_gen.provenance()

    else:
        pytest.skip(f'Toolkit {toolkit} not available.')


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
        pytest.param((workflow_components.StandardConformerGenerator, 'max_conformers', 1), id='StandardConformers'),
        pytest.param((workflow_components.EnumerateTautomers, 'max_tautomers', 2), id='EnumerateTautomers'),
        pytest.param((workflow_components.EnumerateStereoisomers, 'undefined_only', True), id='EnumerateStereoisomers')
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


@pytest.mark.parametrize('toolkit',
                         [
                             pytest.param(('openeye', OpenEyeToolkitWrapper), id='openeye'),
                             pytest.param(('rdkit', RDKitToolkitWrapper), id='rdkit')
                         ])
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

        mols = get_molecues(incude_undefined_stereo=False, include_conformers=False)

        result = conf_gen.apply(mols)

        assert result.component_name == conf_gen.component_name
        assert result.component_description == conf_gen.dict()
        # make sure each molecule has a conformer that passed
        for molecule in result.molecules:
            assert molecule.n_conformers == 1

        for molecule in result.filtered:
            assert molecule.n_conformers == 0

    else:
        pytest.skip(f'Toolkit {toolkit_name} not available.')


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


@pytest.mark.parametrize('toolkit',
                         [
                             pytest.param(('openeye', OpenEyeToolkitWrapper), id='openeye'),
                             pytest.param(('rdkit', RDKitToolkitWrapper), id='rdkit')
                         ])
def test_enumerating_stereoisomers_validator(toolkit):
    """
    Test the validators in enumerating stereoisomers.
    """

    toolkit_name, toolkit_class = toolkit
    if toolkit_class.is_available():

        enumerate_stereo = workflow_components.EnumerateStereoisomers()
        with pytest.raises(ValueError):
            enumerate_stereo.toolkit = 'ambertools'

        enumerate_stereo.toolkit = toolkit_name
        assert toolkit_name in enumerate_stereo.provenance()

        enumerate_stereo.undefined_only = 'y'
        assert enumerate_stereo.undefined_only is True

        enumerate_stereo.max_isomers = 1.1
        assert enumerate_stereo.max_isomers == 1

    else:
        pytest.skip(f'Toolkit {toolkit_name} is not available.')


@pytest.mark.parametrize('toolkit',
                         [
                             pytest.param(('openeye', OpenEyeToolkitWrapper), id='openeye'),
                             pytest.param(('rdkit', RDKitToolkitWrapper), id='rdkit')
                         ])
def test_enumerating_tautomers_validator(toolkit):
    """
    Test the validators in enumerating tautomers.
    """

    toolkit_name, toolkit_class = toolkit

    if toolkit_class.is_available():

        enumerate_tautomers = workflow_components.EnumerateTautomers()

        with pytest.raises(ValueError):
            enumerate_tautomers.toolkit = 'ambertools'

        enumerate_tautomers.toolkit = toolkit_name
        enumerate_tautomers.max_tautomers = 1.1
        assert enumerate_tautomers.max_tautomers == 1

        assert toolkit_name in enumerate_tautomers.provenance()

    else:
        pytest.skip(f'Toolkit {toolkit_name} not available.')



