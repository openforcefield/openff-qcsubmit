"""
Tests for building and running workflows, exporting and importing settings.
"""
import pytest

import tempfile
from qcsubmit.factories import BasicDatasetFactory, OptimizationDatasetFactory, TorsiondriveDatasetFactory
from qcsubmit import workflow_components
from qcsubmit.utils import get_data
from qcsubmit.exceptions import InvalidWorkflowComponentError
from pydantic import ValidationError
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

        factory.export_workflow()

