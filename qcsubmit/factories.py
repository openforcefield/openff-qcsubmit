from typing import List, Union, Dict
from pydantic import BaseModel, validator
import yaml
import json

import os

from qcsubmit import workflow_components
from qcsubmit.datasets import ComponentResult, BasicDataSet
from qcsubmit.exceptions import UnsupportedFiletypeError, InvalidWorkflowComponentError, MissingWorkflowComponentError, \
    InvalidClientError, DriverError
from qcsubmit.procedures import GeometricProcedure

from openforcefield.topology import Molecule

"""
Tools for aiding the construction and submission of QCFractal datasets.

TODO 

Make a base class that has very basic functionality and would idealy be used in the case of a plain dataset submission.
This class should be able to:
    - generate an input for basic dataset
    - expand states
    - filter the molecules
    - set the QM options from file
    - serialise the current options to file
    - generate the required number of conformers, we should not bin the input conformer if there are some, there must be atleast one for this to work.
    - deduplicate submissions, condense the conformers down to one molecule if there are some on each molecule

We also require some subclasses which can control there own options and operations:
    - fragmenter
    - QMOptions
    - Geometric/optimisation options
    - Torsiondrive/scan options
"""


class BasicDatasetFactory(BaseModel):
    """
    Basic dataset generator factory used to build work flows using workflow components before executing them to generate
    a dataset.

    The basic dataset is ideal for large collections of single point calculations using any of the energy, gradient or
    hessian drivers.

    The main metadata features here are concerned with the QM settings to be used which includes the driver.

    Attributes:
        method:  The QM theory to use during dataset calculations.
        basis:   The basis set to use during dataset calculations.
        program: The program which will be used during the calculations.
        maxiter: The maximum amount of SCF cycles allowed.
        driver:  The driver that should be used in the calculation ie energy/gradient/hessian.
        scf_properties: A list of QM properties which should be calculated and collected during the driver calculation.
        client:  The name of the client the data will be sent to for privet clusters this should be the file name where
            the data is stored.
        priority: The priority with which the dataset should be calculated.
        tag: The tag name which should be given to the collection.
        workflow: A dictionary which holds the workflow components to be executed in the set order.
    """

    method: str = 'B3LYP-D3BJ'
    basis: str = 'DZVP'
    program: str = 'psi4'
    maxiter: int = 200
    driver: str = 'energy'
    scf_properties: List[str] = ['dipole', 'qudrupole', 'wiberg_lowdin_indices']
    spec_name: str = 'default'
    spec_description: str = 'Standard OpenFF optimization quantum chemistry specification.'
    client: str = 'public'
    priority: str = 'normal'
    tag: str = 'openff'
    workflow: Dict[str, workflow_components.CustomWorkflowComponent] = {}

    # hidden variable not included in the schema
    _file_readers = {'json': json.load, 'yaml': yaml.full_load}
    _file_writers = {'json': json.dump, 'yaml': yaml.dump}

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        title = 'QCFractalDatasetFactory'

    @validator('driver')
    def _check_driver(cls, driver):
        """Make sure that the driver is supported."""
        available_drivers = ['energy', 'gradient', 'hessian']
        if driver not in available_drivers:
            raise DriverError(f'The requested driver ({driver}) is not in the list of available '
                              f'drivers: {available_drivers}')
        return driver

    @validator('client')
    def _check_client(cls, client):
        """Make sure the client is valid."""
        if isinstance(client, str):
            if client == 'public' or os.path.exists(client):
                return client
        raise InvalidClientError('The client must be set to public or a file path to some client settings.')

    def clear_workflow(self) -> None:
        """
        Reset the workflow to by empty.
        """
        self.workflow = {}

    def add_workflow_component(self, components: Union[List[workflow_components.CustomWorkflowComponent], workflow_components.CustomWorkflowComponent]) -> None:
        """
        Take the workflow components validate them then insert them into the workflow.

        Parameters:
            components: A list of or an individual qcsubmit.workflow_compoents.CustomWokflowComponent which are to be
                validated and added to the current workflow.

        Raises:
            InvalidWorkflowComponentError: If an invalid workflow component is attempted to be added to the workflow.
        """

        if not isinstance(components, list):
            # we have one component make it into a list
            components = [components]

        for component in components:
            if isinstance(component, workflow_components.CustomWorkflowComponent):
                if component.component_name not in self.workflow.keys():
                    self.workflow[component.component_name] = component
                else:
                    # we should increment the name and add it to the workflow
                    if '@' in component.component_name:
                        name, number = component.component_name.split('@')
                    else:
                        name, number = component.component_name, 0
                    # set the new name
                    component.component_name = f'{name}@{int(number) + 1}'
                    self.workflow[component.component_name] = component

            else:
                raise InvalidWorkflowComponentError(f'Component {component} rejected as it is not a sub '
                                                    f'class of CustomWorkflowComponent.')

    def get_workflow_component(self, component_name: str) -> workflow_components.CustomWorkflowComponent:
        """
        Find the workflow component by its component_name attribute.

        Parameters:
            component_name: The name of the component to be gathered from the workflow.

        Returns:
            The instance of the requested component from the workflow.

        Raises:
            MissingWorkflowComponentError: If the component could not be found by its component name in the workflow.
        """

        component = self.workflow.get(component_name, None)
        if component is None:
            raise MissingWorkflowComponentError(f'The requested component {component_name} '
                                                f'was not registered into the workflow.')

        return component

    def remove_workflow_component(self, component_name: str) -> None:
        """
        Find and remove the component via its component_name attribute.

        Parameters:
            component_name: The name of the component to be gathered from the workflow.

        Raises:
            MissingWorkflowComponentError: If the component could not be found by its component name in the workflow.
        """

        try:
            del self.workflow[component_name]

        except KeyError:
            raise MissingWorkflowComponentError(f'The requested component {component_name} '
                                                f'could not be removed as it was not registered.')

    def import_workflow(self, workflow: Union[str, Dict], clear_existing: bool = True) -> None:
        """
        Instance the workflow from a workflow object or from an input file.

        Parameters:
            workflow: The name of the file the workflow should be created from or a workflow dictionary.
            clear_existing: If the current workflow should be deleted and replaced or extended.
        """

        if clear_existing:
            self.clear_workflow()

        if isinstance(workflow, str):
            workflow = self._read_file(workflow)

        if isinstance(workflow, dict):
            # this should be a workflow dict that we can just load
            # if this is from the settings file make sure to unpack the dict first.
            workflow = workflow.get('workflow', workflow)

        # load in the workflow
        for key, value in workflow.items():
            # check if this is not the first instance of the component
            if '@' in key:
                name = key.split('@')[0]
            else:
                name = key
            component = getattr(workflow_components, name, None)
            if component is not None:
                self.add_workflow_component(component.parse_obj(value))

    def _read_file(self, file_name: str) -> Dict:
        """
        Method to help with file reading and returning the data object from the file.

        Parameters:
            file_name: The name of the file to be read.

        Returns:
            The dictionary representation of the json or yaml file.

        Raises:
            UnsupportedFiletypeError: The file give was not of the supported types ie json or yaml.
            RuntimeError: The given file could not be found to be opened.
        """

        # check that the file exists
        if os.path.exists(file_name):
            file_type = self._get_file_type(file_name)
            try:
                reader = self._file_readers[file_type]
                with open(file_name) as input_data:
                    data = reader(input_data)

                    return data

            except KeyError:
                raise UnsupportedFiletypeError(f'The requested file type {file_type} is not supported '
                                               f'currently we can write to {self._file_writers}.')
        else:
            raise RuntimeError(f'The file {file_name} could not be found.')

    def export_workflow(self, file_name: str) -> None:
        """
        Export the workflow components and their settings to file so that they can be loaded latter.

        Parameters:
            file_name: The name of the file the workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: If the file type is not supported.
        """

        file_type = self._get_file_type(file_name=file_name)

        # try and get the file writer
        workflow = self.dict()['workflow']
        try:
            writer = self._file_writers[file_type]
            with open(file_name, 'w') as output:
                if file_type == 'json':
                    writer(workflow, output, indent=2)
                else:
                    writer(workflow, output)
        except KeyError:
            raise UnsupportedFiletypeError(f'The requested file type {file_type} is not supported, '
                                           f'currently we can write to {self._file_writers}.')

    def _get_file_type(self, file_name: str) -> str:
        """
        Helper function to work out the file type being requested from the file name.

        Parameters:
            file_name: The name of the file from which we should work out the file type.

        Returns:
            The file type extension.
        """

        file_type = file_name.split('.')[-1]
        return file_type

    def export_settings(self, file_name: str) -> None:
        """
        Export the current model to file this will include the workflow as well along with each components settings.

        Parameters:
            file_name: The name of the file the settings and workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: When the file type requested is not supported.
        """

        file_type = self._get_file_type(file_name=file_name)

        # try and get the file writer
        try:
            writer = self._file_writers[file_type]
            with open(file_name, 'w') as output:
                if file_type == 'json':
                    writer(self.dict(), output, indent=2)
                else:
                    writer(self.dict(), output)
        except KeyError:
            raise UnsupportedFiletypeError(f'The requested file type {file_type} is not supported, '
                                           f'currently we can write to {self._file_writers}.')

    def import_settings(self, settings: Union[str, Dict], clear_workflow: bool = True) -> None:
        """
        Import settings and workflow from a file.

        Parameters:
            settings: The name of the file the settings should be extracted from or the reference to a settings
                dictionary.
            clear_workflow: If the current workflow should be extended or replaced.
        """

        if isinstance(settings, str):
            data = self._read_file(settings)

            # take the workflow out and import the settings
            workflow = data.pop('workflow')

        elif isinstance(settings, dict):
            workflow = settings.pop('workflow')
            data = settings

        else:
            raise RuntimeError(f'The input type could not be converted into a settings dictionary.')

        # now set the factory meta settings
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                continue

        # now we want to add the workflow back in
        self.import_workflow(workflow=workflow, clear_existing=clear_workflow)

    def add_compute(self, dataset_name: str, await_result: bool = False) -> None:
        """
        A method that can add compute to an existing collection, this involves registering the QM settings and keywords
        and running the compute.

        Parameters:
            dataset_name: The name of the collection in the qcarchive instance that the compute should be added to.
            await_result: If the function should block until the calculations have finished.

        """
        import qcportal as ptl

        if self.client == 'public':
            client = ptl.FractalClient()
        else:
            client = ptl.FractalClient.from_file(self.client)

        try:
            collection = client.get_collection('Dataset', dataset_name)
            kw = ptl.models.KeywordSet(values=self.dict(include={'maxiter', 'scf_properties'}))
            collection.add_keywords(alias=self.spec_name, program=self.program, keyword=kw, default=True)

            # save the keywords
            collection.save()

            # submit the calculations
            response = collection.compute(method=self.method, basis=self.basis, keywords=self.spec_name,
                                          program=self.program, tag=self.tag, priority=self.priority)

        except KeyError:
            raise KeyError(f'The collection: {dataset_name} could not be found, you can only add compute to existing'
                           f'collections.')

    def create_dataset(self, dataset_name: str, molecules: Union[str, List[Molecule], Molecule]) -> BasicDataSet:
        """
        Process the input molecules through the given workflow then create and populate the dataset class which acts as
        a local representation for the collection in qcarchive and has the ability to submit its self to a local or
        public instance.

        Parameters:
             dataset_name: The name that will be given to the collection on submission to an archive instance.
             molecules: The list of molecules which should be processed by the workflow and added to the dataset, this
                can also be a file name which is to be unpacked by the openforcefield toolkit.

        Example:
            How to make a dataset from a list of molecules

            ```python
            >>> from qcsubmit.factories import BasicDatasetFactory
            >>> from qcsubmit import workflow_components
            >>> from openforcefield.topology import Molecule
            >>> factory = BasicDatasetFactory()
            >>> gen = workflow_components.StandardConformerGenerator()
            >>> gen.clear_exsiting = True
            >>> gen.max_conformers = 1
            >>> factory.add_workflow_component(gen)
            >>> smiles = ['C', 'CC', 'CCO']
            >>> mols = [Molecule.from_smiles(smile) for smile in smiles]
            >>> dataset = factory.create_dataset(dataset_name='My collection', molecules=mols)
            ```

        Returns:
            A [DataSet][qcsubmit.datasets.DataSet] instance populated with the molecules that have passed through the
                workflow.

        Important:
            The dataset once created does not allow mutation.
        """
        #TODO set up a logging system to report the components

        #  check if we have been given an input file with molecules inside
        if isinstance(molecules, str):
            if os.path.exists(molecules):
                workflow_molecules = ComponentResult(component_name=self.Config.title,
                                                     component_description={'component_name': self.Config.title},
                                                     input_file=molecules)

        elif isinstance(molecules, Molecule):
            workflow_molecules = ComponentResult(component_name=self.Config.title,
                                                 component_description={'component_name': self.Config.title},
                                                 molecules=[molecules])

        else:
            workflow_molecules = ComponentResult(component_name=self.Config.title,
                                                 component_description={'component_name': self.Config.title},
                                                 molecules=molecules)

        # now we need to start passing the workflow molecules to each module in the workflow
        filtered_molecules = {}
        # if the workflow has components run it
        if self.workflow:
            for component_name, component in self.workflow.items():
                workflow_molecules = component.apply(molecules=workflow_molecules.molecules)

                filtered_molecules[workflow_molecules.component_name] = {'component_description': workflow_molecules.component_description,
                                                                         'molecules': workflow_molecules.filtered}

        # first we need to instance the dataset and assign the metadata
        object_meta = self.dict(exclude={'workflow'})

        # the only data missing is the collection name so add it here.
        object_meta['dataset_name'] = dataset_name
        dataset = BasicDataSet.parse_obj(object_meta)

        # now add the molecules to the correct attributes
        for molecule in workflow_molecules.molecules:
            # order the molecule
            order_mol = molecule.canonical_order_atoms()
            # now submit the molecule
            dataset.add_molecule(index=self.create_index(molecule=order_mol),
                                 molecule=order_mol,
                                 cmiles=self.create_cmiles_metadata(molecule=order_mol))

        # now we need to add the filtered molecules
        for component_name, result in filtered_molecules.items():
            dataset.filter_molecules(molecules=result['molecules'], component_description=result['component_description'])

        return dataset

    def create_cmiles_metadata(self, molecule: Molecule) -> Dict[str, str]:
        """
        Create the Cmiles metadata for the molecule in this dataset.

        Parameters:
            molecule: The molecule for which the cmiles data will be generated.

        Returns:
            The Cmiles identifiers generated for the input molecule.

        Note:
            The Cmiles identifiers currently include:

            - `canonical_smiles`
            - `canonical_isomeric_smiles`
            - `canonical_explicit_hydrogen_smiles`
            - `canonical_isomeric_explicit_hydrogen_smiles`
            - `canonical_isomeric_explicit_hydrogen_mapped_smiles`
            - `molecular_formula`
            - `standard_inchi`
            - `inchi_key`
        """

        cmiles = {'canonical_smiles': molecule.to_smiles(isomeric=False, explicit_hydrogens=False, mapped=False),
                  'canonical_isomeric_smiles': molecule.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False),
                  'canonical_explicit_hydrogen_smiles': molecule.to_smiles(isomeric=False, explicit_hydrogens=True, mapped=False),
                  'canonical_isomeric_explicit_hydrogen_smiles': molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=False),
                  'canonical_isomeric_explicit_hydrogen_mapped_smiles': molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True),
                  'molecular_formula': molecule.hill_formula,
                  'standard_inchi': molecule.to_inchi(fixed_hydrogens=False),
                  'inchi_key': molecule.to_inchikey(fixed_hydrogens=False)}

        return cmiles

    def create_index(self, molecule: Molecule) -> str:
        """
        Create an index for the current molecule.

        Parameters:
            molecule: The molecule for which the dataset index will be generated.

        Returns:
            The canonical isomeric smiles for the molecule which is used as the dataset index.

        Important:
            Each dataset can have a different indexing system depending on the data, in this basic dataset each conformer
            of a molecule is expanded into its own entry separately indexed entry. This is handled by the dataset however
            so we just generate a general index for the molecule before adding to the dataset.
        """

        index = molecule.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False)
        return index


class OptimizationDatasetFactory(BasicDatasetFactory):
    """
    This factory produces OptimisationDatasets which include settings associated with geometric which is used to run the
    optimisation.

    Attributes:

    """

    # set the driver to be gradient this should not be changed when running
    driver = 'gradient'

    # use the default geometric settings during optimisation
    optimisation_program: GeometricProcedure = GeometricProcedure()


class TorsiondriveDatasetFactory(OptimizationDatasetFactory):
    """
    This factory produces TorsiondriveDatasets which include settings associated with geometric which is used to run
    the optimisation.
    """

    grid_spacings: List[int] = [15]

    # set the default settings for a torsiondrive calculation.
    optimisation_program = GeometricProcedure.parse_obj({'enforce': 0.1, 'reset': True, 'qccnv': True, 'epsilon': 0.0})

    def create_index(self, molecule: Molecule) -> str:
        """
        Create a specific torsion index for the molecule, this will use the atom map on the molecule.

        Parameters:
            molecule:  The molecule for which the dataset index will be generated.

        Returns:
            The canonical mapped isomeric smiles, where the mapped indices are on the atoms in the torsion.

        Important:
            This dataset uses a non-standard indexing with 4 atom mapped indices representing the atoms in the torsion
            to be rotated.
        """

        assert 'atom_map' in molecule.properties.keys()

        index = molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        return index
