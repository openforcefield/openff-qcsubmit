from typing import List, Union, Dict
from pydantic import BaseModel, validator
import yaml
import json

import os

from qcsubmit import workflow_components
from qcsubmit.datasets import ComponentResult
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


class UnsuportedFiletypeError(Exception):
    pass


class QCFractalDatasetFactory(BaseModel):
    """
    Base class to control the metadata settings used during generating the work flow.
    This class should not need to be subclassed in order to add new componets or change the oder this should be done,
    via the workflow componet functionality.

    The main metadata features here are concerned with the QM settings to be used which includes the driver.

    Attributes
    ----------
    theory : str, default=B3LYP-D3BJ
        The QM theory used during the calculation
    basis : str, default=DZVP
        The basis set to use during the calculation
    enumerate_tautomers : bool, default=False
        If True, will enumerate tautomers
    max_conformers : int, default=20
        Maximum number of conformers generated per molecule
    input_filters : list, default=[]
        Filters to be applied to input molecules
    submit_filters : list, default=[]
        Filters to be applied to molecules prior to submission
    compute_wbo : bool, default=False
        If True, compute Wiberg-Lowdin bond orders

    """

    theory: str = 'B3LYP-D3BJ'  # the default level of theory for openff
    basis: str = 'DZVP'  # the default basis for openff
    program: str = 'psi4'
    maxiter: int = 200
    driver: str = 'energy'
    scf_properties: List[str] = ['dipole', 'qudrupole', 'wiberg_lowdin_indices']
    client: str = 'public'
    priority: str = 'normal'
    tags: str = 'openff'
    workflow: Dict[str, workflow_components.CustomWorkflowComponet] = {}

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
            raise ValueError(f'The requested driver ({driver}) is not in the list of available '
                             f'drivers: {available_drivers}')
        return driver

    @validator('client')
    def _check_client(cls, client):
        """Make sure the client is valid."""
        if isinstance(client, str):
            if client == 'public' or os.path.exists(client):
                return client
        raise ValueError('The client must be set to public or a file path to some client settings.')

    def clear_workflow(self):
        self.workflow = {}

    def add_workflow_component(self, components: Union[List[workflow_components.CustomWorkflowComponet], workflow_components.CustomWorkflowComponet]):
        """Take the workflow components and insert them into the workflow."""

        if not isinstance(components, list):
            # we have one componenet make it into a list
            components = [components]

        for component in components:
            if isinstance(component, workflow_components.CustomWorkflowComponet):
                if component.componet_name not in self.workflow.keys():
                    self.workflow[component.componet_name] = component
                else:
                    # we should increment the name and add it to the workflow
                    if '@' in component.componet_name:
                        name, number = component.componet_name.split('@')
                    else:
                        name, number = component.componet_name, 0
                    # set the new name
                    component.componet_name = f'{name}@{int(number) + 1}'
                    self.workflow[component.componet_name] = component

            else:
                print(f'Component {component} rejected as it is not subclass of CustomWorkflowComponent.')

    # def insert_workflow_component(self, component: CustomWorkflowComponet, index: int):
    #     "Insert the component at the given index shifting values to the right."
    #
    #     if self._check_workflow(component):
    #         self._workflow.insert(index=index, object=component)

    def get_workflow_component(self, component_name: str):
        "Find the workflow compneent by its compnent_name."

        component = self.workflow.get(component_name, None)
        if component is None:
            raise workflow_components.ComponentMissingError(f'The requested component {component_name} '
                                                            f'was not registeried into the workflow.')

        return component

    def remove_workflow_component(self, component_name: str):
        "Find and remove the component via its name."

        try:
            del self.workflow[component_name]

        except KeyError:
            raise workflow_components.ComponentMissingError(f'The requested component {component_name} '
                                                            f'could not be removed as it was not registerd.')

    def import_workflow(self, workflow: Union[str, Dict], clear_exsisting: bool = True):
        "Instance the workflow from a workflow object or from an input file."

        if clear_exsisting:
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
                raise UnsuportedFiletypeError(f'The requested file type {file_type} is not supported currently we can write to {self._file_writers}.')
        else:
            raise RuntimeError(f'The file {file_name} could not be found.')

    def export_workflow(self, file: str):
        "write the workflow components to file so that they can be loaded latter"

        file_type = self._get_file_type(file=file)

        # try and get the file writer
        workflow = self.dict()['workflow']
        try:
            writer = self._file_writers[file_type]
            with open(file, 'w') as output:
                if file_type == 'json':
                    writer(workflow, output, indent=2)
                else:
                    writer(workflow, output)
        except KeyError:
            raise UnsuportedFiletypeError(f'The requested file type {file_type} is not supported, '
                                          f'currently we can write to {self._file_writers}.')

    def _get_file_type(self, file: str) -> str:
        "take the file name and work out the type of file we want to write to."

        file_type = file.split('.')[-1]
        return file_type

    def export_settings(self, file: str):
        "Export the current model to file this will include the workflow as well"
        file_type = self._get_file_type(file=file)

        # try and get the file writer
        try:
            writer = self._file_writers[file_type]
            with open(file, 'w') as output:
                if file_type == 'json':
                    writer(self.dict(), output, indent=2)
                else:
                    writer(self.dict(), output)
        except KeyError:
            raise UnsuportedFiletypeError(f'The requested file type {file_type} is not supported, '
                                          f'currently we can write to {self._file_writers}.')

    def import_settings(self, settings: Union[str, Dict], clear_workflow: bool = True):
        "import the settings in a file."

        if isinstance(settings, str):
            data = self._read_file(settings)

            # take the workflow out and inport the settings
            workflow = data.pop('workflow')

        elif isinstance(settings, dict):
            workflow = settings.pop('worklow')
            data = settings

        else:
            raise RuntimeError(f'The input type could not be converted into a settings dictionary.')

        # now set the factory meta settings
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                continue

        self.clear_workflow()
        # now we want to add the workflow back in
        self.import_workflow(workflow=workflow, clear_exsisting=clear_workflow)

    def create_dataset(self, dataset_name: str, molecules: Union[str, List[Molecule], Molecule]):
        "the main function which will create the dataset and the maetadata"

        #  check if we have been given an input file with molecules inside
        if isinstance(molecules, str):
            if os.path.exists(molecules):
                workflow_molecules = ComponentResult(component_name='inital',
                                                     component_description='initial',
                                                     component_fail_reason='none',
                                                     input_file=molecules)

        elif isinstance(molecules, Molecule):
            workflow_molecules = ComponentResult(component_name='inital',
                                                 component_description='initial',
                                                 component_fail_reason='none',
                                                 molecules=[molecules])

        else:
            workflow_molecules = ComponentResult(component_name='inital',
                                                 component_description='initial',
                                                 component_fail_reason='none',
                                                 molecules=molecules)

        # now we need to start passing the workflow molecules to each module in the workflow
        filtered_molecules = {}
        for componet_name, component in self.workflow.items():
            workflow_molecules = component.apply(molecules=workflow_molecules.molecules)

            filtered_molecules[workflow_molecules.component_name] = {'component_description': workflow_molecules.component_description,
                                                                     'component_fail_message': workflow_molecules.component_fail_reason,
                                                                     'molecules': workflow_molecules.filtered}

        # now we should print out the final molecules
        print('The final component result molecules', workflow_molecules.molecules)
        print('The final component failed molecules', workflow_molecules.filtered)
        print('The filtered molecules', filtered_molecules)

    def _create_cmiles_metadata(self, molecule: Molecule) -> Dict[str, str]:
        """
        Create the Cmiles metadata for the molecule in this dataset.

        Parameters:
            molecule: openforcefield.topology.Molecule,
                The molecule for which the cmiles data will be generated.

        Returns:
            cmiles: dict[str, str]
            The Cmiles index information generated for the input molecule.
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

    def _create_index(self, molecule: Molecule):
        """
        Create an index for the current molecule.

        Parameters:
            molecule: openforcefield.topology.Molecule,
                The molecule for which the dataset index will be generated.

        Returns:
            index: str
                The canonical isomeric smiles for the molecule which is used as the dataset index.

        Note:
            Each dataset can have a different indexing system depending on the data, in this basic dataset each conformer
            of a molecule is expanded into its own entry and submitted thus a unique index must be made.
        """

        index = molecule.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False)
        return index

    # @validator('client')
    # def _validate_client(cls, client):
    #     """
    #     Here we try and instance the client if it is not already an instance of the FratalClient class.
    #     """
    #     if not isinstance(client, FractalClient) or client != 'public':
    #         raise ValidationError('The client must be a supported')
    #     else:
    #         return client

#     def apply_filter(self, oemols, filter_name):
#         """
#         Apply the specified filter to the provided molecules
#
#         Available Filters
#         -----------------
#         * OpenEye filters: 'BlockBuster', 'Drug', 'Fragment', 'Lead', 'PAINS'
#           https://docs.eyesopen.com/toolkits/python/molproptk/filter_theory.html
#
#         .. todo ::
#
#            * Add QM filters to ensure computational tractability
#
#         Parameters
#         ----------
#         oemols : list of OEMol
#             The molecules to be filtered
#         filter : str
#             Filter to be applied
#
#         Returns
#         -------
#         oemols : list of OEMol
#             List of molecules after filtering (originals are used)
#         """
#         if filter_name in ['Blockbuster', 'Drug', 'Fragment', 'Lead', 'PAINS']:
#             # Apply OpenEye Filters
#             from openeye import oemolprop
#             filter = oemolprop.OEFilter(getattr(oemolprop, 'OEFilterType_' + filter_name)
#         else:
#             raise Exception(f'Filter type {filter_name} unknown')
#
#         new_oemols = list()
#         for oemol in oemols:
#             if filter(oemol):
#                 new_oemols.append(oemol)
#         print('Retained {len(new_oemols)} molecules after application of {filter_name} filter')
#         return new_oemols
#
#     def apply_filters(self, oemols, filters):
#         """
#         Apply filters to the provided molecules.
#
#         Parameters
#         ----------
#         oemols : list of OEMol
#             The molecules to be filtered
#         filters : list of str
#             List of filters to be applied
#
#         Returns
#         -------
#         oemols : list of OEMol
#             List of molecules after filtering (originals are used)
#         """
#         for filter in filters:
#             oemols = self.apply_filter(oemols, filter_name)
#         return oemols
#
#     def create_oemols(self, molecules):
#         """
#         Create new list of OEMols from source molecules.
#
#         Parameters
#         ----------
#         molecules : list of OEMols, SMILES, or QCScheme molecules
#             Molecules to be processed
#
#         Returns
#         -------
#         oemols : list of OEMols
#             List of OEMol (copies will be made if input is OEMols)
#         """
#         # Render input molecules to oemols
#         from openeye import oechem
#         oemols = list()
#         for molecule in molecules:
#             if isinstance(molecules, oechem.OEMol) or isinstance(molecules, oechem.OEGraphMol):
#                 oemol = oechem.OEMol(molecule)
#                 #oechem.OEFindRingAtomsAndBonds(oemol)
#                 #oechem.OEAssignAromaticFlags(oemol, oechem.OEAroModel_OpenEye)
#                 oemols.append(oemol)
#             elif type(molecule) == 'str':
#                 oemol = oechem.OEMol()
#                 oechem.OEParseSmiles(molecule, smiles)
#                 oemols.append(oemol)
#             elif type(molecule) == dict:
#                 # QCArchive JSON
#                 oemol = cmiles.utils.load_molecule(molecule)
#                 oemols.append(oemol)
#
#         return oemols
#
#     def expand_states(self, oemols):
#         """
#         Expand tautomers and stereochemistry according to class settings
#
#         Parameters
#         ----------
#         oemols : list of OEMols
#             List of molecules to expand
#
#         Returns
#         -------
#         oemols : list of OEMols
#             List of OEMol (copies will be made)
#         """
#         new_oemols = list()
#         for oemol in oemols:
#             enumerated_oemols = fragmenter.states.enumerate_states(oemol, tautomers=self.enumerate_tautomers, stereoisomers=self.enumerate_stereochemistry, return_mols=True)
#             new_oemols.extend(enumerated_oemols)
#         print('Generated {len(new_oemols)} molecules by enumerating states (tautomers: {self.enumerate_tautomers, stereochemistry: {self.enumerate_stereochemistry})')
#         return new_oemols
#
#     def fragment(self, oemols):
#         """
#         Fragment the provided molecules with Wiberg bond order based fragmenter
#
#         .. todo ::
#
#         * Use multiprocessing to parallelize?
#         * Memoize molecules?
#
#         Parameters
#         ----------
#         oemols : list of OEMols
#             List of molecules to fragment
#
#         Returns
#         -------
#         oemols : list of OEMols
#             Fragments of the original molecules (new molecules will be created)
#         """
#         new_oemols = list()
#         for oemol in oemols:
#             f = fragmenter.fragment.WBOFragmenter(mol)
#             f.fragment()
#             for bond in f.fragments:
#                 new_oemols.append(f.fragments[bond])
#         print('Generated {len(new_oemols)} fragments')
#         return new_oemols
#
#     def deduplicate(self, oemols):
#         """
#         Eliminate duplicate molecules from the provided list, returning unique molecules
#
#         Uniqueness is judged by equivalence of canonical isomeric SMILES
#
#         Parameters
#         ----------
#         oemols : list of OEMols
#             List of molecules to expand
#
#         Returns
#         -------
#         oemols : list of OEMols
#             List of unique OEMols (originals will be used)
#         """
#         from openeye import oechem
#         smiles_set = set()
#         new_oemols = list()
#         for oemol in oemols:
#             smiles = oechem.OEMolToSmiles(oemol)
#             if smiles not in smiles_set:
#                 new_oemols.append(oemol)
#                 smiles_set.add(smiles)
#         print('{len(new_oemols)} remain after removing duplicates')
#         return new_oemols
#
#     def expand_conformers(self, molecules):
#         """
#         Expand conformers
#
#         Parameters
#         ----------
#         oemols : list of OEMols
#             List of molecules to expand
#
#         Returns
#         -------
#         oemols : list of OEMols
#             List of multiconformer OEMol (copies will be created)
#         """
#         nconformers = 0
#         new_oemols = list()
#         for oemol in oemols:
#             try:
#                 # Omega fails for some molecules.
#                 expanded_oemol = fragmenter.chemi.generate_conformers(oemol, max_confs=self.max_conformers)
#                 nconformers += expanded_oemol.NumConfs()
#                 new_oemols.append(expanded_oemol)
#             except RuntimeError:
#                 from openeye import oechem
#                 smiles = oechem.OEMolToSmiles(oemol)
#                 logging.info('Omega failed to generate conformers for {}'.format(smiles))
#                 continue
#
#         oemols = new_oemols
#         print('Generated {nconformers} in in total by enumerating conformers (max_conformers: {self.max_conformers})')
#
#     def process_molecules(self, oemols):
#         """
#         Process molecules by enumeration, fragmentation, deduplication, filtering, and conformer expansion.
#
#         Molecule and atom ordering is preserved.
#
#         Processing proceeds in this ares:
#
#         * Apply input filters
#         * Expand tautomers and stereoisomers
#         * Fragment molecules, if required
#         * De-duplicate fragments
#         * Apply submit filters
#         * Expand conformers
#
#         Parameters
#         ----------
#         oemols : list of OEMols
#             Molecules to be processed for enumeration, fragmentation, deduplication, filtering, and conformer expansion.
#
#         Returns
#         -------
#         oemols : list of multiconformer OEMols
#             List of multiconformer OEMols following enumeration, fragmentation, and filtering
#         """
#         # Apply input filters
#         oemols = apply_filters(oemols, self.input_filters)
#
#         # Expand tautomers and stereoisomers
#         oemols = expand_states(oemols)
#
#         # Fragment if requested
#         if self.fragment:
#             oemols = fragment(oemols)
#
#         # De-duplicate molecules
#         oemols = deduplicate(oemols)
#
#         # Apply submission filters
#         oemols = apply_filters(oemols, self.submit_filters)
#
#         # Next, expand conformers
#         oemols = expand_conformers(oemols)
#
#         return oemols
#
#     def create_dataset(self, name, molecules):
#         """
#         Create a dataset via enumeration, fragmentation, deduplication, filtering, and conformer expansion.
#
#         Parameters
#         ----------
#         molecules : list of OEMols or SMILES
#             Molecules to be processed for enumeration, fragmentation, deduplication, filtering, and conformer expansion.
#
#         Returns
#         -------
#         dataset : QCFractalDataset or subclass of appropriate type
#             The dataset following enumeration, fragmentation, deduplication, filtering, and conformer expansion.
#
#         """
#         # Create OEMols from input molecules
#         input_oemols = create_oemols(molecules)
#         # Process via enumeration, fragmentation, deduplication, filtering, and conformer expansion
#         oemols = self.process(input_oemols)
#         # Create a dataset of the appropriate type
#         dataset = self.Dataset(name, description, input_oemols, oemols)
#         # Configure the dataset
#         dataset.compute_wbo = self.compute_wbo
#
#         return dataset
#
# class OptimizationDatasetFactory(QCFractalDatasetFactory):
#     """
#     Helper for preparing an OptimizationDataset
#
#     Attributes
#     ----------
#     compute_hessians : bool, default=False
#         If True, will compute Hessians after optimization
#
#     """
#     def __init__(self):
#         super().__init__(self)
#         self.compute_hessians = False
#         self.Dataset = OptimizationDataset
#
#     def create_dataset(self, name, description, molecules):
#         # Create the dataset
#         dataset = self.create_dataset(name, description, molecules)
#         # Configure the dataset
#         dataset.compute_hessians = self.compute_hessians
#         # Expand QCSchema dict for OptimizationDataset
#         optimization_input = list()
#         for oemol in dataset.oemols:
#             qcschema_dict = dataset.mol_to_qcschema_dict(oemol)
#             optimization_input.append(qcschema_dict)
#         dataset.optimization_input = optimization_input
#
# class TorsionDriveDatasetFactory(QCFractalDatasetFactory):
#     """
#     Helper for preparing a 1D TorsionDriveDataset
#
#     Attributes
#     ----------
#     max_conformers : int, optional, default=2
#         Number of conformers
#     grid_spacing : float, optional, default=15
#         Grid spacing (degrees) for 1D torsion drive
#
#     """
#     def __init__(self):
#         super().__init__(self)
#         self.max_conformers = 2 # override
#         self.grid_spacing = 15
#         self.Dataset = TorsionDriveDataset
#
#     def enumerate_torsions(oemol):
#         """Enumerate torsions that can be driven
#
#         Enumerates torsions that are
#         * Marked as rotatable
#         * Not in rings
#         * Prioritizes heaviest atoms involved in the torsion to be driven
#
#         Returns
#         -------
#         torsion_idx_list : list of (int,int,int,int)
#             List of rotatable torsions in molecule, if any
#             Expressed as (atom1, atom2, atom3, atom4) where atom* are zero-indexed atom indices within oemol
#         """
#         torsion_idx_list = list()
#
#         for oebond in oemol.GetBonds():
#             if not oebond.IsRotor(): continue
#             if oebond.IsInRing(): continue
#             # Find heaviest (i,j,k,l) tuple
#             max_mass = -1
#             optimal_torsion_idx = None
#             jatom = oebond.GetBgn()
#             katom = oebond.GetEnd()
#             for iatom in jatom.GetAtoms():
#                 for latom in katom.GetAtoms():
#                     torsion_idx = (iatom.GetIdx(), jatom.GetIdx(), katom.GetIdx(), latom.GetIdx())
#                     if len(set(torsion_idx)) != 4: continue
#                     mass = iatom.GetAtomicNum() + latom.GetAtomicNum()
#                     if mass > max_mass:
#                         max_mass = mass
#                         optimal_torsion_idx = torsion_idx
#             torsion_idx_list.append(optimal_torsion_idx)
#
#         return torsion_idx_list
#
#     def generate_selected_torsions(oemols):
#         """Construct QCSchema dict for TorsionDriveDataset
#
#         Parameters
#         ----------
#         oemols : list of multi-conformer OEMol
#             The molecules whose torsions are to be driven
#
#         Returns
#         -------
#         torsions_dict: dict
#             Dictionary for selected torsions, has this structure:
#             {
#                 index : {
#                     'atom_indices': [ (0,1,2,3) ],
#                     'initial_molecules': [ Molecule1a, Molecule1b, .. ],
#                     'attributes': {'canonical_explicit_hydrogen_smiles': .., 'canonical_isomeric_smiles': .., ..}
#                 },
#                 ..
#             }
#
#         Note
#         ----
#         The 'atom_indices' in return dict value is a list with only one item, because we select only 1-D torsion for now.
#
#         """
#         # generate torsion_dict
#         torsions_dict = dict()
#         ntorsions = 0
#         for index, oemol in enumerate(oemols):
#             qcschema_dict = mol_to_qcschema_dict(oemol)
#             torsion_idx_list = enumerate_torsions(oemol)
#
#             # Skip this if no torsions are found
#             if len(torsion_idx_list) == 0: continue
#
#             # Append torsions to drive
#             qcschema_dict['atom_indices'] = torsion_idx_list
#             ntorsions += len(torsion_idx_list)
#
#         logging.info(f'{ntorsions} torsions added')
#         return torsions_dict
#
#     def create_dataset(self, name, description, molecules):
#         # Create the dataset
#         dataset = self.create_dataset(name, description, molecules)
#         # Configure the dataset
#         dataset.compute_hessians = self.compute_hessians
#         # Expand QCSchema dict for TorsionDriveDataset
#         dataset.torsions = list()
#         optimization_input = list()
#         for oemol in dataset.oemols:
#             qcschema_dict = dataset.mol_to_qcschema_dict(oemol)
#             optimization_input.append(qcschema_dict)
