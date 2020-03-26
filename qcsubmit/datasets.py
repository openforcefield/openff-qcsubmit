from typing import Dict, List, Union, Optional

from qcsubmit.exceptions import UnsupportedFiletypeError

from pydantic import BaseModel
from openforcefield.topology import Molecule
import qcportal as ptl


class ComponentResult:
    """
    Class to contain molecules after the execution of a workflow component this automatically applies de-duplication to
    the molecules. For example if a molecule is already in the molecules list it will not be added but any conformers
    will be kept and transferred.

    If a molecule in the molecules list is then filtered it will be removed from the molecules list.
    """

    def __init__(self, component_name: str,  component_description: Dict, molecules: Optional[Union[List[Molecule], Molecule]] = None, input_file: Optional[str] = None):
        """Register the list of molecules to process."""

        # use a set to automatically remove duplicates
        self.molecules: List[Molecule] = []
        self.filtered: List[Molecule] = []
        self.component_name: str = component_name
        self.component_description: Dict = component_description

        assert molecules or input_file is None, 'Provide either a list of molecules or an input file name.'

        # if we have an input file load it
        if input_file is not None:
            molecules = Molecule.from_file(file_path=input_file, allow_undefined_stereo=True)

        # now lets process the molecules and add them to the class
        if molecules is not None:
            for molecule in molecules:
                self.add_molecule(molecule)

    def add_molecule(self, molecule: Molecule):
        """
        Add a molecule to the molecule list after checking that it is not present already. If it is de-duplicate the
        record and condense the conformers.
        """

        import numpy as np
        from simtk import unit

        if molecule in self.molecules:
            if molecule.n_conformers != 0:
                # we need to align the molecules and transfer the coords
                mol_id = self.molecules.index(molecule)
                # get the mapping
                isomorphic, mapping = Molecule.are_isomorphic(self.molecules[mol_id], molecule, return_atom_map=True)
                assert isomorphic is True
                for conformer in molecule.conformers:
                    new_conformer = np.zeros((molecule.n_atoms, 3))
                    for i in range(molecule.n_atoms):
                        new_conformer[i] = conformer[mapping[i]].value_in_unit(unit.angstrom)
                    self.molecules[mol_id].add_conformer(new_conformer * unit.angstrom)

            else:
                # molecule already in list and coords not present so just return
                return

        else:
            self.molecules.append(molecule)

    def filter_molecule(self, molecule: Molecule):
        """
        Filter out a molecule that has not passed this workflow component. If the molecule is already in the pass list
        remove it and ensure it is only in the filtered list.
        """

        try:
            self.molecules.remove(molecule)

        except ValueError:
            pass

        finally:
            if molecule not in self.filtered:
                self.filtered.append(molecule)
            else:
                return


class BasicResult(BaseModel):
    pass


class BasicDataSet(BaseModel):
    """
    The general qcfractal dataset class which contains all of the molecules and information about them prior to submission.
    The class is a simple holder of the dataset and information about it and can do simple checks on the data before submitting it such as ensuring that the molecules have cmiles information
    and a unique index to be identified by.

    Note:
        The molecules in this dataset are all expanded so that different conformers are unique submissions.
    """

    dataset_name: str = 'BasicDataSet'
    method: str = 'B3LYP-D3BJ'  # the default level of theory for openff
    basis: str = 'DZVP'  # the default basis for openff
    program: str = 'psi4'
    maxiter: int = 200
    driver: str = 'energy'
    scf_properties: List[str] = ['dipole', 'qudrupole', 'wiberg_lowdin_indices']
    spec_name: str = 'default'
    spec_description: str = 'Standard OpenFF optimization quantum chemistry specification.'
    client: str = 'public'
    priority: str = 'normal'
    tag: str = 'openff'
    dataset: Dict[str, Dict] = {}  # the molecules which are to be submitted
    filtered_molecules: Dict[str, Dict] = {}  # the molecules which have been filtered out

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @property
    def filtered(self) -> Molecule:
        """
        A generator for the molecules that have been filtered.

        Returns:
            An openforcefield.topology.Molecule representation of the molecule that has been filtered.

        Note:
            Modifying the molecule will have no effect on the data stored.
        """

        for component, data in self.filtered_molecules.items():
            for smiles in data['molecules']:
                yield Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    @property
    def n_filtered(self) -> int:
        """
        Calculate the total number of molecules filtered by the components used in this workflow.

        Returns:
            The total number of molecules filtered.
        """
        filterd = sum([len(data['molecules']) for data in self.filtered_molecules.values()])
        return filterd

    @property
    def n_records(self) -> int:
        """
        Return the amount of records that will be created on submission of the dataset.

        Returns:
            The amount of records that will be added to the collection.

        Note:
            The number returned will be different depending on the dataset used.

        Important:
            The number of records and molecules added is not always the same this can be checked using `n_molecules`.
        """

        n_records = sum([len(data['initial_molecules']) for data in self.dataset.values()])
        return n_records

    @property
    def n_molecules(self) -> int:
        """
        Calculate the total number of molecules which will be submitted as part of this dataset.

        Returns:
            The number of molecules in the dataset.

        Important:
            The number of molecule records submitted is not always the same as the amount of records created, this can
            also be checked using `n_records`. Here we give the number of unique molecules not excluding conformers.

        Note:
            The number returned will be different depending on the dataset submitted.
        """

        n_molecules = len(self.dataset)
        return n_molecules

    @property
    def molecules(self) -> Molecule:
        """
        A generator that creates the molecules one by one from the dataset note that editing the molecule will not
        edit it in the dataset.

        Returns:
            The instance of the molecule from the dataset.
        """

        from simtk import unit
        import numpy as np

        for index_name, molecule_data in self.dataset.items():
            # create the molecule from the cmiles data
            offmol = Molecule.from_mapped_smiles(mapped_smiles=molecule_data['attributes']['canonical_isomeric_explicit_hydrogen_mapped_smiles'], allow_undefined_stereo=True)
            offmol.name = index_name
            for conformer in molecule_data['initial_molecules']:
                geometry = unit.Quantity(np.array(conformer.geometry), unit=unit.bohr)
                offmol.add_conformer(geometry.in_units_of(unit.angstrom))
            yield offmol

    @property
    def n_components(self) -> int:
        """
        Return the amount of components that have been ran during generating the dataset.

        Returns:
            The number of components that were ran while generating the dataset.
        """

        n_filtered = len(self.filtered_molecules)
        return n_filtered

    @property
    def components(self) -> List[Dict]:
        """
        Gather the details of the components that were ran during the creation of this dataset.

        Returns:
            A list of dictionaries containing inormation about the components ran during the generation of the dataset.
        """

        components = [data['component_description'] for data in self.filtered_molecules.values()]
        return components

    def filter_molecules(self, molecules: Union[Molecule, List[Molecule]], component_description: Dict) -> None:
        """
        Filter a molecule or list of molecules by the component they failed.

        Parameters:
            molecules: A molecule or list of molecules to be filtered.
            component_description: The dict representation of the component that filtered this set of molecules.
        """
        if isinstance(molecules, Molecule):
            # make into a list
            molecules = [molecules]

        self.filtered_molecules[component_description['component_name']] = {'component_description': component_description,
                                                                            'molecules': [molecule.to_smiles(isomeric=True, explicit_hydrogens=True) for molecule in molecules]}

    def add_molecule(self, index: str, molecule: Molecule, cmiles: Dict[str, str]) -> None:
        """
        Add a molecule to the dataset under the given index with the passed cmiles.

        Parameters:
            index: The molecule index that was generated by the factory.
            molecule: The instance of the [openforcefield.topology.Molecule][molecule] which contains its conformer
                information.
            cmiles: The cmiles dictionary containing all of the relevant identifier tags for the molecule.

        Important:
            Each molecule in this basic dataset should have all of its conformers expanded out into separate entries.
            Thus here we take the general molecule index and increment it.
        """

        schema_mols = [molecule.to_qcschema(conformer=conformer) for conformer in range(molecule.n_conformers)]

        self.dataset[index] = {'attributes': cmiles,
                               'initial_molecules': schema_mols}

    def submit(self, await_result: bool = False) -> BasicResult:
        """
        Submit the dataset to the chosen qcarchive address and finish or wait for the results and return the
        corresponding result class.

        Parameters:
            await_result: If the user wants to wait for the calculation to finish before returning.

        Returns:
            Either `None` if we are not waiting for the results or a BasicResult instance with all of the completed
            calculations.
        """

        client = self._activate_client()
        # work out if we are extending a collection
        try:
            collection = client.get_collection('Dataset', self.dataset_name)
        except KeyError:
            collection = ptl.collections.Dataset(name=self.dataset_name, client=client,
                                                 default_driver=self.driver, default_program=self.program)

        # store the keyword set into the collection
        kw = ptl.models.KeywordSet(values=self.dict(include={'maxiter', 'scf_properties'}))
        collection.add_keywords(alias=self.spec_name, program=self.program, keyword=kw, default=True)

        # save the keywords
        collection.save()

        # now add the molecules to the database
        for index, data in self.dataset.items():
            for i, molecule in enumerate(data['initial_molecules']):
                name = index + f'_{i}'
                collection.add_entry(name=name, molecule=molecule, attributes=data['attributes'])

        # save the added entries
        collection.save()

        # submit the calculations
        response = collection.compute(method=self.method, basis=self.basis, keywords=self.spec_name,
                                      program=self.program, tag=self.tag, priority=self.priority)

        result = BasicResult()
        while await_result:

            pass

        return result

    def coverage_report(self, forcefields: Union[str, List[str]]) -> Dict:
        """
        Produce a coverage report of all of the parameters that are exercised by the molecules in the dataset.

        Parameters:
            forcefields: The name of the openforcefield force field which should be included in the coverage report.

        Returns:
            A dictionary for each of the force fields which break down which parameters are exercised by their
            parameter type.
        """

        results = {}

        pass

    def _activate_client(self) -> ptl.FractalClient:
        """
        Make the fractal client and connect to the requested instance.

        Returns:
            A qcportal.FractalClient instance.
        """

        if self.client == 'public':
            return ptl.FractalClient()
        else:
            return ptl.FractalClient.from_file(self.client)

    def molecules_to_file(self, file_name: str, file_type: str) -> None:
        """
        Write the molecules to the requested file type.

        Important:
            The supported file types are:

            - SMI
            - INCHI
            - INCKIKEY
        """

        file_writers = {'smi': self._molecules_to_smiles, 'inchi': self._molecules_to_inchi,
                        'inchikey': self._molecules_to_inchikey}

        try:
            # get the list of molecules
            molecules = file_writers[file_type]()

            with open(file_name, 'w') as output:
                for molecule in molecules:
                    output.write(f'{molecule}\n')
        except KeyError:
            raise UnsupportedFiletypeError(f'The requested file type {file_type} is not supported, supported types are'
                                           f'{file_writers.keys()}.')

    def _molecules_to_smiles(self) -> List[str]:
        """
        Create a list of molecules canonical smiles.
        """

        smiles = [data['attributes']['canonical_smiles'] for data in self.dataset.values()]
        return smiles
    
    def _molecules_to_inchi(self) -> List[str]:
        """
        Create a list of the molecules standard InChI.
        """

        inchi = [data['attributes']['standard_inchi'] for data in self.dataset.values()]
        return inchi

    def _molecules_to_inchikey(self) -> List[str]:
        """
        Create a list of the molecules standard InChIKey.
        """

        inchikey = [data['attributes']['inchi_key'] for data in self.dataset.values()]
        return inchikey


# class QCFractalDataset(object):
#     """
#     Abstract base class for QCFractal dataset.
#
#     Attributes
#     ----------
#     name : str
#         The dataset name
#     description : str
#         A detailed description of the dataset
#     input_oemols : list of OEMol
#         Original molecules prior to processing
#     oemols : list of multi-conformer OEMol
#         Unique molecules in the dataset
#
#     """
#
#     def __init__(self, name, description, input_oemols, oemols):
#         """
#         Create a new QCFractalDataset
#
#         Parameters
#         ----------
#         name : str
#             The dataset name
#         description : str
#             A detailed description of the dataset
#         input_oemols : list of OEMol
#             The original molecules provided to the generator for dataset construction.
#         oemols : list of multi-conformer OEMol
#             Molecules that survived after enumeration, fragmentation, deduplication, filtering, and conformer expansion.
#         """
#         self.name = name
#         self.description = description
#
#         # Store copies of molecules
#         from openeye import oechem
#         self.input_oemols = [ oechem.OEMol(oemol) for oemol in input_oemols ]
#         self.oemols = [ oechem.OEMol(oemol) for oemol in oemols ]
#
#     def mol_to_qcschema_dict(self, oemol):
#         """
#         Render a given OEMol as a QCSchema dict.
#
#         {
#             'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ],
#             'cmiles_identifiers' : ...
#         }
#
#         Returns
#         -------
#         qcschema_dict : dict
#             The dict containing all conformations as a list in qcschma_dict['initial_molecules']
#             and CMILES identifiers as qcschema_dict['cmiles_identifiers']
#         """
#         # Generate CMILES ids
#         import cmiles
#         try:
#             cmiles_ids = cmiles.get_molecule_ids(oemol)
#         except:
#             from openeye import oechem
#             smiles = oechem.OEMolToSmiles(oemol)
#             logging.info('cmiles failed to generate molecule ids {}'.format(smiles))
#             self.cmiles_failures.add(smiles)
#             #continue
#
#         # Extract mapped SMILES
#         mapped_smiles = cmiles_ids['canonical_isomeric_explicit_hydrogen_mapped_smiles']
#
#         # Create QCSchema for all conformers defined in the molecule
#         qcschema_molecules = [ cmiles.utils.mol_to_map_ordered_qcschema(conformer, mapped_smiles) for conformer in oemol.GetConfs() ]
#
#         # Create the QCSchema dict that includes both the specified molecules and CMILES ids
#         qcschema_dict = {
#             'initial_molecules': qcschema_molecules,
#             'cmiles_identifiers': cmiles_ids
#             }
#
#         return qcschema_dict
#
#     def render_molecules(self, filename, rows=10, cols=6):
#         """
#         Create a PDF showing all unique molecules in this dataset.
#
#         Parmeters
#         ---------
#         filename : str
#             Name of file to be written (ending in .pdf or .png)
#         rows : int, optional, default=10
#             Number of rows
#         cols : int, optional, default=6
#             Number of columns
#         """
#         from openeye import oedepict
#
#         # Configure display settings
#         itf = oechem.OEInterface()
#         PageByPage = True
#         suppress_h = True
#         ropts = oedepict.OEReportOptions(rows, cols)
#         ropts.SetHeaderHeight(25)
#         ropts.SetFooterHeight(25)
#         ropts.SetCellGap(2)
#         ropts.SetPageMargins(10)
#         report = oedepict.OEReport(ropts)
#         cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
#         opts = oedepict.OE2DMolDisplayOptions(cellwidth, cellheight, oedepict.OEScale_Default * 0.5)
#         opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
#         pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
#         opts.SetDefaultBondPen(pen)
#         oedepict.OESetup2DMolDisplayOptions(opts, itf)
#
#         # Render molecules
#         for oemol in self.oemols:
#             # Render molecule
#             cell = report.NewCell()
#             oemol_copy = oechem.OEMol(oemol)
#             oedepict.OEPrepareDepiction(oemol_copy, False, suppress_h)
#             disp = oedepict.OE2DMolDisplay(oemol_copy, opts)
#             oedepict.OERenderMolecule(cell, disp)
#
#         # Write the report
#         oedepict.OEWriteReport(filename, report)
#
#     def write_smiles(self, filename, mapped=False):
#         """
#         Write canonical isomeric SMILES entries for all unique molecules in this set.
#
#         Parameters
#         ----------
#         filename : str
#             Filename to which SMILES are to be written
#         mapped : bool, optional, default=False
#             If True, will write explicit hydrogen canonical isomeric tagged SMILES
#         """
#         if filename.endswith('.gz'):
#             import gzip
#             open_fun = gzip.open
#         else:
#             open_fun = open
#
#         import cmiles
#         with open_fun(filename, 'w') as outfile:
#             for oemol in self.oemols:
#                 smiles = cmiles.utils.mol_to_smiles(oemol, mapped=mapped)
#                 outfile.write(smiles + '\n')
#
#     def to_json(self, filename):
#         raise Exception('Abstract base class does not implement this method')
#
#     def submit(self,
#                  address: Union[str, 'FractalServer'] = 'api.qcarchive.molssi.org:443',
#                  username: Optional[str] = None,
#                  password: Optional[str] = None,
#                  verify: bool = True):
#         """
#         Submit the dataset to QCFractal server for computation.
#         """
#         raise Exception('Not implemented')
#
# class OptimizationDataset(QCFractalDataset):
#
#     def to_json(self, filename):
#         """
#         Render the OptimizationDataset to QCSchema JSON
#
#         [
#           {
#              'cmiles_identifiers' : ...,
#              'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ]
#           },
#
#           ...
#         ]
#
#         Parameters
#         ----------
#         filename : str
#             Filename (ending in .json or .json.gz) to be written
#
#         """
#         if filename.endswith('.json.gz'):
#             import gzip
#             open_fun = gzip.open
#         else:
#             open_fun = open
#
#         import json
#         with open_fun(filename, 'w') as outfile:
#             outfile.write(json.dumps(self.optimization_input, indent=2, sort_keys=True).encode('utf-8'))
#
# class TorsionDriveDataset(QCFractalDataset):
#
#     def to_json(self, filename):
#         """
#         Render the TorsionDriveDataset to QCSchema JSON
#
#         [
#           "index" : {
#              'atom_indices' : ...,
#              'cmiles_identifiers' : ...,
#              'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ]
#           },
#
#           ...
#         ]
#
#         Parameters
#         ----------
#         filename : str
#             Filename (ending in .json or .json.gz) to be written
#
#         """
#         if filename.endswith('.json.gz'):
#             import gzip
#             open_fun = gzip.open
#         else:
#             open_fun = open
#
#         import json
#         with open_fun(filename, 'w') as outfile:
#             outfile.write(json.dumps(self.qcschema_dict, indent=2, sort_keys=True).encode('utf-8'))
#
#     def render_molecules(self, filename, rows=10, cols=6):
#         """
#         Create a PDF showing all unique molecules in this dataset.
#
#         Parmeters
#         ---------
#         filename : str
#             Name of file to be written (ending in .pdf or .png)
#         rows : int, optional, default=10
#             Number of rows
#         cols : int, optional, default=6
#             Number of columns
#         """
#         from openeye import oedepict
#
#         # Configure display settings
#         itf = oechem.OEInterface()
#         PageByPage = True
#         suppress_h = True
#         ropts = oedepict.OEReportOptions(rows, cols)
#         ropts.SetHeaderHeight(25)
#         ropts.SetFooterHeight(25)
#         ropts.SetCellGap(2)
#         ropts.SetPageMargins(10)
#         report = oedepict.OEReport(ropts)
#         cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
#         opts = oedepict.OE2DMolDisplayOptions(cellwidth, cellheight, oedepict.OEScale_Default * 0.5)
#         opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
#         pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
#         opts.SetDefaultBondPen(pen)
#         oedepict.OESetup2DMolDisplayOptions(opts, itf)
#
#         # Render molecules
#         for json_molecule in json_molecules.values():
#             # Create oemol
#             import cmiles
#             oemol = cmiles.utils.load_molecule(json_molecule['initial_molecules'][0])
#
#             # Get atom indices
#             atom_indices = json_molecule['atom_indices'][0]
#
#             # Render molecule
#             cell = report.NewCell()
#             oemol_copy = oechem.OEMol(oemol)
#             oedepict.OEPrepareDepiction(oemol_copy, False, suppress_h)
#             disp = oedepict.OE2DMolDisplay(oemol_copy, opts)
#
#             # Highlight central torsion bond and atoms selected to be driven for torsion
#             class NoAtom(oechem.OEUnaryAtomPred):
#                 def __call__(self, atom):
#                     return False
#             class AtomInTorsion(oechem.OEUnaryAtomPred):
#                 def __call__(self, atom):
#                     return atom.GetIdx() in atom_indices
#             class NoBond(oechem.OEUnaryBondPred):
#                 def __call__(self, bond):
#                     return False
#             class BondInTorsion(oechem.OEUnaryBondPred):
#                 def __call__(self, bond):
#                     return (bond.GetBgn().GetIdx() in atom_indices) and (bond.GetEnd().GetIdx() in atom_indices)
#             class CentralBondInTorsion(oechem.OEUnaryBondPred):
#                 def __call__(self, bond):
#                     return (bond.GetBgn().GetIdx() in atom_indices[1:3]) and (bond.GetEnd().GetIdx() in atom_indices[1:3])
#
#             atoms = mol.GetAtoms(AtomInTorsion())
#             bonds = mol.GetBonds(NoBond())
#             abset = oechem.OEAtomBondSet(atoms, bonds)
#             oedepict.OEAddHighlighting(disp, oechem.OEColor(oechem.OEYellow), oedepict.OEHighlightStyle_BallAndStick, abset)
#
#             atoms = mol.GetAtoms(NoAtom())
#             bonds = mol.GetBonds(CentralBondInTorsion())
#             abset = oechem.OEAtomBondSet(atoms, bonds)
#             oedepict.OEAddHighlighting(disp, oechem.OEColor(oechem.OEOrange), oedepict.OEHighlightStyle_BallAndStick, abset)
#
#             oedepict.OERenderMolecule(cell, disp)
#
#         # Write the report
#         oedepict.OEWriteReport(filename, report)