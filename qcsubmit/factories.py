import fragmenter
import cmiles
import openeye

import warnings
import logging

"""
Tools for aiding the construction and submission of QCFractal datasets.

"""

class QCFractalDatasetFactory(object):
    """
    Base class with helper functions for creating QCFractal submission datasets.

    Objects from this class should never be created on their own.
    Use their subclasses instead.

    Attributes
    ----------
    fragment : bool, default=False
        If True, will fragment the molecule prior to submission
    enumerate_stereochemistry : bool, default=True
        If True, will enumerate unspecified stereochemistry
    enumerate_tautomers : bool, default=False
        If True, will enumerate tautomers
    max_conformers : int, default=20
        Maximum number of conformers generated per fragment
    input_filters : list, default=[]
        Filters to be applied to input molecules
    submit_filters : list, default=[]
        Filters to be applied to molecules prior to submission
    compute_wbo : bool, default=False
        If True, compute Wiberg-Lowdin bond orders

    """
    def __init__(self):
        """Base class constructor for generating QCFractal datasets.
        """
        self.fragment = False
        self.enumerate_stereochemistry = True
        self.enumerate_tautomers = False
        self.max_conformers = 20
        self.input_filters = list()
        self.submit_filters = list()
        self.compute_wbo = False

    def apply_filter(self, oemols, filter_name):
        """
        Apply the specified filter to the provided molecules

        Available Filters
        -----------------
        * OpenEye filters: 'BlockBuster', 'Drug', 'Fragment', 'Lead', 'PAINS'
          https://docs.eyesopen.com/toolkits/python/molproptk/filter_theory.html

        .. todo ::

           * Add QM filters to ensure computational tractability

        Parameters
        ----------
        oemols : list of OEMol
            The molecules to be filtered
        filter : str
            Filter to be applied

        Returns
        -------
        oemols : list of OEMol
            List of molecules after filtering (originals are used)
        """
        if filter_name in ['Blockbuster', 'Drug', 'Fragment', 'Lead', 'PAINS']:
            # Apply OpenEye Filters
            from openeye import oemolprop
            filter = oemolprop.OEFilter(getattr(oemolprop, 'OEFilterType_' + filter_name)
        else:
            raise Exception(f'Filter type {filter_name} unknown')

        new_oemols = list()
        for oemol in oemols:
            if filter(oemol):
                new_oemols.append(oemol)
        print('Retained {len(new_oemols)} molecules after application of {filter_name} filter')
        return new_oemols

    def apply_filters(self, oemols, filters):
        """
        Apply filters to the provided molecules.

        Parameters
        ----------
        oemols : list of OEMol
            The molecules to be filtered
        filters : list of str
            List of filters to be applied

        Returns
        -------
        oemols : list of OEMol
            List of molecules after filtering (originals are used)
        """
        for filter in filters:
            oemols = self.apply_filter(oemols, filter_name)
        return oemols

    def create_oemols(self, molecules):
        """
        Create new list of OEMols from source molecules.

        Parameters
        ----------
        molecules : list of OEMols, SMILES, or QCScheme molecules
            Molecules to be processed

        Returns
        -------
        oemols : list of OEMols
            List of OEMol (copies will be made if input is OEMols)
        """
        # Render input molecules to oemols
        from openeye import oechem
        oemols = list()
        for molecule in molecules:
            if isinstance(molecules, oechem.OEMol) or isinstance(molecules, oechem.OEGraphMol):
                oemol = oechem.OEMol(molecule)
                #oechem.OEFindRingAtomsAndBonds(oemol)
                #oechem.OEAssignAromaticFlags(oemol, oechem.OEAroModel_OpenEye)
                oemols.append(oemol)
            elif type(molecule) == 'str':
                oemol = oechem.OEMol()
                oechem.OEParseSmiles(molecule, smiles)
                oemols.append(oemol)
            elif type(molecule) == dict:
                # QCArchive JSON
                oemol = cmiles.utils.load_molecule(molecule)
                oemols.append(oemol)

        return oemols

    def expand_states(self, oemols):
        """
        Expand tautomers and stereochemistry according to class settings

        Parameters
        ----------
        oemols : list of OEMols
            List of molecules to expand

        Returns
        -------
        oemols : list of OEMols
            List of OEMol (copies will be made)
        """
        new_oemols = list()
        for oemol in oemols:
            enumerated_oemols = fragmenter.states.enumerate_states(oemol, tautomers=self.enumerate_tautomers, stereoisomers=self.enumerate_stereochemistry, return_mols=True)
            new_oemols.extend(enumerated_oemols)
        print('Generated {len(new_oemols)} molecules by enumerating states (tautomers: {self.enumerate_tautomers, stereochemistry: {self.enumerate_stereochemistry})')
        return new_oemols

    def fragment(self, oemols):
        """
        Fragment the provided molecules with Wiberg bond order based fragmenter

        .. todo ::

        * Use multiprocessing to parallelize?
        * Memoize molecules?

        Parameters
        ----------
        oemols : list of OEMols
            List of molecules to fragment

        Returns
        -------
        oemols : list of OEMols
            Fragments of the original molecules (new molecules will be created)
        """
        new_oemols = list()
        for oemol in oemols:
            f = fragmenter.fragment.WBOFragmenter(mol)
            f.fragment()
            for bond in f.fragments:
                new_oemols.append(f.fragments[bond])
        print('Generated {len(new_oemols)} fragments')
        return new_oemols

    def deduplicate(self, oemols):
        """
        Eliminate duplicate molecules from the provided list, returning unique molecules

        Uniqueness is judged by equivalence of canonical isomeric SMILES

        Parameters
        ----------
        oemols : list of OEMols
            List of molecules to expand

        Returns
        -------
        oemols : list of OEMols
            List of unique OEMols (originals will be used)
        """
        from openeye import oechem
        smiles_set = set()
        new_oemols = list()
        for oemol in oemols:
            smiles = oechem.OEMolToSmiles(oemol)
            if smiles not in smiles_set:
                new_oemols.append(oemol)
                smiles_set.add(smiles)
        print('{len(new_oemols)} remain after removing duplicates')
        return new_oemols

    def expand_conformers(self, molecules):
        """
        Expand conformers

        Parameters
        ----------
        oemols : list of OEMols
            List of molecules to expand

        Returns
        -------
        oemols : list of OEMols
            List of multiconformer OEMol (copies will be created)
        """
        nconformers = 0
        new_oemols = list()
        for oemol in oemols:
            try:
                # Omega fails for some molecules.
                expanded_oemol = fragmenter.chemi.generate_conformers(oemol, max_confs=self.max_conformers)
                nconformers += expanded_oemol.NumConfs()
                new_oemols.append(expanded_oemol)
            except RuntimeError:
                from openeye import oechem
                smiles = oechem.OEMolToSmiles(oemol)
                logging.info('Omega failed to generate conformers for {}'.format(smiles))
                continue

        oemols = new_oemols
        print('Generated {nconformers} in in total by enumerating conformers (max_conformers: {self.max_conformers})')

    def process_molecules(self, oemols):
        """
        Process molecules by enumeration, fragmentation, deduplication, filtering, and conformer expansion.

        Molecule and atom ordering is preserved.

        Processing proceeds in this ares:

        * Apply input filters
        * Expand tautomers and stereoisomers
        * Fragment molecules, if required
        * De-duplicate fragments
        * Apply submit filters
        * Expand conformers

        Parameters
        ----------
        oemols : list of OEMols
            Molecules to be processed for enumeration, fragmentation, deduplication, filtering, and conformer expansion.

        Returns
        -------
        oemols : list of multiconformer OEMols
            List of multiconformer OEMols following enumeration, fragmentation, and filtering
        """
        # Apply input filters
        oemols = apply_filters(oemols, self.input_filters)

        # Expand tautomers and stereoisomers
        oemols = expand_states(oemols)

        # Fragment if requested
        if self.fragment:
            oemols = fragment(oemols)

        # De-duplicate molecules
        oemols = deduplicate(oemols)

        # Apply submission filters
        oemols = apply_filters(oemols, self.submit_filters)

        # Next, expand conformers
        oemols = expand_conformers(oemols)

        return oemols

    def create_dataset(self, name, molecules):
        """
        Create a dataset via enumeration, fragmentation, deduplication, filtering, and conformer expansion.

        Parameters
        ----------
        molecules : list of OEMols or SMILES
            Molecules to be processed for enumeration, fragmentation, deduplication, filtering, and conformer expansion.

        Returns
        -------
        dataset : QCFractalDataset or subclass of appropriate type
            The dataset following enumeration, fragmentation, deduplication, filtering, and conformer expansion.

        """
        # Create OEMols from input molecules
        input_oemols = create_oemols(molecules)
        # Process via enumeration, fragmentation, deduplication, filtering, and conformer expansion
        oemols = self.process(input_oemols)
        # Create a dataset of the appropriate type
        dataset = self.Dataset(name, description, input_oemols, oemols)
        # Configure the dataset
        dataset.compute_wbo = self.compute_wbo

        return dataset

class OptimizationDatasetFactory(QCFractalDatasetFactory):
    """
    Helper for preparing an OptimizationDataset

    Attributes
    ----------
    compute_hessians : bool, default=False
        If True, will compute Hessians after optimization

    """
    def __init__(self):
        super().__init__(self)
        self.compute_hessians = False
        self.Dataset = OptimizationDataset

    def create_dataset(self, name, description, molecules):
        # Create the dataset
        dataset = self.create_dataset(name, description, molecules)
        # Configure the dataset
        dataset.compute_hessians = self.compute_hessians
        # Expand QCSchema dict for OptimizationDataset
        optimization_input = list()
        for oemol in dataset.oemols:
            qcschema_dict = dataset.mol_to_qcschema_dict(oemol)
            optimization_input.append(qcschema_dict)
        dataset.optimization_input = optimization_input

class TorsionDriveDatasetFactory(QCFractalDatasetFactory):
    """
    Helper for preparing a 1D TorsionDriveDataset

    Attributes
    ----------
    max_conformers : int, optional, default=2
        Number of conformers
    grid_spacing : float, optional, default=15
        Grid spacing (degrees) for 1D torsion drive

    """
    def __init__(self):
        super().__init__(self)
        self.max_conformers = 2 # override
        self.grid_spacing = 15
        self.Dataset = TorsionDriveDataset

    def enumerate_torsions(oemol):
        """Enumerate torsions that can be driven

        Enumerates torsions that are
        * Marked as rotatable
        * Not in rings
        * Prioritizes heaviest atoms involved in the torsion to be driven

        Returns
        -------
        torsion_idx_list : list of (int,int,int,int)
            List of rotatable torsions in molecule, if any
            Expressed as (atom1, atom2, atom3, atom4) where atom* are zero-indexed atom indices within oemol
        """
        torsion_idx_list = list()

        for oebond in oemol.GetBonds():
            if not oebond.IsRotor(): continue
            if oebond.IsInRing(): continue
            # Find heaviest (i,j,k,l) tuple
            max_mass = -1
            optimal_torsion_idx = None
            jatom = oebond.GetBgn()
            katom = oebond.GetEnd()
            for iatom in jatom.GetAtoms():
                for latom in katom.GetAtoms():
                    torsion_idx = (iatom.GetIdx(), jatom.GetIdx(), katom.GetIdx(), latom.GetIdx())
                    if len(set(torsion_idx)) != 4: continue
                    mass = iatom.GetAtomicNum() + latom.GetAtomicNum()
                    if mass > max_mass:
                        max_mass = mass
                        optimal_torsion_idx = torsion_idx
            torsion_idx_list.append(optimal_torsion_idx)

        return torsion_idx_list

    def generate_selected_torsions(oemols):
        """Construct QCSchema dict for TorsionDriveDataset

        Parameters
        ----------
        oemols : list of multi-conformer OEMol
            The molecules whose torsions are to be driven

        Returns
        -------
        torsions_dict: dict
            Dictionary for selected torsions, has this structure:
            {
                index : {
                    'atom_indices': [ (0,1,2,3) ],
                    'initial_molecules': [ Molecule1a, Molecule1b, .. ],
                    'attributes': {'canonical_explicit_hydrogen_smiles': .., 'canonical_isomeric_smiles': .., ..}
                },
                ..
            }

        Note
        ----
        The 'atom_indices' in return dict value is a list with only one item, because we select only 1-D torsion for now.

        """
        # generate torsion_dict
        torsions_dict = dict()
        ntorsions = 0
        for index, oemol in enumerate(oemols):
            qcschema_dict = mol_to_qcschema_dict(oemol)
            torsion_idx_list = enumerate_torsions(oemol)

            # Skip this if no torsions are found
            if len(torsion_idx_list) == 0: continue

            # Append torsions to drive
            qcschema_dict['atom_indices'] = torsion_idx_list
            ntorsions += len(torsion_idx_list)

        logging.info(f'{ntorsions} torsions added')
        return torsions_dict

    def create_dataset(self, name, description, molecules):
        # Create the dataset
        dataset = self.create_dataset(name, description, molecules)
        # Configure the dataset
        dataset.compute_hessians = self.compute_hessians
        # Expand QCSchema dict for TorsionDriveDataset
        dataset.torsions = list()
        optimization_input = list()
        for oemol in dataset.oemols:
            qcschema_dict = dataset.mol_to_qcschema_dict(oemol)
            optimization_input.append(qcschema_dict)
