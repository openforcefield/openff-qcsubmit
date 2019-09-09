import fragmenter
import cmiles
import openeye

import warnings
import logging
from typing import Any, Dict, List, Optional, Union

class QCFractalDataset(object):
    """
    Abstract base class for QCFractal dataset.

    Attributes
    ----------
    name : str
        The dataset name
    description : str
        A detailed description of the dataset
    input_oemols : list of OEMol
        Original molecules prior to processing
    oemols : list of multi-conformer OEMol
        Unique molecules in the dataset

    """

    def __init__(self, name, description, input_oemols, oemols):
        """
        Create a new QCFractalDataset

        Parameters
        ----------
        name : str
            The dataset name
        description : str
            A detailed description of the dataset
        input_oemols : list of OEMol
            The original molecules provided to the generator for dataset construction.
        oemols : list of multi-conformer OEMol
            Molecules that survived after enumeration, fragmentation, deduplication, filtering, and conformer expansion.
        """
        self.name = name
        self.description = description

        # Store copies of molecules
        from openeye import oechem
        self.input_oemols = [ oechem.OEMol(oemol) for oemol in input_oemols ]
        self.oemols = [ oechem.OEMol(oemol) for oemol in oemols ]

    def mol_to_qcschema_dict(self, oemol):
        """
        Render a given OEMol as a QCSchema dict.

        {
            'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ],
            'cmiles_identifiers' : ...
        }

        Returns
        -------
        qcschema_dict : dict
            The dict containing all conformations as a list in qcschma_dict['initial_molecules']
            and CMILES identifiers as qcschema_dict['cmiles_identifiers']
        """
        # Generate CMILES ids
        import cmiles
        try:
            cmiles_ids = cmiles.get_molecule_ids(oemol)
        except:
            from openeye import oechem
            smiles = oechem.OEMolToSmiles(oemol)
            logging.info('cmiles failed to generate molecule ids {}'.format(smiles))
            self.cmiles_failures.add(smiles)
            continue

        # Extract mapped SMILES
        mapped_smiles = cmiles_ids['canonical_isomeric_explicit_hydrogen_mapped_smiles']

        # Create QCSchema for all conformers defined in the molecule
        qcschema_molecules = [ cmiles.utils.mol_to_map_ordered_qcschema(conformer, mapped_smiles) for conformer in oemol.GetConfs() ]

        # Create the QCSchema dict that includes both the specified molecules and CMILES ids
        qcschema_dict = {
            'initial_molecules': qcschema_molecules,
            'cmiles_identifiers': cmiles_ids
            }

        return qcschema_dict

    def render_molecules(self, filename, rows=10, cols=6):
        """
        Create a PDF showing all unique molecules in this dataset.

        Parmeters
        ---------
        filename : str
            Name of file to be written (ending in .pdf or .png)
        rows : int, optional, default=10
            Number of rows
        cols : int, optional, default=6
            Number of columns
        """
        from openeye import oedepict

        # Configure display settings
        itf = oechem.OEInterface()
        PageByPage = True
        suppress_h = True
        ropts = oedepict.OEReportOptions(rows, cols)
        ropts.SetHeaderHeight(25)
        ropts.SetFooterHeight(25)
        ropts.SetCellGap(2)
        ropts.SetPageMargins(10)
        report = oedepict.OEReport(ropts)
        cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
        opts = oedepict.OE2DMolDisplayOptions(cellwidth, cellheight, oedepict.OEScale_Default * 0.5)
        opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
        pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
        opts.SetDefaultBondPen(pen)
        oedepict.OESetup2DMolDisplayOptions(opts, itf)

        # Render molecules
        for oemol in self.oemols:
            # Render molecule
            cell = report.NewCell()
            oemol_copy = oechem.OEMol(oemol)
            oedepict.OEPrepareDepiction(oemol_copy, False, suppress_h)
            disp = oedepict.OE2DMolDisplay(oemol_copy, opts)
            oedepict.OERenderMolecule(cell, disp)

        # Write the report
        oedepict.OEWriteReport(filename, report)

    def write_smiles(self, filename, mapped=False):
        """
        Write canonical isomeric SMILES entries for all unique molecules in this set.

        Parameters
        ----------
        filename : str
            Filename to which SMILES are to be written
        mapped : bool, optional, default=False
            If True, will write explicit hydrogen canonical isomeric tagged SMILES
        """
        if filename.endswith('.gz'):
            import gzip
            open_fun = gzip.open
        else:
            open_fun = open

        import cmiles
        with open_fun(filename, 'w') as outfile:
            for oemol in self.oemols:
                smiles = cmiles.utils.mol_to_smiles(oemol, mapped=mapped)
                outfile.write(smiles + '\n')

    def to_json(self, filename):
        raise Exception('Abstract base class does not implement this method')

    def submit(self,
                 address: Union[str, 'FractalServer'] = 'api.qcarchive.molssi.org:443',
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 verify: bool = True):
        """
        Submit the dataset to QCFractal server for computation.
        """
        raise Exception('Not implemented')

class OptimizationDataset(QCFractalDataset):

    def to_json(self, filename):
        """
        Render the OptimizationDataset to QCSchema JSON

        [
          {
             'cmiles_identifiers' : ...,
             'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ]
          },

          ...
        ]

        Parameters
        ----------
        filename : str
            Filename (ending in .json or .json.gz) to be written

        """
        if filename.endswith('.json.gz'):
            import gzip
            open_fun = gzip.open
        else:
            open_fun = open

        import json
        with open_fun(filename, 'w') as outfile:
            outfile.write(json.dumps(self.optimization_input, indent=2, sort_keys=True).encode('utf-8'))

class TorsionDriveDataset(QCFractalDataset):

    def to_json(self, filename):
        """
        Render the TorsionDriveDataset to QCSchema JSON

        [
          "index" : {
             'atom_indices' : ...,
             'cmiles_identifiers' : ...,
             'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ]
          },

          ...
        ]

        Parameters
        ----------
        filename : str
            Filename (ending in .json or .json.gz) to be written

        """
        if filename.endswith('.json.gz'):
            import gzip
            open_fun = gzip.open
        else:
            open_fun = open

        import json
        with open_fun(filename, 'w') as outfile:
            outfile.write(json.dumps(self.qcschema_dict, indent=2, sort_keys=True).encode('utf-8'))

    def render_molecules(self, filename, rows=10, cols=6):
        """
        Create a PDF showing all unique molecules in this dataset.

        Parmeters
        ---------
        filename : str
            Name of file to be written (ending in .pdf or .png)
        rows : int, optional, default=10
            Number of rows
        cols : int, optional, default=6
            Number of columns
        """
        from openeye import oedepict

        # Configure display settings
        itf = oechem.OEInterface()
        PageByPage = True
        suppress_h = True
        ropts = oedepict.OEReportOptions(rows, cols)
        ropts.SetHeaderHeight(25)
        ropts.SetFooterHeight(25)
        ropts.SetCellGap(2)
        ropts.SetPageMargins(10)
        report = oedepict.OEReport(ropts)
        cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
        opts = oedepict.OE2DMolDisplayOptions(cellwidth, cellheight, oedepict.OEScale_Default * 0.5)
        opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
        pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
        opts.SetDefaultBondPen(pen)
        oedepict.OESetup2DMolDisplayOptions(opts, itf)

        # Render molecules
        for json_molecule in json_molecules.values():
            # Create oemol
            import cmiles
            oemol = cmiles.utils.load_molecule(json_molecule['initial_molecules'][0])

            # Get atom indices
            atom_indices = json_molecule['atom_indices'][0]

            # Render molecule
            cell = report.NewCell()
            oemol_copy = oechem.OEMol(oemol)
            oedepict.OEPrepareDepiction(oemol_copy, False, suppress_h)
            disp = oedepict.OE2DMolDisplay(oemol_copy, opts)

            # Highlight central torsion bond and atoms selected to be driven for torsion
            class NoAtom(oechem.OEUnaryAtomPred):
                def __call__(self, atom):
                    return False
            class AtomInTorsion(oechem.OEUnaryAtomPred):
                def __call__(self, atom):
                    return atom.GetIdx() in atom_indices
            class NoBond(oechem.OEUnaryBondPred):
                def __call__(self, bond):
                    return False
            class BondInTorsion(oechem.OEUnaryBondPred):
                def __call__(self, bond):
                    return (bond.GetBgn().GetIdx() in atom_indices) and (bond.GetEnd().GetIdx() in atom_indices)
            class CentralBondInTorsion(oechem.OEUnaryBondPred):
                def __call__(self, bond):
                    return (bond.GetBgn().GetIdx() in atom_indices[1:3]) and (bond.GetEnd().GetIdx() in atom_indices[1:3])

            atoms = mol.GetAtoms(AtomInTorsion())
            bonds = mol.GetBonds(NoBond())
            abset = oechem.OEAtomBondSet(atoms, bonds)
            oedepict.OEAddHighlighting(disp, oechem.OEColor(oechem.OEYellow), oedepict.OEHighlightStyle_BallAndStick, abset)

            atoms = mol.GetAtoms(NoAtom())
            bonds = mol.GetBonds(CentralBondInTorsion())
            abset = oechem.OEAtomBondSet(atoms, bonds)
            oedepict.OEAddHighlighting(disp, oechem.OEColor(oechem.OEOrange), oedepict.OEHighlightStyle_BallAndStick, abset)

            oedepict.OERenderMolecule(cell, disp)

        # Write the report
        oedepict.OEWriteReport(filename, report)
