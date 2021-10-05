"""WorkFlow related utility classes and functions."""
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openff.toolkit.topology as off
import tqdm
from pydantic import Field, validator

try:
    from openmm import unit
except ImportError:
    from simtk import unit

from openff.qcsubmit.common_structures import DatasetConfig, ResultsConfig
from openff.qcsubmit.validators import check_environments


def order_torsion(torsion: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Order a torsion string to ensure the central bond is ordered.
    """
    if torsion[1:3] == tuple(sorted(torsion[1:3])):
        return torsion
    else:
        return torsion[::-1]


def order_scan_range(scan_range: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """
    Order the scan range.
    """
    if scan_range is not None:
        return tuple(sorted(scan_range))
    else:
        return None


class Scan1D(ResultsConfig):
    """
    A class to hold information on 1D scans to be computed.
    """

    smarts1: str = Field(
        ...,
        description="The numerically tagged SMARTs pattern used to select the torsion.",
    )
    scan_range1: Optional[Tuple[int, int]] = Field(
        None, description="The scan range that should be given to this torsion drive."
    )
    scan_increment: List[int] = Field(
        15, description="The angle in degrees between each grid point in the scan."
    )

    _check_smarts1 = validator("smarts1", allow_reuse=True)(check_environments)
    _order_scan_range1 = validator("scan_range1", allow_reuse=True)(order_scan_range)


class Scan2D(Scan1D):
    """
    A class to hold information on 2D scans to be computed.
    """

    smarts2: str = Field(
        ...,
        description="The second numerically tagged SMARTs pattern used to select a torsion.",
    )
    scan_range2: Optional[Tuple[int, int]] = Field(
        None,
        description="The scan range which should be given to the second torsion drive.",
    )
    scan_increment: List[int] = [15, 15]

    _check_smarts2 = validator("smarts2", allow_reuse=True)(check_environments)
    _order_scan_range2 = validator("scan_range2", allow_reuse=True)(order_scan_range)


class ImproperScan(ResultsConfig):
    """
    A class to hold information on Improper scans to be computed.
    """

    smarts: str = Field(
        ...,
        description="The numerically tagged SMARTs pattern used to select the improper torsion.",
    )
    central_smarts: str = Field(
        ...,
        description="The numerically tagged SMARTSs pattern used to select the central"
        "of the improper torsion.",
    )
    scan_range: Optional[Tuple[int, int]] = Field(
        None, description="The scan range which should be used for the improper."
    )
    scan_increment: List[int] = Field(
        15, description="The angle in degrees between each grid point in the scan."
    )

    _chack_smarts = validator("smarts", "central_smarts", allow_reuse=True)(
        check_environments
    )
    _check_scan_range = validator("scan_range", allow_reuse=True)(order_scan_range)


class SingleTorsion(ResultsConfig):
    """
    A class used to mark torsions that will be driven for torsiondrive datasets.

    Note:
        This is only for 1D torsiondrives.
    """

    torsion1: Tuple[int, int, int, int] = Field(
        ..., description="The torsion which is to be driven."
    )
    scan_range1: Optional[Tuple[int, int]] = Field(
        None, description="The scan range used in the torsion drive"
    )
    scan_increment: List[int] = Field(
        [15], description="The value in degrees between each grid point in the scan."
    )
    symmetry_group1: Tuple[int, int] = Field(
        ...,
        description="The symmetry of the central atoms in the torsion used to deduplicate symmetrical torsions.",
    )

    _order_torsion1 = validator("torsion1", allow_reuse=True)(order_torsion)
    _order_scan_range1 = validator("scan_range1", allow_reuse=True)(order_scan_range)
    _order_symmetry_group1 = validator("symmetry_group1", allow_reuse=True)(
        order_scan_range
    )

    @property
    def central_bond(self) -> Tuple[int, int]:
        """Get the sorted index of the central bond."""

        return tuple(sorted(self.torsion1[1:3]))

    @property
    def get_dihedrals(self) -> List[Tuple[int, int, int, int]]:
        """
        Get the formatted representation of the dihedrals to scan over.
        """
        return [
            self.torsion1,
        ]

    @property
    def get_scan_range(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get the formatted representation of the dihedral scan ranges.
        """
        if self.scan_range1 is not None:
            return [
                self.scan_range1,
            ]
        else:
            return self.scan_range1

    @property
    def get_atom_map(self) -> Dict[int, int]:
        """
        Create an atom map which will tag the correct dihedral atoms.
        """
        return dict((atom, i) for i, atom in enumerate(self.torsion1))


class DoubleTorsion(SingleTorsion):
    """A class used to mark coupled torsions which should be scanned."""

    torsion2: Tuple[int, int, int, int] = Field(
        ...,
        description="The torsion tuple of the second dihedral to be drive at the same time as the first.",
    )
    scan_range2: Optional[Tuple[int, int]] = Field(
        None,
        description="The separate scan range that should be used for the second dihedral.",
    )
    scan_increment: List[int] = [15, 15]
    symmetry_group2: Tuple[int, int] = Field(
        ...,
        description="The symmetry group of the second torsion, used to deduplicate torsions.",
    )

    _order_torsion2 = validator("torsion2", allow_reuse=True)(order_torsion)
    _order_scan_range2 = validator("scan_range2", allow_reuse=True)(order_scan_range)
    _order_symmetry_group2 = validator("symmetry_group2", allow_reuse=True)(
        order_scan_range
    )

    @property
    def central_bond(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get the 4 integer tuple of the two central bonds."""

        central_bond = tuple(
            sorted(
                [
                    tuple(sorted(self.torsion1[1:3])),
                    tuple(sorted(self.torsion2[1:3])),
                ]
            )
        )
        return central_bond

    @property
    def get_dihedrals(self) -> List[Tuple[int, int, int, int]]:
        """
        Get the formatted representation of the dihedrals to scan over.
        """
        return [
            self.torsion1,
            self.torsion2,
        ]

    @property
    def get_scan_range(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get the formatted representation of the dihedral scan ranges.
        """
        if self.scan_range1 is not None and self.scan_range2 is not None:
            return [
                self.scan_range1,
                self.scan_range2,
            ]
        else:
            return None

    @property
    def get_atom_map(self) -> Dict[int, int]:
        """
        Create an atom map which will tag the correct dihedral atoms.
        """
        atom_map = {}
        i = 0
        for torsion in self.get_dihedrals:
            for atom in torsion:
                atom_map[atom] = i
                i += 1
        return atom_map


class ImproperTorsion(ResultsConfig):
    """
    A class to keep track of improper torsions being scanned.
    """

    central_atom: int = Field(
        ..., description="The index of the central atom of an improper torsion."
    )
    improper: Tuple[int, int, int, int] = Field(
        ..., description="The tuple of the atoms in the improper torsion."
    )
    scan_range: Optional[Tuple[int, int]] = Field(
        None,
        description="The scan range of the improper dihedral which should normally be limited.",
    )
    scan_increment: List[int] = Field(
        [15], description="The value in degrees between each grid point in the scan."
    )
    symmetry_group: Tuple[int, int, int, int] = Field(
        ...,
        description="The symmetry group of the improper used to deduplicate improper torsions.",
    )

    _order_scan_range = validator("scan_range", allow_reuse=True)(order_scan_range)

    @validator("symmetry_group")
    def _sort_symmetry(
        cls, symmetry_group: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Sort the symmetry group for easier comparison."""
        return tuple(sorted(symmetry_group))

    @property
    def get_dihedrals(self) -> List[Tuple[int, int, int, int]]:
        """
        Get the formatted representation of the dihedrals to scan over.
        """
        return [
            self.improper,
        ]

    @property
    def get_scan_range(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get the formatted representation of the dihedral scan ranges.
        """
        if self.scan_range is not None:
            return [
                self.scan_range,
            ]
        else:
            return self.scan_range

    @property
    def get_atom_map(self) -> Dict[int, int]:
        """
        Create an atom map which will tag the correct dihedral atoms.
        """
        return dict((atom, i) for i, atom in enumerate(self.improper))


class TorsionIndexer(DatasetConfig):
    """
    A class to keep track of the torsions highlighted for scanning, with methods for combining and deduplication.
    """

    torsions: Dict[Tuple[int, int], SingleTorsion] = Field(
        {},
        description="A dictionary of the torsions to be scanned grouped by the central bond in the torsion.",
    )
    double_torsions: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]], DoubleTorsion
    ] = Field(
        {},
        description="A dictionary of the 2D torsions to be scanned grouped by the sorted combination of the central bonds.",
    )
    impropers: Dict[int, ImproperTorsion] = Field(
        {},
        description="A dictionary of the improper torsions to be scanned grouped by the central atom in the torsion.",
    )

    @property
    def get_dihedrals(
        self,
    ) -> List[Union[SingleTorsion, DoubleTorsion, ImproperTorsion]]:
        """
        Return a list of all of the dihedrals tagged making it easy to loop over.
        """
        all_torsions = []
        all_torsions.extend(list(self.torsions.values()))
        all_torsions.extend(list(self.double_torsions.values()))
        all_torsions.extend(list(self.impropers.values()))
        return all_torsions

    @property
    def torsion_groups(self) -> List[Tuple[int, int]]:
        """Return a list of all of the currently covered torsion symmetry groups. Note this only includes central bond atoms."""
        return [torsion.symmetry_group1 for torsion in self.torsions.values()]

    @property
    def double_torsion_groups(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return a list of all of the currently covered pairs of torsion symmetry groups."""
        return [
            (double_torsion.symmetry_group1, double_torsion.symmetry_group2)
            for double_torsion in self.double_torsions.values()
        ]

    @property
    def improper_groups(self) -> List[Tuple[int, int, int, int]]:
        """Return a list of all of the currently covered improper torsion symmetry groups."""
        return [improper.symmetry_group for improper in self.impropers.values()]

    def add_torsion(
        self,
        torsion: Tuple[int, int, int, int],
        symmetry_group: Tuple[int, int],
        scan_range: Optional[Tuple[int, int]] = None,
        scan_increment: List[int] = [15],
        overwrite: bool = False,
    ) -> None:
        """
        Add a single torsion to the torsion indexer if this central bond has not already been tagged and the torsion
        is symmetry unique.
        """
        torsion = SingleTorsion(
            torsion1=torsion,
            scan_range1=scan_range,
            scan_increment=scan_increment,
            symmetry_group1=symmetry_group,
        )

        if (
            torsion.central_bond not in self.torsions
            and torsion.symmetry_group1 not in self.torsion_groups
        ):
            self.torsions[torsion.central_bond] = torsion
        elif overwrite:
            self.torsions[torsion.central_bond] = torsion

    def add_double_torsion(
        self,
        torsion1: Tuple[int, int, int, int],
        torsion2: Tuple[int, int, int, int],
        symmetry_group1: Tuple[int, int],
        symmetry_group2: Tuple[int, int],
        scan_range1: Optional[Tuple[int, int]] = None,
        scan_range2: Optional[Tuple[int, int]] = None,
        scan_increment: List[int] = [15, 15],
        overwrite: bool = False,
    ) -> None:
        """
        Add a double torsion to the indexer if this central bond combination has not been tagged and the torsion pair are
        symmetry unique.
        """

        double_torsion = DoubleTorsion(
            torsion1=torsion1,
            torsion2=torsion2,
            symmetry_group1=symmetry_group1,
            symmetry_group2=symmetry_group2,
            scan_range1=scan_range1,
            scan_range2=scan_range2,
            scan_increment=scan_increment,
        )

        if (
            double_torsion.central_bond not in self.double_torsions
            and (double_torsion.symmetry_group1, double_torsion.symmetry_group2)
            not in self.double_torsion_groups
        ):
            self.double_torsions[double_torsion.central_bond] = double_torsion
        elif overwrite:
            self.double_torsions[double_torsion.central_bond] = double_torsion

    def add_improper(
        self,
        central_atom: int,
        improper: Tuple[int, int, int, int],
        symmetry_group: Tuple[int, int, int, int],
        scan_range: Optional[Tuple[int, int]] = None,
        scan_increment: List[int] = [15],
        overwrite: bool = False,
    ) -> None:
        """
        Add an improper torsion to the indexer if its central atom is not already covered and the improper is symmetry
        unique.
        """

        improper_torsion = ImproperTorsion(
            central_atom=central_atom,
            symmetry_group=symmetry_group,
            improper=improper,
            scan_range=scan_range,
            scan_increment=scan_increment,
        )

        if (
            improper_torsion.central_atom not in self.impropers
            and improper_torsion.symmetry_group not in self.improper_groups
        ):
            self.impropers[improper_torsion.central_atom] = improper_torsion
        elif overwrite:
            self.impropers[improper_torsion.central_atom] = improper_torsion

    def update(
        self,
        torsion_indexer: "TorsionIndexer",
        reorder_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Update the current torsion indexer with another.

        Parameters:
            torsion_indexer: The other torsionindxer that should be used to update the current object.
            reorder_mapping: The mapping between the other and current molecule should the order need updating.
        """

        # we need to use the reorder_mapping to change the objects before adding them if required
        for torsion in torsion_indexer.torsions.values():
            self.add_torsion(
                torsion=self._reorder_torsion(torsion.torsion1, reorder_mapping)
                if reorder_mapping is not None
                else torsion.torsion1,
                scan_range=torsion.scan_range1,
                scan_increment=torsion.scan_increment,
                symmetry_group=torsion.symmetry_group1,
            )

        for double_torsion in torsion_indexer.double_torsions.values():
            self.add_double_torsion(
                torsion1=self._reorder_torsion(double_torsion.torsion1, reorder_mapping)
                if reorder_mapping is not None
                else double_torsion.torsion1,
                torsion2=self._reorder_torsion(double_torsion.torsion2, reorder_mapping)
                if reorder_mapping is not None
                else double_torsion.torsion2,
                scan_range1=double_torsion.scan_range1,
                scan_range2=double_torsion.scan_range2,
                scan_increment=double_torsion.scan_increment,
                symmetry_group1=double_torsion.symmetry_group1,
                symmetry_group2=double_torsion.symmetry_group2,
            )

        for improper in torsion_indexer.impropers.values():
            self.add_improper(
                central_atom=reorder_mapping[improper.central_atom]
                if reorder_mapping is not None
                else improper.central_atom,
                improper=self._reorder_torsion(improper.improper, reorder_mapping)
                if reorder_mapping is not None
                else improper.improper,
                scan_range=improper.scan_range,
                scan_increment=improper.scan_increment,
                symmetry_group=improper.symmetry_group,
            )

    @staticmethod
    def _reorder_torsion(
        torsion: Tuple[int, int, int, int], mapping: Dict[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Reorder the given torsion based on the mapping.

        Parameters:
            torsion: The other molecules torsion that should be remapped.
            mapping: The mapping between the other molecule and the current.
        """

        return tuple([mapping[index] for index in torsion])

    @property
    def n_torsions(self) -> int:
        """Return the number of torsions highlighted."""

        return len(self.torsions)

    @property
    def n_double_torsions(self) -> int:
        """
        Return the number of double torsions highlighted.
        """

        return len(self.double_torsions)

    @property
    def n_impropers(self) -> int:
        """
        Return the number of imporpers highlighted.
        """

        return len(self.impropers)


class ComponentResult:
    """
    Class to contain molecules after the execution of a workflow component this automatically applies de-duplication to
    the molecules. For example if a molecule is already in the molecules list it will not be added but any conformers
    will be kept and transferred.


    If a molecule in the molecules list is then filtered it will be removed from the molecules list.
    """

    def __init__(
        self,
        component_name: str,
        component_description: Dict[str, str],
        component_provenance: Dict[str, str],
        molecules: Optional[Union[List[off.Molecule], off.Molecule]] = None,
        input_file: Optional[str] = None,
        input_directory: Optional[str] = None,
        skip_unique_check: Optional[bool] = False,
        verbose: bool = True,
    ):
        """Register the list of molecules to process.

        Args:
            component_name:
                The name of the component that produced this result.
            component_description:
                The dictionary representation of the component which details the function and running parameters.
            component_provenance:
                The dictionary of the modules used and there version number when running the component.
            molecules:
                The list of molecules that have been possessed by a component and returned as a result.
            input_file:
                The name of the input file used to produce the result if not from a component.
            input_directory:
                The name of the input directory which contains input molecule files.
            verbose:
                If the timing information and progress bar should be shown while doing deduplication.
            skip_unique_check:
                Set to True if it is sure that all molecules will be unique in this result
        """

        self._molecules: Dict[str, off.Molecule] = {}
        self._filtered: Dict[str, off.Molecule] = {}
        self.component_name: str = component_name
        self.component_description: Dict = component_description
        self.component_provenance: Dict = component_provenance
        self.skip_unique_check: bool = skip_unique_check
        self._unit_conversion: Dict[str, unit.Unit] = {
            "nanometer": unit.nanometers,
            "nanometers": unit.nanometers,
            "angstrom": unit.angstrom,
            "angstroms": unit.angstrom,
            "bohr": unit.bohr,
            "bohrs": unit.bohr,
        }

        assert (
            molecules is None or input_file is None
        ), "Provide either a list of molecules or an input file name."

        # if we have an input file load it
        if input_file is not None:
            if "hdf5" in input_file:
                molecules = self._read_hdf5(input_file=input_file)
            else:
                molecules = off.Molecule.from_file(
                    file_path=input_file, allow_undefined_stereo=True
                )
            if not isinstance(molecules, list):
                molecules = [
                    molecules,
                ]

        if input_directory is not None:
            molecules = []
            for file in os.listdir(input_directory):
                # each file could have many molecules in it so combine
                mols = off.Molecule.from_file(
                    file_path=os.path.join(input_directory, file),
                    allow_undefined_stereo=True,
                )
                try:
                    molecules.extend(mols)
                except TypeError:
                    molecules.append(mols)

        # now lets process the molecules and add them to the class
        if molecules is not None:
            for molecule in tqdm.tqdm(
                molecules,
                total=len(molecules),
                ncols=80,
                desc="{:30s}".format("Deduplication"),
                disable=not verbose,
            ):
                self.add_molecule(molecule)

    def _read_hdf5(self, input_file: str) -> List[off.Molecule]:
        """
        Read a set of molecules and conformers from an hdf5 file in a specified format.
        """
        import h5py

        molecules = []

        f = h5py.File(input_file, "r")
        for name, entry in f.items():
            mapped_smiles = entry["smiles"][0].decode("utf-8")
            molecule: off.Molecule = off.Molecule.from_mapped_smiles(
                mapped_smiles, allow_undefined_stereo=True
            )
            molecule.name = name
            units = self._unit_conversion[entry["conformations"].attrs["units"].lower()]
            # now add the conformers
            for conformer in entry["conformations"]:
                molecule.add_conformer(coordinates=conformer * units)

            molecules.append(molecule)
        f.close()

        return molecules

    @property
    def molecules(self) -> List[off.Molecule]:
        """
        Get the list of molecules which can be iterated over.
        """
        return list(self._molecules.values())

    @property
    def filtered(self) -> List[off.Molecule]:
        """
        Get the list of molecule that have been filtered to iterate over.
        """
        return list(self._filtered.values())

    @property
    def n_molecules(self) -> int:
        """
        The number of molecules saved in the result.
        """

        return len(self._molecules)

    @property
    def n_conformers(self) -> int:
        """
        The number of conformers stored in the molecules.
        """

        conformers = sum(
            [molecule.n_conformers for molecule in self._molecules.values()]
        )
        return conformers

    @property
    def n_filtered(self) -> int:
        """
        The number of filtered molecules.
        """
        return len(self._filtered)

    def add_molecule(self, molecule: off.Molecule) -> bool:
        """
        Add a molecule to the molecule list after checking that it is not present already. If it is de-duplicate the
        record and condense the conformers and metadata.

        Args:
            molecule:
                The molecule and its conformers which we should try and add to the result.

        Returns:
            `True` if the molecule is already present and `False` if not.
        """
        # always strip the atom map as it is not preserved in a workflow
        if "atom_map" in molecule.properties:
            del molecule.properties["atom_map"]

        # make a unique molecule hash independent of atom order or conformers
        molecule_hash = molecule.to_inchikey(fixed_hydrogens=True)

        if not self.skip_unique_check and molecule_hash in self._molecules:
            # we need to align the molecules and transfer the coords and properties
            # get the mapping, drop some comparisons to match inchikey
            isomorphic, mapping = off.Molecule.are_isomorphic(
                molecule,
                self._molecules[molecule_hash],
                return_atom_map=True,
                formal_charge_matching=False,
                bond_order_matching=False,
            )
            assert isomorphic is True
            # transfer any torsion indexes for similar fragments
            if "dihedrals" in molecule.properties:
                # we need to transfer the properties; get the current molecule dihedrals indexer
                # if one is missing create a new one
                current_indexer = self._molecules[molecule_hash].properties.get(
                    "dihedrals", TorsionIndexer()
                )

                # update it with the new molecule info
                current_indexer.update(
                    torsion_indexer=molecule.properties["dihedrals"],
                    reorder_mapping=mapping,
                )

                # store it back
                self._molecules[molecule_hash].properties["dihedrals"] = current_indexer

            if molecule.n_conformers != 0:

                # transfer the coordinates
                for conformer in molecule.conformers:
                    new_conformer = np.zeros((molecule.n_atoms, 3))
                    for i in range(molecule.n_atoms):
                        new_conformer[i] = conformer[mapping[i]].value_in_unit(
                            unit.angstrom
                        )

                    new_conf = unit.Quantity(value=new_conformer, unit=unit.angstrom)

                    # check if the conformer is already on the molecule
                    for old_conformer in self._molecules[molecule_hash].conformers:
                        if old_conformer.tolist() == new_conf.tolist():
                            break
                    else:
                        self._molecules[molecule_hash].add_conformer(
                            new_conformer * unit.angstrom
                        )
            else:
                # molecule already in list and coords not present so just return
                return True

        else:
            if molecule.n_conformers == 0:
                # make sure this is a list to avoid errors
                molecule._conformers = []
            self._molecules[molecule_hash] = molecule
            return False

    def filter_molecule(self, molecule: off.Molecule):
        """
        Filter out a molecule that has not passed this workflow component. If the molecule is already in the pass list
        remove it and ensure it is only in the filtered list.

        Args:
            molecule:
                The molecule which should be filtered.
        """

        molecule_hash = molecule.to_inchikey(fixed_hydrogens=True)
        try:
            del self._molecules[molecule_hash]

        except KeyError:
            pass

        finally:
            if molecule not in self._filtered:
                self._filtered[molecule_hash] = molecule

    def __repr__(self):
        return f"ComponentResult(name={self.component_name}, molecules={self.n_molecules}, filtered={self.n_filtered})"

    def __str__(self):
        return f"<ComponentResult name='{self.component_name}' molecules='{self.n_molecules}' filtered='{self.n_filtered}'>"
