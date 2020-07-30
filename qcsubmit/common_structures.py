"""
This file contains common starting structures which can be mixed into datasets, results and factories.
"""
import getpass
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import qcportal as ptl
from qcelemental.models.results import WavefunctionProtocolEnum
from pydantic import BaseModel, HttpUrl, constr, validator
from qcfractal.interface import FractalClient

from qcsubmit.exceptions import DatasetInputError


class DatasetConfig(BaseModel):
    """
    The basic configurations for all datasets.
    """

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = True
        validate_assignment: bool = True
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}


class ResultsConfig(BaseModel):
    """
    A basic config class for results structures.
    """

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}


class QCSpec(ResultsConfig):

    method: constr(strip_whitespace=True) = "B3LYP-D3BJ"
    basis: Optional[constr(strip_whitespace=True)] = "DZVP"
    program: str = "psi4"
    spec_name: str = "default"
    spec_description: str = "Standard OpenFF optimization quantum chemistry specification."
    store_wavefunction: WavefunctionProtocolEnum = WavefunctionProtocolEnum.none

    def dict(
        self,
        *,
        include: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
        exclude: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> 'DictStrAny':

        data = super().dict(exclude={"store_wavefunction"})
        data["store_wavefunction"] = self.store_wavefunction.value
        return data


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


class SingleTorsion(ResultsConfig):
    """
    A class used to mark torsions that will be driven for torsiondrive datasets.
    """

    torsion1: Tuple[int, int, int, int]
    scan_range1: Optional[Tuple[int, int]] = None

    _order_torsion1 = validator("torsion1", allow_reuse=True)(order_torsion)
    _order_scan_range1 = validator("scan_range1", allow_reuse=True)(order_scan_range)

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

    torsion2: Tuple[int, int, int, int]
    scan_range2: Optional[Tuple[int, int]] = None

    _order_torsion2 = validator("torsion2", allow_reuse=True)(order_torsion)
    _order_scan_range2 = validator("scan_range2", allow_reuse=True)(order_scan_range)

    @property
    def central_bond(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get the 4 integer tuple of the two central bonds."""

        central_bond = tuple(
            sorted(
                [tuple(sorted(self.torsion1[1:3])), tuple(sorted(self.torsion2[1:3])),]
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

    central_atom: int
    improper: Tuple[int, int, int, int]
    scan_range: Optional[Tuple[int, int]] = None

    _order_scan_range = validator("scan_range", allow_reuse=True)(order_scan_range)

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

    torsions: Dict[Tuple[int, int], SingleTorsion] = {}
    double_torsions: Dict[Tuple[Tuple[int, int], Tuple[int, int]], DoubleTorsion] = {}
    imporpers: Dict[int, ImproperTorsion] = {}

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
        all_torsions.extend(list(self.imporpers.values()))
        return all_torsions

    def add_torsion(
        self,
        torsion: Tuple[int, int, int, int],
        scan_range: Optional[Tuple[int, int]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add a single torsion to the torsion indexer if this central bond has not already been tagged.
        """
        torsion = SingleTorsion(torsion1=torsion, scan_range1=scan_range)

        if torsion.central_bond not in self.torsions:
            self.torsions[torsion.central_bond] = torsion
        elif overwrite:
            self.torsions[torsion.central_bond] = torsion

    def add_double_torsion(
        self,
        torsion1: Tuple[int, int, int, int],
        torsion2: Tuple[int, int, int, int],
        scan_range1: Optional[Tuple[int, int]] = None,
        scan_range2: Optional[Tuple[int, int]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add a double torsion to the indexer if this central bond combination has not been tagged.
        """

        double_torsion = DoubleTorsion(
            torsion1=torsion1,
            torsion2=torsion2,
            scan_range1=scan_range1,
            scan_range2=scan_range2,
        )

        if double_torsion.central_bond not in self.double_torsions:
            self.double_torsions[double_torsion.central_bond] = double_torsion
        elif overwrite:
            self.double_torsions[double_torsion.central_bond] = double_torsion

    def add_improper(
        self,
        central_atom: int,
        improper: Tuple[int, int, int, int],
        scan_range: Optional[Tuple[int, int]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add an improper torsion to the indexer if its central atom is not already covered.
        """

        improper_torsion = ImproperTorsion(
            central_atom=central_atom, improper=improper, scan_range=scan_range
        )

        if improper_torsion.central_atom not in self.imporpers:
            self.imporpers[improper_torsion.central_atom] = improper_torsion
        elif overwrite:
            self.imporpers[improper_torsion.central_atom] = improper_torsion

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

        if reorder_mapping is None:
            self.torsions.update(torsion_indexer.torsions)
            self.double_torsions.update(torsion_indexer.double_torsions)
            self.imporpers.update(torsion_indexer.imporpers)

        else:
            # we need to use the reorder_mapping to change the objects before adding them
            for torsion in torsion_indexer.torsions.values():
                new_torsion = self._reorder_torsion(torsion.torsion1, reorder_mapping)
                self.add_torsion(torsion=new_torsion, scan_range=torsion.scan_range1)

            for double_torsion in torsion_indexer.double_torsions.values():
                new_torsion1 = self._reorder_torsion(
                    double_torsion.torsion1, reorder_mapping
                )
                new_torsion2 = self._reorder_torsion(
                    double_torsion.torsion2, reorder_mapping
                )
                self.add_double_torsion(
                    torsion1=new_torsion1,
                    torsion2=new_torsion2,
                    scan_range1=double_torsion.scan_range1,
                    scan_range2=double_torsion.scan_range2,
                )

            for improper in torsion_indexer.imporpers.values():
                new_improper = self._reorder_torsion(improper.improper, reorder_mapping)
                new_central = reorder_mapping[improper.central_atom]
                self.add_improper(
                    new_central, new_improper, scan_range=improper.scan_range
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

        return len(self.imporpers)


class IndexCleaner:
    """
    This class offers the ability to clean a molecule index that already has a numeric tag useful for datasets and
    results.
    """

    @staticmethod
    def _clean_index(index: str) -> Tuple[str, int]:
        """
        Take an index and clean it by checking if it already has an enumerator in it return the core index and any
        numeric tags if no tag is found the tag is set to 0.

        Parameters:
            index: The index for the entry which should be checked, if no numeric tag can be found return 0.

        Returns:
            A tuple of the core index and the numeric tag it starts from.

        Note:
            This function allows the dataset to add more conformers to a molecule set so long as the index the molecule
            is stored under is a new index not in the database for example if 3 conformers for ethane exist then the
            new index should start from 'CC-3'.
        """
        # tags take the form '-no'
        match = re.search("-[0-9]+$", index)
        if match is not None:
            core = index[: match.span()[0]]
            # drop the -
            tag = int(match.group()[1:])
        else:
            core = index
            tag = 0

        return core, tag


class ClientHandler:
    """
    This mixin class offers the ability to handle activating qcportal Fractal client instances.
    """

    @staticmethod
    def _activate_client(client) -> ptl.FractalClient:
        """
        Make the fractal client and connect to the requested instance.

        Parameters:
            client: The name of the file containing the client information or the client instance.

        Returns:
            A qcportal.FractalClient instance.
        """

        if isinstance(client, ptl.FractalClient):
            return client
        elif isinstance(client, FractalClient):
            return client
        elif client == "public":
            return ptl.FractalClient()
        else:
            return ptl.FractalClient.from_file(client)


class Metadata(DatasetConfig):
    """
    A general metadata class which is required to be filled in before submitting a dataset to the qcarchive.
    """

    submitter: str = getpass.getuser()
    creation_date: date = datetime.today().date()
    collection_type: Optional[str] = None
    dataset_name: Optional[str] = None
    short_description: Optional[constr(min_length=8, regex="[a-zA-Z]")] = None
    long_description_url: Optional[HttpUrl] = None
    long_description: Optional[constr(min_length=8, regex="[a-zA-Z]")] = None
    elements: Set[str] = set()

    def validate_metadata(self, raise_errors: bool = False) -> Optional[List[str]]:
        """
        Before submitting this function should be called to highlight any incomplete fields.
        """

        empty_fields = []
        for field in self.__fields__:
            attr = getattr(self, field)
            if attr is None:
                empty_fields.append(field)

        if empty_fields and raise_errors:
            raise DatasetInputError(
                f"The metadata has the following incomplete fields {empty_fields}"
            )
        else:
            return empty_fields
