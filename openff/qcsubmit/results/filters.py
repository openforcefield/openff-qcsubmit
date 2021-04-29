import abc
import logging
from collections import defaultdict
from typing import List, Optional, Set, TypeVar, Union

import numpy
from openff.toolkit.topology import Molecule
from pydantic import BaseModel, Field, PrivateAttr, root_validator, validator
from qcelemental.molutil import guess_connectivity
from qcportal.models.records import RecordBase, RecordStatusEnum
from simtk import unit
from typing_extensions import Literal

from openff.qcsubmit.results.results import _BaseResult, _BaseResultCollection
from openff.qcsubmit.validators import check_allowed_elements

T = TypeVar("T", bound=_BaseResultCollection)

logger = logging.getLogger(__name__)


class ResultFilter(BaseModel, abc.ABC):
    """The base class for a filter which will retain selection of QC records based on
    a specific criterion.
    """

    @abc.abstractmethod
    def _apply(self, result_collection: "T") -> "T":
        """The internal implementation of the ``apply`` method which should apply this
        filter to a results collection and return a new collection containing only the
        retained entries.

        Notes:
            The ``result_collection`` passed to this function will be a copy and
            so can be modified in place if needed.

        Args:
            result_collection: The collection to apply the filter to.

        Returns:
            The collection containing only the retained entries.
        """
        raise NotImplementedError()

    def apply(self, result_collection: "T") -> "T":
        """Apply this filter to a results collection, returning a new collection
        containing only the retained entries.

        Args:
            result_collection: The collection to apply the filter to.

        Returns:
            The collection containing only the retained entries.
        """

        filtered_collection = self._apply(result_collection.copy(deep=True))

        filtered_collection.entries = {
            address: entries
            for address, entries in filtered_collection.entries.items()
            if len(entries) > 0
        }

        logger.info(
            f"{abs(filtered_collection.n_results - result_collection.n_results)} "
            f"results were removed after applying a {self.__class__.__name__} filter."
        )

        if "applied-filters" not in filtered_collection.provenance:
            filtered_collection.provenance["applied-filters"] = {}

        n_existing_filters = len(filtered_collection.provenance["applied-filters"])
        filter_name = f"{self.__class__.__name__}-{n_existing_filters}"

        filtered_collection.provenance["applied-filters"][filter_name] = {**self.dict()}

        return filtered_collection


class CMILESResultFilter(ResultFilter, abc.ABC):
    """The base class for a filter which will retain selection of QC records based
    solely on the CMILES ( / InChI key) associated with the record itself, and not
    the actual record.

    If the filter needs to access information from the QC record itself the
    ``ResultRecordFilter`` class should be used instead as it more efficiently
    retrieves the records and associated molecule objects from the QCFractal
    instances.
    """

    @abc.abstractmethod
    def _filter_function(self, result: "_BaseResult") -> bool:
        """A method which should return whether to retain a particular result based
        on some property of the result object.
        """
        raise NotImplementedError()

    def _apply(self, result_collection: "T") -> "T":

        result_collection.entries = {
            address: [*filter(self._filter_function, entries)]
            for address, entries in result_collection.entries.items()
        }

        return result_collection


class ResultRecordFilter(ResultFilter, abc.ABC):
    """The base class for filters which will operate on QC records and their
    corresponding molecules directly."""

    @abc.abstractmethod
    def _filter_function(
        self, result: "_BaseResult", record: RecordBase, molecule: Molecule
    ) -> bool:
        """A method which should return whether to retain a particular result based
        on some property of the associated QC record.
        """
        raise NotImplementedError()

    def _apply(self, result_collection: "T") -> "T":

        all_records_and_molecules = defaultdict(list)

        for record, molecule in result_collection.to_records():
            all_records_and_molecules[record.client.address].append((record, molecule))

        filtered_results = {}

        for address, entries in result_collection.entries.items():

            entries_by_id = {entry.record_id: entry for entry in entries}

            records_and_molecules = all_records_and_molecules[address]

            filtered_ids = [
                record.id
                for record, molecule in records_and_molecules
                if self._filter_function(entries_by_id[record.id], record, molecule)
            ]

            filtered_results[address] = [
                entry for entry in entries if entry.record_id in filtered_ids
            ]

        result_collection.entries = filtered_results

        return result_collection


class SMILESFilter(CMILESResultFilter):
    """A filter which will remove or retain records which were computed for molecules
    described by specific SMILES patterns.
    """

    _inchi_keys_to_include: Optional[Set[str]] = PrivateAttr(None)
    _inchi_keys_to_exclude: Optional[Set[str]] = PrivateAttr(None)

    smiles_to_include: Optional[List[str]] = Field(
        None,
        description="Only QC records computed for molecules whose SMILES representation "
        "appears in this list will be retained. This option is mutually exclusive with "
        "``smiles_to_exclude``.",
    )
    smiles_to_exclude: Optional[List[str]] = Field(
        None,
        description="Any QC records computed for molecules whose SMILES representation "
        "appears in this list will be discarded. This option is mutually exclusive with "
        "``smiles_to_include``.",
    )

    @root_validator
    def _validate_mutually_exclusive(cls, values):

        smiles_to_include = values.get("smiles_to_include")
        smiles_to_exclude = values.get("smiles_to_exclude")

        message = (
            "exactly one of `smiles_to_include` and `smiles_to_exclude` must be "
            "specified"
        )

        assert smiles_to_include is not None or smiles_to_exclude is not None, message
        assert smiles_to_include is None or smiles_to_exclude is None, message

        return values

    @staticmethod
    def _smiles_to_inchi_key(smiles: str) -> str:
        return Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_inchikey(
            fixed_hydrogens=False
        )

    def _filter_function(self, entry: "_BaseResult") -> bool:

        return (
            entry.inchi_key in self._inchi_keys_to_include
            if self._inchi_keys_to_include is not None
            else entry.inchi_key not in self._inchi_keys_to_exclude
        )

    def _apply(self, result_collection: "T") -> "T":

        self._inchi_keys_to_include = (
            None
            if self.smiles_to_include is None
            else {
                self._smiles_to_inchi_key(smiles) for smiles in self.smiles_to_include
            }
        )
        self._inchi_keys_to_exclude = (
            None
            if self.smiles_to_exclude is None
            else {
                self._smiles_to_inchi_key(smiles) for smiles in self.smiles_to_exclude
            }
        )

        return super(SMILESFilter, self)._apply(result_collection)


class SMARTSFilter(CMILESResultFilter):
    """A filter which will remove or retain records which were computed for molecules
    which match specific SMARTS patterns.
    """

    smarts_to_include: Optional[List[str]] = Field(
        None,
        description="Only QC records computed for molecules that match one or more of "
        "the SMARTS patterns in this list will be retained. This option is mutually "
        "exclusive with ``smarts_to_exclude``.",
    )
    smarts_to_exclude: Optional[List[str]] = Field(
        None,
        description="Any QC records computed for molecules that match one or more of "
        "the SMARTS patterns in this list will be discarded. This option is mutually "
        "exclusive with ``smarts_to_include``.",
    )

    @root_validator
    def _validate_mutually_exclusive(cls, values):

        smarts_to_include = values.get("smarts_to_include")
        smarts_to_exclude = values.get("smarts_to_exclude")

        message = (
            "exactly one of `smarts_to_include` and `smarts_to_exclude` must be "
            "specified"
        )

        assert smarts_to_include is not None or smarts_to_exclude is not None, message
        assert smarts_to_include is None or smarts_to_exclude is None, message

        return values

    def _filter_function(self, entry: "_BaseResult") -> bool:

        molecule: Molecule = Molecule.from_mapped_smiles(
            entry.cmiles, allow_undefined_stereo=True
        )

        if self.smarts_to_include is not None:

            return any(
                len(molecule.chemical_environment_matches(smarts)) > 0
                for smarts in self.smarts_to_include
            )

        return all(
            len(molecule.chemical_environment_matches(smarts)) == 0
            for smarts in self.smarts_to_exclude
        )


class HydrogenBondFilter(ResultRecordFilter):
    """A filter which will remove or retain records which were computed for molecules
    which match specific SMARTS patterns.

    Notes:
        * For ``BasicResultCollection`` objects the single conformer associated with
          each result record will be checked for hydrogen bonds.
        * For ``OptimizationResultCollection`` objects the minimum energy conformer
          associated with each optimization record will be checked for hydrogen bonds.
        * For ``TorsionDriveResultCollection`` objects the minimum energy conformer
          at each grid angle will be checked for hydrogen bonds.
    """

    method: Literal["baker-hubbard"] = Field(
        "baker-hubbard", description="The method to use to detect any hydrogen bonds."
    )

    def _filter_function(
        self, result: "_BaseResult", record: RecordBase, molecule: Molecule
    ) -> bool:

        import mdtraj

        conformers = numpy.array(
            [
                conformer.value_in_unit(unit.nanometers).tolist()
                for conformer in molecule.conformers
            ]
        )

        mdtraj_topology = mdtraj.Topology.from_openmm(
            molecule.to_topology().to_openmm()
        )
        mdtraj_trajectory = mdtraj.Trajectory(
            conformers * unit.nanometers, mdtraj_topology
        )

        if self.method == "baker-hubbard":
            h_bonds = mdtraj.baker_hubbard(mdtraj_trajectory, freq=0.0, periodic=False)
        else:
            raise NotImplementedError()

        return len(h_bonds) == 0


class ConnectivityFilter(ResultRecordFilter):
    """A filter which will remove records whose corresponding molecules changed their
    connectivity during the computation, e.g. a proton transfer occurred.

    The connectivity will be percived from the 'final' conformer (see the Notes section)
    using the ``qcelemental.molutil.guess_connectivity`` function.

    Notes:
        * For ``BasicResultCollection`` objects no filtering will occur.
        * For ``OptimizationResultCollection`` objects the molecules final connectivity
          will be perceived using the minimum energy conformer.
        * For ``TorsionDriveResultCollection`` objects the connectivty will be
          will be perceived using the minimum energy conformer conformer at each grid
          angle.
    """

    tolerance: float = Field(
        1.2, description="Tunes the covalent radii metric safety factor."
    )

    def _filter_function(
        self, result: "_BaseResult", record: RecordBase, molecule: Molecule
    ) -> bool:

        qc_molecules = [
            molecule.to_qcschema(conformer=i) for i in range(molecule.n_conformers)
        ]

        expected_connectivity = {
            tuple(sorted([bond.atom1_index, bond.atom2_index]))
            for bond in molecule.bonds
        }

        for qc_molecule in qc_molecules:

            actual_connectivity = {
                tuple(sorted(connection))
                for connection in guess_connectivity(
                    qc_molecule.symbols, qc_molecule.geometry, self.tolerance
                )
            }

            if actual_connectivity == expected_connectivity:
                continue

            return False

        return True


class RecordStatusFilter(ResultRecordFilter):
    """A filter which will only retain records if their status matches a specified
    value.
    """

    status: RecordStatusEnum = Field(
        RecordStatusEnum.complete,
        description="Records whose status match this value will be retained.",
    )

    def _filter_function(
        self, result: "_BaseResult", record: RecordBase, molecule: Molecule
    ) -> bool:

        return record.status.value.upper() == self.status.value.upper()


class ChargeFilter(CMILESResultFilter):
    """A filter which will only retain records if their formal charge matches allowed values or is not in the
    exclude list."""

    charges_to_include: Optional[List[int]] = Field(
        None,
        description="Only molecules with a net formal charge in this list will be kept. "
        "This option is mutually exclusive with ``charges_to_exclude``.",
    )

    charges_to_exclude: Optional[List[int]] = Field(
        None,
        description="Any molecules with a net formal charge which matches any of these values will be removed. "
        "This option is mutually exclusive with ``charges_to_include``.",
    )

    @root_validator
    def _validate_mutually_exclusive(cls, values):
        charges_to_include = values.get("charges_to_include")
        charges_to_exclude = values.get("charges_to_exclude")

        message = (
            "exactly one of `charges_to_include` and `charges_to_exclude` must be "
            "specified"
        )

        assert charges_to_include is not None or charges_to_exclude is not None, message
        assert charges_to_include is None or charges_to_exclude is None, message

        return values

    def _filter_function(self, entry: "_BaseResult") -> bool:

        molecule: Molecule = Molecule.from_mapped_smiles(
            entry.cmiles, allow_undefined_stereo=True
        )
        total_charge = molecule.total_charge.value_in_unit(unit.elementary_charge)

        if self.charges_to_include is not None:

            return total_charge in self.charges_to_include

        return total_charge not in self.charges_to_exclude


class ElementFilter(CMILESResultFilter):
    """A filter which will only retain records that contain the requested elements."""

    _allowed_atomic_numbers: Optional[Set[int]] = PrivateAttr(None)

    allowed_elements: List[Union[int, str]] = Field(
        ...,
        description="The list of allowed elements as symbols or atomic number ints.",
    )

    _check_elements = validator("allowed_elements", each_item=True, allow_reuse=True)(
        check_allowed_elements
    )

    def _filter_function(self, entry: "_BaseResult") -> bool:
        molecule: Molecule = Molecule.from_mapped_smiles(
            entry.cmiles, allow_undefined_stereo=True
        )
        # get a set of atomic numbers
        mol_atoms = set([atom.atomic_number for atom in molecule.atoms])
        # get the difference between mol atoms and allowed atoms
        return not bool(mol_atoms.difference(self._allowed_atomic_numbers))

    def _apply(self, result_collection: "T") -> "T":
        from simtk.openmm.app import Element

        self._allowed_atomic_numbers = set(
            [
                Element.getBySymbol(ele).atomic_number if isinstance(ele, str) else ele
                for ele in self.allowed_elements
            ]
        )

        return super(ElementFilter, self)._apply(result_collection)
