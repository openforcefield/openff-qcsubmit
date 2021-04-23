import abc
import logging
from typing import TYPE_CHECKING, List, Optional, TypeVar

from openff.toolkit.topology import Molecule
from pydantic import BaseModel, Field, root_validator

if TYPE_CHECKING:

    from openff.qcsubmit.results.results import _BaseResult, _BaseResultCollection

    T = TypeVar("T", bound=_BaseResultCollection)

logger = logging.getLogger(__name__)


class ResultFilter(BaseModel, abc.ABC):
    """The base class for a filter which will retain selection of QC records based on
    a specific criterion.
    """

    @abc.abstractmethod
    def _apply(self, result_collection: "T") -> "T":
        """The internal implementation of thr ``apply`` method which should apply this
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


class SMILESFilter(ResultFilter):
    """A filter which will remove or retain records which were computed for molecules
    described by specific SMILES patterns.
    """

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
        return Molecule.from_smiles(smiles).to_inchikey(fixed_hydrogens=False)

    def _apply(self, result_collection: "T") -> "T":

        keys_to_include = (
            None
            if self.smiles_to_include is None
            else {
                self._smiles_to_inchi_key(smiles) for smiles in self.smiles_to_include
            }
        )
        keys_to_exclude = (
            None
            if self.smiles_to_exclude is None
            else {
                self._smiles_to_inchi_key(smiles) for smiles in self.smiles_to_exclude
            }
        )

        def _filter_function(entry: "_BaseResult") -> bool:

            return (
                entry.inchi_key in keys_to_include
                if keys_to_include is not None
                else entry.inchi_key not in keys_to_exclude
            )

        result_collection.entries = {
            address: [*filter(_filter_function, entries)]
            for address, entries in result_collection.entries.items()
        }

        return result_collection


class SMARTSFilter(ResultFilter):
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

    def _apply(self, result_collection: "T") -> "T":
        def _filter_function(entry: "_BaseResult") -> bool:

            molecule: Molecule = Molecule.from_mapped_smiles(entry.cmiles)

            if self.smarts_to_include is not None:

                return any(
                    len(molecule.chemical_environment_matches(smarts)) > 0
                    for smarts in self.smarts_to_include
                )

            return all(
                len(molecule.chemical_environment_matches(smarts)) == 0
                for smarts in self.smarts_to_exclude
            )

        result_collection.entries = {
            address: [*filter(_filter_function, entries)]
            for address, entries in result_collection.entries.items()
        }

        return result_collection
