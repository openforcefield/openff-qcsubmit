"""
A module which contains convenience classes for referencing, retrieving and filtering
results from a QCFractal instance.
"""
from __future__ import annotations

import abc
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

try:
    from openmm import unit
except ImportError:
    from simtk import unit

import numpy
import qcportal
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from pydantic import BaseModel, Field, validator
from qcportal.dataset_models import BaseDataset as QCPDataset
from qcportal.optimization import OptimizationDataset, OptimizationRecord
from qcportal.record_models import BaseRecord, RecordStatusEnum
from qcportal.singlepoint import (
    SinglepointDataset,
    SinglepointDatasetNewEntry,
    SinglepointDriver,
    SinglepointRecord,
)
from qcportal.torsiondrive import TorsiondriveDataset, TorsiondriveRecord
from typing_extensions import Literal

from openff.qcsubmit.common_structures import Metadata, MoleculeAttributes, QCSpec
from openff.qcsubmit.datasets import BasicDataset
from openff.qcsubmit.exceptions import RecordTypeError
from openff.qcsubmit.results.caching import (
    batched_indices,
    cached_fractal_client,
    cached_query_basic_results,
    cached_query_molecules,
    cached_query_optimization_results,
    cached_query_torsion_drive_results,
)
from openff.qcsubmit.utils.smirnoff import smirnoff_coverage, smirnoff_torsion_coverage

if TYPE_CHECKING:
    from openff.qcsubmit.results.filters import ResultFilter

T = TypeVar("T")
S = TypeVar("S")


class _BaseResult(BaseModel, abc.ABC):
    """The base model for storing information about a QC record generated by
    QCFractal."""

    type: Literal["base"]

    record_id: int = Field(
        ...,
        description="The unique id assigned to the record referenced by this result.",
    )

    cmiles: str = Field(
        ...,
        description="The canonical, isomeric, explicit hydrogen, mapped SMILES "
        "representation of the molecule that this record was created for.",
    )
    inchi_key: str = Field(
        ...,
        description="The fixed hydrogen layer InChI key generated from the ``cmiles`` representation. This "
        "may be used as a hash for the molecule referenced by this record.",
    )

    @property
    def molecule(self) -> Molecule:
        """Returns an OpenFF molecule object created from this records
        CMILES which is in the correct order to align with the QCArchive records.
        """
        return Molecule.from_mapped_smiles(self.cmiles, allow_undefined_stereo=True)


class _BaseResultCollection(BaseModel, abc.ABC):
    """The base model for storing references to a collection of QC records generated by
    QCFractal which are not necessarily stored in the same datasets.
    """

    entries: Dict[str, List[_BaseResult]] = Field(
        ...,
        description="The entries stored in this collection in a dictionary of the form "
        "``collection.entries['qcfractal_address'] = [record_1, ..., record_N]``.",
    )

    provenance: Dict[str, Any] = Field(
        {},
        description="A dictionary which can contain provenance information about "
        "how and why this collection was curated.",
    )

    @validator("entries")
    def _validate_entries(cls, values):
        for client_address, entries in values.items():
            record_ids = {entry.record_id for entry in entries}
            assert len(entries) == len(
                record_ids
            ), f"duplicate entries were found for {client_address}"

        return values

    @property
    def n_results(self) -> int:
        """Returns the number of results in this collection."""
        return sum(len(entries) for entries in self.entries.values())

    @property
    def n_molecules(self) -> int:
        """Returns the number of unique molecules referenced by this collection."""
        return len(
            {entry.inchi_key for entries in self.entries.values() for entry in entries}
        )

    @classmethod
    @abc.abstractmethod
    def from_datasets(
        cls: T,
        datasets: Union[QCPDataset, Iterable[QCPDataset]],
        spec_name: str = "default",
    ) -> T:
        """Retrieve the COMPLETE record ids referenced by the specified datasets.

        Args:
            datasets: The datasets to retrieve the records from.
            spec_name: The name of the QC specification that the records to retrieve
                should have been computed using.

        Returns:
            A results collection object containing the record ids and a minimal amount
            of associated metadata such as the CMILES of the associated molecule.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_server(
        cls: T,
        client: qcportal.PortalClient,
        datasets: Union[str, Iterable[str]],
        spec_name: str = "default",
    ) -> T:
        """Retrieve (and deduplicate) the COMPLETE record ids referenced by the
        specified datasets.

        Args:
            client: The fractal client that should be used to interface with the running
                QCFractal instance which stores the datasets and their associated
                results records.
            datasets: The names of the datasets
            spec_name: The name of the QC specification that the records to retrieve
                should have been computed using.

        Returns:
            A results collection object containing the record ids and a minimal amount
            of associated metadata such as the CMILES of the associated molecule.
        """
        raise NotImplementedError()

    @staticmethod
    def _validate_record_types(
        records: List[SinglepointRecord], expected_type: Type[BaseRecord]
    ):
        """A helper method which raises a ``RecordTypeError`` if all records in the list
        are not of the specified type."""

        incorrect_types = [
            record for record in records if not isinstance(record, expected_type)
        ]

        if len(incorrect_types) > 0:
            incorrect_types_dict = defaultdict(set)

            for record in incorrect_types:
                incorrect_types_dict[record.__class__.__name__].add(record.id)

            raise RecordTypeError(
                f"The collection contained a records which were of the wrong type. "
                f"This collection should only store {expected_type.__name__} records."
                f"{dict(**incorrect_types_dict)}"
            )

    @abc.abstractmethod
    def to_records(self) -> List[Tuple[BaseRecord, Molecule]]:
        """Returns the native QCPortal record objects for each of the records referenced
        in this collection along with a corresponding OpenFF molecule object.
        """
        raise NotImplementedError()

    def filter(self, *filters: "ResultFilter"):
        """Filter this collection by applying a set of filters sequentially, returning
        a new collection containing only the retained entries.

        Notes:
            Information about any applied filters will be stored in the provenance
            dictionary in the 'applied-filters' field. Any existing information in
            this field will be overwritten here.

        Args:
            filters: The filters to apply, in the order to apply them in.

        Returns:
            The collection containing only the retained entries.

        Example::

            >>> filtered_collection = self.filter(
            >>>     RecordStatusFilter(status=RecordStatusEnum.complete),
            >>>     ConnectivityFilter(tolerance=1.2),
            >>>     UnperceivableStereoFilter(),
            >>>     ElementFilter(allowed_elements=elements),
            >>>     ConformerRMSDFilter(max_conformers=max_opt_conformers),
            >>> )
        """

        filtered_collection = self.copy(deep=True)

        for collection_filter in filters:
            filtered_collection = collection_filter.apply(filtered_collection)

        return filtered_collection

    def smirnoff_coverage(self, force_field: ForceField, verbose: bool = False):
        """Returns a summary of how many unique molecules within this collection
        would be assigned each of the parameters in a force field.

        Notes:
            * Parameters which would not be assigned to any molecules in the collection
              will not be included in the returned summary.

        Args:
            force_field: The force field containing the parameters to summarize.
            verbose: If true a progress bar will be shown on screen.

        Returns:
            A dictionary of the form ``coverage[handler_name][parameter_smirks] = count``
            which stores the number of unique molecules within this collection that
            would be assigned to each parameter.
        """

        # We filter by inchi key to make sure that we don't double count molecules
        # with different orderings.
        unique_molecules = set(
            {
                entry.molecule.to_smiles(isomeric=False, mapped=False): entry.molecule
                for entries in self.entries.values()
                for entry in entries
            }.values()
        )

        return smirnoff_coverage(unique_molecules, force_field, verbose)


# TODO - SinglepointResult?
class BasicResult(_BaseResult):
    """A class which stores a reference to, and allows the retrieval of, data from
    a single result record stored in a QCFractal instance."""

    type: Literal["basic"] = "basic"


# TODO - SinglepointResultCollection?
class BasicResultCollection(_BaseResultCollection):
    """A class which stores a reference to, and allows the retrieval of, data from
    a single result record stored in a QCFractal instance."""

    type: Literal["BasicResultCollection"] = "BasicResultCollection"

    entries: Dict[str, List[BasicResult]] = Field(
        ...,
        description="The entries stored in this collection in a dictionary of the form "
        "``collection.entries['qcfractal_address'] = [record_1, ..., record_N]``.",
    )

    @classmethod
    def from_datasets(
        cls,
        datasets: Union[SinglepointDataset, Iterable[SinglepointDataset]],
        spec_name: str = "default",
    ) -> BasicResultCollection:
        if isinstance(datasets, QCPDataset):
            datasets = [datasets]

        if not all(isinstance(dataset, SinglepointDataset) for dataset in datasets):
            raise TypeError(
                "A ``BasicResultCollection`` can only be created from ``SinglepointDataset`` objects."
            )

        result_records = defaultdict(dict)

        for dataset in datasets:
            client = dataset.client

            # Fetch all entries for use later. These get stored internally
            # in the dataset class
            dataset.fetch_entries()

            if spec_name not in dataset.specifications:
                raise KeyError(
                    f"The {dataset.name} dataset does not contain a '{spec_name}' compute specification"
                )

            for entry_name, spec_name, record in dataset.iterate_records(
                specification_names=spec_name, status=RecordStatusEnum.complete
            ):
                entry = dataset.get_entry(entry_name)
                molecule = entry.molecule

                cmiles = (
                    molecule.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
                )
                if not cmiles:
                    cmiles = molecule.extras.get(
                        "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                    )
                if not cmiles:
                    print(f"MISSING CMILES! entry = {entry_name}")
                    continue

                inchi_key = entry.attributes.get("fixed_hydrogen_inchi_key")

                # Undefined stereochemistry is not expected however there
                # may be some TK specific edge cases we don't want
                # exceptions for such as OE and nitrogen stereochemistry.
                if inchi_key is None:
                    tmp_mol = Molecule.from_mapped_smiles(
                        cmiles, allow_undefined_stereo=True
                    )
                    inchi_key = tmp_mol.to_inchikey(fixed_hydrogens=True)

                br = BasicResult(
                    record_id=record.id, cmiles=cmiles, inchi_key=inchi_key
                )
                result_records[client.address][record.id] = br

        return cls(
            entries={
                address: [*records.values()]
                for address, records in result_records.items()
            }
        )

    @classmethod
    def from_server(
        cls,
        client: qcportal.PortalClient,
        datasets: Union[str, Iterable[str]],
        spec_name: str = "default",
    ) -> BasicResultCollection:
        if isinstance(datasets, str):
            datasets = [datasets]

        # noinspection PyTypeChecker
        return cls.from_datasets(
            [
                client.get_dataset("singlepoint", dataset_name)
                for dataset_name in datasets
            ],
            spec_name,
        )

    def to_records(self) -> List[Tuple[SinglepointRecord, Molecule]]:
        """Returns the native QCPortal record objects for each of the records referenced
        in this collection along with a corresponding OpenFF molecule object.

        Each molecule will contain the conformer referenced by the record.
        """

        records_and_molecules = []

        for client_address, records in self.entries.items():
            client = cached_fractal_client(address=client_address)

            # TODO - batching/chunking (maybe in portal?)
            for record in records:
                rec = client.get_singlepoints(record.record_id, include=["molecule"])

                # OpenFF molecule
                molecule: Molecule = Molecule.from_mapped_smiles(
                    record.cmiles, allow_undefined_stereo=True
                )

                molecule.add_conformer(
                    numpy.array(rec.molecule.geometry, float).reshape(-1, 3) * unit.bohr
                )

                records_and_molecules.append((rec, molecule))

        return records_and_molecules


class OptimizationResult(_BaseResult):
    """A class which stores a reference to, and allows the retrieval of, data from
    a single optimization result record stored in a QCFractal instance."""

    type: Literal["optimization"] = "optimization"


class OptimizationResultCollection(_BaseResultCollection):
    """A class which stores a reference to, and allows the retrieval of, data from
    a single optimization result record stored in a QCFractal instance."""

    type: Literal["OptimizationResultCollection"] = "OptimizationResultCollection"

    entries: Dict[str, List[OptimizationResult]] = Field(
        ...,
        description="The entries stored in this collection in a dictionary of the form "
        "``collection.entries['qcfractal_address'] = [record_1, ..., record_N]``.",
    )

    @classmethod
    def from_datasets(
        cls,
        datasets: Union[OptimizationDataset, Iterable[OptimizationDataset]],
        spec_name: str = "default",
    ) -> "OptimizationResultCollection":
        if isinstance(datasets, QCPDataset):
            datasets = [datasets]

        if not all(isinstance(dataset, OptimizationDataset) for dataset in datasets):
            raise TypeError(
                "A ``OptimizationResultCollection`` can only be created from "
                "``OptimizationDataset`` objects."
            )

        result_records = defaultdict(dict)

        for dataset in datasets:
            client = dataset.client

            # Fetch all entries for use later. These get stored internally
            # in the dataset class
            dataset.fetch_entries()

            if spec_name not in dataset.specifications:
                raise KeyError(
                    f"The {dataset.name} dataset does not contain a '{spec_name}' compute specification"
                )

            for entry_name, spec_name, record in dataset.iterate_records(
                specification_names=spec_name, status=RecordStatusEnum.complete
            ):
                entry = dataset.get_entry(entry_name)
                molecule = entry.initial_molecule

                cmiles = entry.attributes[
                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                ]
                inchi_key = molecule.extras.get("fixed_hydrogen_inchi_key")

                if inchi_key is None:
                    try:
                        mol = Molecule.from_mapped_smiles(
                            entry.attributes[
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ],
                            allow_undefined_stereo=True,
                        )
                    except ValueError:
                        warnings.warn(
                            f"Skipping entry {entry.name} with invalid CMILES {entry.attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles']}",
                            UserWarning,
                        )
                        continue
                    inchi_key = mol.to_inchikey(fixed_hydrogens=True)
                opt_rec = OptimizationResult(
                    record_id=record.id, cmiles=cmiles, inchi_key=inchi_key
                )
                result_records[client.address][record.id] = opt_rec

        return cls(
            entries={
                address: [*entries.values()]
                for address, entries in result_records.items()
            }
        )

    @classmethod
    def from_server(
        cls,
        client: qcportal.PortalClient,
        datasets: Union[str, Iterable[str]],
        spec_name: str = "default",
    ) -> OptimizationResultCollection:
        if isinstance(datasets, str):
            datasets = [datasets]

        # noinspection PyTypeChecker
        return cls.from_datasets(
            [
                client.get_dataset("Optimization", dataset_name)
                for dataset_name in datasets
            ],
            spec_name,
        )

    def to_records(self) -> List[Tuple[OptimizationRecord, Molecule]]:
        """Returns the native QCPortal record objects for each of the records referenced
        in this collection along with a corresponding OpenFF molecule object.

        Each molecule will contain the minimum energy conformer referenced by the
        record.
        """

        records_and_molecules = []

        for client_address, records in self.entries.items():
            client = cached_fractal_client(address=client_address)

            # TODO - batching/chunking (maybe in portal?)
            for record in records:
                rec = client.get_optimizations(
                    record.record_id, include=["initial_molecule"]
                )

                # OpenFF molecule
                molecule: Molecule = Molecule.from_mapped_smiles(
                    record.cmiles, allow_undefined_stereo=True
                )

                molecule.add_conformer(
                    numpy.array(rec.initial_molecule.geometry, float).reshape(-1, 3)
                    * unit.bohr
                )

                records_and_molecules.append((rec, molecule))

        return records_and_molecules

    def to_basic_result_collection(self) -> BasicResultCollection:
        """Returns a basic results collection which references results records which
        were created from the *final* structure of one of the optimizations in this
        collection, and used the same program, method, and basis as the parent
        optimization record.

        Returns:
            The results collection referencing records created from the final optimized
            structures referenced by this collection.
        """

        records_and_molecules = self.to_records()

        result_records = defaultdict(list)

        for record, molecule in records_and_molecules:
            result_records[record.client.address].append(
                (record.trajectory_element(-1), molecule)
            )

        result_entries = defaultdict(list)

        for client_address in result_records:
            for record, molecule in result_records[client_address]:
                result_entries[client_address].append(
                    BasicResult(
                        record_id=record.id,
                        cmiles=molecule.to_smiles(
                            isomeric=True, explicit_hydrogens=True, mapped=True
                        ),
                        inchi_key=molecule.to_inchikey(fixed_hydrogens=True),
                    )
                )

        return BasicResultCollection(entries=result_entries)

    def create_basic_dataset(
        self,
        dataset_name: str,
        description: str,
        tagline: str,
        driver: SinglepointDriver,
        metadata: Optional[Metadata] = None,
        qc_specifications: Optional[List[QCSpec]] = None,
    ) -> BasicDataset:
        """Create a basic dataset from the results of the current dataset.

        Notes:
            * This may be used, for example, to evaluate the hessians of each optimized
              geometry.

        Parameters:
            dataset_name: The name that will be given to the new dataset.
            tagline: The tagline that should be given to the new dataset.
            description: The description that should be given to the new dataset.
            driver: The driver to be used on the basic dataset.
            metadata: The metadata for the new dataset.
            qc_specifications: The QC specifications to be used on the new dataset. If no value
                is provided, the default OpenFF QCSpec will be added.

        Returns:
            The created basic dataset.
        """

        records_by_cmiles: Dict[
            str, List[Tuple[OptimizationRecord, Molecule]]
        ] = defaultdict(list)

        for record, molecule in self.to_records():
            records_by_cmiles[
                molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
            ].append((record, molecule))

        dataset = BasicDataset(
            dataset_name=dataset_name,
            description=description,
            dataset_tagline=tagline,
            driver=driver,
            metadata={} if metadata is None else metadata,
            qc_specifications={"default": QCSpec()}
            if qc_specifications is None
            else {
                qc_specification.spec_name: qc_specification
                for qc_specification in qc_specifications
            },
        )

        for records in records_by_cmiles.values():
            base_record, base_molecule = records[0]
            base_molecule._conformers = [m.conformers[0] for _, m in records]

            dataset.add_molecule(
                index=base_molecule.to_smiles(
                    isomeric=True, explicit_hydrogens=False, mapped=False
                ),
                molecule=base_molecule,
                attributes=MoleculeAttributes.from_openff_molecule(base_molecule),
                extras=base_record.extras,
                keywords=base_record.specification.keywords,
            )

        return dataset


class TorsionDriveResult(_BaseResult):
    """A class which stores a reference to, and allows the retrieval of, data from
    a single torsion drive result record stored in a QCFractal instance."""

    type: Literal["torsion"] = "torsion"


class TorsionDriveResultCollection(_BaseResultCollection):
    """A class which stores a reference to, and allows the retrieval of, data from
    a single torsion drive result record stored in a QCFractal instance."""

    type: Literal["TorsionDriveResultCollection"] = "TorsionDriveResultCollection"

    entries: Dict[str, List[TorsionDriveResult]] = Field(
        ...,
        description="The entries stored in this collection in a dictionary of the form "
        "``collection.entries['qcfractal_address'] = [record_1, ..., record_N]``.",
    )

    @classmethod
    def from_datasets(
        cls,
        datasets: Union[TorsiondriveDataset, Iterable[TorsiondriveDataset]],
        spec_name: str = "default",
    ) -> "TorsionDriveResultCollection":
        if isinstance(datasets, QCPDataset):
            datasets = [datasets]

        if not all(isinstance(dataset, TorsiondriveDataset) for dataset in datasets):
            raise TypeError(
                "A ``TorsionDriveResultCollection`` can only be created from "
                "``TorsiondriveDataset`` objects."
            )

        result_records = defaultdict(dict)

        for dataset in datasets:
            client = dataset.client

            # Fetch all entries for use later. These get stored internally
            # in the dataset class
            dataset.fetch_entries()

            if spec_name not in dataset.specifications:
                raise KeyError(
                    f"The {dataset.name} dataset does not contain a '{spec_name}' compute specification"
                )

            for entry_name, spec_name, record in dataset.iterate_records(
                specification_names=spec_name, status=RecordStatusEnum.complete
            ):
                entry = dataset.get_entry(entry_name)

                cmiles = entry.attributes[
                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                ]
                inchi_key = entry.attributes.get("fixed_hydrogen_inchi_key")

                if inchi_key is None:
                    tmp_mol = Molecule.from_mapped_smiles(
                        cmiles, allow_undefined_stereo=True
                    )
                    inchi_key = tmp_mol.to_inchikey(fixed_hydrogens=True)

                td_rec = TorsionDriveResult(
                    record_id=record.id, cmiles=cmiles, inchi_key=inchi_key
                )
                result_records[client.address][record.id] = td_rec

        return cls(
            entries={
                address: [*entries.values()]
                for address, entries in result_records.items()
            }
        )

    @classmethod
    def from_server(
        cls,
        client: qcportal.PortalClient,
        datasets: Union[str, Iterable[str]],
        spec_name: str = "default",
    ) -> "TorsionDriveResultCollection":
        if isinstance(datasets, str):
            datasets = [datasets]

        # noinspection PyTypeChecker
        return cls.from_datasets(
            [
                client.get_dataset("Torsiondrive", dataset_name)
                for dataset_name in datasets
            ],
            spec_name,
        )

    def to_records(self) -> List[Tuple[TorsiondriveRecord, Molecule]]:
        """Returns the native QCPortal record objects for each of the records referenced
        in this collection along with a corresponding OpenFF molecule object.

        Each molecule will contain the minimum energy conformer referenced by the
        record.
        """

        records_and_molecules = []

        for client_address, records in self.entries.items():
            client = cached_fractal_client(address=client_address)

            for record in records:
                rec = client.get_torsiondrives(record.record_id)

                # OpenFF molecule
                molecule: Molecule = Molecule.from_mapped_smiles(
                    record.cmiles, allow_undefined_stereo=True
                )

                # Map of torsion drive keys to minimum optimization
                qc_grid_molecules = [
                    (k, v.final_molecule) for k, v in rec.minimum_optimizations.items()
                ]

                # order the ids so the conformers follow the torsiondrive scan range
                # x[0] is the torsiondrive key, ie Tuple[float]
                # We can sort by the whole tuple (although there should only be one value)
                qc_grid_molecules.sort(key=lambda x: x[0])

                molecule._conformers = [
                    numpy.array(qc_molecule.geometry, float).reshape(-1, 3) * unit.bohr
                    for _, qc_molecule in qc_grid_molecules
                ]

                molecule.properties["grid_ids"] = [x[0] for x in qc_grid_molecules]

                records_and_molecules.append((rec, molecule))

        return records_and_molecules

    def create_optimization_dataset(
        self,
        dataset_name: str,
        description: str,
        tagline: str,
        metadata: Optional[Metadata] = None,
    ) -> OptimizationDataset:
        """Create an optimization dataset from the results of the current torsion drive
        dataset. This will result in many constrained optimizations for each molecule.

        Note:
            The final geometry of each torsiondrive constrained optimization is supplied
            as a starting geometry.

        Parameters:
            dataset_name: The name that will be given to the new dataset.
            tagline: The tagline that should be given to the new dataset.
            description: The description that should be given to the new dataset.
            metadata: The metadata for the new dataset.

        Returns:
            The created optimization dataset.
        """
        raise NotImplementedError()

    def smirnoff_coverage(
        self, force_field: ForceField, verbose: bool = False, driven_only: bool = False
    ):
        """Returns a summary of how many unique molecules within this collection
        would be assigned each of the parameters in a force field.

        Notes:
            * Parameters which would not be assigned to any molecules in the collection
              will not be included in the returned summary.

        Args:
            force_field: The force field containing the parameters to summarize.
            verbose: If true a progress bar will be shown on screen.
            driven_only: Whether to only include parameters that are applied (at least
                partially) across a central torsion bond that was driven.

                This option is still experimental and may in cases not count parameters
                that are actually applied or double count certain applied parameters.

        Returns:
            A dictionary of the form ``coverage[handler_name][parameter_smirks] = count``
            which stores the number of unique molecules within this collection that
            would be assigned to each parameter.

            If ``driven_only`` is true, then ``count`` will be the number of unique
            driven torsions, rather than unique molecules.
        """

        if not driven_only:
            return super(TorsionDriveResultCollection, self).smirnoff_coverage(
                force_field, verbose
            )

        records_and_molecules = self.to_records()

        molecules = [
            (molecule, tuple(record.specification.keywords.dihedrals[0]))
            for record, molecule in records_and_molecules
        ]

        return smirnoff_torsion_coverage(molecules, force_field, verbose)
