"""
A module which contains convenience classes for referencing, retrieving and filtering
results from a QCFractal instance.
"""
import abc
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

import qcportal
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from pydantic import BaseModel, Field, validator
from qcportal.collections import OptimizationDataset, TorsionDriveDataset
from qcportal.collections.collection import Collection as QCCollection
from qcportal.collections.dataset import Dataset, MoleculeEntry
from qcportal.models import OptimizationRecord, ResultRecord, TorsionDriveRecord
from qcportal.models.common_models import DriverEnum, ObjectId
from qcportal.models.records import RecordBase
from qcportal.models.rest_models import QueryStr
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

    record_id: ObjectId = Field(
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
        datasets: Union[QCCollection, Iterable[QCCollection]],
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
        client: qcportal.FractalClient,
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

    @classmethod
    def _validate_record_types(
        cls, records: List[ResultRecord], expected_type: Type[RecordBase]
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
    def to_records(self) -> List[Tuple[RecordBase, Molecule]]:
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


class BasicResult(_BaseResult):
    """A class which stores a reference to, and allows the retrieval of, data from
    a single result record stored in a QCFractal instance."""

    type: Literal["basic"] = "basic"


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
        datasets: Union[Dataset, Iterable[Dataset]],
        spec_name: str = "default",
    ) -> "BasicResultCollection":
        if isinstance(datasets, QCCollection):
            datasets = [datasets]

        if not all(isinstance(dataset, Dataset) for dataset in datasets):
            raise TypeError(
                "A ``BasicResultCollection`` can only be created from ``Dataset`` "
                "objects."
            )

        result_records = defaultdict(dict)
        molecules = {}

        for dataset in datasets:
            client = dataset.client

            dataset_specs = {
                spec: {
                    "method": method,
                    "basis": basis,
                    "program": program,
                    "keywords": spec,
                }
                for _, program, method, basis, spec in dataset.data.history
            }

            if spec_name not in dataset_specs:
                raise KeyError(
                    f"The {dataset.data.name} dataset does not contain a '{spec_name}' "
                    f"compute specification"
                )

            # query the database to get all of the result records requested
            query = dataset.get_records(
                **dataset_specs[spec_name],
                status=[
                    "COMPLETE",
                ],
            )

            entries: Dict[str, MoleculeEntry] = {
                entry.name: entry for entry in dataset.data.records
            }

            # Query the server for the molecules associated with these entries.
            # We only try to pull down ones which haven't already been retrieved.
            molecules.update(
                {
                    molecule.id: molecule
                    for molecule in cached_query_molecules(
                        client.address,
                        [
                            entry.molecule_id
                            for entry in entries.values()
                            if entry.molecule_id not in molecules
                        ],
                    )
                }
            )

            result_records[client.address].update(
                {
                    result.id: BasicResult(
                        record_id=result.id,
                        cmiles=molecules[entries[index].molecule_id].extras[
                            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                        ],
                        inchi_key=Molecule.from_mapped_smiles(
                            molecules[entries[index].molecule_id].extras[
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ],
                            # Undefined stereochemistry is not expected however there
                            # may be some TK specific edge cases we don't want
                            # exceptions for such as OE and nitrogen stereochemistry.
                            allow_undefined_stereo=True,
                        ).to_inchikey(fixed_hydrogens=True),
                    )
                    for index, (result,) in query.iterrows()
                    if isinstance(result, ResultRecord)
                    and result.status.value.upper() == "COMPLETE"
                }
            )

        return cls(
            entries={
                address: [*entries.values()]
                for address, entries in result_records.items()
            }
        )

    @classmethod
    def from_server(
        cls,
        client: qcportal.FractalClient,
        datasets: Union[str, Iterable[str]],
        spec_name: str = "default",
    ) -> "BasicResultCollection":
        if isinstance(datasets, str):
            datasets = [datasets]

        # noinspection PyTypeChecker
        return cls.from_datasets(
            [
                client.get_collection("Dataset", dataset_name)
                for dataset_name in datasets
            ],
            spec_name,
        )

    def to_records(self) -> List[Tuple[ResultRecord, Molecule]]:
        """Returns the native QCPortal record objects for each of the records referenced
        in this collection along with a corresponding OpenFF molecule object.

        Each molecule will contain the conformer referenced by the record.
        """

        records_and_molecules = [
            result
            for client_address, entries in self.entries.items()
            for result in cached_query_basic_results(client_address, entries)
        ]

        records, _ = zip(*records_and_molecules)

        self._validate_record_types(records, ResultRecord)

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
        if isinstance(datasets, QCCollection):
            datasets = [datasets]

        if not all(isinstance(dataset, OptimizationDataset) for dataset in datasets):
            raise TypeError(
                "A ``OptimizationResultCollection`` can only be created from "
                "``OptimizationDataset`` objects."
            )

        result_records = defaultdict(dict)

        for dataset in datasets:
            client = dataset.client
            query = dataset.query(spec_name)

            for entry in dataset.data.records.values():
                if not (
                    (entry.name in query)
                    and (query[entry.name].status.value.upper() == "COMPLETE")
                ):
                    continue
                inchi_key = entry.attributes.get("fixed_hydrogen_inchi_key")
                if inchi_key is None:
                    try:
                        mol = Molecule.from_mapped_smiles(
                            entry.attributes[
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ],
                            allow_undefined_stereo=True,
                        )
                    except ValueError:
                        continue
                    inchi_key = mol.to_inchikey(fixed_hydrogens=True)

                result_records[client.address][
                    query[entry.name].id
                ] = OptimizationResult(
                    record_id=query[entry.name].id,
                    cmiles=entry.attributes[
                        "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                    ],
                    inchi_key=inchi_key,
                )

            # result_records[client.address].update(
            #    {
            #        :
            #    }
            # )

        return cls(
            entries={
                address: [*entries.values()]
                for address, entries in result_records.items()
            }
        )

    @classmethod
    def from_server(
        cls,
        client: qcportal.FractalClient,
        datasets: Union[str, Iterable[str]],
        spec_name: str = "default",
    ) -> "OptimizationResultCollection":
        if isinstance(datasets, str):
            datasets = [datasets]

        # noinspection PyTypeChecker
        return cls.from_datasets(
            [
                client.get_collection("OptimizationDataset", dataset_name)
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

        records_and_molecules = [
            result
            for client_address, entries in self.entries.items()
            for result in cached_query_optimization_results(client_address, entries)
        ]

        records, _ = zip(*records_and_molecules)

        self._validate_record_types(records, OptimizationRecord)

        return records_and_molecules

    def to_basic_result_collection(
        self, driver: Optional[QueryStr] = None
    ) -> BasicResultCollection:
        """Returns a basic results collection which references results records which
        were created from the *final* structure of one of the optimizations in this
        collection, and used the same program, method, and basis as the parent
        optimization record.

        Args:
            driver: Optionally specify the driver to filter by.

        Returns:
            The results collection referencing records created from the final optimized
            structures referenced by this collection.
        """

        records_and_molecules = self.to_records()

        final_molecule_ids = defaultdict(lambda: defaultdict(list))
        final_molecules = defaultdict(dict)

        for record, molecule in records_and_molecules:
            spec = (
                record.qc_spec.program,
                record.qc_spec.method,
                record.qc_spec.basis,
                record.qc_spec.keywords,
            )

            final_molecule_ids[record.client.address][spec].append(
                record.final_molecule
            )
            final_molecules[record.client.address][record.final_molecule] = molecule

        result_entries = defaultdict(list)

        for client_address in final_molecule_ids:
            client = cached_fractal_client(client_address)

            result_records = [
                record
                for (
                    program,
                    method,
                    basis,
                    keywords,
                ), molecules_ids in final_molecule_ids[client_address].items()
                for batch_ids in batched_indices(molecules_ids, client.query_limit)
                for record in client.query_results(
                    molecule=batch_ids,
                    driver=driver,
                    program=program,
                    method=method,
                    basis=basis,
                    keywords=keywords,
                )
            ]

            for record in result_records:
                molecule = final_molecules[client_address][record.molecule]

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
        driver: DriverEnum,
        metadata: Optional[Metadata] = None,
        qc_specs: Optional[List[QCSpec]] = None,
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
            qc_specs: The QC specifications to be used on the new dataset. If no value
                is provided, the default OpenFF QCSpec will be added.

        Returns:
            The created basic dataset.
        """

        records_by_cmiles = defaultdict(list)

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
            if qc_specs is None
            else {qc_spec.spec_name: qc_spec for qc_spec in qc_specs},
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
                keywords=base_record.keywords,
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
        datasets: Union[TorsionDriveDataset, Iterable[TorsionDriveDataset]],
        spec_name: str = "default",
    ) -> "TorsionDriveResultCollection":
        if isinstance(datasets, QCCollection):
            datasets = [datasets]

        if not all(isinstance(dataset, TorsionDriveDataset) for dataset in datasets):
            raise TypeError(
                "A ``TorsionDriveResultCollection`` can only be created from "
                "``TorsionDriveDataset`` objects."
            )

        result_records = defaultdict(dict)

        for dataset in datasets:
            client = dataset.client
            query = dataset.query(spec_name)

            result_records[client.address].update(
                {
                    query[entry.name].id: TorsionDriveResult(
                        record_id=query[entry.name].id,
                        cmiles=entry.attributes[
                            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                        ],
                        inchi_key=entry.attributes.get("fixed_hydrogen_inchi_key")
                        or Molecule.from_mapped_smiles(
                            entry.attributes[
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ],
                            allow_undefined_stereo=True,
                        ).to_inchikey(fixed_hydrogens=True),
                    )
                    for entry in dataset.data.records.values()
                    if entry.name in query
                    and query[entry.name].status.value.upper() == "COMPLETE"
                }
            )

        return cls(
            entries={
                address: [*entries.values()]
                for address, entries in result_records.items()
            }
        )

    @classmethod
    def from_server(
        cls,
        client: qcportal.FractalClient,
        datasets: Union[str, Iterable[str]],
        spec_name: str = "default",
    ) -> "TorsionDriveResultCollection":
        if isinstance(datasets, str):
            datasets = [datasets]

        # noinspection PyTypeChecker
        return cls.from_datasets(
            [
                client.get_collection("TorsionDriveDataset", dataset_name)
                for dataset_name in datasets
            ],
            spec_name,
        )

    def to_records(self) -> List[Tuple[TorsionDriveRecord, Molecule]]:
        """Returns the native QCPortal record objects for each of the records referenced
        in this collection along with a corresponding OpenFF molecule object.

        Each molecule will contain the minimum energy conformer referenced by the
        record.
        """

        records_and_molecules = [
            result
            for client_address, entries in self.entries.items()
            for result in cached_query_torsion_drive_results(client_address, entries)
        ]

        records, _ = zip(*records_and_molecules)

        self._validate_record_types(records, TorsionDriveRecord)

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
            (molecule, tuple(record.keywords.dihedrals[0]))
            for record, molecule in records_and_molecules
        ]

        return smirnoff_torsion_coverage(molecules, force_field, verbose)
