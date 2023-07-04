"""A module which contains helpers for retrieving data from QCA and caching the
results in memory for future requests."""
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy
from cachetools import Cache, LRUCache
from openff.toolkit.topology import Molecule
from openff.units import unit
from qcportal import PortalClient
from qcportal.molecules import Molecule as QCMolecule
from qcportal.optimization import OptimizationRecord
from qcportal.record_models import BaseRecord
from qcportal.singlepoint import SinglepointRecord
from qcportal.torsiondrive import TorsiondriveRecord

if TYPE_CHECKING:
    from openff.qcsubmit.results.results import (
        BasicResult,
        OptimizationResult,
        TorsionDriveResult,
        _BaseResult,
    )

    R = TypeVar("R", bound=_BaseResult)

S = TypeVar("S", bound=BaseRecord)
T = TypeVar("T")

RecordAndMolecule = Tuple[BaseRecord, Molecule]

_record_cache = LRUCache(maxsize=20000)

_molecule_cache = LRUCache(maxsize=240000)

_grid_id_cache = LRUCache(maxsize=240000)

logger = logging.getLogger(__name__)


def clear_results_caches():
    """Clear the internal results caches."""

    _record_cache.clear()
    _molecule_cache.clear()
    _grid_id_cache.clear()


def batched_indices(indices: List[T], batch_size: int) -> List[List[T]]:
    """Split a list of indices into batches.

    Args:
        indices: The indices to split.
        batch_size: The size to split the indices into.

    Returns:
        A list of list batches of indices.
    """
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


@lru_cache()
def cached_fractal_client(address: str) -> PortalClient:
    """Returns a cached copy of a fractal client."""

    try:
        return PortalClient(address)

    except ConnectionRefusedError as e:
        # Try to handle the case when connecting to a local snowflake.
        try:
            return PortalClient(address, verify=False)
        except ConnectionRefusedError:
            raise e


def _cached_client_query(
    client_address: str,
    query_ids: List[T],
    query_name: str,
    query_cache: Cache,
    query_limit: int,
    cache_predicate: Optional[Callable[[Any], bool]] = None,
) -> List[S]:
    """A helper method to cache calls to ``PortalClient.query_XXX`` methods.

    Args:
        client_address: The address of the running QCFractal instance to query.
        query_ids: The ids to query.
        query_name: The name of the query function.
        query_cache: The cache associated with a query. Records should be indexable
            by ``query_cache[(client_address, query_id)]``.
        query_limit: The query limit for the particular query name.
        cache_predicate: A function which returns whether and object should be added
            to the cache.

    Returns:
        The returned query objects.
    """

    client_address = client_address.rstrip("/")

    missing_query_ids = list(
        {
            query_id
            for query_id in query_ids
            if (client_address, query_id) not in query_cache
        }
    )

    found_queries = [
        query_cache[(client_address, query_id)]
        for query_id in query_ids
        if (client_address, query_id) in query_cache
    ]

    client = cached_fractal_client(client_address)

    logger.debug(f"starting {query_name} to {client_address}")

    batch_query_ids = batched_indices(missing_query_ids, query_limit)
    logger.debug(f"query split into {len(batch_query_ids)} batches")

    for i, batch_ids in enumerate(batch_query_ids):
        logger.debug(f"starting batch query {i}")

        for query in getattr(client, query_name)(batch_ids):
            found_queries.append(query)

            if cache_predicate is not None and not cache_predicate(query):
                continue

            query_cache[(client_address, query.id)] = query

        logger.debug(f"finished batch query {i}")

    logger.debug(f"finished {query_name} to {client_address}")

    return found_queries


def cached_query_procedures(client_address: str, record_ids: List[str]) -> List[S]:
    """A cached version of ``PortalClient.query_procedures``.

    Args:
        client_address: The address of the running QCFractal instance to query.
        record_ids: The ids of the records to query.

    Returns:
        The returned records.
    """
    client_address = client_address.rstrip("/")
    client = cached_fractal_client(client_address)

    query_limit = client.api_limits["get_records"]

    return _cached_client_query(
        client_address,
        record_ids,
        "query_procedures",
        _record_cache,
        query_limit,
        lambda record: record.status.value.upper() == "COMPLETE",
    )


def cached_query_molecules(
    client_address: str, molecule_ids: List[str]
) -> List[QCMolecule]:
    """A cached version of ``PortalClient.query_molecules``.

    Args:
        client_address: The address of the running QCFractal instance to query.
        molecule_ids: The ids of the molecules to query.

    Returns:
        The returned molecules.
    """
    client_address = client_address.rstrip("/")
    client = cached_fractal_client(client_address)

    query_limit = client.api_limits["get_molecules"]

    return _cached_client_query(
        client_address,
        molecule_ids,
        "query_molecules",
        _molecule_cache,
        query_limit,
    )


def _cached_query_single_structure_results(
    client_address: str, results: List["R"], molecule_attribute: str
) -> List[Tuple[S, Molecule]]:
    """A utility function to batch query a server for the records and their
    corresponding molecules referenced by a list of result entries.

    Args:
        client_address: The address of the running QCFractal instance to query.
        results: The result objects to query.
        molecule_attribute: The name of the field on the record which stores the
            id of the molecule associated with the record.

    Returns:
        A list of tuples of the returned records and molecules.
    """

    logger.debug(f"retrieving records from {client_address}")

    qc_records: Dict[str, S] = {
        qc_record.id: qc_record
        for qc_record in cached_query_procedures(
            client_address, [result.record_id for result in results]
        )
    }

    logger.debug(f"finished retrieving records from {client_address}")
    logger.debug("retrieving corresponding molecules")

    qc_record_to_molecule_id = {
        qc_record.id: getattr(qc_record, molecule_attribute)
        for qc_record in qc_records.values()
    }

    qc_molecules = {
        molecule.id: molecule
        for molecule in cached_query_molecules(
            client_address, [*qc_record_to_molecule_id.values()]
        )
    }

    logger.debug("finished retrieving corresponding molecules")

    return_values = []

    for result in results:
        qc_record = qc_records[result.record_id]
        qc_molecule = qc_molecules[qc_record_to_molecule_id[result.record_id]]

        molecule: Molecule = Molecule.from_mapped_smiles(
            result.cmiles, allow_undefined_stereo=True
        )

        molecule.add_conformer(
            numpy.array(qc_molecule.geometry, float).reshape(-1, 3) * unit.bohr
        )

        return_values.append((qc_record, molecule))

    return return_values


def cached_query_basic_results(
    client_address: str, results: List["BasicResult"]
) -> List[Tuple[SinglepointRecord, Molecule]]:
    """Returns the QC record and corresponding molecule object associated with each
    of the specified result entries.

    The molecule will contain the conformer referenced by the record.

    Args:
        client_address: The address of the running QCFractal instance to query.
        results: The result objects to query.

    Returns:
        A list of tuples of the returned records and molecules.
    """

    return _cached_query_single_structure_results(client_address, results, "molecule")


def cached_query_optimization_results(
    client_address: str, results: List["OptimizationResult"]
) -> List[Tuple[OptimizationRecord, Molecule]]:
    """Returns the QC record and corresponding molecule object associated with each
    of the specified result entries.

    The molecule will contain the minimum energy conformer referenced by the record.

    Args:
        client_address: The address of the running QCFractal instance to query.
        results: The result objects to query.

    Returns:
        A list of tuples of the returned records and molecules.
    """

    return _cached_query_single_structure_results(
        client_address, results, "final_molecule"
    )


def cached_query_torsion_drive_results(
    client_address: str, results: List["TorsionDriveResult"]
) -> List[Tuple[TorsiondriveRecord, Molecule]]:
    """Returns the QC record and corresponding molecule object associated with each
    of the specified result entries.

    The molecule will contain the minimum energy conformer at each grid id. The grid
    ids themselves will be stored in ``molecule.properties["grid_ids"]``, such
    that ``molecule.conformers[i]`` corresponds to the minimum energy structure
    at grid id ``molecule.properties["grid_ids"][i]``.

    Args:
        client_address: The address of the running QCFractal instance to retrieve the
            results from.
        results: The results to retrieve.
    """

    logger.debug(f"retrieving records from {client_address}")

    qc_records: Dict[str, TorsiondriveRecord] = {
        qc_record.id: qc_record
        for qc_record in cached_query_procedures(
            client_address, [result.record_id for result in results]
        )
    }
    return_values = []

    for result in results:
        qc_record = qc_records[result.record_id]

        qc_grid_molecules = [
            (grid_point, opt.final_molecule)
            for grid_point, opt in qc_record.minimum_optimizations.items()
        ]
        # order the ids so the conformers follow the torsiondrive scan range
        qc_grid_molecules.sort(
            key=lambda s: tuple(float(x) for x in s.strip("[]").split(", "))
        )

        molecule: Molecule = Molecule.from_mapped_smiles(
            result.cmiles, allow_undefined_stereo=True
        )
        molecule._conformers = [
            numpy.array(qc_molecule.geometry, float).reshape(-1, 3) * unit.bohr
            for _, qc_molecule in qc_grid_molecules
        ]

        molecule.properties["grid_ids"] = [x[0] for x in qc_grid_molecules]

        return_values.append((qc_record, molecule))

    return return_values
