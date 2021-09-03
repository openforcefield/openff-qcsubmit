import json

import numpy
import pytest
import requests_mock
from openff.toolkit.topology import Molecule
from qcportal.models import ObjectId, OptimizationRecord, ResultRecord

try:
    from openmm import unit
except ImportError:
    from simtk import unit

from openff.qcsubmit.results import BasicResult, OptimizationResult, TorsionDriveResult
from openff.qcsubmit.results.caching import (
    _grid_id_cache,
    _molecule_cache,
    _record_cache,
    batched_indices,
    cached_fractal_client,
    cached_query_basic_results,
    cached_query_molecules,
    cached_query_optimization_results,
    cached_query_procedures,
    cached_query_torsion_drive_results,
)


def test_batched_indices():
    assert batched_indices([1, 2, 3, 4], 3) == [[1, 2, 3], [4]]


def test_cached_fractal_client_bad_address():

    with pytest.raises(ConnectionRefusedError):
        cached_fractal_client("http://localhost:1234/")


def test_cached_fractal_client_snowflake():

    from qcfractal import FractalSnowflake

    snowflake = FractalSnowflake(start_server=False)
    client = cached_fractal_client(snowflake.client().address)

    assert client is not None


def test_cached_query_procedures(public_client):

    assert len(_record_cache) == 0

    record_ids = ["32651863", "32651864"]

    records = cached_query_procedures(public_client.address, record_ids)

    assert len(records) == 2
    assert {record.id for record in records} == {*record_ids}

    assert len(_record_cache) == 2

    # The request mocker would raise an exception if the client tries to reach out
    # to the server.
    with requests_mock.Mocker():
        cached_query_procedures(public_client.address, record_ids)


def test_cached_query_molecule(public_client):

    assert len(_molecule_cache) == 0

    molecule_ids = ["25696236", "25696152"]

    molecules = cached_query_molecules(public_client.address, molecule_ids)

    assert len(molecules) == 2
    assert {molecule.id for molecule in molecules} == {*molecule_ids}

    assert len(_molecule_cache) == 2

    # The request mocker would raise an exception if the client tries to reach out
    # to the server.
    with requests_mock.Mocker():
        cached_query_molecules(public_client.address, molecule_ids)


@pytest.mark.parametrize(
    "result, query_function",
    [
        (
            BasicResult(
                record_id=ObjectId("32651863"),
                cmiles="[H:3][C:1]([H:4])([H:5])[O:2][H:6]",
                inchi_key="",
            ),
            cached_query_basic_results,
        ),
        (
            OptimizationResult(
                record_id=ObjectId("25724668"),
                cmiles="[C:1]([H:2])([H:3])([H:4])[H:5]",
                inchi_key="",
            ),
            cached_query_optimization_results,
        ),
    ],
)
def test_record_to_molecule(
    result, query_function, public_client
):

    expected_molecule = Molecule.from_mapped_smiles(result.cmiles)

    records = query_function(public_client.address, [result])
    assert len(records) == 1

    record, molecule = records[0]

    assert molecule.n_conformers == 1

    if isinstance(record, ResultRecord):
        expected_qc_molecule = record.get_molecule()
    elif isinstance(record, OptimizationRecord):
        expected_qc_molecule = record.get_final_molecule()
    else:
        raise RuntimeError()

    assert numpy.allclose(
        molecule.conformers[0].value_in_unit(unit.bohr),
        expected_qc_molecule.geometry.reshape((molecule.n_atoms, 3))
    )

    are_isomorphic, _ = Molecule.are_isomorphic(molecule, expected_molecule)
    assert are_isomorphic

    # The request mocker would raise an exception if the client tries to reach out
    # to the server.
    with requests_mock.Mocker():
        query_function(public_client.address, [result])


def test_cached_query_torsion_drive_results(public_client):

    assert len(_grid_id_cache) == 0

    result = TorsionDriveResult(
        record_id=ObjectId("36633243"),
        cmiles="[H:6][N:5]([H:7])[C:3](=[O:4])[C:1]#[N:2]",
        inchi_key="",
    )

    expected_molecule = Molecule.from_mapped_smiles(result.cmiles)

    records = cached_query_torsion_drive_results(public_client.address, [result])
    assert len(records) == 1

    record, molecule = records[0]

    assert molecule.n_conformers == 24

    assert "grid_ids" in molecule.properties

    expected_qc_molecules = {
        json.dumps(grid_id): expected_qc_molecule
        for grid_id, expected_qc_molecule in record.get_final_molecules().items()
    }

    for grid_id, conformer in zip(molecule.properties["grid_ids"], molecule.conformers):

        assert numpy.allclose(
            conformer.value_in_unit(unit.bohr),
            expected_qc_molecules[grid_id].geometry.reshape((molecule.n_atoms, 3))
        )

    assert len(molecule.properties["grid_ids"]) == 24

    are_isomorphic, _ = Molecule.are_isomorphic(molecule, expected_molecule)
    assert are_isomorphic

    assert len(_grid_id_cache) == 24

    # The request mocker would raise an exception if the client tries to reach out
    # to the server.
    with requests_mock.Mocker():
        cached_query_torsion_drive_results(public_client.address, [result])
