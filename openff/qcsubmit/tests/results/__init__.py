import copy

import numpy
from openff.toolkit.topology import Molecule
from pydantic import BaseModel
from qcelemental.models import DriverEnum
from qcportal.models import (
    ObjectId,
    OptimizationRecord,
    OptimizationSpecification,
    QCSpecification,
    ResultRecord,
    TorsionDriveRecord,
)
from qcportal.models.records import RecordStatusEnum
from qcportal.models.torsiondrive import TDKeywords

try:
    from openmm import unit
except ImportError:
    from simtk import unit

from openff.qcsubmit.results import (
    BasicResult,
    BasicResultCollection,
    OptimizationResult,
    OptimizationResultCollection,
    TorsionDriveResult,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.results import _BaseResult


class _FractalClient(BaseModel):

    address: str


def _mock_molecule(entry: _BaseResult, n_conformers: int = 1) -> Molecule:

    molecule: Molecule = Molecule.from_smiles(entry.cmiles)

    for _ in range(n_conformers):

        molecule.add_conformer(
            numpy.arange(molecule.n_atoms * 3).reshape((molecule.n_atoms, 3))
            * unit.angstrom
        )

    return molecule


def mock_basic_result_collection(molecules, monkeypatch) -> BasicResultCollection:

    collection = BasicResultCollection(
        entries={
            address: [
                BasicResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=molecule.to_smiles(mapped=True),
                    inchi_key=molecule.to_inchikey(fixed_hydrogens=True),
                )
                for i, molecule in enumerate(molecules[address])
            ]
            for address in molecules
        }
    )

    monkeypatch.setattr(
        BasicResultCollection,
        "to_records",
        lambda self: [
            (
                ResultRecord(
                    id=entry.record_id,
                    program="psi4",
                    driver=DriverEnum.gradient,
                    method="scf",
                    basis="sto-3g",
                    molecule=entry.record_id,
                    status=RecordStatusEnum.complete,
                    client=_FractalClient(address=address),
                ),
                molecules[address][int(entry.record_id) - 1]
            )
            for address, entries in self.entries.items()
            for entry in entries
        ],
    )

    return collection


def mock_optimization_result_collection(
    molecules, monkeypatch
) -> OptimizationResultCollection:

    collection = OptimizationResultCollection(
        entries={
            address: [
                OptimizationResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=molecule.to_smiles(mapped=True),
                    inchi_key=molecule.to_inchikey(fixed_hydrogens=True),
                )
                for i, molecule in enumerate(molecules[address])
            ]
            for address in molecules
        }
    )

    monkeypatch.setattr(
        OptimizationResultCollection,
        "to_records",
        lambda self: [
            (
                OptimizationRecord(
                    id=entry.record_id,
                    program="psi4",
                    qc_spec=QCSpecification(
                        driver=DriverEnum.gradient,
                        method="scf",
                        basis="sto-3g",
                        program="psi4",
                    ),
                    initial_molecule=ObjectId(entry.record_id),
                    final_molecule=ObjectId(entry.record_id),
                    status=RecordStatusEnum.complete,
                    energies=[numpy.random.random()],
                    client=_FractalClient(address=address),
                ),
                molecules[address][int(entry.record_id) - 1],
            )
            for address, entries in self.entries.items()
            for entry in entries
        ],
    )

    return collection


def mock_torsion_drive_result_collection(
    molecules, monkeypatch
) -> TorsionDriveResultCollection:

    collection = TorsionDriveResultCollection(
        entries={
            address: [
                TorsionDriveResult(
                    record_id=ObjectId(str(i + 1)),
                    cmiles=molecule.to_smiles(mapped=True),
                    inchi_key=molecule.to_inchikey(fixed_hydrogens=True),
                )
                for i, molecule in enumerate(molecules[address])
            ]
            for address in molecules
        }
    )

    monkeypatch.setattr(
        TorsionDriveResultCollection,
        "to_records",
        lambda self: [
            (
                TorsionDriveRecord(
                    id=entry.record_id,
                    qc_spec=QCSpecification(
                        driver=DriverEnum.gradient,
                        method="scf",
                        basis="sto-3g",
                        program="psi4",
                    ),
                    optimization_spec=OptimizationSpecification(
                        program="geometric", keywords={}
                    ),
                    initial_molecule=[
                        ObjectId(i + 1)
                        for i in range(
                            molecules[address][int(entry.record_id) - 1].n_conformers
                        )
                    ],
                    status=RecordStatusEnum.complete,
                    client=_FractalClient(address=address),
                    keywords=TDKeywords(dihedrals=[], grid_spacing=[]),
                    final_energy_dict={},
                    optimization_history={},
                    minimum_positions={},
                ),
                molecules[address][int(entry.record_id) - 1],
            )
            for address, entries in self.entries.items()
            for entry in entries
        ],
    )

    def get_molecules(self):

        return_value = []

        for molecule_id in self.initial_molecule:

            molecule = copy.deepcopy(molecules[self.client.address][int(self.id) - 1])
            molecule._conformers = [molecule.conformers[int(molecule_id) - 1]]

            return_value.append(molecule.to_qcschema())

        return return_value

    monkeypatch.setattr(
        TorsionDriveRecord, "get_final_molecules", lambda self: get_molecules(self)
    )

    return collection
