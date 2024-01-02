import copy
import datetime

import numpy
from openff.toolkit.topology import Molecule
from openff.units import unit
from qcelemental.models import DriverEnum
from qcelemental.models.procedures import TDKeywords
from qcportal.optimization import OptimizationRecord, OptimizationSpecification
from qcportal.record_models import RecordStatusEnum
from qcportal.singlepoint import QCSpecification, SinglepointRecord
from qcportal.torsiondrive import TorsiondriveRecord, TorsiondriveSpecification

from openff.qcsubmit._pydantic import BaseModel
from openff.qcsubmit.results import (
    BasicResult,
    BasicResultCollection,
    OptimizationResult,
    OptimizationResultCollection,
    TorsionDriveResult,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.results import _BaseResult


class _PortalClient(BaseModel):
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
                    record_id=i + 1,
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
                SinglepointRecord(
                    id=entry.record_id,
                    specification=QCSpecification(
                        program="psi4",
                        driver=DriverEnum.gradient,
                        method="scf",
                        basis="sto-3g",
                    ),
                    molecule_id=entry.record_id,
                    is_service=False,
                    created_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
                    modified_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
                    # compute_history=list(),
                    status=RecordStatusEnum.complete,
                    client=_PortalClient(address=address),
                ),
                molecules[address][int(entry.record_id) - 1],
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
                    record_id=i + 1,
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
                # OptimizationRecord.construct(
                # OptimizationRecord.construct(
                OptimizationRecord(
                    # OptimizationRecord(
                    specification=OptimizationSpecification(
                        program="geometric",
                        qc_specification=QCSpecification(
                            driver=DriverEnum.gradient,
                            method="scf",
                            basis="sto-3g",
                            program="psi4",
                        ),
                    ),
                    id=entry.record_id,
                    initial_molecule_id=entry.record_id,
                    final_molecule_id=entry.record_id,
                    status=RecordStatusEnum.complete,
                    energies=[numpy.random.random()],
                    is_service=False,
                    created_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
                    modified_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
                    # compute_history=list(),
                    # ),
                    client=_PortalClient(address=address),
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
                    record_id=i + 1,
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
                TorsiondriveRecord(
                    id=entry.record_id,
                    specification=TorsiondriveSpecification(
                        program="torsiondrive",
                        keywords=TDKeywords(dihedrals=[], grid_spacing=[]),
                        optimization_specification=OptimizationSpecification(
                            program="geometric",
                            keywords={},
                            qc_specification=QCSpecification(
                                driver=DriverEnum.gradient,
                                method="scf",
                                basis="sto-3g",
                                program="psi4",
                            ),
                        ),
                    ),
                    initial_molecules_ids_=[
                        i + 1
                        for i in range(
                            molecules[address][int(entry.record_id) - 1].n_conformers
                        )
                    ],
                    status=RecordStatusEnum.complete,
                    client=_PortalClient(address=address),
                    is_service=True,
                    created_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
                    modified_on=datetime.datetime(2022, 4, 21, 0, 0, 0),
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
        TorsiondriveRecord, "minimum_optimizations", lambda self: get_molecules(self)
    )

    return collection
