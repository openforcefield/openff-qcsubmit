"""
Test submissions to a local qcarchive instance using different compute backends, RDKit, OpenMM, PSI4

Here we use the qcfractal snowflake fixture to set up the database.
"""

import pytest
from openff.toolkit.topology import Molecule
from qcelemental.models.procedures import OptimizationProtocols
from qcengine.testing import has_program
from qcportal import PortalClient
from qcportal.record_models import RecordStatusEnum

from openff.qcsubmit import workflow_components
from openff.qcsubmit._pydantic import ValidationError
from openff.qcsubmit.common_structures import (
    DDXSettings,
    MoleculeAttributes,
    PCMSettings,
    SCFProperties,
)
from openff.qcsubmit.constraints import Constraints
from openff.qcsubmit.datasets import (
    BasicDataset,
    OptimizationDataset,
    TorsiondriveDataset,
)
from openff.qcsubmit.datasets.dataset_utils import (
    legacy_qcsubmit_ds_type_to_next_qcf_ds_type,
)
from openff.qcsubmit.exceptions import DatasetInputError, MissingBasisCoverageError
from openff.qcsubmit.factories import (
    BasicDatasetFactory,
    OptimizationDatasetFactory,
    TorsiondriveDatasetFactory,
)
from openff.qcsubmit.procedures import GeometricProcedure
from openff.qcsubmit.results import (
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.utils import get_data
import regex as re


def await_results(client, timeout=120, check_fn=PortalClient.get_singlepoints, ids=[1]):
    import time

    for i in range(timeout):
        time.sleep(1)
        recs = check_fn(client, ids)
        from pprint import pprint

        finished = 0
        from qcportal.record_models import OutputTypeEnum

        for rec in recs:
            print(rec.status)
            # would be nice to replace this with match, but black wasn't
            # accepting it
            if rec.status in [
                RecordStatusEnum.error,
                RecordStatusEnum.invalid,
                RecordStatusEnum.cancelled,
                RecordStatusEnum.deleted,
            ]:
                print("stderr", rec._get_output(OutputTypeEnum.stderr))
                print("stdout", rec._get_output(OutputTypeEnum.stdout))
                print("error: ")
                pprint(rec._get_output(OutputTypeEnum.error))
                raise RuntimeError(f"calculation failed: {rec}")
            elif rec.status in [RecordStatusEnum.running, RecordStatusEnum.waiting]:
                pass  # still running
            elif rec.status == RecordStatusEnum.complete:
                finished += 1
            else:
                raise RuntimeError(
                    f"Unrecognized status ({rec.status}) for record: {rec}"
                )
        if finished == len(recs):
            return True
    else:
        raise RuntimeError("Did not finish calculation in time")


def await_services(client, max_iter=10):
    import time
    from pprint import pprint

    from qcportal.record_models import OutputTypeEnum

    for x in range(1, max_iter + 1):
        recs = [
            *client.query_singlepoints(),
            *client.query_optimizations(),
            *client.query_torsiondrives(),
        ]
        finished = 0
        for rec in recs:
            if rec.status == RecordStatusEnum.error:
                print("stderr", rec._get_output(OutputTypeEnum.stderr))
                print("stdout", rec._get_output(OutputTypeEnum.stdout))
                print("error: ")
                pprint(rec._get_output(OutputTypeEnum.error))
                raise RuntimeError(f"calculation failed: {rec}")
            if rec.status not in [RecordStatusEnum.running, RecordStatusEnum.waiting]:
                finished += 1
        if finished == len(recs):
            return True
        time.sleep(1)
    raise RuntimeError("Did not finish calculation in time")


def check_added_specs(ds, dataset):
    """Make sure each of the dataset specs were correctly added to qcportal."""
    for spec_name, specification in ds.specifications.items():
        spec = dataset.qc_specifications[spec_name]
        assert specification.specification.driver == dataset.driver
        assert specification.specification.program == spec.program
        assert specification.specification.method == spec.method
        assert specification.specification.basis == spec.basis
        assert specification.description == spec.spec_description
        break
    else:
        raise RuntimeError(
            f"The requested compute specification was not found in the dataset {ds.specifications}"
        )


def check_metadata(ds, dataset):
    "Check the metadata, tags, and provenance of ds compared to dataset"
    meta = ds.metadata
    assert meta["long_description"] == dataset.metadata.long_description
    assert meta["short_description"] == dataset.metadata.short_description
    assert ds.tags == dataset.dataset_tags
    assert ds.provenance == dataset.provenance


@pytest.mark.parametrize(
    "specification",
    [
        pytest.param(
            ({"method": "hf", "basis": "3-21g", "program": "psi4"}, "energy"),
            id="PSI4 hf 3-21g energy",
        ),
        pytest.param(
            (
                {
                    "method": "openff-2.1.0",
                    "basis": "smirnoff",
                    "program": "openmm",
                },
                "energy",
            ),
            id="SMIRNOFF openff-2.1.0 energy",
        ),
        pytest.param(
            ({"method": "uff", "basis": None, "program": "rdkit"}, "gradient"),
            id="RDKit UFF gradient",
        ),
    ],
)
def test_basic_submissions_single_spec(fulltest_client, specification):
    """Test submitting a basic dataset to a snowflake server."""

    client = fulltest_client

    qc_spec, driver = specification

    program = qc_spec["program"]
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    # keep the QM cost down by using fewer conformers
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")[:2]

    factory = BasicDatasetFactory(driver=driver)
    factory.add_qc_spec(
        **qc_spec,
        spec_name="default",
        spec_description="testing the single points",
        overwrite=True,
    )

    dataset = factory.create_dataset(
        dataset_name=f"Test single points info {program}, {driver}",
        molecules=molecules,
        description="Test basics dataset",
        tagline="Testing single point datasets",
    )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=client)
    await_results(client)

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # check the metadata
    check_metadata(ds=ds, dataset=dataset)

    # make sure all specifications were added
    check_added_specs(ds=ds, dataset=dataset)

    # check the compute was run with the requested specification
    for spec in dataset.qc_specifications.values():
        query = list(
            ds.iterate_records(
                specification_names="default",
            )
        )
        # make sure all of the conformers were submitted
        assert len(query) == len(molecules)
        for name, _, record in query:
            assert record.status == RecordStatusEnum.complete
            assert record.error is None
            assert record.return_result is not None
            assert record.specification.dict(
                include={"method", "program", "basis"}
            ) == spec.dict(include={"method", "program", "basis"})


def test_basic_submissions_property_driver(fulltest_client, water):
    """Make sure the keywords are formatted properly if we use the property driver."""

    if not has_program("psi4"):
        pytest.skip("Program psi4 not found.")

    client = fulltest_client

    dataset = BasicDataset(
        dataset_name="testing properties",
        dataset_tagline="testing properties driver",
        description="testing properties driver",
        driver="properties",
    )
    dataset.clear_qcspecs()
    dataset.add_qc_spec(
        method="hf",
        basis="sto-3g",
        program="psi4",
        scf_properties=[
            SCFProperties.DipolePolarizabilities,
            SCFProperties.Dipole,
            SCFProperties.MBISCharges,
        ],
        spec_name="hf/sto3g",
        spec_description="Quick hf spec",
    )

    dataset.add_molecule(index="water", molecule=water)
    # make sure the keywords are formatted correctly
    qc_keywords = dataset.qc_specifications["hf/sto3g"].qc_keywords(properties=True)
    assert "dipole_polarizabilities" in qc_keywords["function_kwargs"]["properties"]

    # submit and check the dipole polarizability was calculated
    dataset.submit(client=client)
    await_results(client)

    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )
    # make sure all specifications were added
    check_added_specs(ds=ds, dataset=dataset)

    record = ds.get_record(entry_name="water", specification_name="hf/sto3g")
    assert record.status == RecordStatusEnum.complete
    assert record.error is None
    # make sure normal scf properties were calculated
    assert "mbis charges" in record.properties
    # check the method specific dipole was added
    assert "hf dipole" in record.properties
    # make sure the response property was calculated
    assert "dipole polarizability xx" in record.properties
    # check the specification on the record
    assert record.specification.program == "psi4"
    assert record.specification.driver == "properties"
    assert record.specification.method == "hf"


def test_basic_submissions_multiple_spec(fulltest_client, conformer_water):
    """Test submitting a basic dataset to a snowflake server with multiple qcspecs."""

    client = fulltest_client

    qc_specs = [
        {
            "method": "openff-2.0.0",
            "basis": "smirnoff",
            "program": "openmm",
            "spec_name": "openff-2.0.0",
        },
        {
            "method": "uff",
            "basis": None,
            "program": "rdkit",
            "spec_name": "uff",
        },
    ]

    factory = BasicDatasetFactory(driver="energy")
    factory.clear_qcspecs()
    for spec in qc_specs:
        factory.add_qc_spec(**spec, spec_description="testing the single points")

    dataset = factory.create_dataset(
        dataset_name="Test single points multiple specs",
        molecules=conformer_water,
        description="Test basics dataset",
        tagline="Testing single point datasets",
    )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=client)
    # 2 conformers * 2 specs check all 4 results
    await_results(client, ids=[1, 2, 3, 4])

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # check the metadata
    check_metadata(ds=ds, dataset=dataset)

    # check the specifications were added correctly
    check_added_specs(ds=ds, dataset=dataset)

    # check the results of each spec
    for spec_name, spec in dataset.qc_specifications.items():
        query = list(ds.iterate_records(specification_names=[spec_name]))
        assert len(query) == 2
        for name, _, record in query:
            assert record.status == RecordStatusEnum.complete
            assert record.error is None
            assert record.return_result is not None
            assert record.specification.dict(
                include={"method", "program", "basis"}
            ) == spec.dict(include={"method", "program", "basis"})


@pytest.mark.parametrize(
    "solvent_model, solvent_energy, solvent_evidence",
    [
        pytest.param(
            PCMSettings(units="au", medium_Solvent="water"),
            "pcm polarization energy",
            "Solvent name:          Water",
            id="PCM",
        ),
        pytest.param(
            DDXSettings(ddx_solvent_epsilon=4),
            "dd solvation energy",
            "solvent_epsilon         = 4.0",
            id="DDX Epsilon",
        ),
        pytest.param(
            DDXSettings(ddx_solvent="1-bromooctane"),
            "dd solvation energy",
            "solvent_epsilon         = 5.0244",
            id="DDX Solvent",
        ),
    ],
)
def test_basic_submissions_single_solvent_spec(
    fulltest_client, solvent_model, solvent_energy, solvent_evidence, water
):
    """Test submitting a basic dataset to a snowflake server with pcm water in the specification."""

    client = fulltest_client

    program = "psi4"
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    factory = BasicDatasetFactory(driver="energy")
    factory.add_qc_spec(
        method="hf",
        basis="sto-3g",
        program=program,
        spec_name="default",
        spec_description="testing the single points with pcm",
        implicit_solvent=solvent_model,
        overwrite=True,
    )

    # only use one molecule due to the time it takes to run with pcm
    dataset = factory.create_dataset(
        dataset_name="Test single points with pcm water",
        molecules=water,
        description="Test basics dataset with pcm water",
        tagline="Testing single point datasets with pcm water",
    )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset with pcm water"

    # now submit again
    dataset.submit(client=client)

    await_results(client)

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    check_metadata(ds, dataset)

    # check the qc spec
    check_added_specs(ds=ds, dataset=dataset)

    for spec_name, spec in dataset.qc_specifications.items():
        query = list(
            ds.iterate_records(
                specification_names=spec_name,
            )
        )
        assert len(query) == 1  # only used 1 molecule above
        for name, _, record in query:
            assert record.status == RecordStatusEnum.complete
            assert record.error is None
            assert record.return_result is not None
            # make sure the PCM result was captured
            assert record.properties[solvent_energy] < 0
            # make sure the correct solvent was used
            assert solvent_evidence in record.stdout
            assert record.specification.dict(
                include={"method", "basis", "program"}
            ) == spec.dict(include={"method", "basis", "program"})


@pytest.mark.parametrize(
    "dataset_data",
    [
        pytest.param((BasicDatasetFactory, BasicDataset), id="Dataset"),
        pytest.param(
            (OptimizationDatasetFactory, OptimizationDataset), id="OptimizationDataset"
        ),
        pytest.param(
            (TorsiondriveDatasetFactory, TorsiondriveDataset), id="TorsiondriveDataset"
        ),
    ],
)
def test_adding_compute(fulltest_client, dataset_data):
    """
    Test adding new compute to each of the dataset types using none psi4 programs.
    """
    client = fulltest_client
    mol = Molecule.from_smiles("CO")
    factory_type, dataset_type = dataset_data
    # make and clear out the qc specs
    factory = factory_type()
    factory.clear_qcspecs()
    factory.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="default",
        spec_description="default spec for openff",
    )
    dataset = factory.create_dataset(
        dataset_name=f"Test adding compute to {factory_type}",
        molecules=mol,
        description=f"Testing adding compute to a {dataset_type} dataset",
        tagline="tests for adding compute.",
    )

    # Submit the initial openFF compute
    dataset.submit(client=client)
    # make sure that the compute has finished
    await_services(fulltest_client, max_iter=30)

    # make a dataset with new compute and submit it
    # transfer the metadata to compare the elements
    compute_dataset = dataset_type(
        dataset_name=dataset.dataset_name,
        metadata=dataset.metadata,
        dataset_tagline=dataset.dataset_tagline,
        description=dataset.description,
    )
    compute_dataset.clear_qcspecs()
    # now add the new compute spec
    compute_dataset.add_qc_spec(
        method="uff",
        basis=None,
        program="rdkit",
        spec_name="rdkit",
        spec_description="rdkit basic spec",
    )

    # make sure the dataset has no molecules and submit it
    assert compute_dataset.dataset == {}
    # this should expand the compute of the initial dataset
    compute_dataset.submit(client=client)
    # make sure that the compute has finished
    await_services(fulltest_client, max_iter=30)

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # check the metadata
    check_metadata(ds, dataset)

    # update all specs into one dataset for comparison
    dataset.add_qc_spec(**compute_dataset.qc_specifications["rdkit"].dict())

    # For each dataset type check the compute result
    if dataset.type == "DataSet":
        # check the basic dataset specs
        check_added_specs(ds=ds, dataset=dataset)
        # Make sure the compute for this spec has finished and matches what we requested
        for spec_name, spec in dataset.qc_specifications.items():
            query = ds.iterate_records(specification_names=spec_name)
            for entry_name, _, rec in query:
                assert rec.status.value.upper() == "COMPLETE"
                assert rec.error is None
                assert rec.return_result is not None
                assert rec.specification.program == spec.program
                assert rec.specification.method == spec.method
                assert rec.specification.basis == spec.basis
                assert rec.specification.driver == dataset.driver

    elif dataset.type == "OptimizationDataset":
        # check the qc spec
        for spec_name, specification in ds.specifications.items():
            spec = dataset.qc_specifications[spec_name]
            s = specification.specification
            assert s.qc_specification.driver == dataset.driver
            assert s.qc_specification.program == spec.program
            assert s.qc_specification.method == spec.method
            assert s.qc_specification.basis == spec.basis
            assert specification.description == spec.spec_description

            # check the keywords
            got = s.keywords
            want = dataset._get_specifications()[spec_name].keywords
            assert got == want

            # query the dataset
            query = ds.iterate_records(specification_names="default")

            for name, spec, record in query:
                input_spec = dataset.qc_specifications[spec]
                assert record.status == RecordStatusEnum.complete
                assert record.error is None
                assert len(record.trajectory) > 1
                # check the specification of a result in the opt
                opt_single_point = record.trajectory[-1]
                assert opt_single_point.specification.program == input_spec.program
                assert opt_single_point.specification.method == input_spec.method
                assert opt_single_point.specification.basis == input_spec.basis

    if dataset.type == "TorsionDriveDataset":
        # check the qc spec
        for spec_name, specification in ds.specifications.items():
            spec = dataset.qc_specifications[spec_name]
            s = specification.specification
            assert (
                s.optimization_specification.qc_specification.driver == dataset.driver
            )
            assert s.optimization_specification.program == "geometric"
            assert s.optimization_specification.qc_specification.program == spec.program
            assert s.optimization_specification.qc_specification.method == spec.method
            assert s.optimization_specification.qc_specification.basis == spec.basis
            assert specification.description == spec.spec_description

            # check the keywords
            got = s.keywords
            want = dataset._get_specifications()[spec_name].keywords
            assert got == want

            # query the dataset
            query = ds.iterate_records(specification_names="default")

            for name, spec, record in query:
                assert record.status == RecordStatusEnum.complete
                assert record.error is None
                assert len(record.trajectory) > 1


def test_basic_submissions_wavefunction(fulltest_client, conformer_water):
    """
    Test submitting a basic dataset with a wavefunction protocol and make sure it is executed.
    """
    # only a psi4 test
    if not has_program("psi4"):
        pytest.skip("Program psi4 not found.")

    client = fulltest_client

    factory = BasicDatasetFactory(driver="energy")
    factory.clear_qcspecs()
    factory.add_qc_spec(
        method="hf",
        basis="sto-3g",
        program="psi4",
        spec_name="default",
        spec_description="wavefunction spec",
        store_wavefunction="orbitals_and_eigenvalues",
    )

    dataset = factory.create_dataset(
        dataset_name="Test single points with wavefunction",
        molecules=conformer_water,
        description="Test basics dataset",
        tagline="Testing single point datasets with wavefunction",
    )

    # submit the dataset
    # now submit again
    dataset.submit(client=client)

    await_results(client)

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # check the metadata
    check_metadata(ds, dataset)

    # get the last ran spec
    check_added_specs(ds=ds, dataset=dataset)

    query = list(
        ds.iterate_records(
            specification_names="default",
        )
    )
    assert len(query) == 2
    for _, _, result in query:
        assert result.status == RecordStatusEnum.complete
        assert result.error is None
        assert result.return_result is not None
        wavefunction = result.wavefunction
        assert wavefunction.basis.name.lower() == "sto-3g"
        assert wavefunction.scf_orbitals_a is not None


def test_optimization_submissions_with_constraints(fulltest_client):
    """
    Make sure that the constraints are added to the optimization and enforced.
    """
    client = fulltest_client
    ethane = Molecule.from_file(get_data("ethane.sdf"), "sdf")
    dataset = OptimizationDataset(
        dataset_name="Test optimizations with constraint",
        description="Test optimization dataset with constraints",
        dataset_tagline="Testing optimization datasets",
    )
    # add just mm spec
    dataset.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="default",
        spec_description="mm default spec",
        overwrite=True,
    )
    # build some constraints
    constraints = Constraints()
    constraints.add_set_constraint(
        constraint_type="dihedral", indices=[2, 0, 1, 5], value=60, bonded=True
    )
    constraints.add_freeze_constraint(
        constraint_type="distance", indices=[0, 1], bonded=True
    )
    # add the molecule
    index = ethane.to_smiles()
    dataset.add_molecule(index=index, molecule=ethane, constraints=constraints)

    # now submit again
    dataset.submit(client=client)

    await_results(client, check_fn=PortalClient.get_optimizations)

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )
    query = ds.iterate_records(specification_names="default")
    for name, spec, record in query:
        assert record.status is RecordStatusEnum.complete
        assert record.error is None
        assert len(record.trajectory) > 1
        break
    else:
        raise RuntimeError("The requested compute was not found")

    # now make sure the constraints worked
    final_molecule = record.final_molecule
    initial_molecule = record.initial_molecule
    assert pytest.approx(final_molecule.measure((2, 0, 1, 5)), abs=1e-2) == 60
    assert initial_molecule.measure((0, 1)) == pytest.approx(
        final_molecule.measure((0, 1))
    )


@pytest.mark.parametrize(
    "specification",
    [
        pytest.param(
            ({"method": "hf", "basis": "3-21g", "program": "psi4"}, "gradient"),
            id="PSI4 hf 3-21g gradient",
        ),
        pytest.param(
            (
                {
                    "method": "openff_unconstrained-1.0.0",
                    "basis": "smirnoff",
                    "program": "openmm",
                },
                "gradient",
            ),
            id="SMIRNOFF openff_unconstrained-1.0.0 gradient",
        ),
        pytest.param(
            ({"method": "uff", "basis": None, "program": "rdkit"}, "gradient"),
            id="RDKit UFF gradient",
        ),
    ],
)
def test_optimization_submissions(fulltest_client, specification):
    """Test submitting an Optimization dataset to a snowflake server."""

    client = fulltest_client

    qc_spec, driver = specification
    program = qc_spec["program"]
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = OptimizationDatasetFactory(driver=driver)
    factory.add_qc_spec(
        **qc_spec, spec_name="default", spec_description="test", overwrite=True
    )

    dataset = factory.create_dataset(
        dataset_name=f"Test optimizations info {program}, {driver}",
        molecules=molecules[:2],
        description="Test optimization dataset",
        tagline="Testing optimization datasets",
    )

    # force a metadata validation error
    dataset.metadata.long_description = None

    # only save final gradients, results
    dataset.protocols = OptimizationProtocols(trajectory="initial_and_final")

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test optimization dataset"

    # now submit again
    dataset.submit(client=client)

    await_results(
        client, check_fn=PortalClient.get_optimizations, timeout=240, ids=[1, 2]
    )

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # check the metadata
    check_metadata(ds, dataset)

    # check the qc spec
    # ds is a qcportal OptimizationDataset, and dataset is our
    # OptimizationDataset, kinda confusing
    for spec_name, specification in ds.specifications.items():
        spec = dataset.qc_specifications[spec_name]

        s = specification.specification
        assert s.qc_specification.driver == dataset.driver
        assert s.qc_specification.program == spec.program
        assert s.qc_specification.method == spec.method
        assert s.qc_specification.basis == spec.basis
        assert specification.description == spec.spec_description

        # check the keywords
        got = s.keywords
        want = dataset._get_specifications()[spec_name].keywords
        assert got == want

    for spec in dataset.qc_specifications.values():
        # query the dataset
        query = ds.iterate_records(specification_names="default")

        for name, spec, record in query:
            assert record.status == RecordStatusEnum.complete
            assert record.error is None

            # since we only chose to keep `initial_and_final` trajectory,
            # should only have two results
            assert len(record.trajectory) == 2
            # if we used psi4 make sure the properties were captured
            if program == "psi4":
                result = record.trajectory[0]
                assert "current dipole" in result.properties.keys()
                assert "scf quadrupole" in result.properties.keys()


@pytest.mark.parametrize(
    "opt_keywords",
    [
        pytest.param(
            (
                "CUSTOM",
                [
                    "energy",
                    "1e-8",
                    "maxiter",
                ],
                3,
                "custom convergence with maxiter",
            ),
            id="Custom convergence with maxiter",
        ),
        pytest.param(
            (
                "GAU_VERYTIGHT",
                [
                    "maxiter",
                ],
                3,
                "Default conv with maxiter",
            ),
            id="Default conv with maxiter",
        ),
        pytest.param(
            (
                "CUSTOM",
                [
                    "energy",
                    "1e-4",
                    "grms",
                    "3e-2",
                    "gmax",
                    "4.5e-1",
                    "drms",
                    "1.2e-3",
                    "dmax",
                    "1.8e-1",
                ],
                300,
                "Custom convergence, no maxiter",
            ),
            id="Custom convergence, no maxiter",
        ),
    ],
)
def test_optimization_submissions_convergence(fulltest_client, opt_keywords):
    """Test submitting an Optimization dataset with custom convergence options."""

    client = fulltest_client

    convergence_set, converge, maxit, ds_suffix = opt_keywords

    ethane = Molecule.from_file(get_data("ethane.sdf"), "sdf")

    dataset = OptimizationDataset(
        dataset_name="Test optimizations with converge " + ds_suffix,
        description="Test optimization dataset with constraints" + ds_suffix,
        dataset_tagline="Testing optimization datasets" + ds_suffix,
    )

    dataset.clear_qcspecs()
    dataset.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="test_spec",
        spec_description="test_spec",
        overwrite=True,
    )

    # add the molecule
    index = ethane.to_smiles()
    dataset.add_molecule(index=index, molecule=ethane)


    # Add the GeometricProcedure so we can submit the dataset
    dataset.optimization_procedure = GeometricProcedure(
        program="geometric",
        maxiter=maxit,
        convergence_set=convergence_set,
        converge=converge,
    )

    # only save final gradients, results, if --converge maxiter not requested
    if "maxiter" not in converge:
        dataset.protocols = OptimizationProtocols(trajectory="initial_and_final")

    # now submit
    dataset.submit(client=client)

    await_results(
        client,
        check_fn=PortalClient.get_optimizations,
        timeout=240,  
    )

    # make sure of the results are complete
    ds = client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    query = ds.iterate_records(specification_names="test_spec")

    # Dictionary to help parse GeomeTRIC output, relating the `--converge` keywords to output prints
    keyword_to_stdout = {
        'energy': r"\|Delta-E\|",
        'grms':r"RMS-Grad ", 
        'gmax':r"Max-Grad ",
        'drms':r"RMS-Disp ",
        'dmax':r"Max-Disp "
    }
    convergence_set_to_value = {
        'GAU_VERYTIGHT': ['energy' ,'1.0e-6','grms','1.0e-6','gmax','2.0e-6','drms','4.0e-6','dmax','6.0e-6'],
        'CUSTOM': converge
    }

    for name, spec, record in query:
        assert record.status == RecordStatusEnum.complete
        assert record.error is None

        # Check that the converge keywords were passed to the record's input
        assert [
            key.lower() for key in record.specification.keywords["converge"]
        ] == dataset.optimization_procedure.converge

        if convergence_set != 'CUSTOM':
            # Confirm that convergence_set was passed to record input
            assert record.specification.keywords["convergence_set"] == dataset.optimization_procedure.convergence_set   

        else:   
            # Confirm that convergence_set is absent from record input
            assert 'convergence_set' not in record.specification.keywords

        # Parse the geomeTRIC output to check the convergence criteria were passed to GeomeTRIC
        geometric_output = record.stdout

        # Confirm that GeomeTRIC is using the requested convergence criteria
        for i,key in enumerate(convergence_set_to_value[convergence_set]):
            try: float(key) # Only want to check the string flags
            except ValueError:
                if key != 'maxiter':
                    output_key = keyword_to_stdout[key]
                    matches = re.findall(r"{} \< \d\.\d\de\-\d\d".format(output_key),geometric_output)
                    assert len(matches) == 1
                    assert float(matches[0].split()[-1]) == float(convergence_set_to_value[convergence_set][i+1])

        # Checking --converge maxiter
        using_converge_maxiter = (len(re.findall(r"Converge-on-maxiter set: Will exit with success if maximum number of iterations \({}\) is reached".format(maxit),geometric_output)) == 1)
        converged_due_to_maxiter = (len(re.findall(r"Exiting normally because --converge maxiter was set",geometric_output)) == 1)

        if "maxiter" in converge:
            # Length of trajectory is the number of steps. Should be equal to maxiter + 1
            # if --converge maxiter was requested
            assert len(record.trajectory) == dataset.optimization_procedure.maxiter + 1

            # Confirm that maxiter is set and the correct max number of iterations was passed
            assert using_converge_maxiter

            # Confirm that it actually did exit due to maxiter
            assert converged_due_to_maxiter
       
        else:
            # If not using maxiter, should only have two results
            # since we only chose to keep `initial_and_final` trajectory
            assert len(record.trajectory) == 2  

            # Confirm that maxiter is NOT set
            assert not using_converge_maxiter

            # Confirm that it did NOT exit due to maxiter
            assert not converged_due_to_maxiter

@pytest.mark.xfail(
    reason="Known issue with recent versions of pcm https://github.com/PCMSolver/pcmsolver/issues/206"
)
def test_optimization_submissions_with_pcm(fulltest_client):
    """Test submitting an Optimization dataset to a snowflake server with PCM."""
    program = "psi4"
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    # use a single small molecule due to the extra time PCM takes
    molecules = Molecule.from_smiles("N")

    factory = OptimizationDatasetFactory(driver="gradient")
    factory.add_qc_spec(
        method="hf",
        basis="sto-3g",
        program=program,
        spec_name="default",
        spec_description="test",
        implicit_solvent=PCMSettings(units="au", medium_Solvent="water"),
        overwrite=True,
    )

    dataset = factory.create_dataset(
        dataset_name="Test optimizations info with pcm water",
        molecules=molecules,
        description="Test optimization dataset",
        tagline="Testing optimization datasets",
    )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=fulltest_client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=fulltest_client)

    await_services(fulltest_client, max_iter=240)
    # snowflake.await_results()

    # make sure of the results are complete
    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # check the metadata
    check_metadata(ds, dataset)

    # check the qc spec
    for spec_name, specification in ds.specifications.items():
        spec = dataset.qc_specifications[spec_name]

        s = specification.specification
        assert s.qc_specification.driver == dataset.driver
        assert s.qc_specification.program == spec.program
        assert s.qc_specification.method == spec.method
        assert s.qc_specification.basis == spec.basis
        assert specification.description == spec.spec_description

        # check the keywords
        got = s.keywords
        want = dataset._get_specifications()[spec_name].keywords
        assert got == want

        # query the dataset
        query = ds.iterate_records(specification_names="default")

        for name, spec, record in query:
            assert record.status == RecordStatusEnum.complete
            assert record.error is None
            assert len(record.trajectory) > 1
            result = record.trajectory[-1]

            assert "scf dipole" in result.properties.keys()
            assert "scf quadrupole" in result.properties.keys()
            # make sure the PCM result was captured
            assert result.properties["pcm polarization energy"] < 0


def test_torsiondrive_scan_keywords(fulltest_client):
    """
    Test running torsiondrives with unique keyword settings which overwrite the global grid spacing and scan range.
    """
    molecules = Molecule.from_smiles("CO")
    factory = TorsiondriveDatasetFactory()
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[*:1]~[#6:2]-[#8:3]~[*:4]")
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    factory.add_qc_spec(
        method="openff_unconstrained-1.1.0",
        basis="smirnoff",
        program="openmm",
        spec_description="scan range test",
        spec_name="openff-1.1.0",
    )
    dataset = factory.create_dataset(
        dataset_name="Torsiondrive scan keywords",
        molecules=molecules,
        description="Testing scan keywords which overwrite the global settings",
        tagline="Testing scan keywords which overwrite the global settings",
    )

    # now set the keywords
    keys = list(dataset.dataset.keys())
    entry = dataset.dataset[keys[0]]
    entry.keywords = {"grid_spacing": [5], "dihedral_ranges": [(-10, 10)]}

    # now submit
    dataset.submit(client=fulltest_client)
    await_services(fulltest_client, max_iter=30)

    # make sure of the results are complete
    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # get the entry
    query = list(ds.iterate_records(specification_names="openff-1.1.0"))
    assert len(query) == 1  # only used 1 molecule above
    for _, _, record in query:
        assert record.status == RecordStatusEnum.complete
        assert record.error is None
        assert record.final_energies is not None
        assert record.specification.keywords.grid_spacing == [5]
        assert record.specification.keywords.grid_spacing != dataset.grid_spacing
        assert record.specification.keywords.dihedral_ranges == [(-10, 10)]
        assert record.specification.keywords.dihedral_ranges != dataset.dihedral_ranges


def test_torsiondrive_constraints(fulltest_client):
    """
    Make sure constraints are correctly passed to optimisations in torsiondrives.
    """

    # client = snowflake.client()
    molecule = Molecule.from_file(get_data("3_torsions.sdf"))
    dataset = TorsiondriveDataset(
        dataset_name="Torsiondrive constraints",
        dataset_tagline="Testing torsiondrive constraints",
        description="Testing torsiondrive constraints.",
    )
    dataset.clear_qcspecs()
    dataset.add_qc_spec(
        method="uff",
        basis=None,
        program="rdkit",
        spec_name="uff",
        spec_description="tdrive constraints",
    )
    # use a restricted range to keep the scan fast
    dataset.add_molecule(
        index="1",
        molecule=molecule,
        attributes=MoleculeAttributes.from_openff_molecule(molecule=molecule),
        dihedrals=[(0, 1, 3, 4)],
        keywords={"dihedral_ranges": [(-5, 20)]},
    )
    entry = dataset.dataset["1"]
    # add the constraints
    entry.add_constraint(
        constraint="freeze", constraint_type="dihedral", indices=[7, 0, 1, 5]
    )
    entry.add_constraint(
        constraint="freeze", constraint_type="dihedral", indices=[0, 1, 5, 6]
    )

    dataset.submit(client=fulltest_client)
    await_services(fulltest_client, max_iter=300)

    # make sure the result is complete
    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    query = ds.iterate_records(
        specification_names="uff",
    )
    for name, spec, record in query:
        constraints = record.optimizations[(0,)][0].specification.keywords[
            "constraints"
        ]
        # constraints = opt.keywords["constraints"]
        # make sure both the freeze and set constraints are passed on
        assert "set" in constraints
        assert "freeze" in constraints
        # make sure both freeze constraints are present
        assert len(constraints["freeze"]) == 2
        assert constraints["freeze"][0]["indices"] == [5, 1, 0, 7]
        # make sure the dihedral has not changed
        assert pytest.approx(
            record.minimum_optimizations[(0,)].final_molecule.measure((5, 1, 0, 7)),
            abs=1e-2,
        ) == record.initial_molecules[0].measure((5, 1, 0, 7))


@pytest.mark.parametrize(
    "specification",
    [
        pytest.param(
            (
                {
                    "method": "openff_unconstrained-1.1.0",
                    "basis": "smirnoff",
                    "program": "openmm",
                },
                "gradient",
            ),
            id="SMIRNOFF openff_unconstrained-1.0.0 gradient",
        ),
        pytest.param(
            ({"method": "mmff94", "basis": None, "program": "rdkit"}, "gradient"),
            id="RDKit mmff94 gradient",
        ),
    ],
)
def test_torsiondrive_submissions(fulltest_client, specification):
    """
    Test submitting a torsiondrive dataset and computing it.
    """

    # client = snowflake.client()

    qc_spec, driver = specification
    program = qc_spec["program"]
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecule = Molecule.from_mapped_smiles("[H:1][C:2]([H:3])([H:4])[O:5][H:6]")

    factory = TorsiondriveDatasetFactory(driver=driver)
    factory.add_qc_spec(
        **qc_spec, spec_name="default", spec_description="test", overwrite=True
    )

    dataset = factory.create_dataset(
        dataset_name=f"Test torsiondrives info {program}, {driver}",
        molecules=[],
        description="Test torsiondrive dataset",
        tagline="Testing torsiondrive datasets",
    )
    dataset.add_molecule(
        index="foo",
        molecule=molecule,
        dihedrals=[[0, 1, 4, 5]],
        keywords={"dihedral_ranges": [(-180, 91)], "grid_spacing": [180]},
    )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=fulltest_client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=fulltest_client)

    await_services(fulltest_client, max_iter=120)
    # snowflake.await_services(max_iter=50)

    # make sure of the results are complete
    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    # check the metadata
    check_metadata(ds, dataset)

    for spec_name, specification in ds.specifications.items():
        spec = dataset.qc_specifications[spec_name]

        s = specification.specification.optimization_specification

        assert s.qc_specification.driver == dataset.driver
        assert s.qc_specification.program == spec.program
        assert s.qc_specification.method == spec.method
        assert s.qc_specification.basis == spec.basis

        assert specification.description == spec.spec_description

        # check the torsiondrive spec keywords
        got = ds.specifications[spec_name].specification.keywords
        want = dataset._get_specifications()[spec_name].keywords
        assert got == want

        # check the qc spec keywords
        got = ds.specifications[
            spec_name
        ].specification.optimization_specification.qc_specification.keywords
        want = dataset._get_specifications()[
            spec_name
        ].optimization_specification.qc_specification.keywords
        assert "maxiter" in got
        assert "scf_properties" in got
        assert got == want

        #

        # query the dataset
        for entry_name, spec_name, record in ds.iterate_records():
            # this will take some time so make sure it is running with no error
            assert record.status.value == "complete", print(record.dict())
            assert record.error is None
            assert len(record.final_energies) == 2


@pytest.mark.parametrize(
    "factory_type",
    [
        pytest.param(BasicDatasetFactory, id="BasicDataset ignore_errors"),
        pytest.param(
            OptimizationDatasetFactory, id="OptimizationDataset ignore_errors"
        ),
        pytest.param(
            TorsiondriveDatasetFactory, id="TorsiondriveDataset ignore_errors"
        ),
    ],
)
def test_ignore_errors_all_datasets(fulltest_client, factory_type, capsys):
    """
    For each dataset make sure that when the basis is not fully covered the dataset raises warning errors, and verbose information
    """

    # molecule containing boron
    molecule = Molecule.from_smiles("OB(O)C1=CC=CC=C1")
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[#6:1]~[#6:2]-[B:3]~[#8:4]")
    factory = factory_type()
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="parsley",
        spec_description="standard parsley spec",
    )
    dataset = factory.create_dataset(
        dataset_name=f"Test ignore_error for {factory.type}",
        molecules=molecule,
        description="Test ignore errors dataset",
        tagline="Testing ignore errors datasets",
    )

    # make sure the dataset raises an error here
    with pytest.raises(MissingBasisCoverageError):
        dataset.submit(client=fulltest_client, ignore_errors=False)

    # now we want to try again and make sure warnings are raised
    with pytest.warns(UserWarning):
        dataset.submit(client=fulltest_client, ignore_errors=True, verbose=True)

    info = capsys.readouterr()
    assert (
        info.out == f"Number of new entries: {dataset.n_records}/{dataset.n_records}\n"
    )


@pytest.mark.parametrize(
    "factory_type",
    [
        pytest.param(BasicDatasetFactory, id="Basicdataset"),
        pytest.param(OptimizationDatasetFactory, id="Optimizationdataset"),
    ],
)
def test_index_not_changed(fulltest_client, factory_type):
    """
    Make sure that when we submit molecules from a dataset/optimizationdataset with one input conformer that the index is not changed.
    """
    factory = factory_type()
    factory.clear_qcspecs()

    # add only mm specs
    factory.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="parsley",
        spec_description="standard parsley spec",
    )

    molecule = Molecule.from_smiles("C")
    # make sure we only have one conformer
    molecule.generate_conformers(n_conformers=1)
    dataset = factory.create_dataset(
        dataset_name=f"Test index change for {factory.type}",
        molecules=molecule,
        description="Test index change dataset",
        tagline="Testing index changes datasets",
    )

    # now change the index name to something unique
    entry = dataset.dataset.pop(list(dataset.dataset.keys())[0])
    entry.index = "my_unique_index"
    dataset.dataset[entry.index] = entry

    dataset.submit(client=fulltest_client)

    # pull the dataset and make sure our index is present
    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )

    if dataset.type == "DataSet":
        query = ds.get_record("my_unique_index", "parsley")
        assert query is not None
    else:
        assert "my_unique_index" in ds.entry_names


@pytest.mark.parametrize(
    "factory_type",
    [
        pytest.param(OptimizationDatasetFactory, id="OptimizationDataset index clash"),
        pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDataset index clash"),
    ],
)
def test_adding_dataset_entry_fail(fulltest_client, factory_type, capsys):
    """
    Make sure that the new entries is not incremented if we can not add a molecule to the server due to a name clash.
    TODO add basic dataset into the testing if the api changes to return an error when adding the same index twice
    """
    # client = snowflake.client()
    molecule = Molecule.from_smiles("CO")
    molecule.generate_conformers(n_conformers=1)
    factory = factory_type()
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[*:1]~[#6:2]-[#8:3]~[*:4]")
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="parsley",
        spec_description="standard parsley spec",
    )
    dataset = factory.create_dataset(
        dataset_name=f"Test index clash for {factory.type}",
        molecules=molecule,
        description="Test ignore errors dataset",
        tagline="Testing ignore errors datasets",
    )

    # make sure all expected index get submitted
    dataset.submit(client=fulltest_client, verbose=True)
    info = capsys.readouterr()
    assert (
        info.out == f"Number of new entries: {dataset.n_records}/{dataset.n_records}\n"
    )

    # now add a new spec and try and submit again
    dataset.clear_qcspecs()
    dataset.add_qc_spec(
        method="mmff94",
        basis=None,
        program="rdkit",
        spec_name="mff94",
        spec_description="mff94 force field in rdkit",
    )
    dataset.submit(client=fulltest_client, verbose=True)
    info = capsys.readouterr()
    assert info.out == f"Number of new entries: 0/{dataset.n_records}\n"


@pytest.mark.parametrize(
    "factory_type",
    [
        pytest.param(
            OptimizationDatasetFactory, id="OptimizationDataset expand compute"
        ),
        pytest.param(
            TorsiondriveDatasetFactory, id="TorsiondriveDataset expand compute"
        ),
    ],
)
def test_expanding_compute(fulltest_client, factory_type):
    """
    Make sure that if we expand the compute of a dataset tasks are generated.
    """
    # client = snowflake.client()
    molecule = Molecule.from_smiles("CC")
    molecule.generate_conformers(n_conformers=1)
    factory = factory_type()
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[*:1]~[#6:2]-[#6:3]~[*:4]")
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="default",
        spec_description="standard parsley spec",
    )
    dataset = factory.create_dataset(
        dataset_name=f"Test compute expand {factory.type}",
        molecules=molecule,
        description="Test compute expansion",
        tagline="Testing compute expansion",
    )

    # make sure all expected index get submitted
    dataset.submit(client=fulltest_client)
    # grab the dataset and check the history
    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )
    assert ds.specifications.keys() == {"default"}

    # now make another dataset to expand the compute
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(
        method="openff-1.2.0",
        basis="smirnoff",
        program="openmm",
        spec_name="parsley2",
        spec_description="standard parsley spec",
    )
    dataset = factory.create_dataset(
        dataset_name=f"Test compute expand {factory.type}",
        molecules=[],
        description="Test compute expansion",
        tagline="Testing compute expansion",
    )
    # now submit again
    dataset.submit(client=fulltest_client)

    # now grab the dataset again and check the tasks list
    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )
    assert ds.specifications.keys() == {"default", "parsley2"}
    # make sure a record has been made
    assert len([*ds.iterate_records()]) == 2


@pytest.mark.parametrize(
    "factory_type,result_collection_type",
    [
        [BasicDatasetFactory, BasicResultCollection],
        [OptimizationDatasetFactory, OptimizationResultCollection],
        [TorsiondriveDatasetFactory, TorsionDriveResultCollection],
    ],
)
def test_invalid_cmiles(fulltest_client, factory_type, result_collection_type):
    molecule = Molecule.from_mapped_smiles("[H:4][C:2](=[O:1])[O:3][H:5]")
    molecule.generate_conformers(n_conformers=1)
    factory = factory_type()
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(
        method="openff-1.0.0",
        basis="smirnoff",
        program="openmm",
        spec_name="default",
        spec_description="standard parsley spec",
    )
    dataset = factory.create_dataset(
        dataset_name=f"Test invalid cmiles {factory.type}",
        molecules=[],
        description="Test invalid cmiles",
        tagline="Testing invalid cmiles",
    )
    if factory_type is TorsiondriveDatasetFactory:
        dataset.add_molecule(
            index="foo",
            molecule=molecule,
            dihedrals=[[0, 1, 2, 4]],
            keywords={"dihedral_ranges": [(0, 20)], "grid_spacing": [15]},
        )
    else:
        dataset.add_molecule(index="foo", molecule=molecule)

    dataset.submit(client=fulltest_client)
    if factory_type is BasicDatasetFactory:
        await_results(fulltest_client)
    else:
        await_services(fulltest_client, max_iter=120)

    ds = fulltest_client.get_dataset(
        legacy_qcsubmit_ds_type_to_next_qcf_ds_type[dataset.type], dataset.dataset_name
    )
    assert ds.specifications.keys() == {"default"}
    results = result_collection_type.from_datasets(datasets=ds)
    assert results.n_molecules == 1
    records = results.to_records()
    assert len(records) == 1
    # Single points and optimizations look here
    fulltest_client.modify_molecule(
        1,
        identifiers={
            "canonical_isomeric_explicit_hydrogen_mapped_smiles": "[H:4][C:2](=[O:1])[OH:3]"
        },
        overwrite_identifiers=True,
    )
    # Do this to flush the local cache and fetch the modified molecule from the server
    entries = [*ds.iterate_entries(force_refetch=True)]
    # Torsiondrives look here
    entries[0].attributes[
        "canonical_isomeric_explicit_hydrogen_mapped_smiles"
    ] = "[H:4][C:2](=[O:1])[OH:3]"
    ds._cache_data.update_entries(entries)
    results = result_collection_type.from_datasets(datasets=ds)
    assert results.n_molecules == 1
    with pytest.warns(UserWarning, match="invalid CMILES"):
        records = results.to_records()
    assert len(records) == 0
