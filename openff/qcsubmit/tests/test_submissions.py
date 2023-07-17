"""
Test submissions to a local qcarchive instance using different compute backends, RDKit, OpenMM, PSI4

Here we use the qcfractal snowflake fixture to set up the database.
"""

import pytest
from openff.toolkit.topology import Molecule
from qcengine.testing import has_program
from qcfractalcompute.testing_helpers import QCATestingComputeThread
from qcportal import PortalClient

from openff.qcsubmit import workflow_components
from openff.qcsubmit.common_structures import Metadata, MoleculeAttributes, PCMSettings
from openff.qcsubmit.constraints import Constraints
from openff.qcsubmit.datasets import (
    BasicDataset,
    OptimizationDataset,
    TorsiondriveDataset,
)
from openff.qcsubmit.exceptions import (
    DatasetInputError,
    MissingBasisCoverageError,
    QCSpecificationError,
)
from openff.qcsubmit.factories import (
    BasicDatasetFactory,
    OptimizationDatasetFactory,
    TorsiondriveDatasetFactory,
)
from openff.qcsubmit.utils import get_data

def await_results(fulltest_client):
    import time

    from qcportal.record_models import RecordStatusEnum
    for i in range(120):
        time.sleep(1)
        #rec = fulltest_client.get_singlepoints(ids[0])
        #print(type(dataset))
        #print(dir(dataset))
        rec = fulltest_client.get_singlepoints(1)
        if rec.status not in [RecordStatusEnum.running, RecordStatusEnum.waiting]:
            break
    else:
        raise RuntimeError("Did not finish calculation in time")


@pytest.mark.parametrize("specification", [
    #pytest.param(({"method": "hf", "basis": "3-21g", "program": "psi4"}, "energy"), id="PSI4 hf 3-21g energy"),
    pytest.param(({"method": "smirnoff99Frosst-1.1.0", "basis": "smirnoff", "program": "openmm"}, "energy"), id="SMIRNOFF smirnoff99Frosst-1.1.0 energy"),
    #pytest.param(({"method": "uff", "basis": None, "program": "rdkit"}, "gradient"), id="RDKit UFF gradient")
])
def test_basic_submissions_single_spec(fulltest_client, specification):
    """Test submitting a basic dataset to a snowflake server."""


    #client = snowflake.client()
    client = fulltest_client

    qc_spec, driver = specification

    program = qc_spec["program"]
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = BasicDatasetFactory(driver=driver)
    factory.add_qc_spec(**qc_spec, spec_name="default",
                        spec_description="testing the single points",
                        overwrite=True)

    dataset = factory.create_dataset(dataset_name=f"Test single points info {program}, {driver}",
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
    #ct = QCATestingComputeThread(snowflake.config)
    #snowflake.activate_manager()
    #snowflake.start_job_runner()

    #fulltest_client.await_results()
    await_results(fulltest_client)

    # make sure of the results are complete
    #ds = client.get_dataset("Dataset", dataset.dataset_name)
    ds = client.get_dataset("singlepoint", dataset.dataset_name)
    #ds = client.get_dataset_by_id(1)

    # check the metadata
    #meta = Metadata(**ds.data.metadata)
    meta = ds.metadata
    #assert meta == dataset.metadata
    print(f'{meta=}')
    print(f'{dataset.metadata=}')

    assert meta['long_description'] == dataset.description
    assert meta['short_description'] == dataset.dataset_tagline
    assert ds.tags == dataset.dataset_tags

    # check the provenance
    assert ds.provenance == dataset.provenance

    # check the qc spec
    #assert ds.data.default_driver == dataset.driver

    # get the last ran spec
    print(f"{ds.specifications=}")
    for spec_name, specification in ds.specifications.items():# data.history:
        print(f'{specification=}')
        #driver, program, method, basis, spec_name = specification
        spec = dataset.qc_specifications[spec_name]
        assert specification.specification.driver == dataset.driver
        assert specification.specification.program == spec.program
        assert specification.specification.method == spec.method
        assert specification.specification.basis == spec.basis
        break
    else:
        raise RuntimeError(f"The requested compute was not found in the history {ds.data.history}")

    for spec in dataset.qc_specifications.values():
        #query = ds.get_records(
        query=ds.iterate_records(
            specification_names="default",
            #method=spec.method,
            #basis=spec.basis,
            #program=spec.program,
        )
        # make sure all of the conformers were submitted
        assert len(list(query)) == len(molecules)
        for name, spec, record in query:
            #result = query.loc[index].record
            assert record.status == RecordStatusEnum.complete
            #assert result.status.value.upper() == "COMPLETE"
            assert record.error is None
            assert record.return_result is not None


def test_basic_submissions_multiple_spec(snowflake):
    """Test submitting a basic dataset to a snowflake server with multiple qcspecs."""

    client = snowflake.client()

    qc_specs = [{"method": "openff-1.0.0", "basis": "smirnoff", "program": "openmm", "spec_name": "openff"},
                {"method": "gaff-2.11", "basis": "antechamber", "program": "openmm", "spec_name": "gaff"}]

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = BasicDatasetFactory(driver="energy")
    factory.clear_qcspecs()
    for spec in qc_specs:
        factory.add_qc_spec(**spec,
                            spec_description="testing the single points"
                            )

    dataset = factory.create_dataset(dataset_name="Test single points multiple specs",
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

    snowflake.await_results()

    # make sure of the results are complete
    ds = client.get_dataset("Dataset", dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    assert ds.data.description == dataset.description
    assert ds.data.tagline == dataset.dataset_tagline
    assert ds.data.tags == dataset.dataset_tags

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # check the qc spec
    assert ds.data.default_driver == dataset.driver

    # get the last ran spec
    for specification in ds.data.history:
        driver, program, method, basis, spec_name = specification
        spec = dataset.qc_specifications[spec_name]
        assert driver == dataset.driver
        assert program == spec.program
        assert method == spec.method
        assert basis == spec.basis

    for spec in dataset.qc_specifications.values():
        query = ds.get_records(
            method=spec.method,
            basis=spec.basis,
            program=spec.program,
        )
        # make sure all conformers are submitted
        assert len(query.index) == len(molecules)
        for index in query.index:
            result = query.loc[index].record
            assert result.status.value.upper() == "COMPLETE"
            assert result.error is None
            assert result.return_result is not None


def test_basic_submissions_single_pcm_spec(snowflake):
    """Test submitting a basic dataset to a snowflake server with pcm water in the specification."""

    client = snowflake.client()

    program = "psi4"
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = BasicDatasetFactory(driver="energy")
    factory.add_qc_spec(method="hf", basis="sto-3g", program=program, spec_name="default",
                        spec_description="testing the single points with pcm",
                        implicit_solvent=PCMSettings(units="au", medium_Solvent="water"),
                        overwrite=True)

    # only use one molecule due to the time it takes to run with pcm
    dataset = factory.create_dataset(dataset_name="Test single points with pcm water",
                                     molecules=molecules[0],
                                     description="Test basics dataset with pcm water",
                                     tagline="Testing single point datasets with pcm water",
                                     )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=client)

    snowflake.await_results()

    # make sure of the results are complete
    ds = client.get_dataset("Dataset", dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    assert ds.data.description == dataset.description
    assert ds.data.tagline == dataset.dataset_tagline
    assert ds.data.tags == dataset.dataset_tags

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # check the qc spec
    assert ds.data.default_driver == dataset.driver

    # get the last ran spec
    for specification in ds.data.history:
        driver, program, method, basis, spec_name = specification
        spec = dataset.qc_specifications[spec_name]
        assert driver == dataset.driver
        assert program == spec.program
        assert method == spec.method
        assert basis == spec.basis
        break
    else:
        raise RuntimeError(f"The requested compute was not found in the history {ds.data.history}")

    for spec in dataset.qc_specifications.values():
        query = ds.get_records(
            method=spec.method,
            basis=spec.basis,
            program=spec.program,
        )
        for index in query.index:
            result = query.loc[index].record
            assert result.status.value.upper() == "COMPLETE"
            assert result.error is None
            assert result.return_result is not None
            # make sure the PCM result was captured
            assert result.extras["qcvars"]["PCM POLARIZATION ENERGY"] < 0


def test_adding_specifications(snowflake):
    """
    Test adding specifications to datasets.
    Here we are testing multiple scenarios:
    1) Adding an identical specification to a dataset
    2) Adding a spec with the same name as another but with different options
    3) overwrite a spec which was added but never used.
    """
    client = snowflake.client()
    mol = Molecule.from_smiles("CO")
    # make a dataset
    factory = OptimizationDatasetFactory()
    opt_dataset = factory.create_dataset(dataset_name="Specification error check", molecules=mol,
                                         description="test adding new compute specs to datasets",
                                         tagline="test adding new compute specs")
    opt_dataset.clear_qcspecs()
    # add a new mm spec
    opt_dataset.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm",
                            spec_description="default openff spec", spec_name="openff-1.0.0")

    # submit the optimizations and let the compute run
    opt_dataset.submit(client=client)
    snowflake.await_results()
    snowflake.await_services()

    # grab the collection
    ds = client.get_dataset(opt_dataset.type, opt_dataset.dataset_name)

    # now try and add the specification again this should return True
    assert opt_dataset._add_dataset_specification(spec=opt_dataset.qc_specifications["openff-1.0.0"],
                                                  procedure_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                                  dataset=ds) is True

    # now change part of the spec but keep the name the same
    opt_dataset.clear_qcspecs()
    opt_dataset.add_qc_spec(method="openff-1.2.1", basis="smirnoff", spec_name="openff-1.0.0", program="openmm",
                            spec_description="openff-1.2.1 with wrong name.")

    # now try and add this specification with the same name but different settings
    with pytest.raises(QCSpecificationError):
        opt_dataset._add_dataset_specification(spec=opt_dataset.qc_specifications["openff-1.0.0"],
                                               procedure_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                               dataset=ds)

    # now add a new specification but no compute and make sure it is overwritten
    opt_dataset.clear_qcspecs()
    opt_dataset.add_qc_spec(method="ani1x", basis=None, program="torchani", spec_name="ani", spec_description="a ani spec")
    assert opt_dataset._add_dataset_specification(spec=opt_dataset.qc_specifications["ani"],
                                                  procedure_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                                  dataset=ds) is True

    # now change the spec slightly and add again
    opt_dataset.clear_qcspecs()
    opt_dataset.add_qc_spec(method="ani1ccx", basis=None, program="torchani", spec_name="ani",
                            spec_description="a ani spec")
    assert opt_dataset._add_dataset_specification(spec=opt_dataset.qc_specifications["ani"],
                                                  procedure_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                                  dataset=ds) is True


@pytest.mark.parametrize("dataset_data", [
    pytest.param((BasicDatasetFactory, BasicDataset), id="Dataset"),
    pytest.param((OptimizationDatasetFactory, OptimizationDataset), id="OptimizationDataset"),
    pytest.param((TorsiondriveDatasetFactory, TorsiondriveDataset), id="TorsiondriveDataset")
])
def test_adding_compute(snowflake, dataset_data):
    """
    Test adding new compute to each of the dataset types using none psi4 programs.
    """
    client = snowflake.client()
    mol = Molecule.from_smiles("CO")
    factory_type, dataset_type = dataset_data
    # make and clear out the qc specs
    factory = factory_type()
    factory.clear_qcspecs()
    factory.add_qc_spec(method="openff-1.0.0",
                        basis="smirnoff",
                        program="openmm",
                        spec_name="default",
                        spec_description="default spec for openff")
    dataset = factory.create_dataset(dataset_name=f"Test adding compute to {factory_type}",
                                     molecules=mol,
                                     description=f"Testing adding compute to a {dataset_type} dataset",
                                     tagline="tests for adding compute.")

    # now submit again
    dataset.submit(client=client)
    # make sure that the compute has finished
    snowflake.await_results()
    snowflake.await_services(max_iter=50)

    # now lets make a dataset with new compute and submit it
    # transfer the metadata to compare the elements
    compute_dataset = dataset_type(dataset_name=dataset.dataset_name, metadata=dataset.metadata, dataset_tagline=dataset.dataset_tagline, description=dataset.description)
    compute_dataset.clear_qcspecs()
    # now add the new compute spec
    compute_dataset.add_qc_spec(method="uff",
                                basis=None,
                                program="rdkit",
                                spec_name="rdkit",
                                spec_description="rdkit basic spec")

    # make sure the dataset has no molecules and submit it
    assert compute_dataset.dataset == {}
    compute_dataset.submit(client=client)
    # make sure that the compute has finished
    snowflake.await_results()
    snowflake.await_services(max_iter=50)

    # make sure of the results are complete
    ds = client.get_dataset(dataset.type, dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    assert ds.data.description == dataset.description
    assert ds.data.tagline == dataset.dataset_tagline
    assert ds.data.tags == dataset.dataset_tags

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # update all specs into one dataset
    dataset.add_qc_spec(**compute_dataset.qc_specifications["rdkit"].dict())
    # get the last ran spec
    if dataset.type == "DataSet":
        for specification in ds.data.history:
            driver, program, method, basis, spec_name = specification
            spec = dataset.qc_specifications[spec_name]
            assert driver == dataset.driver
            assert program == spec.program
            assert method == spec.method
            assert basis == spec.basis

        for spec in dataset.qc_specifications.values():
            query = ds.get_records(
                method=spec.method,
                basis=spec.basis,
                program=spec.program,
            )
            for index in query.index:
                result = query.loc[index].record
                assert result.status.value.upper() == "COMPLETE"
                assert result.error is None
                assert result.return_result is not None
    else:
        # check the qc spec
        for qc_spec in dataset.qc_specifications.values():
            spec = ds.data.specs[qc_spec.spec_name]

            assert spec.description == qc_spec.spec_description
            assert spec.qc_spec.driver == dataset.driver
            assert spec.qc_spec.method == qc_spec.method
            assert spec.qc_spec.basis == qc_spec.basis
            assert spec.qc_spec.program == qc_spec.program

            # check the keywords
            keywords = client.query_keywords(spec.qc_spec.keywords)[0]

            assert keywords.values["maxiter"] == qc_spec.maxiter
            assert keywords.values["scf_properties"] == qc_spec.scf_properties

            # query the dataset
            ds.query(qc_spec.spec_name)

            for index in ds.df.index:
                record = ds.df.loc[index].default
                # this will take some time so make sure it is running with no error
                assert record.status.value == "COMPLETE", print(record.dict())
                assert record.error is None


def test_basic_submissions_wavefunction(snowflake):
    """
    Test submitting a basic dataset with a wavefunction protocol and make sure it is executed.
    """
    # only a psi4 test
    if not has_program("psi4"):
        pytest.skip("Program psi4 not found.")

    client = snowflake.client()
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = BasicDatasetFactory(driver="energy")
    factory.clear_qcspecs()
    factory.add_qc_spec(method="hf",
                        basis="sto-6g",
                        program="psi4",
                        spec_name="default",
                        spec_description="wavefunction spec",
                        store_wavefunction="orbitals_and_eigenvalues")

    dataset = factory.create_dataset(dataset_name="Test single points with wavefunction",
                                     molecules=molecules,
                                     description="Test basics dataset",
                                     tagline="Testing single point datasets with wavefunction",
                                     )

    # submit the dataset
    # now submit again
    dataset.submit(client=client)

    snowflake.await_results()

    # make sure of the results are complete
    ds = client.get_dataset("Dataset", dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    assert ds.data.description == dataset.description
    assert ds.data.tagline == dataset.dataset_tagline
    assert ds.data.tags == dataset.dataset_tags

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # check the qc spec
    assert ds.data.default_driver == dataset.driver

    # get the last ran spec
    for specification in ds.data.history:
        driver, program, method, basis, spec_name = specification
        spec = dataset.qc_specifications[spec_name]
        assert driver == dataset.driver
        assert program == spec.program
        assert method == spec.method
        assert basis == spec.basis

    for spec in dataset.qc_specifications.values():
        query = ds.get_records(
            method=spec.method,
            basis=spec.basis,
            program=spec.program,
        )
        for index in query.index:
            result = query.loc[index].record
            assert result.status.value.upper() == "COMPLETE"
            assert result.error is None
            assert result.return_result is not None
            basis = result.get_wavefunction("basis")
            assert basis.name.lower() == "sto-6g"
            orbitals = result.get_wavefunction("orbitals_a")
            assert orbitals.shape is not None


def test_optimization_submissions_with_constraints(snowflake):
    """
    Make sure that the constraints are added to the optimization and enforced.
    """
    client = snowflake.client()
    ethane = Molecule.from_file(get_data("ethane.sdf"), "sdf")
    dataset = OptimizationDataset(dataset_name="Test optimizations with constraint", description="Test optimization dataset with constraints", dataset_tagline="Testing optimization datasets")
    # add just mm spec
    dataset.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm", spec_name="default", spec_description="mm default spec", overwrite=True)
    # build some constraints
    constraints = Constraints()
    constraints.add_set_constraint(constraint_type="dihedral", indices=[2, 0, 1, 5], value=60, bonded=True)
    constraints.add_freeze_constraint(constraint_type="distance", indices=[0, 1], bonded=True)
    # add the molecule
    index = ethane.to_smiles()
    dataset.add_molecule(index=index, molecule=ethane, constraints=constraints)

    # now submit again
    dataset.submit(client=client)

    snowflake.await_results()

    # make sure of the results are complete
    ds = client.get_dataset("OptimizationDataset", dataset.dataset_name)
    record = ds.get_record(ds.df.index[0], "default")
    assert "constraints" in record.keywords
    assert record.status.value == "COMPLETE"
    assert record.error is None
    assert len(record.trajectory) > 1

    # now make sure the constraints worked
    final_molecule = record.get_final_molecule()
    assert pytest.approx(final_molecule.measure((2, 0, 1, 5)), abs=1e-2) == 60
    assert record.get_initial_molecule().measure((0, 1)) == pytest.approx(final_molecule.measure((0, 1)))


@pytest.mark.parametrize("specification", [
    pytest.param(({"method": "hf", "basis": "3-21g", "program": "psi4"}, "gradient"), id="PSI4 hf 3-21g gradient"),
    pytest.param(({"method": "openff_unconstrained-1.0.0", "basis": "smirnoff", "program": "openmm"}, "gradient"), id="SMIRNOFF openff_unconstrained-1.0.0 gradient"),
    pytest.param(({"method": "uff", "basis": None, "program": "rdkit"}, "gradient"), id="RDKit UFF gradient")
])
def test_optimization_submissions(snowflake, specification):
    """Test submitting an Optimization dataset to a snowflake server."""

    client = snowflake.client()

    qc_spec, driver = specification
    program = qc_spec["program"]
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = OptimizationDatasetFactory(driver=driver)
    factory.add_qc_spec(**qc_spec, spec_name="default", spec_description="test", overwrite=True)

    dataset = factory.create_dataset(dataset_name=f"Test optimizations info {program}, {driver}",
                                     molecules=molecules[:2],
                                     description="Test optimization dataset",
                                     tagline="Testing optimization datasets",
                                     )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=client)

    snowflake.await_results()

    # make sure of the results are complete
    ds = client.get_dataset("OptimizationDataset", dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # check the qc spec
    for qc_spec in dataset.qc_specifications.values():
        spec = ds.data.specs[qc_spec.spec_name]

        assert spec.description == qc_spec.spec_description
        assert spec.qc_spec.driver == dataset.driver
        assert spec.qc_spec.method == qc_spec.method
        assert spec.qc_spec.basis == qc_spec.basis
        assert spec.qc_spec.program == qc_spec.program

        # check the keywords
        keywords = client.query_keywords(spec.qc_spec.keywords)[0]

        assert keywords.values["maxiter"] == qc_spec.maxiter
        assert keywords.values["scf_properties"] == qc_spec.scf_properties

        # query the dataset
        ds.query(qc_spec.spec_name)

        for index in ds.df.index:
            record = ds.df.loc[index].default
            assert record.status.value == "COMPLETE"
            assert record.error is None
            assert len(record.trajectory) > 1
            # if we used psi4 make sure the properties were captured
            if program == "psi4":
                result = record.get_trajectory()[0]
                assert "CURRENT DIPOLE X" in result.extras["qcvars"].keys()
                assert "SCF QUADRUPOLE XX" in result.extras["qcvars"].keys()


def test_optimization_submissions_with_pcm(snowflake):
    """Test submitting an Optimization dataset to a snowflake server with PCM."""

    client = snowflake.client()

    program = "psi4"
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    # use a single small molecule due to the extra time PCM takes
    molecules = Molecule.from_smiles("C")

    factory = OptimizationDatasetFactory(driver="gradient")
    factory.add_qc_spec(method="hf", basis="sto-3g", program=program, spec_name="default", spec_description="test",
                        implicit_solvent=PCMSettings(units="au", medium_Solvent="water"),
                        overwrite=True)

    dataset = factory.create_dataset(dataset_name="Test optimizations info with pcm water",
                                     molecules=molecules,
                                     description="Test optimization dataset",
                                     tagline="Testing optimization datasets",
                                     )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=client)

    snowflake.await_results()

    # make sure of the results are complete
    ds = client.get_dataset("OptimizationDataset", dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # check the qc spec
    for qc_spec in dataset.qc_specifications.values():
        spec = ds.data.specs[qc_spec.spec_name]

        assert spec.description == qc_spec.spec_description
        assert spec.qc_spec.driver == dataset.driver
        assert spec.qc_spec.method == qc_spec.method
        assert spec.qc_spec.basis == qc_spec.basis
        assert spec.qc_spec.program == qc_spec.program

        # check the keywords
        keywords = client.query_keywords(spec.qc_spec.keywords)[0]

        assert keywords.values["maxiter"] == qc_spec.maxiter
        assert keywords.values["scf_properties"] == qc_spec.scf_properties

        # query the dataset
        ds.query(qc_spec.spec_name)

        for index in ds.df.index:
            record = ds.df.loc[index].default
            assert record.status.value == "COMPLETE"
            assert record.error is None
            assert len(record.trajectory) > 1
            result = record.get_trajectory()[0]
            assert "CURRENT DIPOLE X" in result.extras["qcvars"].keys()
            assert "SCF QUADRUPOLE XX" in result.extras["qcvars"].keys()
            # make sure the PCM result was captured
            assert result.extras["qcvars"]["PCM POLARIZATION ENERGY"] < 0


def test_torsiondrive_scan_keywords(snowflake):
    """
    Test running torsiondrives with unique keyword settings which overwrite the global grid spacing and scan range.
    """

    client = snowflake.client()
    molecules = Molecule.from_smiles("CO")
    factory = TorsiondriveDatasetFactory()
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[*:1]~[#6:2]-[#8:3]~[*:4]")
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    factory.add_qc_spec(method="openff_unconstrained-1.1.0", basis="smirnoff", program="openmm", spec_description="scan range test", spec_name="openff-1.1.0")
    dataset = factory.create_dataset(dataset_name="Torsiondrive scan keywords", molecules=molecules,
                                     description="Testing scan keywords which overwrite the global settings",
                                     tagline="Testing scan keywords which overwrite the global settings")

    # now set the keywords
    keys = list(dataset.dataset.keys())
    entry = dataset.dataset[keys[0]]
    entry.keywords = {"grid_spacing": [5],
                      "dihedral_ranges": [(-10, 10)]}

    # now submit
    dataset.submit(client=client)
    snowflake.await_services(max_iter=50)

    # make sure of the results are complete
    ds = client.get_dataset("TorsionDriveDataset", dataset.dataset_name)

    # get the entry
    record = ds.get_record(ds.df.index[0], "openff-1.1.0")
    assert record.keywords.grid_spacing == [5]
    assert record.keywords.grid_spacing != dataset.grid_spacing
    assert record.keywords.dihedral_ranges == [(-10, 10)]
    assert record.keywords.dihedral_ranges != dataset.dihedral_ranges


def test_torsiondrive_constraints(snowflake):
    """
    Make sure constraints are correctly passed to optimisations in torsiondrives.
    """

    client = snowflake.client()
    molecule = Molecule.from_file(get_data("TRP.mol2"))
    dataset = TorsiondriveDataset(dataset_name="Torsiondrive constraints", dataset_tagline="Testing torsiondrive constraints", description="Testing torsiondrive constraints.")
    dataset.clear_qcspecs()
    dataset.add_qc_spec(method="uff", basis=None, program="rdkit", spec_name="uff", spec_description="tdrive constraints")
    # use a restricted range to keep the scan fast
    dataset.add_molecule(index="1", molecule=molecule, attributes=MoleculeAttributes.from_openff_molecule(molecule=molecule), dihedrals=[(4, 6, 8, 28)], keywords={"dihedral_ranges": [(-165, -145)]})
    entry = dataset.dataset["1"]
    # add the constraints
    entry.add_constraint(constraint="freeze", constraint_type="dihedral", indices=[6, 8, 10, 13])
    entry.add_constraint(constraint="freeze", constraint_type="dihedral", indices=[8, 10, 13, 14])

    dataset.submit(client=client, processes=1)
    snowflake.await_services(max_iter=50)

    # make sure the result is complete
    ds = client.get_dataset("TorsionDriveDataset", dataset.dataset_name)

    record = ds.get_record(ds.df.index[0], "uff")
    opt = client.query_procedures(id=record.optimization_history['[-150]'])[0]
    constraints = opt.keywords["constraints"]
    # make sure both the freeze and set constraints are passed on
    assert "set" in constraints
    assert "freeze" in constraints
    # make sure both freeze constraints are present
    assert len(constraints["freeze"]) == 2
    assert constraints["freeze"][0]["indices"] == [6, 8, 10, 13]
    # make sure the dihedral has not changed
    assert pytest.approx(opt.get_initial_molecule().measure((6, 8, 10, 13))) == opt.get_final_molecule().measure((6, 8, 10, 13))


@pytest.mark.parametrize("specification", [
    pytest.param(({"method": "openff_unconstrained-1.1.0", "basis": "smirnoff", "program": "openmm"}, "gradient"), id="SMIRNOFF openff_unconstrained-1.0.0 gradient"),
    pytest.param(({"method": "mmff94", "basis": None, "program": "rdkit"}, "gradient"), id="RDKit mmff94 gradient")
])
def test_torsiondrive_submissions(snowflake, specification):
    """
    Test submitting a torsiondrive dataset and computing it.
    """

    client = snowflake.client()

    qc_spec, driver = specification
    program = qc_spec["program"]
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecules = Molecule.from_smiles("CO")

    factory = TorsiondriveDatasetFactory(driver=driver)
    factory.add_qc_spec(**qc_spec, spec_name="default", spec_description="test", overwrite=True)

    dataset = factory.create_dataset(dataset_name=f"Test torsiondrives info {program}, {driver}",
                                     molecules=molecules,
                                     description="Test torsiondrive dataset",
                                     tagline="Testing torsiondrive datasets",
                                     )

    # force a metadata validation error
    dataset.metadata.long_description = None

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client)

    # re-add the description so we can submit the data
    dataset.metadata.long_description = "Test basics dataset"

    # now submit again
    dataset.submit(client=client)

    snowflake.await_services(max_iter=50)

    # make sure of the results are complete
    ds = client.get_dataset("TorsionDriveDataset", dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # check the qc spec
    for qc_spec in dataset.qc_specifications.values():
        spec = ds.data.specs[qc_spec.spec_name]

        assert spec.description == qc_spec.spec_description
        assert spec.qc_spec.driver == dataset.driver
        assert spec.qc_spec.method == qc_spec.method
        assert spec.qc_spec.basis == qc_spec.basis
        assert spec.qc_spec.program == qc_spec.program

        # check the keywords
        keywords = client.query_keywords(spec.qc_spec.keywords)[0]

        assert keywords.values["maxiter"] == qc_spec.maxiter
        assert keywords.values["scf_properties"] == qc_spec.scf_properties

        # query the dataset
        ds.query(qc_spec.spec_name)

        for index in ds.df.index:
            record = ds.df.loc[index].default
            # this will take some time so make sure it is running with no error
            assert record.status.value == "COMPLETE", print(record.dict())
            assert record.error is None
            assert len(record.final_energy_dict) == 24


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="BasicDataset ignore_errors"),
    pytest.param(OptimizationDatasetFactory, id="OptimizationDataset ignore_errors"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDataset ignore_errors"),
])
def test_ignore_errors_all_datasets(snowflake, factory_type, capsys):
    """
    For each dataset make sure that when the basis is not fully covered the dataset raises warning errors, and verbose information
    """
    client = snowflake.client()
    # molecule containing boron
    molecule = Molecule.from_smiles("OB(O)C1=CC=CC=C1")
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[#6:1]~[#6:2]-[B:3]~[#8:4]")
    factory = factory_type()
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm", spec_name="parsley", spec_description="standard parsley spec")
    dataset = factory.create_dataset(dataset_name=f"Test ignore_error for {factory.type}",
                                     molecules=molecule,
                                     description="Test ignore errors dataset",
                                     tagline="Testing ignore errors datasets",
                                     )

    # make sure the dataset raises an error here
    with pytest.raises(MissingBasisCoverageError):
        dataset.submit(client=client, ignore_errors=False)

    # now we want to try again and make sure warnings are raised
    with pytest.warns(UserWarning):
        dataset.submit(client=client, ignore_errors=True, verbose=True)

    info = capsys.readouterr()
    assert info.out == f"Number of new entries: {dataset.n_records}/{dataset.n_records}\n"


@pytest.mark.parametrize("factory_type", [
    pytest.param(BasicDatasetFactory, id="Basicdataset"),
    pytest.param(OptimizationDatasetFactory, id="Optimizationdataset")
])
def test_index_not_changed(snowflake, factory_type):
    """
    Make sure that when we submit molecules from a dataset/optimizationdataset with one input conformer that the index is not changed.
    """
    factory = factory_type()
    factory.clear_qcspecs()
    client = snowflake.client()
    #client = snowflake.client()
    # add only mm specs
    factory.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm", spec_name="parsley",
                        spec_description="standard parsley spec")

    molecule = Molecule.from_smiles("C")
    # make sure we only have one conformer
    molecule.generate_conformers(n_conformers=1)
    dataset = factory.create_dataset(dataset_name=f"Test index change for {factory.type}",
                                     molecules=molecule,
                                     description="Test index change dataset",
                                     tagline="Testing index changes datasets",
                                     )

    # now change the index name to something unique
    entry = dataset.dataset.pop(list(dataset.dataset.keys())[0])
    entry.index = "my_unique_index"
    dataset.dataset[entry.index] = entry

    dataset.submit(client=client)

    # pull the dataset and make sure our index is present
    ds = client.get_dataset(dataset.type, dataset.dataset_name)

    if dataset.type == "DataSet":
        query = ds.get_records(method="openff-1.0.0", basis="smirnoff", program="openmm")
        assert "my_unique_index" in query.index
    else:
        assert "my_unique_index" in ds.df.index


@pytest.mark.parametrize("factory_type", [
    pytest.param(OptimizationDatasetFactory, id="OptimizationDataset index clash"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDataset index clash"),
])
def test_adding_dataset_entry_fail(snowflake, factory_type, capsys):
    """
    Make sure that the new entries is not incremented if we can not add a molecule to the server due to a name clash.
    TODO add basic dataset into the testing if the api changes to return an error when adding the same index twice
    """
    client = snowflake.client()
    molecule = Molecule.from_smiles("CO")
    molecule.generate_conformers(n_conformers=1)
    factory = factory_type()
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[*:1]~[#6:2]-[#8:3]~[*:4]")
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm", spec_name="parsley", spec_description="standard parsley spec")
    dataset = factory.create_dataset(dataset_name=f"Test index clash for {factory.type}",
                                     molecules=molecule,
                                     description="Test ignore errors dataset",
                                     tagline="Testing ignore errors datasets",
                                     )

    # make sure all expected index get submitted
    dataset.submit(client=client)
    info = capsys.readouterr()
    assert info.out == f"Number of new entries: {dataset.n_records}/{dataset.n_records}\n"

    # now add a new spec and try and submit again
    dataset.clear_qcspecs()
    dataset.add_qc_spec(method="mmff94", basis=None, program="rdkit", spec_name="mff94", spec_description="mff94 force field in rdkit")
    dataset.submit(client=client, verbose=True)
    info = capsys.readouterr()
    assert info.out == f"Number of new entries: 0/{dataset.n_records}\n"


@pytest.mark.parametrize("factory_type", [
    pytest.param(OptimizationDatasetFactory, id="OptimizationDataset expand compute"),
    pytest.param(TorsiondriveDatasetFactory, id="TorsiondriveDataset expand compute"),
])
def test_expanding_compute(snowflake, factory_type):
    """
    Make sure that if we expand the compute of a dataset tasks are generated.
    """
    client = snowflake.client()
    molecule = Molecule.from_smiles("CC")
    molecule.generate_conformers(n_conformers=1)
    factory = factory_type()
    scan_enum = workflow_components.ScanEnumerator()
    scan_enum.add_torsion_scan(smarts="[*:1]~[#6:2]-[#6:3]~[*:4]")
    factory.add_workflow_components(scan_enum)
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm", spec_name="default",
                        spec_description="standard parsley spec")
    dataset = factory.create_dataset(dataset_name=f"Test compute expand {factory.type}",
                                     molecules=molecule,
                                     description="Test compute expansion",
                                     tagline="Testing compute expansion",
                                     )

    # make sure all expected index get submitted
    dataset.submit(client=client)
    # grab the dataset and check the history
    ds = client.get_dataset(dataset.type, dataset.dataset_name)
    assert ds.data.history == {"default"}

    # now make another dataset to expand the compute
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(method="openff-1.2.0", basis="smirnoff", program="openmm", spec_name="parsley2",
                        spec_description="standard parsley spec")
    dataset = factory.create_dataset(dataset_name=f"Test compute expand {factory.type}",
                                     molecules=[],
                                     description="Test compute expansion",
                                     tagline="Testing compute expansion",
                                     )
    # now submit again
    dataset.submit(client=client)

    # now grab the dataset again and check the tasks list
    ds = client.get_dataset(dataset.type, dataset.dataset_name)
    assert ds.data.history == {"default", "parsley2"}
    # make sure a record has been made
    entry = ds.get_entry(ds.df.index[0])
    assert "parsley2" in entry.object_map
