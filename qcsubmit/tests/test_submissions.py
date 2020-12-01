"""
Test submissions to a local qcarchive instance using different compute backends, RDKit, OpenMM, PSI4

Here we use the qcfractal fractal_compute_server fixture to set up the database.
"""

import pytest
from openforcefield.topology import Molecule
from qcengine.testing import has_program
from qcfractal.testing import fractal_compute_server
from qcportal import FractalClient

from qcsubmit.common_structures import Metadata, PCMSettings
from qcsubmit.constraints import Constraints
from qcsubmit.datasets import BasicDataset, OptimizationDataset, TorsiondriveDataset
from qcsubmit.exceptions import (
    DatasetInputError,
    MissingBasisCoverageError,
    QCSpecificationError,
)
from qcsubmit.factories import (
    BasicDatasetFactory,
    OptimizationDatasetFactory,
    TorsiondriveDatasetFactory,
)
from qcsubmit.utils import get_data


@pytest.mark.parametrize("specification", [
    pytest.param(({"method": "hf", "basis": "3-21g", "program": "psi4"}, "energy"), id="PSI4 hf 3-21g energy"),
    pytest.param(({"method": "openff-1.0.0", "basis": "smirnoff", "program": "openmm"}, "energy"), id="SMIRNOFF openff-1.0.0 energy"),
    pytest.param(({"method": "uff", "basis": None, "program": "rdkit"}, "gradient"), id="RDKit UFF gradient")
])
def test_basic_submissions_single_spec(fractal_compute_server, specification):
    """Test submitting a basic dataset to a snowflake server."""

    client = FractalClient(fractal_compute_server)

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

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client, await_result=False)

    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_results()

    # make sure of the results are complete
    ds = client.get_collection("Dataset", dataset.dataset_name)

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


def test_basic_submissions_multiple_spec(fractal_compute_server):
    """Test submitting a basic dataset to a snowflake server with multiple qcspecs."""

    client = FractalClient(fractal_compute_server)

    qc_specs = [{"method": "openff-1.0.0", "basis": "smirnoff", "program": "openmm", "spec_name": "openff"},
                {"method": "gaff-2.11", "basis": "antechamber", "program": "openmm", "spec_name": "gaff"}]

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = BasicDatasetFactory(driver="energy")
    factory.clear_qcspecs()
    for spec in qc_specs:
        factory.add_qc_spec(**spec,
                            spec_description="testing the single points"
                            )

    dataset = factory.create_dataset(dataset_name=f"Test single points multiple specs",
                                     molecules=molecules,
                                     description="Test basics dataset",
                                     tagline="Testing single point datasets",
                                     )

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client, await_result=False)

    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_results()

    # make sure of the results are complete
    ds = client.get_collection("Dataset", dataset.dataset_name)

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


def test_basic_submissions_single_pcm_spec(fractal_compute_server):
    """Test submitting a basic dataset to a snowflake server with pcm water in the specification."""

    client = FractalClient(fractal_compute_server)

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
    dataset = factory.create_dataset(dataset_name=f"Test single points with pcm water",
                                     molecules=molecules[0],
                                     description="Test basics dataset with pcm water",
                                     tagline="Testing single point datasets with pcm water",
                                     )

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client, await_result=False)

    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_results()

    # make sure of the results are complete
    ds = client.get_collection("Dataset", dataset.dataset_name)

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


def test_adding_specifications(fractal_compute_server):
    """
    Test adding specifications to datasets.
    Here we are testing multiple scenarios:
    1) Adding an identical specification to a dataset
    2) Adding a spec with the same name as another but with different options
    3) overwrite a spec which was added but never used.
    """
    client = FractalClient(fractal_compute_server)
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

    opt_dataset.metadata.long_description_url = "https://test.org"
    # submit the optimizations and let the compute run
    opt_dataset.submit(client=client, await_result=False)
    fractal_compute_server.await_results()
    fractal_compute_server.await_services()

    # grab the collection
    ds = client.get_collection(opt_dataset.dataset_type, opt_dataset.dataset_name)

    # now try and add the specification again this should return True
    assert opt_dataset.add_dataset_specification(spec=opt_dataset.qc_specifications["openff-1.0.0"],
                                                 opt_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                                 collection=ds) is True

    # now change part of the spec but keep the name the same
    opt_dataset.clear_qcspecs()
    opt_dataset.add_qc_spec(method="openff-1.2.1", basis="smirnoff", spec_name="openff-1.0.0", program="openmm",
                            spec_description="openff-1.2.1 with wrong name.")

    # now try and add this specification with the same name but different settings
    with pytest.raises(QCSpecificationError):
        opt_dataset.add_dataset_specification(spec=opt_dataset.qc_specifications["openff-1.0.0"],
                                              opt_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                              collection=ds)

    # now add a new specification but no compute and make sure it is overwritten
    opt_dataset.clear_qcspecs()
    opt_dataset.add_qc_spec(method="ani1x", basis=None, program="torchani", spec_name="ani", spec_description="a ani spec")
    assert opt_dataset.add_dataset_specification(spec=opt_dataset.qc_specifications["ani"],
                                                 opt_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                                 collection=ds) is True

    # now change the spec slightly and add again
    opt_dataset.clear_qcspecs()
    opt_dataset.add_qc_spec(method="ani1ccx", basis=None, program="torchani", spec_name="ani",
                            spec_description="a ani spec")
    assert opt_dataset.add_dataset_specification(spec=opt_dataset.qc_specifications["ani"],
                                                 opt_spec=opt_dataset.optimization_procedure.get_optimzation_spec(),
                                                 collection=ds) is True


@pytest.mark.parametrize("dataset_data", [
    pytest.param((BasicDatasetFactory, BasicDataset), id="Dataset"),
    pytest.param((OptimizationDatasetFactory, OptimizationDataset), id="OptimizationDataset"),
    pytest.param((TorsiondriveDatasetFactory, TorsiondriveDataset), id="TorsiondriveDataset")
])
def test_adding_compute(fractal_compute_server, dataset_data):
    """
    Test adding new compute to each of the dataset types using none psi4 programs.
    """
    client = FractalClient(fractal_compute_server)
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

    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)
    # make sure that the compute has finished
    fractal_compute_server.await_results()
    fractal_compute_server.await_services(max_iter=50)

    # now lets make a dataset with new compute and submit it
    # transfer the metadata to compare the elements
    compute_dataset = dataset_type(dataset_name=dataset.dataset_name, metadata=dataset.metadata)
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
    fractal_compute_server.await_results()
    fractal_compute_server.await_services(max_iter=50)

    # make sure of the results are complete
    ds = client.get_collection(dataset.dataset_type, dataset.dataset_name)

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
    if dataset.dataset_type == "DataSet":
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

            assert keywords.values["maxiter"] == dataset.maxiter
            assert keywords.values["scf_properties"] == dataset.scf_properties

            # query the dataset
            ds.query(qc_spec.spec_name)

            for index in ds.df.index:
                record = ds.df.loc[index].default
                # this will take some time so make sure it is running with no error
                assert record.status.value == "COMPLETE", print(record.dict())
                assert record.error is None


def test_basic_submissions_wavefunction(fractal_compute_server):
    """
    Test submitting a basic dataset with a wavefunction protocol and make sure it is executed.
    """
    # only a psi4 test
    if not has_program("psi4"):
        pytest.skip(f"Program psi4 not found.")

    client = FractalClient(fractal_compute_server)
    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = BasicDatasetFactory(driver="energy")
    factory.clear_qcspecs()
    factory.add_qc_spec(method="hf",
                        basis="sto-6g",
                        program="psi4",
                        spec_name="default",
                        spec_description="wavefunction spec",
                        store_wavefunction="orbitals_and_eigenvalues")

    dataset = factory.create_dataset(dataset_name=f"Test single points with wavefunction",
                                     molecules=molecules,
                                     description="Test basics dataset",
                                     tagline="Testing single point datasets with wavefunction",
                                     )
    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # submit the dataset
    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_results()

    # make sure of the results are complete
    ds = client.get_collection("Dataset", dataset.dataset_name)

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


def test_optimization_submissions_with_constraints(fractal_compute_server):
    """
    Make sure that the constraints are added to the optimization and enforced.
    """
    client = FractalClient(fractal_compute_server)
    ethane = Molecule.from_file(get_data("ethane.sdf"), "sdf")
    factory = OptimizationDatasetFactory()
    dataset = OptimizationDataset(dataset_name="Test optimizations with constraint", description="Test optimization dataset with constraints", tagline="Testing optimization datasets")
    # add just mm spec
    dataset.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm", spec_name="default", spec_description="mm default spec", overwrite=True)
    # build some constraints
    constraints = Constraints()
    constraints.add_set_constraint(constraint_type="dihedral", indices=[2, 0, 1, 5], value=60, bonded=True)
    constraints.add_freeze_constraint(constraint_type="distance", indices=[0, 1], bonded=True)
    # add the molecule
    attributes = factory.create_cmiles_metadata(ethane)
    index = ethane.to_smiles()
    dataset.add_molecule(index=index, molecule=ethane, attributes=attributes, constraints=constraints)
    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_results()

    # make sure of the results are complete
    ds = client.get_collection("OptimizationDataset", dataset.dataset_name)
    record = ds.get_record(ds.df.index[0], "default")
    assert "constraints" in record.keywords
    assert record.status.value == "COMPLETE"
    assert record.error is None
    assert len(record.trajectory) > 1

    # now make sure the constraints worked
    final_molecule = record.get_final_molecule()
    assert pytest.approx(60, final_molecule.measure((2, 0, 1, 5)))
    assert pytest.approx(record.get_initial_molecule().measure((0, 1)), final_molecule.measure((0, 1)))


@pytest.mark.parametrize("specification", [
    pytest.param(({"method": "hf", "basis": "3-21g", "program": "psi4"}, "gradient"), id="PSI4 hf 3-21g gradient"),
    pytest.param(({"method": "openff_unconstrained-1.0.0", "basis": "smirnoff", "program": "openmm"}, "gradient"), id="SMIRNOFF openff_unconstrained-1.0.0 gradient"),
    pytest.param(({"method": "uff", "basis": None, "program": "rdkit"}, "gradient"), id="RDKit UFF gradient")
])
def test_optimization_submissions(fractal_compute_server, specification):
    """Test submitting an Optimization dataset to a snowflake server."""

    client = FractalClient(fractal_compute_server)

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

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client, await_result=False)

    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_results()

    # make sure of the results are complete
    ds = client.get_collection("OptimizationDataset", dataset.dataset_name)

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

        assert keywords.values["maxiter"] == dataset.maxiter
        assert keywords.values["scf_properties"] == dataset.scf_properties

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


def test_optimization_submissions_with_pcm(fractal_compute_server):
    """Test submitting an Optimization dataset to a snowflake server with PCM."""

    client = FractalClient(fractal_compute_server)

    program = "psi4"
    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    # use a single small molecule due to the extra time PCM takes
    molecules = Molecule.from_smiles("C")

    factory = OptimizationDatasetFactory(driver="gradient")
    factory.add_qc_spec(method="hf", basis="sto-3g", program=program, spec_name="default", spec_description="test",
                        implicit_solvent=PCMSettings(units="au", medium_Solvent="water"),
                        overwrite=True)

    dataset = factory.create_dataset(dataset_name=f"Test optimizations info with pcm water",
                                     molecules=molecules,
                                     description="Test optimization dataset",
                                     tagline="Testing optimization datasets",
                                     )

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client, await_result=False)

    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_results()

    # make sure of the results are complete
    ds = client.get_collection("OptimizationDataset", dataset.dataset_name)

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

        assert keywords.values["maxiter"] == dataset.maxiter
        assert keywords.values["scf_properties"] == dataset.scf_properties

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


@pytest.mark.parametrize("specification", [
    pytest.param(({"method": "openff_unconstrained-1.1.0", "basis": "smirnoff", "program": "openmm"}, "gradient"), id="SMIRNOFF openff_unconstrained-1.0.0 gradient"),
    pytest.param(({"method": "mmff94", "basis": None, "program": "rdkit"}, "gradient"), id="RDKit mmff94 gradient")
])
def test_torsiondrive_submissions(fractal_compute_server, specification):
    """
    Test submitting a torsiondrive dataset and computing it.
    """

    client = FractalClient(fractal_compute_server)

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

    with pytest.raises(DatasetInputError):
        dataset.submit(client=client, await_result=False)

    # now add a mock url so we can submit the data
    dataset.metadata.long_description_url = "https://test.org"

    # now submit again
    dataset.submit(client=client, await_result=False)

    fractal_compute_server.await_services(max_iter=50)

    # make sure of the results are complete
    ds = client.get_collection("TorsionDriveDataset", dataset.dataset_name)

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

        assert keywords.values["maxiter"] == dataset.maxiter
        assert keywords.values["scf_properties"] == dataset.scf_properties

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
def test_ignore_errors_all_datasets(fractal_compute_server, factory_type):
    """
    For each dataset make sure that when the basis is not fully covered the dataset raises warning errors.
    """
    import warnings
    client = FractalClient(fractal_compute_server)
    # molecule containing boron
    molecule = Molecule.from_smiles("OB(O)C1=CC=CC=C1")
    factory = factory_type()
    factory.clear_qcspecs()
    # add only mm specs
    factory.add_qc_spec(method="openff-1.0.0", basis="smirnoff", program="openmm", spec_name="parsley", spec_description="standard parsley spec")
    dataset = factory.create_dataset(dataset_name=f"Test ignore_error for {factory.Config.title}",
                                     molecules=molecule,
                                     description="Test ignore errors dataset",
                                     tagline="Testing ignore errors datasets",
                                     )

    dataset.metadata.long_description_url = "https://test.org"

    # make sure the dataset rasies an error here
    with pytest.raises(MissingBasisCoverageError):
        dataset.submit(client=client, ignore_errors=False)

    # now we want to try again and make sure warnings are rasied
    with pytest.warns(UserWarning):
        dataset.submit(client=client, ignore_errors=True)
