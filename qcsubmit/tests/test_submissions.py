"""
Test submissions to a local qcarchive instance using different compute backends, RDKit, OpenMM, PSI4, ANI1

Here we use the qcfractal fractal_compute_server fixture to set up the database.
"""

import pytest
from qcfractal.testing import fractal_compute_server
from qcportal import FractalClient

from openforcefield.topology import Molecule
from qcengine.testing import has_program
from qcsubmit.common_structures import Metadata
from qcsubmit.exceptions import DatasetInputError
from qcsubmit.factories import (
    BasicDatasetFactory,
    OptimizationDatasetFactory,
    TorsiondriveDatasetFactory,
)
from qcsubmit.utils import get_data


@pytest.mark.parametrize("specification", [
    pytest.param(("hf", "3-21g", "psi4", "energy"), id="PSI4 hf 3-21g energy"),
    pytest.param(("ani1ccx", None, "torchani", "hessian"), id="ANI1CCX hessian"),
    pytest.param(("openff-1.0.0", "smirnoff", "openmm", "energy"), id="SMIRNOFF openff-1.0.0 energy"),
    pytest.param(("UFF", None, "rdkit", "gradient"), id="RDKit UFF gradient")
])
def test_basic_submissions(fractal_compute_server, specification):
    """Test submitting a basic dataset to a snowflake server."""

    client = FractalClient(fractal_compute_server)

    method, basis, program, driver = specification

    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = BasicDatasetFactory(method=method, basis=basis, program=program, driver=driver)

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
        if spec_name == dataset.spec_name:
            assert driver == dataset.driver
            assert program == dataset.program
            assert method == dataset.method
            assert basis == dataset.basis
            break
    else:
        raise RuntimeError(f"The requested compute was not found in the history {ds.data.history}")



    query = ds.get_records(
        method=dataset.method,
        basis=dataset.basis,
        program=dataset.program,
    )
    for index in query.index:
        result = query.loc[index].record
        assert result.status.value.upper() == "COMPLETE"
        assert result.error is None
        assert result.return_result is not None


@pytest.mark.parametrize("specification", [
    pytest.param(("hf", "3-21g", "psi4", "gradient"), id="PSI4 hf 3-21g gradient"),
    pytest.param(("openff-1.0.0", "smirnoff", "openmm", "gradient"), id="SMIRNOFF openff-1.0.0 gradient"),
    pytest.param(("uff", None, "rdkit", "gradient"), id="RDKit UFF gradient")
])
def test_optimization_submissions(fractal_compute_server, specification):
    """Test submitting an Optimization dataset to a snowflake server."""

    client = FractalClient(fractal_compute_server)

    method, basis, program, driver = specification

    if not has_program(program):
        pytest.skip(f"Program '{program}' not found.")

    molecules = Molecule.from_file(get_data("butane_conformers.pdb"), "pdb")

    factory = OptimizationDatasetFactory(method=method, basis=basis, program=program, driver=driver)

    dataset = factory.create_dataset(dataset_name=f"Test single points info {program}, {driver}",
                                     molecules=molecules[:2],
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
    ds = client.get_collection("OptimizationDataset", dataset.dataset_name)

    # check the metadata
    meta = Metadata(**ds.data.metadata)
    assert meta == dataset.metadata

    # check the provenance
    assert dataset.provenance == ds.data.provenance

    # check the qc spec
    spec = ds.data.specs[dataset.spec_name]

    assert spec.description == dataset.spec_description
    assert spec.qc_spec.driver == dataset.driver
    assert spec.qc_spec.method == dataset.method
    assert spec.qc_spec.basis == dataset.basis
    assert spec.qc_spec.program == dataset.program

    # check the keywords
    keywords = client.query_keywords(spec.qc_spec.keywords)[0]

    assert keywords.values["maxiter"] == dataset.maxiter
    assert keywords.values["scf_properties"] == dataset.scf_properties

    # query the dataset
    ds.query(dataset.spec_name)

    for index in ds.df.index:
        assert ds.df.loc[index].default.status.value == "COMPLETE"
        assert ds.df.loc[index].default.error is None
        assert len(ds.df.loc[index].default.trajectory) > 1
