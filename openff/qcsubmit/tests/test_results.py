"""
Test the results packages when collecting from the public qcarchive.
"""

import pytest
from qcportal import FractalClient

from openff.qcsubmit.results import (
    BasicCollectionResult,
    OptimizationCollectionResult,
    TorsionDriveCollectionResult,
)
from openff.qcsubmit.testing import temp_directory


@pytest.fixture
def public_client():
    """Setup a new connection to the public qcarchive client."""

    return FractalClient()


def test_optimization_default_results(public_client):
    """
    Test collecting results from the public qcarchive.
    The optimization collection is small containing only 576 records however some are incomplete and only 16
    unique molecules should be pulled.
    This also tests the ability to rebuild a non default geometric procedure from server.
    """

    result = OptimizationCollectionResult.from_server(
        client=public_client,
        spec_name="default",
        dataset_name="OpenFF Protein Fragments v1.0",
        include_trajectory=False,
        final_molecule_only=False)

    assert result.n_molecules == 16
    assert result.n_results == 576
    assert result.method == "b3lyp-d3bj"
    assert result.basis == "def2-tzvp"
    assert result.dataset_name == "OpenFF Protein Fragments v1.0"
    assert result.program == "psi4"

    # by default the result should pull the initial and final molecule only
    for optimization in result.collection.values():
        for entry in optimization.entries:
            assert len(entry.trajectory) == 2

            for single_result in entry.trajectory:
                assert single_result.wbo is not None
                assert single_result.mbo is not None


def test_optimization_trajectory_results(public_client):
    """
    Test downloading an optimization result with the trajectory.
    """

    # just take the first molecule in the set as the download can be slow
    result = OptimizationCollectionResult.from_server(
        client=public_client,
        spec_name="default",
        dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
        include_trajectory=True,
        final_molecule_only=False,
        subset=["cn1c(cccc1=o)c2ccccc2oc-0"])

    # make sure one unique molecule is found
    assert result.n_molecules == 1
    assert result.n_results == 1

    # get the optimization result
    opt_result = result.collection["Cn1c(cccc1=O)c2ccccc2OC"]
    # check connectivity changes
    assert opt_result.detect_connectivity_changes_heuristic() == {0: False}
    assert opt_result.detect_connectivity_changes_wbo() == {0: False}

    # check hydrogen bonds
    assert opt_result.detect_hydrogen_bonds_heuristic() == {0: False}
    assert opt_result.detect_hydrogen_bonds_wbo() == {0: False}

    # make sure the full trajectory was pulled out
    assert opt_result.n_entries == 1
    traj = opt_result.entries[0]

    molecule = traj.get_trajectory()

    assert molecule.n_conformers == 59
    assert len(traj.energies) == molecule.n_conformers


def test_optimizationdataset_new_basicdataset(public_client):
    """
    Test creating a new basicdataset from the result of an optimization dataset.
    """
    result = OptimizationCollectionResult.from_server(
        client=public_client,
        spec_name="default",
        dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
        include_trajectory=True,
        final_molecule_only=False,
        subset=["cn1c(cccc1=o)c2ccccc2oc-0"])
    new_dataset = result.create_basic_dataset(dataset_name="new basicdataset",
                                              description="test new optimizationdataset",
                                              tagline='new optimization dataset',
                                              driver="energy")
    assert new_dataset.dataset_name == "new basicdataset"
    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 1
    result_geom = result.collection["Cn1c(cccc1=O)c2ccccc2OC"].entries[0].final_molecule.molecule.geometry
    # make sure the geometry is correct
    assert new_dataset.dataset["cn1c(cccc1=o)c2ccccc2oc-0"].initial_molecules[0].geometry.all() == result_geom.all()


def test_optimizationdataset_new_optimization(public_client):
    """
    Test creating a new optimizationdataset from the result of an optimization dataset.
    """
    result = OptimizationCollectionResult.from_server(
        client=public_client,
        spec_name="default",
        dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
        include_trajectory=True,
        final_molecule_only=False,
        subset=["cn1c(cccc1=o)c2ccccc2oc-0"])
    new_dataset = result.create_optimization_dataset(dataset_name="new optimizationdataset",
                                                     description="new test optimizationdataset",
                                                     tagline="new optimizationdataset")
    assert new_dataset.dataset_name == "new optimizationdataset"
    assert new_dataset.n_records == 1
    assert new_dataset.n_molecules == 1
    result_geom = result.collection["Cn1c(cccc1=O)c2ccccc2OC"].entries[0].final_molecule.molecule.geometry
    assert new_dataset.dataset["cn1c(cccc1=o)c2ccccc2oc-0"].initial_molecules[0].geometry.all() == result_geom.all()


def test_optimization_final_only_result(public_client):
    """
    Test gathering a result with only the final molecules in the records.
    """

    result = OptimizationCollectionResult.from_server(
        client=public_client,
        spec_name="default",
        dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
        include_trajectory=False,
        final_molecule_only=True)

    for optimization in result.collection.values():
        for entry in optimization.entries:
            assert len(entry.trajectory) == 1


@pytest.mark.parametrize("compression", [
    pytest.param("xz", id="lzma"),
    pytest.param("gz", id="gz"),
    pytest.param("bz2", id="bz2"),
    pytest.param(None, id="no compression")
])
def test_optimization_export_round_trip_compression(public_client, compression):
    """Test exporting the results to file and back."""

    with temp_directory():
        result = OptimizationCollectionResult.from_server(
            client=public_client,
            spec_name="default",
            dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
            include_trajectory=False,
            final_molecule_only=True)
        file_name = "dataset.json"
        result.export_results(filename=file_name, compression=compression)

        # now load the dataset back in
        if compression is not None:
            name = "".join([file_name, ".", compression])
        else:
            name = file_name
        result2 = OptimizationCollectionResult.parse_file(name)

        assert result.dict(exclude={"collection"}) == result2.dict(exclude={"collection"})


def test_basicdataset_result(public_client):
    """
    Test downloading a default dataset.
    """

    result = BasicCollectionResult.from_server(
        client=public_client,
        dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
        spec_name="default",
        method="b3lyp-d3bj",
        basis="dzvp",
    )

    assert result.driver.value == "hessian"
    assert result.method == "b3lyp-d3bj"
    assert result.basis == "dzvp"
    assert result.n_molecules == 150
    assert result.n_results == 298

    # now make sure the number of geometries and entries are the same
    for basic_result in result.collection.values():
        # test the lowest energy entry
        entry = basic_result.get_lowest_energy_entry()
        # now search for the lowest
        low_energy = 0
        for mol_entry in basic_result.entries:
            if mol_entry.energy < low_energy:
                low_energy = mol_entry.energy
        assert low_energy == entry.energy


def test_basicdataset_export_round_trip(public_client):
    """
    Test basic dataset round tripping to file.
    """

    with temp_directory():
        result = BasicCollectionResult.from_server(
            client=public_client,
            dataset_name="OpenFF Gen 2 Opt Set 1 Roche",
            spec_name="default",
            method="b3lyp-d3bj",
            basis="dzvp",
        )

        result.export_results("dataset.json")

        result2 = BasicCollectionResult.parse_file("dataset.json")

        assert result.dict(exclude={"collection"}) == result2.dict(exclude={"collection"})
        for molecule in result.collection:
            assert molecule in result2.collection


def test_torsiondrivedataset_result_default(public_client):
    """
    Test downloading a basic torsiondrive dataset from the archive.
    """
    import numpy as np
    from simtk import unit

    result = TorsionDriveCollectionResult.from_server(client=public_client,
                                                      spec_name="default",
                                                      dataset_name="TorsionDrive Paper",
                                                      include_trajectory=False,
                                                      final_molecule_only=False)

    # now we need to make sure that each optimization traj has only one molecule in it.
    for torsiondrive in result.collection.values():
        for optimization in torsiondrive.optimization.values():
            assert len(optimization.trajectory) == 2

    # make sure the three torsiondrives are pulled
    assert len(result.collection) == 3

    # now check the utility functions
    torsiondrive = result.collection["[ch2:3]([ch2:2][oh:4])[oh:1]_12"]
    assert torsiondrive.final_energies is not None
    # make sure there is an energy of every result
    assert len(torsiondrive.final_energies) == len(torsiondrive.optimization)
    mol = torsiondrive.molecule
    molecule = torsiondrive.get_torsiondrive()
    assert mol == molecule
    # make sure the conformers are loaded onto the molecule
    assert molecule.n_conformers == len(torsiondrive.optimization)
    # now check each conformer
    ordered_results = torsiondrive.get_ordered_results()
    for conformer, single_result in zip(molecule.conformers, ordered_results):
        assert np.allclose(conformer.in_units_of(unit.bohr).tolist(), single_result[1].molecule.geometry.tolist())

    # now make sure the lowest energy optimization is recognized
    lowest_result = torsiondrive.get_lowest_energy_optimisation()
    all_energies = list(torsiondrive.final_energies.values())
    assert lowest_result.final_energy == min(all_energies)


def test_torsiondrivedataset_final_result_only(public_client):
    """
    Make sure the final_molecule_only keyword is working
    """

    result = TorsionDriveCollectionResult.from_server(client=public_client,
                                                      spec_name="default",
                                                      dataset_name="TorsionDrive Paper",
                                                      include_trajectory=False,
                                                      final_molecule_only=True)

    # now we need to make sure that each optimization traj has only one molecule in it.
    for torsiondrive in result.collection.values():
        for optimization in torsiondrive.optimization.values():
            assert len(optimization.trajectory) == 1


def test_torsiondrivedataset_traj_subset(public_client):
    """
    Make sure the full trajectories are pulled when requested for a subset of molecules in a collection.
    """

    result = TorsionDriveCollectionResult.from_server(client=public_client,
                                                      spec_name="default",
                                                      dataset_name="TorsionDrive Paper",
                                                      include_trajectory=True,
                                                      final_molecule_only=False,
                                                      subset=["[ch2:3]([ch2:2][oh:4])[oh:1]_12"])

    # make sure one torsiondrive was pulled down
    assert len(result.collection) == 1
    # now make sure the full trajectory is pulled
    torsiondrive = result.collection["[ch2:3]([ch2:2][oh:4])[oh:1]_12"]
    for optimization in torsiondrive.optimization.values():
        assert len(optimization.trajectory) > 2


def test_torsiondrivedataset_export(public_client):
    """
    Make sure that the torsiondrive datasets can be exported.
    """

    with temp_directory():
        result = TorsionDriveCollectionResult.from_server(client=public_client,
                                                          spec_name="default",
                                                          dataset_name="TorsionDrive Paper",
                                                          include_trajectory=False,
                                                          final_molecule_only=True)

        result.export_results("dataset.json")

        result2 = TorsionDriveCollectionResult.parse_file("dataset.json")

        assert result.dict(exclude={"collection"}) == result2.dict(exclude={"collection"})
        for molecule in result.collection:
            assert molecule in result2.collection


def test_torsiondrivedataset_new_torsiondrive(public_client):
    """
    Test making a new torsiondrive dataset from the results of another.
    """

    result = TorsionDriveCollectionResult.from_server(client=public_client,
                                                      spec_name="default",
                                                      dataset_name="TorsionDrive Paper",
                                                      include_trajectory=True,
                                                      final_molecule_only=False,
                                                      subset=["[ch2:3]([ch2:2][oh:4])[oh:1]_12"])
    # make a new torsiondrive dataset
    new_dataset = result.create_torsiondrive_dataset(dataset_name="new torsiondrive dataset",
                                                     description="a test torsiondrive dataset",
                                                     tagline="a test torsiondrive dataset.")
    assert new_dataset.dataset_name == "new torsiondrive dataset"
    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 1
    entry = new_dataset.dataset["[ch2:3]([ch2:2][oh:4])[oh:1]_12"]
    # make sure all starting molecules are present
    assert len(entry.initial_molecules) == 24


def test_torsiondrivedataset_new_optimization(public_client):
    """
    Tast making a new optimization dataset of constrained optimizations from the results of a torsiondrive dataset.
    """

    result = TorsionDriveCollectionResult.from_server(client=public_client,
                                                      spec_name="default",
                                                      dataset_name="TorsionDrive Paper",
                                                      include_trajectory=True,
                                                      final_molecule_only=False,
                                                      subset=["[ch2:3]([ch2:2][oh:4])[oh:1]_12"])
    # make a new torsiondrive dataset
    new_dataset = result.create_optimization_dataset(dataset_name="new optimization dataset",
                                                     description="a test optimization dataset",
                                                     tagline="a test optimization dataset.")

    assert new_dataset.dataset_name == "new optimization dataset"
    assert new_dataset.n_molecules == 1
    assert new_dataset.n_records == 24
    dihedrals = set()
    for entry in new_dataset.dataset.values():
        assert entry.constraints.has_constraints is True
        assert len(entry.constraints.set) == 1
        dihedrals.add(entry.constraints.set[0].value)

    # now sort the dihedrals and make sure they are all present
    dihs = sorted(dihedrals)
    refs = [x for x in range(-165, 195, 15)]
    assert dihs == refs


def test_torsiondrive_new_basicdataset(public_client):
    """
    Test creating a new basicdataset of the final geometries of the current torsiondrive results class.
    """
    result = TorsionDriveCollectionResult.from_server(client=public_client,
                                                      spec_name="default",
                                                      dataset_name="TorsionDrive Paper",
                                                      include_trajectory=True,
                                                      final_molecule_only=False,
                                                      subset=["[ch2:3]([ch2:2][oh:4])[oh:1]_12"])
    new_dataset = result.create_basic_dataset(dataset_name="new basicdataset",
                                              description="new basicdataset",
                                              tagline="new basicdataset",
                                              driver="gradient")
    assert new_dataset.dataset_name == "new basicdataset"
    assert new_dataset.n_molecules == 1
    # make sure all of the molecule geometries are unpacked
    assert new_dataset.n_records == 24
    assert new_dataset.driver.value == "gradient"
