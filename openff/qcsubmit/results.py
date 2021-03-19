"""
A module with classes that can be used to collect results from the qcarchive and have them locally for filtering and
analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import qcelemental as qcel
import qcportal as ptl
from openff.toolkit.topology import Molecule
from pydantic import constr, validator
from qcelemental.models.types import Array
from qcportal.models import OptimizationRecord, ResultRecord
from qcportal.models.common_models import DriverEnum
from simtk import unit

from openff.qcsubmit.common_structures import IndexCleaner, Metadata, ResultsConfig
from openff.qcsubmit.exceptions import UnsupportedFiletypeError
from openff.qcsubmit.procedures import GeometricProcedure
from openff.qcsubmit.serializers import deserialize, serialize


class SingleResult(ResultsConfig):
    """
    This is a very basic result class that captures the coordinates of the calculation along with the main result
    and any extras that were calculated using scf properties.
    """

    molecule: ptl.models.Molecule
    wbo: Optional[Array[np.ndarray]] = None
    mbo: Optional[Array[np.ndarray]] = None
    id: int
    energy: Optional[float] = None
    gradient: Optional[Array[np.ndarray]] = None
    hessian: Optional[Array[np.ndarray]] = None
    extras: Optional[Dict] = None
    index: Optional[str] = None

    @validator("wbo", "mbo")
    def _check_wbo_and_hessian(cls, array):
        """
        Take the input wbo/hessian which is normally a list and cast it to a np.ndarry of the correct shape.
        """
        if array is None:
            return array
        else:
            atoms = np.sqrt(len(array)).astype(int)
            return array.reshape((atoms, -1))

    @validator("gradient")
    def _check_gradient(cls, gradient):
        """
        Take the gradient which is normally a flat list and cast it to the correct shape.
        """
        if gradient is None:
            return gradient
        else:
            return gradient.reshape((-1, 3))

    @classmethod
    def from_result_and_molecule(
        cls, result: ptl.models.ResultRecord, molecule: ptl.models.Molecule, index: str
    ) -> "SingleResult":
        """
        Instance the class from a result and corresponding molecule.

        Parameters:
            result: The qcportal results record where we pull out extra information.
            molecule: The qcportal molecule record which the result was computed for.
            index: An optional index that should be given to the result.
        """

        extras = result.extras.get("qcvars", None)
        return cls(
            molecule=molecule,
            wbo=extras.get("WIBERG_LOWDIN_INDICES", None)
            if extras is not None
            else None,
            mbo=extras.get("MAYER_INDICES", None) if extras is not None else None,
            energy=result.properties.return_energy,
            gradient=result.return_result
            if result.driver.value == "gradient"
            else None,
            hessian=result.return_result if result.driver.value == "hessian" else None,
            id=result.id,
            index=index,
        )

    def guess_connectivity(self) -> List[Tuple[int, int]]:
        """
        Use the qcelemental procedure to guess the connectivity.
        """

        conn = qcel.molutil.guess_connectivity(
            self.molecule.symbols, self.molecule.geometry
        )
        return conn

    def get_wbo_connectivity(
        self, wbo_threshold: float = 0.5
    ) -> List[Tuple[int, int, float]]:
        """
        Build the connectivity using the wbo for the final molecule.

        Returns:
            A list of tuples of the bond connections along with the WBO.
        """

        if self.wbo is None:
            return []
        bonds = []
        for i in range(self.wbo.shape[0]):
            for j in range(self.wbo.shape[1]):
                if self.wbo[i, j] > wbo_threshold:
                    # this is a bond
                    bonds.append((i, j, self.wbo[i, j]))

        return bonds

    def find_hydrogen_bonds_heuristic(
        self,
    ) -> List[Tuple[int, int]]:
        """
        Find hydrogen bonds in the final molecule using the Baker-Hubbard method.


        Returns:
        h_bonds : List[Tuple[int, int]]
            A list of atom indexes (acceptor and hydrogen) involved in hydrogen bonds.
        """

        cutoff = 4.72432  # angstrom to bohr cutoff
        # set up the required information
        h_acceptors = ["N", "O"]
        h_donors = ["N", "O"]
        molecule = self.molecule
        n_atoms = len(molecule.symbols)

        # create a distance matrix in bohr
        distance_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i):
                distance_matrix[i, j] = distance_matrix[j, i] = molecule.measure([i, j])

        h_bonds = set()
        # we need to make a new connectivity table
        for bond in self.guess_connectivity():
            # work out if we have a polar hydrogen
            if (
                molecule.symbols[bond[0]] in h_donors
                or molecule.symbols[bond[1]] in h_donors
            ):
                if molecule.symbols[bond[0]] == "H":
                    hydrogen_index = bond[0]
                    donor_index = bond[1]
                elif molecule.symbols[bond[1]] == "H":
                    hydrogen_index = bond[1]
                    donor_index = bond[0]
                else:
                    continue

                for i, distance in enumerate(distance_matrix[hydrogen_index]):
                    if donor_index != i:
                        if distance < cutoff and molecule.symbols[i] in h_acceptors:
                            # now check the angle
                            if molecule.measure([donor_index, hydrogen_index, i]) > 120:
                                hbond = tuple(sorted([hydrogen_index, i]))
                                h_bonds.add(hbond)

        return list(h_bonds)

    def find_hydrogen_bonds_wbo(
        self, hbond_threshold: float = 0.04, bond_threshold: float = 0.5
    ) -> List[Tuple[int, int]]:
        """
        Calculate if an internal hydrogen has formed using the WBO and return where it formed for the final molecule
        in the optimization.

        Note:
            The threshold is very low to be counted as a hydrogen bond.

        Parameters:
        hbond_threshold: float, optional, default=0.05
            The minimum wbo overlap to define a hydrogen bond by.

        Returns:
        h_bonds: List[Tuple[int, int]]
              A list of tuples of the atom indexes that have formed hydrogen bonds.
        """

        h_acceptors = ["N", "O"]
        h_donors = ["N", "O"]
        # get the molecule
        molecule = self.molecule
        if self.wbo is None:
            # if the wbo is missing return None
            return None
        wbo_bonds = self.get_wbo_connectivity(wbo_threshold=bond_threshold)
        # now loop over the molecule bonds and make sure we find a bond in the array
        h_bonds = set()
        for bond in wbo_bonds:
            # work out if we have a polar hydrogen
            if (
                molecule.symbols[bond[0]] in h_donors
                or molecule.symbols[bond[1]] in h_donors
            ):
                # look for an hydrogen atom
                if molecule.symbols[bond[0]] == "H":
                    hydrogen_index = bond[0]
                elif molecule.symbols[bond[1]] == "H":
                    hydrogen_index = bond[1]
                else:
                    continue

                # now loop over the columns and find the bond
                for i, bond_order in enumerate(self.wbo[hydrogen_index]):
                    if hbond_threshold < bond_order < 0.5:
                        if molecule.symbols[i] in h_acceptors:
                            hbond = tuple(sorted([hydrogen_index, i]))
                            h_bonds.add(hbond)

        return list(h_bonds)


class BasicResult(ResultsConfig):
    """
    The basic result condenses results for the same molecule in different conformers together and offers utility
    methods.
    """

    entries: List[SingleResult] = []
    attributes: Dict[str, Any] = {}
    index: str

    def add_single_result(
        self, result: ptl.models.ResultRecord, molecule: ptl.models.Molecule, index: str
    ) -> None:
        """
        Create and add a single result to the collection from the result, molecule and index.
        """

        single_result = SingleResult.from_result_and_molecule(
            result=result, molecule=molecule, index=index
        )
        self.add_entry(single_result)

    def add_entry(self, entry: SingleResult) -> None:
        """
        Add a new SingleResult to the result record.
        """

        self.entries.append(entry)

    @property
    def molecule(self) -> Molecule:
        """
        Build an openforcefield.topology.Molecule from the cmiles which is in the correct order to align with the
        QCArchive records.

        Returns:
            The openforcefield molecule representation of the molecule.
        """

        mol = Molecule.from_mapped_smiles(
            self.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
        )

        return mol

    def get_molecule_geometries(self) -> Molecule:
        """
        Gather all of the entries together and place the conformers on a single molecule object.

        Returns:
            molecule: A molecule instance with all of the conformers collapsed onto it.
        """

        molecule = self.molecule
        for entry in self.entries:
            geometry = unit.Quantity(entry.molecule.geometry, unit=unit.bohr)
            molecule.add_conformer(geometry)

        return molecule

    def get_lowest_energy_entry(self) -> SingleResult:
        """
        Search through the entries in the basic result and get the lowest energy result.

        Returns:
        result: The lowest energy result for this particular molecule.
        """

        results = [(result.energy, i) for i, result in enumerate(self.entries)]
        results.sort(key=lambda x: x[0])
        lowest_index = results[0][1]
        return self.entries[lowest_index]

    @property
    def n_entries(self) -> int:
        """
        Returns:
            The number of entries in the BasicResult.
        """
        return len(self.entries)


class BasicCollectionResult(IndexCleaner, ResultsConfig):
    """
    A basic dataset collection of results, these are individual entries and not condensed.
    """

    result_type: constr(regex="BasicCollectionResult") = "BasicCollectionResult"
    method: str
    basis: Optional[str] = None
    dataset_name: str
    program: str
    driver: DriverEnum
    scf_properties: Optional[List[str]]
    maxiter: Optional[int]
    spec_name: str
    provenance: Dict[str, Any] = {}
    dataset_tagline: Optional[str]
    dataset_tags: List[str] = []
    metadata: Dict[str, Any] = {}
    description: Optional[str]
    collection: Dict[str, BasicResult] = {}

    def _create_dataset_meta(self, collection_type: str) -> Metadata:
        """
        Format a new dataset metadata class from given data.
        """
        metadata = Metadata(
            collection_type=collection_type,
            dataset_name=self.dataset_name,
            short_description=self.dataset_tagline,
            long_description=self.description,
        )

        return metadata

    def export_results(self, filename: str, compression: Optional[str] = None) -> None:
        """
        Export the results to json file.

        Parameters:
            filename: The name of the json file which the results should be wrote to.
            compression: The type of compression that should be used.

        Note:
            The compression can also be supplied in the file_name.
        """

        if "json" in filename:
            serialize(serializable=self, file_name=filename, compression=compression)
        else:
            raise UnsupportedFiletypeError("Results can only be exported to json.")

    @classmethod
    def parse_file(cls, file_name: str):
        """
        Overwrite the parse file function to use decompression when needed.
        """
        data = deserialize(file_name=file_name)
        return cls(**data)

    @property
    def n_molecules(self) -> int:
        """
        Returns:
            The number of unique molecules in the basic dataset.
        """
        return len(self.collection.keys())

    @property
    def n_results(self) -> int:
        """
        Returns:
            The number of results in the whole dataset.
        """

        return sum([result.n_entries for result in self.collection.values()])

    def add_basic_result(self, basic_result: BasicResult) -> str:
        """
        Add a basic result to the collection.

        Parameters:
            basic_result: An instance of the BasicResult which collapses results for the same molecule down.

        Returns:
            The index the BasicResult is stored under in the collection.

        """

        if basic_result.index not in self.collection:
            self.collection[basic_result.index] = basic_result
        return basic_result.index

    @staticmethod
    def _gather_metadata(
        collection: ptl.collections.Dataset,
        client: ptl.FractalClient,
        spec_name: Optional[str] = "default",
        program: Optional[str] = None,
        method: Optional[str] = None,
        basis: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Gather metadata needed for BasicDataset results classes.

        These classes work slightly differently as the method and basis are not part of a spec and are separate
        variables. If `None` is given we look through the history for the last method and basis set combination.

        Parameters:
            collection: The dataset that we should pull the metadata from.
            client: The client from which the collection was pulled and the keywords should be pulled from.
            spec_name: The alias of the spec name used to compute the dataset.
            program: The QC program used to run the computation.
            method: The method used in the computation, if `None` is given the last method used will be searched.
            basis: The basis used in the computation if `None` is given the last basis set used will be searched.
        """

        data_model = collection.data
        for history in data_model.history:
            _, ran_program, ran_method, ran_basis, ran_spec = history
            if spec_name == ran_spec:
                scf_keywords = client.query_keywords(
                    data_model.alias_keywords[program or ran_program][spec_name]
                )[0]
                data = {
                    "spec_name": spec_name,
                    "dataset_name": collection.name,
                    "method": method or ran_method,
                    "basis": basis or ran_basis,
                    "program": program or ran_program,
                    "driver": data_model.default_driver,
                    "maxiter": scf_keywords.values["maxiter"],
                    "scf_properties": scf_keywords.values["scf_properties"],
                    "description": data_model.description,
                    "metadata": data_model.metadata,
                    "provenance": data_model.provenance,
                    "dataset_tagline": data_model.tagline,
                    "dataset_tags": data_model.tags,
                }
                return data

    @staticmethod
    def _query_in_chunks(
        query: List[str], result_type: str, client: ptl.FractalClient
    ) -> List[Union[ptl.models.Molecule, ResultRecord, OptimizationRecord]]:
        """
        Take the query list of ids and their type and query the database in query limit chunks.
        """

        client_results = []

        for i in range(0, len(query), client.query_limit):
            # get the results records
            if result_type == "molecule":
                client_results.extend(
                    client.query_molecules(id=query[i : i + client.query_limit])
                )
            elif result_type == "result":
                client_results.extend(
                    client.query_results(id=query[i : i + client.query_limit])
                )
            elif result_type == "procedure":
                client_results.extend(
                    client.query_procedures(id=query[i : i + client.query_limit])
                )
        return client_results

    @classmethod
    def from_server(
        cls,
        client: ptl.FractalClient,
        dataset_name: str,
        spec_name: str = "default",
        program: str = "psi4",
        method: Optional[str] = None,
        basis: Optional[str] = None,
        subset: Optional[List[str]] = None,
    ) -> "BasicCollectionResult":
        """
        Build up the collection result from a OptimizationDataset on a archive client this will also collapse the
        records into entries for the same molecules.

        Parameters:
            client: The fractal client we should contact to pull the results from.
            spec_name: The spec the data was calculated with that we want results for.
            dataset_name: The name of the Optimization set we want to pull down.
            program: The program used to compute the data, this is how the spec if stored in basic datasets.
            method: The method used to compute the requested data, if `None` is given then the last method used in the
                history is pulled.
            basis: The basis used to compute the requested data, if `None` is given then the last basis used in the
                history is pulled.
            subset: The chunk of result indexes that we should pull down.
        """

        # build the result object from metadata
        ds = client.get_collection("Dataset", dataset_name)
        metadata = cls._gather_metadata(
            collection=ds, spec_name=spec_name, client=client
        )
        collection_result = cls(**metadata)

        # query the database to get all of the result records requested
        query = ds.get_records(
            method=collection_result.method,
            basis=collection_result.basis,
            program=collection_result.program,
            subset=subset,
        )

        collection = {}

        # start loop through the records and adding them to the collection
        for index in query.index:
            result = query.loc[index].record
            try:
                if result.status.value.upper() == "COMPLETE":
                    # get the common identifier
                    # note basic datasets have no attributes currently
                    common_name, _ = cls._clean_index(index=index)
                    if common_name not in collection:
                        # build up a list of molecules and results and pull them from the database
                        collection[common_name] = {
                            "index": common_name,
                            "attributes": {},
                            "entries": [],
                        }
                    entry = {
                        "index": index,
                        "id": result.id,
                        "molecule_id": result.molecule,
                        "result": result,
                    }
                    collection[common_name]["entries"].append(entry)
            except AttributeError:
                # the record is an error nan float so skip it
                continue

        # now process the list into chucks that can be queried
        # only molecules have to be pulled
        query_molecules = []
        for data in collection.values():
            for entry in data["entries"]:
                query_molecules.append(entry["molecule_id"])

        # we will get one molecule for each result
        print("requested molecules", len(query_molecules))
        print("requested results", len(query_molecules))
        # now we need to form a list of molecules to query
        client_molecules = cls._query_in_chunks(
            query=query_molecules, result_type="molecule", client=client
        )

        # make into a look up table
        molecules_table = dict((molecule.id, molecule) for molecule in client_molecules)

        # now we need to build up the collection from the gathered results
        for data in collection.values():
            basic_result = BasicResult(
                index=data["index"], attributes=data["attributes"]
            )
            for entry in data["entries"]:
                # create the SingleResult entry
                molecule = molecules_table[entry["molecule_id"]]
                basic_result.add_single_result(
                    result=entry["result"], molecule=molecule, index=entry["index"]
                )

            collection_result.add_basic_result(basic_result=basic_result)

        return collection_result


class OptimizationEntryResult(ResultsConfig):
    """
    The optimization Entry Result is built from a series of SingleResults to form the trajectory.
    """

    trajectory: List[SingleResult] = []
    index: str
    id: int
    cmiles: str
    keywords: Optional[Dict[str, Any]]

    @property
    def energies(self) -> List[float]:
        """
        Returns:
             A  list of energies from the optimization trajectory.
        """
        return [molecule.energy for molecule in self.trajectory]

    @property
    def molecule(self) -> Molecule:
        """
        Build the molecule.
        """
        molecule = Molecule.from_mapped_smiles(self.cmiles)
        return molecule

    @property
    def final_molecule(self) -> SingleResult:
        """
        Get the final molecule.
        """
        return self.trajectory[-1]

    @property
    def initial_molecule(self) -> SingleResult:
        """
        Get the intial molecule.
        """

        return self.trajectory[0]

    @property
    def final_energy(self):
        """
        Get the finial energy of the optimisation.
        """

        return self.final_molecule.energy

    def get_initial_molecule(self) -> Molecule:
        """
        Take a template molecule the calculations were done on and return the molecule at the coordinates requested.
        """
        molecule = self.molecule
        geometry = unit.Quantity(
            self.initial_molecule.molecule.geometry, unit=unit.bohr
        )
        molecule.add_conformer(geometry)

        return molecule

    def get_final_molecule(self) -> Molecule:
        """
        Take a template molecule and return it at the final coordinates in the optimisation.
        """

        molecule = self.molecule
        geometry = unit.Quantity(self.final_molecule.molecule.geometry, unit=unit.bohr)
        molecule.add_conformer(geometry)

        return molecule

    def get_trajectory(self) -> Molecule:
        """
        Take a template molecule and return it with the trajectory attached to it.

        Note:
            If the full trajectory was not pulled on creating the dataset then it will only have the initial molecule
            and final molecule attached.
        """

        molecule = self.molecule
        for conformer in self.trajectory:
            geometry = unit.Quantity(conformer.molecule.geometry, unit=unit.bohr)
            molecule.add_conformer(geometry)

        return molecule

    @classmethod
    def from_server(
        cls,
        optimization_result: ptl.models.OptimizationRecord,
        cmiles: str,
        index: str,
        include_trajectory: bool = False,
        final_molecule_only: bool = False,
    ) -> "OptimizationEntryResult":
        """
        Parse an optimization record to get the required data.

        Parameters:
            optimization_result: The optimizationrecord object we want to download from the archive.
            cmiles: The attributes dictionary of the entry, this is all of the metadata of the entry including the
                cmiles data.
            index: The index of the entry which is being pulled from the archive as we can not back track to get it.
            include_trajectory: If the entire optimization trajectory should vbe pulled from the entry, this can include
                a lot of results.
            final_molecule_only: This will indicate to only pll down the final molecule in the trajectory and overwrites
                pulling the whole trajectory.

        Note:
            Normal execution will only pull the first and last molecule in a trajectory.
        """

        if final_molecule_only:
            traj = [optimization_result.trajectory[-1]]
            molecules = [optimization_result.final_molecule]
        elif include_trajectory:
            traj = optimization_result.trajectory
            molecules = optimization_result.get_molecular_trajectory()
        else:
            traj = [
                optimization_result.trajectory[0],
                optimization_result.trajectory[-1],
            ]
            molecules = [
                optimization_result.initial_molecule,
                optimization_result.final_molecule,
            ]

        result_trajectory = optimization_result.client.query_procedures(traj)
        result_molecules = optimization_result.client.query_molecules(molecules)
        data = {
            "index": index,
            "cmiles": cmiles,
            "id": optimization_result.id,
            "keywords": optimization_result.keywords,
        }

        entry = OptimizationEntryResult(**data)
        # now add in the trajectory
        for data in zip(result_trajectory, result_molecules):
            entry.add_single_result(*data)

        return entry

    def add_single_result(
        self, result: ptl.models.ResultRecord, molecule: ptl.models.Molecule
    ) -> None:
        """
        A helpful method to turn the molecule details and the result record into a SingleResult.
        """

        single_result = SingleResult.from_result_and_molecule(
            result=result, molecule=molecule, index=None
        )
        self.trajectory.append(single_result)

    def detect_connectivity_changes_wbo(self, wbo_threshold: float = 0.5) -> bool:
        """
        Detect if the connectivity has changed from the input cmiles specification or not using the WBO, a bond is
        detected based on the wbo_threshold supplied.

        Note:
            This is only compared for the final geometry.

        Returns:
            `True` if the connectivity has changed or `False` if it has not.
        """
        from openff.toolkit.topology import NotBondedError

        # grab the molecule with its bonds
        molecule = self.molecule
        if self.final_molecule.wbo is None:
            # if the wbo is missing return None
            return None
        # else build the connectivity based on the wbo_threshold
        wbo_bonds = self.final_molecule.get_wbo_connectivity(
            wbo_threshold=wbo_threshold
        )
        for bond in wbo_bonds:
            try:
                molecule.get_bond_between(bond[0], bond[1])
            except NotBondedError:
                return True

        return False

    def detect_connectivity_changes_heuristic(self) -> bool:
        """
        Guess the connectivity then check if it has changed from the initial input.

        Returns:
            `True` if the connectivity has changed based on the distance based rules
            `False` if the connectivity has not changed based on the rules.
        """
        molecule = self.molecule
        # guess the connectivity
        connectivity = self.final_molecule.guess_connectivity()
        # now compare the connectivity
        for bond in molecule.bonds:
            b_tup = tuple([bond.atom1_index, bond.atom2_index])
            if b_tup not in connectivity and reversed(tuple(b_tup)) not in connectivity:
                return True

        return False

    def find_hydrogen_bonds_wbo(
        self, hbond_threshold: float = 0.04, bond_threshold: float = 0.5
    ) -> List[Tuple[int, int]]:
        """
        Calculate if an internal hydrogen has formed using the WBO and return where it formed for the final molecule
        in the optimization.

        Note:
            The threshold is very low to be counted as a hydrogen bond.

        Parameters:
            hbond_threshold: The minimum wbo overlap to define a hydrogen bond by.

        Returns:
            A list of tuples of the atom indexes that have formed hydrogen bonds.
        """

        return self.final_molecule.find_hydrogen_bonds_wbo(
            hbond_threshold=hbond_threshold, bond_threshold=bond_threshold
        )

    def find_hydrogen_bonds_heuristic(
        self,
    ) -> List[Tuple[int, int]]:
        """
        Find hydrogen bonds in the final molecule using the Baker-Hubbard method.


        Returns:
            A list of atom indexes (acceptor and hydrogen) involved in hydrogen bonds.
        """

        return self.final_molecule.find_hydrogen_bonds_heuristic()


class OptimizationResult(BasicResult):
    """
    A Optimiszation result contains metadata about the molecule which is being optimized along with each of the
    optimization entries as each molecule may of been optimised multiple times from different starting conformations.
    """

    entries: List[OptimizationEntryResult] = []

    def add_entry(self, entry: OptimizationEntryResult) -> None:
        """
        Add a new OptimizationEntryResult to the result record.
        """

        self.entries.append(entry)

    def get_lowest_energy_optimisation(self) -> OptimizationEntryResult:
        """
        From all of the entries get the optimization that results in the lowest energy conformer, if any are the same
        the first conformer with this energy is returned.

        Returns:
            The attached OptimizationEntryResult with the lowest energy final conformer.
        """

        opt_results = [
            (opt_rec.final_energy, i) for i, opt_rec in enumerate(self.entries)
        ]
        opt_results.sort(key=lambda x: x[0])
        lowest_index = opt_results[0][1]
        return self.entries[lowest_index]

    def detect_connectivity_changes_wbo(
        self, wbo_threshold: float = 0.5
    ) -> Dict[int, bool]:
        """
        Detect any connectivity changes in the optimization entries and report them.

        Returns:
            A dictionary of the optimisation entry and a bool of the connectivity changed or not.
            `True` indicates the connectivity did change, `False` indicates it did not.
        """

        connectivity_changes = dict(
            (index, opt_rec.detect_connectivity_changes_wbo(wbo_threshold))
            for index, opt_rec in enumerate(self.entries)
        )
        return connectivity_changes

    def detect_connectivity_changes_heuristic(self) -> Dict[int, bool]:
        """
        Detect connectivity changes based on heuristic rules.

        Returns:
            A dictionary of the optimization entry and a bool representing if the connectivity has changed or not.
            `True` indicates the connectivity is now different from the input.
            `False` indicates the connectivity is the same as the input.
        """

        connectivity_changes = dict(
            (index, opt_rec.detect_connectivity_changes_heuristic())
            for index, opt_rec in enumerate(self.entries)
        )
        return connectivity_changes

    def detect_hydrogen_bonds_wbo(self, wbo_threshold: float = 0.04) -> Dict[int, bool]:
        """
        Detect hydrogen bonds in the final molecules using the wbo.

        Returns:
            A dictionary of the optimization entry and a bool representing if an internal hydrogen bond was found.
            `True` indicates a bond was found.
            `False` indicates a bond was not found.

        Note:
            You can also query where the hydrogen bond is formed using the `find_hydrogen_bonds_wbo` function on the
            corresponding entry.
        """
        hydrogen_bonds = {}
        for index, opt_rec in enumerate(self.entries):
            hbonds = opt_rec.find_hydrogen_bonds_wbo(wbo_threshold)
            result = bool(hbonds) if hbonds is not None else None
            hydrogen_bonds[index] = result

        return hydrogen_bonds

    def detect_hydrogen_bonds_heuristic(self) -> Dict[int, bool]:
        """
        Detect hydrogen bonds in the final molecule of the trajectory using the Baker-Hubbard rule based method.

        Returns:
            A dictionary of the optimization entry index and bool representing if an internal hydrogen bond was found.
            `True` indicates a bond was found.
            `False` indicates a bond was not found.

        Note:
            You can also query which atoms the bond was formed between using the `find_hydrogen_bonds_heuristic` method
            on the corresponding entry.
        """

        hydrogen_bonds = {}
        for index, opt_rec in enumerate(self.entries):
            hbonds = opt_rec.find_hydrogen_bonds_heuristic()
            result = bool(hbonds) if hbonds is not None else None
            hydrogen_bonds[index] = result

        return hydrogen_bonds

    @property
    def n_entries(self) -> int:
        """
        Get the number of optimization entries for this molecule.
        """

        return len(self.entries)


class OptimizationCollectionResult(BasicCollectionResult):
    """
    The master collection result which contains many optimizationResults representing the archive collection.
    """

    result_type: constr(regex="OptimizationCollection") = "OptimizationCollection"
    spec_description: str
    optimization_procedure: GeometricProcedure
    collection: Dict[str, OptimizationResult] = {}

    def create_basic_dataset(
        self,
        dataset_name: str,
        description: constr(min_length=8, regex="[a-zA-Z]"),
        tagline: constr(min_length=8, regex="[a-zA-Z]"),
        driver: DriverEnum,
        metadata: Optional[Metadata] = None,
    ) -> "BasicDataset":
        """
        Create a qcsubmit.datasets.BasicDataSet from the Optimization set final geometries.

        Parameters:
            dataset_name:
            description:
            tagline:
            driver: The type of driver the dataset should use.
            metadata: The metadata that should be put into the new dataset.

        Note:
            This is a very easy way to create a hessian dataset from the output of an OptimizationDataset.
        """
        from openff.qcsubmit.datasets import BasicDataset

        # create the dataset basic data
        data = self.dict(
            exclude={
                "collection",
                "description",
                "dataset_name",
                "dataset_tagline",
                "metadata",
                "driver",
            }
        )

        if metadata is not None:
            data["metadata"] = metadata.dict()

        dataset = BasicDataset(
            **data,
            dataset_name=dataset_name,
            description=description,
            dataset_tagline=tagline,
            driver=driver,
        )
        # now we need to add the QC_spec
        dataset.clear_qcspecs()
        dataset.add_qc_spec(
            method=self.method,
            basis=self.basis,
            program=self.program,
            spec_name=self.spec_name,
            spec_description=self.spec_description,
        )

        for common_index, entries in self.collection.items():
            for result in entries.entries:
                dataset.add_molecule(
                    index=result.index,
                    molecule=result.get_final_molecule(),
                    attributes=entries.attributes,
                    keywords=result.keywords,
                )

        return dataset

    def create_optimization_dataset(
        self,
        dataset_name: str,
        description: str,
        tagline: str,
        metadata: Optional[Metadata] = None,
    ) -> "OptimizationDataset":
        """
        Create a qcsubmit.datasets.OptimizationDataset from the current results collection.

        Parameters:
            dataset_name: The name that will be given to the new dataset.
            tagline: The tagline that should be given to the new dataset.
            description: The description that should be given to the new dataset.
            metadata: The metadata for the new dataset.

        Note:
            The dataset is created using the final geometries as the input geometries for the next optimization.
            Past datasets

        Returns:
            The instance of the Optimization dataset.
        """
        from openff.qcsubmit.datasets import OptimizationDataset

        # create the dataset basic data
        data = self.dict(
            exclude={
                "collection",
                "description",
                "dataset_name",
                "dataset_tagline",
                "metadata",
            }
        )

        if metadata is not None:
            data["metadata"] = metadata.dict()

        dataset = OptimizationDataset(
            **data,
            dataset_name=dataset_name,
            description=description,
            dataset_tagline=tagline,
        )
        # now we need to add the QC_spec
        dataset.clear_qcspecs()
        dataset.add_qc_spec(
            method=self.method,
            basis=self.basis,
            program=self.program,
            spec_name=self.spec_name,
            spec_description=self.spec_description,
        )

        # now we need to add the molecules
        for common_index, entries in self.collection.items():
            for result in entries.entries:
                dataset.add_molecule(
                    index=result.index,
                    molecule=result.get_final_molecule(),
                    attributes=entries.attributes,
                    keywords=result.keywords,
                )

        return dataset

    @classmethod
    def _build_proxy_query(
        cls,
        collection: Union[
            ptl.collections.OptimizationDataset, ptl.collections.TorsionDriveDataset
        ],
        spec_name: str,
        subset: List[str],
        client: ptl.FractalClient,
    ) -> pd.DataFrame:
        """
        Build a proxy query for a subset of the data.
        """
        procedures = {}
        for index in subset:
            entry = collection.data.records[index]
            procedures[entry.name] = entry.object_map[spec_name]
        # now do the query
        client_procedures = cls._query_in_chunks(
            query=list(procedures.values()), result_type="procedure", client=client
        )

        # now we need to create the dummy pd dataframe
        proc_lookup = {x.id: x for x in client_procedures}
        data = []
        for name, oid in procedures.items():
            data.append([name, proc_lookup[oid]])
        df = pd.DataFrame(data, columns=["index", spec_name])
        df.set_index("index", inplace=True)
        return df[spec_name]

    @staticmethod
    def _gather_metadata(
        collection: ptl.collections.Dataset,
        spec_name: str,
        client: ptl.FractalClient,
    ) -> Dict[str, str]:
        """
        Gather metadata needed for Optimization and Torsiondrive results classes.
        """
        spec = collection.data.specs[spec_name]
        optimization_procedure = GeometricProcedure.from_opt_spec(
            spec.optimization_spec
        )
        if spec.qc_spec.keywords is not None:
            scf_keywords = client.query_keywords(spec.qc_spec.keywords)[0].values
        else:
            scf_keywords = {}
        data_model = collection.data
        data = {
            "spec_name": spec_name,
            "spec_description": spec.description,
            "dataset_name": collection.name,
            "method": spec.qc_spec.method,
            "basis": spec.qc_spec.basis,
            "program": spec.qc_spec.program,
            "driver": spec.qc_spec.driver,
            "maxiter": scf_keywords.get("maxiter", None),
            "scf_properties": scf_keywords.get("scf_properties", None),
            "optimization_procedure": optimization_procedure,
            "provenance": data_model.provenance,
            "dataset_tags": data_model.tags,
            "tagline": data_model.tagline,
            "metadata": data_model.metadata,
            "description": data_model.description,
        }
        return data

    @classmethod
    def from_server(
        cls,
        client: ptl.FractalClient,
        spec_name: str,
        dataset_name: str,
        include_trajectory: Optional[bool] = False,
        final_molecule_only: Optional[bool] = False,
        subset: Optional[List[str]] = None,
    ) -> "OptimizationCollectionResult":
        """
        Build up the collection result from a OptimizationDataset on a archive client this will also collapse the
        records into entries for the same molecules.

        Parameters:
            client: The fractal client we should contact to pull the results from.
            spec_name: The spec the data was calculated with that we want results for.
            dataset_name: The name of the Optimization set we want to pull down.
            include_trajectory: If we should include the full trajectory when downloading the data, note this can
                significantly increase the amount of client requests.
            final_molecule_only: Only download the final geometry of each entry.
            subset: The chunk of result indexes that we should pull down.
        """

        # build the result object from metadata
        opt_ds = client.get_collection("OptimizationDataset", dataset_name)
        metadata = cls._gather_metadata(
            collection=opt_ds, spec_name=spec_name, client=client
        )
        collection_result = cls(**metadata)

        # query the database to get all of the optimization records
        if subset is None:
            query = opt_ds.query(spec_name)
            subset = list(opt_ds.data.records.keys())
        else:
            query = cls._build_proxy_query(
                collection=opt_ds, spec_name=spec_name, subset=subset, client=client
            )

        collection = {}

        # start loop through the records and adding them to the collection
        for index in subset:
            opt = opt_ds.data.records[index]
            try:
                opt_record = query.loc[opt.name]
                if opt_record.status.value.upper() == "COMPLETE":
                    # get the common identifier
                    common_name = opt.attributes["canonical_isomeric_smiles"]
                    if common_name not in collection:
                        # build up a list of molecules and results and pull them from the database
                        collection[common_name] = {
                            "index": common_name,
                            "attributes": opt.attributes,
                            "entries": [],
                        }
                    if final_molecule_only:
                        traj = [opt_record.trajectory[-1]]
                    elif include_trajectory:
                        traj = opt_record.trajectory
                    else:
                        traj = [opt_record.trajectory[0], opt_record.trajectory[-1]]
                    entry = {
                        "index": index,
                        "id": opt_record.id,
                        "trajectory_records": traj,
                        "cmiles": opt.attributes[
                            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                        ],
                        "keywords": opt_record.keywords,
                    }
                    collection[common_name]["entries"].append(entry)
            except KeyError:
                # this means there is no object map on the entry
                continue

        # now process the list into chucks that can be queried
        query_results = []
        for data in collection.values():
            for entry in data["entries"]:
                for result in entry["trajectory_records"]:
                    # build a list of trajectory ids to query
                    query_results.append(result)

        # we will get one molecule for each result
        print("requested molecules", len(query_results))
        print("requested results", len(query_results))
        # get the results
        client_results = cls._query_in_chunks(
            query=query_results, result_type="result", client=client
        )
        # now we need to form a list of molecules to query
        query_molecules = [result.molecule for result in client_results]
        client_molecules = cls._query_in_chunks(
            query=query_molecules, result_type="molecule", client=client
        )

        # make into a look up table
        molecules_table = dict((molecule.id, molecule) for molecule in client_molecules)
        results_table = dict((result.id, result) for result in client_results)

        # now we need to build up the collection from the gathered results
        for data in collection.values():
            opt_result = OptimizationResult(
                index=data["index"], attributes=data["attributes"]
            )
            for entry in data["entries"]:
                # create the optimization entry
                opt_entry = OptimizationEntryResult(**entry)
                for result_id in entry["trajectory_records"]:
                    # now we need to make the single results
                    # this will create a trajectory
                    result = results_table[result_id]
                    molecule = molecules_table[result.molecule]
                    opt_entry.add_single_result(result, molecule)
                opt_result.add_entry(opt_entry)
            collection_result.add_optimization_result(opt_result)

        return collection_result

    def add_optimization_result(self, result: OptimizationResult) -> str:
        """
        Add an optimization result to the collection if the molecule has been seen before it will be condensed to an
        on the correct result.

        Retunrs:
            The string of the index it has been stored under.
        """

        if result.index not in self.collection:
            self.collection[result.index] = result

        return result.index


class TorsionDriveResult(ResultsConfig):
    """
    This class holds the individual constrained optimisations and is the equivalent to a record in qcarchive.
    """

    optimization: Dict[str, OptimizationEntryResult] = {}
    dihedrals: List[Tuple[int, int, int, int]]
    grid_spacing: List[int]
    energy_upper_limit: float
    dihedral_ranges: Optional[List[Tuple[int]]]
    energy_decrease_thresh: Optional[float]
    index: str
    initial_molecules: List[str]
    attributes: Dict[str, str]
    final_energies: Dict[str, float]

    def add_entry(self, angle: str, entry: OptimizationEntryResult) -> None:
        """
        Add a new OptimizationEntryResult to the result record.
        """

        self.optimization[angle] = entry

    @property
    def molecule(self) -> Molecule:
        """
        Build an openforcefield.topology.Molecule from the cmiles which is in the correct order to align with the
        QCArchive records.

        Returns:
         mol: The openforcefield molecule representation of the molecule.
        """

        mol = Molecule.from_mapped_smiles(
            self.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
        )

        return mol

    def get_input_molecules(self) -> Molecule:
        """
        Return the openforcefield representation of the molecule with all of the input conformers collapsed onto the
        molecule.
        """

        molecule = self.molecule
        for conformer in self.initial_molecules:
            geometry = unit.Quantity(conformer.geometry, unit=unit.bohr)
            molecule.add_conformer(geometry)

        return molecule

    def get_lowest_energy_optimisation(self) -> OptimizationEntryResult:
        """
        Get the OptimizationEntryResult which results in the lowest energy optimisation.
        """

        opt_results = [
            (opt_rec.final_energy, angle)
            for angle, opt_rec in self.optimization.items()
        ]
        opt_results.sort(key=lambda x: x[0])
        lowest_index = opt_results[0][1]
        return self.optimization[lowest_index]

    def get_torsiondrive(self) -> Molecule:
        """
        Get the torsiondrive trajectory collapsed onto a single openforcefield.topology.Molecule for viewing or
        exporting.
        """
        import re

        molecule = self.molecule
        # now sort the angles
        angles = [
            [int(x) for x in re.findall("-*[0-9]+", angle)]
            for angle in self.optimization.keys()
        ]
        angles.sort(key=lambda x: x[0])
        for angle in angles:
            or_molecule = self.optimization[str(angle)].get_final_molecule()
            molecule.add_conformer(or_molecule.conformers[0])
        return molecule

    def get_ordered_results(self) -> List[Tuple[List[int], SingleResult]]:
        """
        Create an ordered list of the optimization results sorted by the angle of the dihedral.
        """
        import re

        results = []
        for angles, optimization in self.optimization.items():
            results.append(
                (
                    [int(x) for x in re.findall("-*[0-9]+", angles)],
                    optimization.final_molecule,
                )
            )
        # now sort the list
        results.sort(key=lambda x: x[0])
        return results

    def detect_connectivity_changes_wbo(
        self, wbo_threshold: float = 0.5
    ) -> Dict[Tuple[int], bool]:
        """
        Detect any connectivity changes in the optimization entries and report them.

        Returns:
            A dictionary of the optimisation entry and a bool of the connectivity changed or not.
            `True` indicates the connectivity did change, `False` indicates it did not.
        """

        connectivity_changes = dict(
            (angle, opt_rec.detect_connectivity_changes_wbo(wbo_threshold))
            for angle, opt_rec in self.optimization.items()
        )
        return connectivity_changes

    def detect_connectivity_changes_heuristic(self) -> Dict[Tuple[int], bool]:
        """
        Detect connectivity changes based on heuristic rules.

        Returns:
            A dictionary of the optimization entry and a bool representing if the connectivity has changed or not.
            `True` indicates the connectivity is now different from the input.
            `False` indicates the connectivity is the same as the input.
        """

        connectivity_changes = dict(
            (angle, opt_rec.detect_connectivity_changes_heuristic())
            for angle, opt_rec in self.optimization.items()
        )
        return connectivity_changes

    def detect_hydrogen_bonds_wbo(
        self, wbo_threshold: float = 0.04
    ) -> Dict[Tuple[int], bool]:
        """
        Detect hydrogen bonds in the final molecules using the wbo.

        Returns:
            A dictionary of the optimization entry and a bool representing if an internal hydrogen bond was found.
            `True` indicates a bond was found.
            `False` indicates a bond was not found.

        Note:
            You can also query where the hydrogen bond is formed using the `find_hydrogen_bonds_wbo` function on the
            corresponding entry.
        """
        hydrogen_bonds = {}
        for angle, opt_rec in self.optimization.items():
            hbonds = opt_rec.find_hydrogen_bonds_wbo(wbo_threshold)
            result = bool(hbonds) if hbonds is not None else None
            hydrogen_bonds[angle] = result

        return hydrogen_bonds

    def detect_hydrogen_bonds_heuristic(self) -> Dict[Tuple[int], bool]:
        """
        Detect hydrogen bonds in the final molecule of the trajectory using the Baker-Hubbard rule based method.

        Returns:
            A dictionary of the optimization entry index and bool representing if an internal hydrogen bond was found.
            `True` indicates a bond was found.
            `False` indicates a bond was not found.

        Note:
            You can also query which atoms the bond was formed between using the `find_hydrogen_bonds_heuristic` method
            on the corresponding entry.
        """

        hydrogen_bonds = {}
        for angle, opt_rec in self.optimization.items():
            hbonds = opt_rec.find_hydrogen_bonds_heuristic()
            result = bool(hbonds) if hbonds is not None else None
            hydrogen_bonds[angle] = result

        return hydrogen_bonds


class TorsionDriveCollectionResult(OptimizationCollectionResult):
    """
    The master collection result which contains many optimizationResults representing the archive collection.
    """

    result_type: constr(regex="TorsionDriveCollection") = "TorsionDriveCollection"
    collection: Dict[str, TorsionDriveResult] = {}

    class Config:
        title = "TorsionDriveCollectionResult"

    def create_optimization_dataset(
        self,
        dataset_name: str,
        description: constr(min_length=8, regex="[a-zA-Z]"),
        tagline: constr(min_length=8, regex="[a-zA-Z]"),
        metadata: Optional[Metadata] = None,
    ) -> "OptimizationDataset":
        """
        Create an optimization dataset from the results of the current torsion drive dataset. This will result in many constrained optimizations for each molecule.

        Parameters:
            dataset_name: The name that the new dataset will be submitted under.
            description: A longer description string of the datasets purpose.
            tagline: A short tag line explaining the datasets purpose that will be displayed on the archive.
            metadata: The required metadata that will be supplied to the dataset.

        Note:
            The final geometry of each torsiondrive constrained optimization is supplied as a starting geometry.
        """
        from openff.qcsubmit.datasets import OptimizationDataset

        data = self.dict(
            exclude={
                "collection",
                "description",
                "dataset_name",
                "dataset_tagline",
                "metadata",
            }
        )

        if metadata is not None:
            data["metadata"] = metadata.dict()

        dataset = OptimizationDataset(
            **data,
            dataset_name=dataset_name,
            description=description,
            dataset_tagline=tagline,
        )
        # now we need to add the QC_spec
        dataset.clear_qcspecs()
        dataset.add_qc_spec(
            method=self.method,
            basis=self.basis,
            program=self.program,
            spec_name=self.spec_name,
            spec_description=self.spec_description,
        )
        # now we need to fill the dataset
        for result in self.collection.values():
            attributes = result.attributes
            # now we need to add a new optimization for each of the geometries in the torsiondrive
            index = result.attributes["canonical_isomeric_smiles"]
            for i, optimization in enumerate(result.optimization.values()):
                dataset.add_molecule(
                    index=index + f"_{i}",
                    molecule=optimization.get_final_molecule(),
                    attributes=attributes,
                    extras={},
                    keywords=optimization.keywords,
                )
        return dataset

    def create_basic_dataset(
        self,
        dataset_name: str,
        description: constr(min_length=8, regex="[a-zA-Z]"),
        tagline: constr(min_length=8, regex="[a-zA-Z]"),
        driver: DriverEnum,
        metadata: Optional[Metadata] = None,
    ) -> "BasicDataset":
        """
        Create a new basicdataset from the results of the current dataset.

        Note:
            The final geometries of the torsiondrive are rolled into a single molecule which is expanded on submission.

         Parameters:
            dataset_name: The name that the new dataset will be submitted under.
            description: A longer description string of the datasets purpose.
            tagline: A short tag line explaining the datasets purpose that will be displayed on the archive.
            metadata: The required metadata that will be supplied to the dataset.
            driver: The driver to be used on the basic dataset.

        """
        from openff.qcsubmit.datasets import BasicDataset

        # create the dataset basic data
        data = self.dict(
            exclude={
                "collection",
                "description",
                "dataset_name",
                "dataset_tagline",
                "metadata",
                "driver",
            }
        )

        if metadata is not None:
            data["metadata"] = metadata.dict()

        dataset = BasicDataset(
            **data,
            dataset_name=dataset_name,
            description=description,
            dataset_tagline=tagline,
            driver=driver,
        )
        # now we need to add the QC_spec
        dataset.clear_qcspecs()
        dataset.add_qc_spec(
            method=self.method,
            basis=self.basis,
            program=self.program,
            spec_name=self.spec_name,
            spec_description=self.spec_description,
        )

        # now we need to fill the dataset
        for result in self.collection.values():
            dataset.add_molecule(
                index=result.attributes["canonical_isomeric_smiles"],
                molecule=result.get_torsiondrive(),
                attributes=result.attributes,
            )

        return dataset

    def create_torsiondrive_dataset(
        self,
        dataset_name: str,
        description: constr(min_length=8, regex="[a-zA-Z]"),
        tagline: constr(min_length=8, regex="[a-zA-Z]"),
        metadata: Optional[Metadata] = None,
    ) -> "TorsiondriveDataset":
        """
        Create a torsiondrive dataset from the results of the current dataset.

        Parameters:
            dataset_name: The name that the new dataset will be submitted under.
            description: A longer description string of the datasets purpose.
            tagline: A short tag line explaining the datasets purpose that will be displayed on the archive.
            metadata: The required metadata that will be supplied to the dataset.

        Note:
            The final geometry of each torsiondrive constrained optimization is supplied as a starting geometry.

        Returns:
            A TorsiondriveDataset dataset instance that can be submited to a client, built from the final geometries
            of the current results torsiondrive dataset.
        """
        from openff.qcsubmit.datasets import TorsiondriveDataset

        # create the dataset basic data
        data = self.dict(
            exclude={
                "collection",
                "description",
                "dataset_name",
                "dataset_tagline",
                "metadata",
            }
        )

        if metadata is not None:
            data["metadata"] = metadata.dict()

        dataset = TorsiondriveDataset(
            **data,
            dataset_name=dataset_name,
            description=description,
            dataset_tagline=tagline,
        )
        # now we need to add the QC_spec
        dataset.clear_qcspecs()
        dataset.add_qc_spec(
            method=self.method,
            basis=self.basis,
            program=self.program,
            spec_name=self.spec_name,
            spec_description=self.spec_description,
        )

        # now we need to fill in the dataset data
        for index, result in self.collection.items():
            # get the torsion drive trajectory onto the molecule
            tdrive_mol = result.get_torsiondrive()
            dataset.add_molecule(
                index=index,
                molecule=tdrive_mol,
                attributes=result.attributes,
                dihedrals=result.dihedrals,
            )
        return dataset

    def add_torsiondrive(self, torsriondrive: TorsionDriveResult) -> str:
        """
        Add a torsiondrive result to the collection under its index.

        Parameters:
        torsriondrive: The instance of the TorsionDriveResult which should be added to the collection.

        Returns:
            The index string the TorsionDriveResult was stored under.
        """
        self.collection[torsriondrive.index] = torsriondrive
        return torsriondrive.index

    @classmethod
    def from_server(
        cls,
        client: ptl.FractalClient,
        spec_name: str,
        dataset_name: str,
        include_trajectory: Optional[bool] = False,
        final_molecule_only: Optional[bool] = False,
        subset: Optional[List[str]] = None,
    ) -> "TorsionDriveCollectionResult":
        """
        Build up the collection result from a TorsionDriveDataset on a archive client this will also collapse the
        records into entries for the same molecules.

        Parameters:
            client: The fractal client we should contact to pull the results from.
            spec_name: The spec the data was calculated with that we want results for.
            dataset_name: The name of the Optimization set we want to pull down.
            include_trajectory: If we should include the full trajectory when downloading the data, note this can significantly increase the
                amount of client requests.
            final_molecule_only: Only download the final geometry of each entry.
            subset: The chunk of result indexes that we should pull down.
        """

        # build the result object from metadata
        td_ds = client.get_collection("TorsionDriveDataset", dataset_name)
        metadata = cls._gather_metadata(
            collection=td_ds, spec_name=spec_name, client=client
        )
        collection_result = cls(**metadata)

        if subset is None:
            query = td_ds.query(spec_name)
            subset = list(td_ds.data.records.keys())
        else:
            query = cls._build_proxy_query(
                collection=td_ds, spec_name=spec_name, subset=subset, client=client
            )

        collection = {}
        query_optimizations = []
        # start looping through the dataset building place holders for the results
        for index in subset:
            tdrive = td_ds.data.records[index]
            try:
                tdrive_record = query.loc[tdrive.name]
                if tdrive_record.status.value.upper() == "COMPLETE":
                    # get the torsiondrive keywords
                    data = tdrive.td_keywords.dict()
                    # add the extra data
                    data["index"] = index
                    data["initial_molecules"] = tdrive.initial_molecules
                    data["attributes"] = tdrive.attributes
                    data["final_energies"] = tdrive_record.final_energy_dict
                    # we need to build  up a list of the optimizations we need to query
                    optimizations = {}
                    for angle, min_pos in tdrive_record.minimum_positions.items():
                        optimizations[angle] = tdrive_record.optimization_history[
                            angle
                        ][min_pos]
                    query_optimizations.extend(list(optimizations.values()))
                    data["optimization_data"] = optimizations
                    # save the place holder information
                    collection[index] = data
            except KeyError:
                # the object map is empty
                continue

        # now we have to query all of the optimizations to get the result
        # and molecule ids
        print("requested optimizations", len(query_optimizations))
        client_optimizations = cls._query_in_chunks(
            query=query_optimizations, result_type="procedure", client=client
        )

        # now we need to build up a list of the molecules and results to request
        query_results = []
        for opt in client_optimizations:
            if final_molecule_only:
                query_results.append(opt.trajectory[-1])
            elif include_trajectory:
                query_results.extend(opt.trajectory)
            else:
                query_results.extend([opt.trajectory[0], opt.trajectory[-1]])

        # we will get one molecule for each result
        print("requested molecules", len(query_results))
        print("requested results", len(query_results))
        client_results = cls._query_in_chunks(
            query=query_results, result_type="result", client=client
        )
        # now we need to form a list of molecules to query
        query_molecules = [result.molecule for result in client_results]
        client_molecules = cls._query_in_chunks(
            query=query_molecules, result_type="molecule", client=client
        )

        # make into a look up table
        optimizations_table = dict((opt.id, opt) for opt in client_optimizations)
        molecules_table = dict((molecule.id, molecule) for molecule in client_molecules)
        results_table = dict((result.id, result) for result in client_results)

        # now we need to build up the collection from the gathered results
        for data in collection.values():
            torsiondrive = TorsionDriveResult(**data)
            # now we just need to make the optimization entries
            for angle, optimization_id in data["optimization_data"].items():
                entry = OptimizationEntryResult(
                    index=torsiondrive.index,
                    id=optimization_id,
                    cmiles=torsiondrive.attributes[
                        "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                    ],
                    keywords=optimizations_table[optimization_id].keywords,
                )

                # now we need to add the trajectory for each optimization
                if final_molecule_only:
                    traj_ids = [optimizations_table[optimization_id].trajectory[-1]]
                elif include_trajectory:
                    traj_ids = optimizations_table[optimization_id].trajectory
                else:
                    opt = optimizations_table[optimization_id]
                    traj_ids = [opt.trajectory[0], opt.trajectory[-1]]
                for result_id in traj_ids:
                    # get the molecule and the result and add it to the collection
                    result = results_table[result_id]
                    molecule = molecules_table[result.molecule]
                    entry.add_single_result(result, molecule)
                # add the constrained optimization to the torsiondrive
                torsiondrive.add_entry(angle=angle, entry=entry)
            # now add the torsiondrive to the collection
            collection_result.add_torsiondrive(torsiondrive)

        return collection_result


OptimizationEntryResult.update_forward_refs()
