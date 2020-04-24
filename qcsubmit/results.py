"""
A module with classes that can be used to collect results from the qcarchive and have them locally for filtering and
analysis.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from openforcefield.topology import Molecule
import numpy as np
from simtk import unit
import qcportal as ptl
import pandas as pd
import qcelemental as qcel


class SingleResult(BaseModel):
    """
    This is a very basic result class that captures the coordinates of the calculation along with the main result
    and any extras that were calculated using scf properties.
    """

    molecule: ptl.models.Molecule
    wbo: List[float]
    id: int
    energy: Optional[float] = None
    gradient: Optional[np.ndarray] = None
    hessian: Optional[List[List[float]]] = None
    extras: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}

    def guess_connectivity(self) -> List[Tuple[int, int]]:
        """
        Use the qcelemental procedure to guess the connectivity.
        """

        conn = qcel.molutil.guess_connectivity(self.molecule.symbols, self.molecule.geometry)
        return conn


class OptimizationEntryResult(BaseModel):
    """
    The optimization Entry Result is built from a series of SingleResults to form the trajectory.
    """

    trajectory: List[SingleResult] = []
    index: str
    id: int
    cmiles: str

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}

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
        geometry = unit.Quantity(self.initial_molecule.molecule.geometry, unit=unit.bohr)
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

        Parameters
        ----------
        optimization_result : qcportal.models.OptimizationRecord
            The optimizationrecord object we want to download from the archive.
        cmiles : Dict[str, str],
            The attributes dictionary of the entry, this is all of the metadata of the entry including the cmiles data.
        index : str,
            The index of the entry which is being pulled from the archive as we can not back track to get it.
        include_trajectory : bool, optional, default=False,
            If the entire optimization trajectory should vbe pulled from the entry, this can include a lot of results.
        final_molecule_only : bool, optional, default=False,
            This will indicate to only pll down the final molecule in the trajectory and overwrites pulling the whole
            trajectory.

        Notes
        -----
            Normal execution will only pull the first and last molecule in a trajectory.
        """

        if final_molecule_only:
            traj = [optimization_result.trajectory[-1]]
            molecules = [optimization_result.final_molecule]
        elif include_trajectory:
            traj = optimization_result.trajectory
            molecules = optimization_result.get_molecular_trajectory()
        else:
            traj = [optimization_result.trajectory[0], optimization_result.trajectory[-1]]
            molecules = [optimization_result.initial_molecule, optimization_result.final_molecule]

        result_trajectory = optimization_result.client.query_procedures(traj)
        result_molecules = optimization_result.client.query_molecules(molecules)
        data = {"index": index, "cmiles": cmiles, "id": optimization_result.id}

        entry = OptimizationEntryResult(**data)
        # now add in the trajectory
        for data in zip(result_trajectory, result_molecules):
            entry.add_single_result(*data)

        return entry

    def add_single_result(self, result: ptl.models.ResultRecord, molecule: ptl.models.Molecule) -> None:
        """
        A helpful method to turn the molecule details and the result record into a SingleResult.
        """

        extras = result.extras.get("qcvars", None)
        single_result = SingleResult(
            molecule=molecule,
            wbo=extras.get("WIBERG_LOWDIN_INDICES", [np.nan]) if extras is not None else [np.nan],
            energy=result.properties.return_energy,
            gradient=result.return_result,
            id=result.id,
        )

        self.trajectory.append(single_result)

    def get_wbo_connectivity(self, wbo_threshold: float = 0.7) -> List[Tuple[int, int, float]]:
        """
        Build the connectivity using the wbo for the final molecule.

        Returns
        -------
            A list of tuples of the bond connections along with the WBO.
        """

        molecule = self.molecule
        if self.final_molecule.wbo == [np.nan]:
            return []
        wbo = np.array(self.final_molecule.wbo).reshape((molecule.n_atoms, molecule.n_atoms))
        bonds = []
        for i in range(molecule.n_atoms):
            for j in range(i):
                if wbo[i, j] > wbo_threshold:
                    # this is a bond
                    bonds.append((i, j, wbo[i, j]))

        return bonds

    def detect_connectivity_changes_wbo(self, wbo_threshold: float = 0.65) -> bool:
        """
        Detect if the connectivity has changed from the input cmiles specification or not using the WBO, a bond is
        detected based on the wbo_threshold supplied.

        Notes
        -----
            This is only compared for the final geometry.

        Returns
        -------
            `True` if the connectivity has changed or `False` if it has not.
        """
        # grab the molecule with its bonds
        molecule = self.molecule
        # cast the wbo into the correct shape
        if self.final_molecule.wbo == [np.nan]:
            # if the wbo is missing return None
            return None
        wbo = np.array(self.final_molecule.wbo).reshape((molecule.n_atoms, molecule.n_atoms))
        # now loop over the molecule bonds and make sure we find a bond in the array
        for bond in molecule.bonds:
            if wbo[bond.atom1_index, bond.atom2_index] < wbo_threshold:
                return True
        else:
            return False

    def detect_connectivity_changes_heuristic(self) -> bool:
        """
        Guess the connectivity then check if it has changed from the initial input.

        Returns
        -------
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

        else:
            return False

    def find_hydrogen_bonds_wbo(self, hbond_threshold: float = 0.05) -> List[Tuple[int, int]]:
        """
        Calculate if an internal hydrogen has formed using the WBO and return where it formed.

        Notes
        -----
            The threshold is very low to be counted as a hydrogen bond.

        Parameters
        ----------
            hbond_threshold: float, optional, default=0.05
                The minimum wbo overlap to define a hydrogen bond by.

        Returns
        -------
            h_bonds: List[Tuple[int, int]]
              A list of tuples of the atom indexes that have formed hydrogen bonds.
        """

        h_acceptors = [7, 8, 16]
        h_donors = [7, 8, 16]
        # get the molecule
        molecule = self.molecule
        # cast the wbo into the correct shape
        if self.final_molecule.wbo == [np.nan]:
            # if the wbo is missing return None
            return None
        wbo = np.array(self.final_molecule.wbo).reshape((molecule.n_atoms, molecule.n_atoms))
        # now loop over the molecule bonds and make sure we find a bond in the array
        h_bonds = set()
        for bond in molecule.bonds:
            # work out if we have a polar hydrogen
            if (
                molecule.atoms[bond.atom1_index].atomic_number in h_donors
                or molecule.atoms[bond.atom2_index].atomic_number in h_donors
            ):
                # look for an hydrogen atom
                if molecule.atoms[bond.atom1_index].atomic_number == 1:
                    hydrogen_index = bond.atom1_index
                elif molecule.atoms[bond.atom2_index].atomic_number == 1:
                    hydrogen_index = bond.atom2_index
                else:
                    continue

                # now loop over the columns and find the bond
                for i, bond_order in enumerate(wbo[hydrogen_index]):
                    if hbond_threshold < bond_order < 0.5:
                        if molecule.atoms[i].atomic_number in h_acceptors:
                            hbond = tuple(sorted([hydrogen_index, i]))
                            h_bonds.add(hbond)

        return list(h_bonds)

    def find_hydrogen_bonds_heuristic(self,) -> List[Tuple[int, int]]:
        """
        Find hydrogen bonds in the final molecule using the Baker-Hubbard method.


        Returns
        -------
        h_bonds : List[Tuple[int, int]]
            A list of atom indexes (acceptor and hydrogen) involved in hydrogen bonds.
        """

        cutoff = 4.72432  # angstrom to bohr cutoff
        # set up the required information
        h_acceptors = ["N", "O", "S"]
        h_donors = ["N", "O", "S"]
        molecule = self.final_molecule.molecule
        n_atoms = self.molecule.n_atoms

        # create a distance matrix in bohr
        distance_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i):
                distance_matrix[i, j] = distance_matrix[j, i] = molecule.measure([i, j])

        h_bonds = set()
        # we need to make a new connectivity table
        for bond in self.final_molecule.guess_connectivity():
            # work out if we have a polar hydrogen
            if molecule.symbols[bond[0]] in h_donors or molecule.symbols[bond[1]] in h_donors:
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


class OptimizationResult(BaseModel):
    """
    A Optimiszation result contains metadata about the molecule which is being optimized along with each of the
    optimization entries as each molecule may of been optimised multiple times from different starting conformations.
    """

    entries: List[OptimizationEntryResult] = []
    attributes: Dict
    index: str

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}

    def add_entry(self, entry: OptimizationEntryResult) -> None:
        """
        Add a new OptimizationEntryResult to the result record.
        """

        self.entries.append(entry)

    @property
    def molecule(self) -> Molecule:
        """
        Build an openforcefield.topology.Molecule from the cmiles which is in the correct order to align with the
        QCArchive records.

        Returns
        -------
        mol : openforcefield.topology.Molecule,
            The openforcefield molecule representation of the molecule.
        """

        mol = Molecule.from_mapped_smiles(self.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"])

        return mol

    def get_lowest_energy_optimisation(self) -> OptimizationEntryResult:
        """
        From all of the entries get the optimization that results in the lowest energy conformer, if any are the same
        the first conformer with this energy is returned.

        Returns
        -------
        OptimizationEntryResult : qcsubmit.results.OptimizationEntryResult,
            The attached OptimizationEntryResult with the lowest energy final conformer.
        """

        opt_results = [(opt_rec.final_energy, i) for i, opt_rec in enumerate(self.entries)]
        opt_results.sort(key=lambda x: x[0])
        lowest_index = opt_results[0][1]
        return self.entries[lowest_index]

    def detect_connectivity_changes_wbo(self, wbo_threshold: float = 0.65) -> Dict[int, bool]:
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

        Returns
        -------
         connectivity_changes : Dict[int, bool],
            A dictionary of the optimization entry and a bool representing if the connectivity has changed or not.
            `True` indicates the connectivity is now different from the input.
            `False` indicates the connectivity is the same as the input.
        """

        connectivity_changes = dict(
            (index, opt_rec.detect_connectivity_changes_heuristic()) for index, opt_rec in enumerate(self.entries)
        )
        return connectivity_changes

    def detect_hydrogen_bonds_wbo(self, wbo_threshold: float = 0.05) -> Dict[int, bool]:
        """
        Detect hydrogen bonds in the final molecules using the wbo.

        Returns
        -------
        hydrogen_bonds :  Dict[int, bool],
            A dictionary of the optimization entry and a bool representing if an internal hydrogen bond was found.
            `True` indicates a bond was found.
            `False` indicates a bond was not found.

        Notes
        -----
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

        Returns
        -------
        hydrogen_bonds : Dict[int, bool],
            A dictionary of the optimization entry index and bool representing if an internal hydrogen bond was found.
            `True` indicates a bond was found.
            `False` indicates a bond was not found.

        Notes
        -----
            You can also query which atoms the bond was formed between using the `find_hydrogen_bonds_heuristic` method
            on the corresponding entry.
        """

         #TODO

    @property
    def n_entries(self) -> int:
        """
        Get the number of optimization entries for this molecule.
        """

        return len(self.entries)


class OptimizationCollectionResult(BaseModel):
    """
    The master collection result which contains many optimizationResults representing the archive collection.
    """

    method: str
    basis: str
    dataset_name: str
    program: str
    driver: str
    scf_properties: List[str]
    spec_name: str
    collection: Dict[str, OptimizationResult] = {}

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}

    @classmethod
    def from_server(
        cls,
        client: ptl.FractalClient,
        spec_name: str,
        dataset_name: str,
        include_trajectory: bool = False,
        final_molecule_only: Optional[bool] = False,
        subset: Optional[List[str]] = None,
    ) -> "OptimizationCollectionResult":
        """
        Build up the collection result from a OptimizationDatset on a archive client this will also collapse the
        records into entries for the same molecules.

        Parameters:
            client: The fractal client we should contact to pull the results from.
            spec_name: The spec the data was calculated with that we want results for.
            dataset_name: The name of the Optimization set we want to pull down.
            include_trajectory: If we should include the full trajectory when downloading the data this can be quite
                slow.
            final_molecule_only: Only down load the final geometry of each entry.
            subset: The chunk of result indexs that we should pull down.
        """

        # build the input object
        opt_ds = client.get_collection("OptimizationDataset", dataset_name)
        qc_spc = opt_ds.data.specs[spec_name].qc_spec.dict()
        scf_properties = client.query_keywords(qc_spc["keywords"])[0].values["scf_properties"]
        data = {
            "spec_name": spec_name,
            "dataset_name": dataset_name,
            "method": qc_spc["method"],
            "basis": qc_spc["basis"],
            "program": qc_spc["program"],
            "driver": qc_spc["driver"].value,
            "scf_properties": scf_properties,
        }

        collection_result = OptimizationCollectionResult(**data)

        # query the database to get all of the optimization records
        if subset is None:
            query = opt_ds.query(spec_name)
        else:
            # build a list of procedures to request
            procedures = {}
            for index in subset:
                entry = opt_ds.data.records[index]
                procedures[entry.name] = entry.object_map[spec_name]
            # now do the query
            client_procedures = []
            for i in range(0, len(procedures), client.query_limit):
                client_procedures.extend(
                    client.query_procedures(id=list(procedures.values())[i : i + client.query_limit])
                )
            # now we need to create the dummy pd dataframe
            proc_lookup = {x.id: x for x in client_procedures}
            data = []
            for name, oid in procedures.items():
                data.append([name, proc_lookup[oid]])
            df = pd.DataFrame(data, columns=["index", spec_name])
            df.set_index("index", inplace=True)
            query = df[spec_name]

        collection = {}
        # build the list of records to check
        if subset is None:
            subset = list(opt_ds.data.records.keys())

        # start loop through the records and adding them to the collection
        for index in subset:
            opt = opt_ds.data.records[index]
            opt_record = query.loc[opt.name]
            if opt_record.status.value.upper() == "COMPLETE":
                # get the common identifier
                common_name = opt.attributes["canonical_isomeric_smiles"]
                if common_name not in collection:
                    # build up a list of molecules and results and pull them from the database
                    collection[common_name] = {"index": common_name, "attributes": opt.attributes, "entries": []}
                if final_molecule_only:
                    traj = [opt_record.trajectory[-1]]
                    molecules = [opt_record.final_molecule]
                else:
                    traj = [opt_record.trajectory[0], opt_record.trajectory[-1]]
                    molecules = [opt_record.initial_molecule, opt_record.final_molecule]
                entry = {
                    "index": index,
                    "id": opt_record.id,
                    "trajectory_records": traj,
                    "cmiles": opt.attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"],
                    "trajectory_molecules": molecules,
                }
                collection[common_name]["entries"].append(entry)

        # now process the list into chucks that can be queried
        query_molecules = []
        query_results = []
        for data in collection.values():
            for entry in data["entries"]:
                for (result, molecule) in zip(entry["trajectory_records"], entry["trajectory_molecules"]):
                    query_molecules.append(molecule)
                    query_results.append(result)

        print("requested molecules", len(query_molecules))
        print("requested results", len(query_results))
        client_molecules = []
        client_results = []
        # chunk the dataset into a query limit bites and request
        for i in range(0, len(query_molecules), client.query_limit):
            client_molecules.extend(client.query_molecules(id=query_molecules[i : i + client.query_limit]))
            client_results.extend(client.query_results(id=query_results[i : i + client.query_limit]))

        # make into a look up table
        molecules_table = dict((molecule.id, molecule) for molecule in client_molecules)
        results_table = dict((result.id, result) for result in client_results)

        # now we need to build up the collection from the gathered results
        for data in collection.values():
            opt_result = OptimizationResult(index=data["index"], attributes=data["attributes"])
            for entry in data["entries"]:
                opt_entry = OptimizationEntryResult(
                    index=entry["index"], id=entry["id"], cmiles=entry["cmiles"]
                )
                for (result_id, molecule_id) in zip(entry["trajectory_records"], entry["trajectory_molecules"]):
                    # now we need to make the single results
                    molecule = molecules_table[molecule_id]
                    result = results_table[result_id]
                    # check if the qcvars are present
                    # can be missing for MM and ANI calculations
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


OptimizationEntryResult.update_forward_refs()
