"""
A module with classes that can be used to collect results from the qcarchive and have them locally for filtering and
analysis.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from openforcefield.topology import Molecule
import numpy as np
from simtk import unit
import qcportal as ptl
import pandas as pd


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


class OptimizationEntryResult(BaseModel):
    """
    The optimization Entry Result is built from a series of SingleResults to form the trajectory.
    """

    trajectory: List[SingleResult]
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
    ) -> "OptimizationEntryResult":
        """
        Parse an optimization record to get the required data.
        """

        if include_trajectory:
            raise NotImplementedError()

        result_trajectory = optimization_result.client.query_procedures(
            [optimization_result.trajectory[0], optimization_result.trajectory[-1]]
        )
        molecules = optimization_result.client.query_molecules(
            [result_trajectory[0].molecule, result_trajectory[1].molecule]
        )
        trajectory = [cls._create_single_result(*data) for data in zip(result_trajectory, molecules)]
        data = {"trajectory": trajectory, "index": index, "cmiles": cmiles, "id": optimization_result.id}

        entry = OptimizationEntryResult.parse_obj(data)

        return entry

    @staticmethod
    def _create_single_result(result: ptl.models.ResultRecord, molecule: ptl.models.Molecule) -> SingleResult:
        """
        A helpful method to turn the molecule details and the result record into a SingleResult.
        """

        data = {
            "coordinates": molecule.geometry,
            "wbo": result.extras["qcvars"]["WIBERG_LOWDIN_INDICES"],
            "energy": result.extras["qcvars"]["CURRENT ENERGY"],
            "gradient": result.return_result,
            "id": result.id,
        }

        result = SingleResult.parse_obj(data)

        return result

    def connectivity_changed(self, wbo_threshold: float = 0.74) -> bool:
        """
        Detect if the connectivity has changed from the input cmiles specification or not using the WBO.

        Note:
            This is only compared for the final geometry.

        Returns:
            `True` if the connectivity has changed or `False` if it has not.
        """
        # grab the molecule with its bonds
        molecule = self.molecule
        # cast the wbo into the correct shape
        wbo = np.array(self.final_molecule.wbo).reshape((molecule.n_atoms, molecule.n_atoms))
        # now loop over the molecule bonds and make sure we find a bond in the array
        for bond in molecule.bonds:
            if wbo[bond.atom1_index, bond.atom2_index] < wbo_threshold:
                return True
        else:
            return False


class OptimizationResult(BaseModel):
    """
    A Optimiszation result contains metadata about the molecule which is being optimized along with each of the
    optimization entries as each molecule may of been optimised multiple times from different starting conformations.
    """

    entries: List[OptimizationEntryResult] = []
    cmiles: Dict
    index: str

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}

    def add_entry(self, entry: OptimizationEntryResult) -> None:
        """

        """
        self.entries.append(entry)

    @property
    def molecule(self) -> Molecule:
        """
        Build the molecule from the cmiles.
        """

        mol = Molecule.from_mapped_smiles(self.cmiles["canonical_isomeric_explicit_hydrogen_mapped_smiles"])

        return mol

    def get_lowest_energy_optimisation(self) -> OptimizationEntryResult:
        """
        From all of the entries get the optimisation that results in the lowest energy conformer, if any are the same
        a random conformer is returned.
        """

        lowest_index = 0
        lowest_energy = 0
        for index, opt_rec in enumerate(self.entries):
            opt_energy = opt_rec.final_energy
            if opt_energy < lowest_energy:
                lowest_energy = opt_energy
                lowest_index = index

        return self.entries[lowest_index]

    def detect_conectivity_changes(self, wbo_threshold: float = 0.74) -> Dict[int, bool]:
        """
        Detect any connectivity changes in the optimization entries and report them.

        Returns:
            A dictionary of the optimisation entry and a bool of the connectivity changed or not.
            `True` indicates the connectivity did change, `False` indicates it did not.
        """

        conectivity_changes = dict(
            (index, opt_rec.connectivity_changed(wbo_threshold)) for index, opt_rec in enumerate(self.entries)
        )
        return conectivity_changes

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

        collection_result = OptimizationCollectionResult.parse_obj(data)

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
                    collection[common_name] = {"index": common_name, "cmiles": opt.attributes, "entries": []}
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
            opt_result = OptimizationResult(index=data["index"], cmiles=data["cmiles"])
            for entry in data["entries"]:
                trajectory = []
                for (result_id, molecule_id) in zip(entry["trajectory_records"], entry["trajectory_molecules"]):
                    # now we need to make the single results
                    molecule = molecules_table[molecule_id]
                    result = results_table[result_id]
                    trajectory.append(
                        SingleResult(
                            molecule=molecule,
                            wbo=result.extras["qcvars"]["WIBERG_LOWDIN_INDICES"],
                            energy=result.extras["qcvars"]["CURRENT ENERGY"],
                            gradient=result.return_result,
                            id=result.id,
                        )
                    )

                opt_entry = OptimizationEntryResult(
                    trajectory=trajectory, index=entry["index"], id=entry["id"], cmiles=entry["cmiles"]
                )
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
