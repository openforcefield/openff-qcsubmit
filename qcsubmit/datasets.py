import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import qcportal as ptl
from pydantic import BaseModel, validator

import openforcefield.topology as off
from qcsubmit.exceptions import UnsupportedFiletypeError

from .procedures import GeometricProcedure
from .results import SingleResult
from .exceptions import DatasetInputError


class ComponentResult:
    """
    Class to contain molecules after the execution of a workflow component this automatically applies de-duplication to
    the molecules. For example if a molecule is already in the molecules list it will not be added but any conformers
    will be kept and transferred.


    If a molecule in the molecules list is then filtered it will be removed from the molecules list.
    """

    def __init__(
        self,
        component_name: str,
        component_description: Dict[str, str],
        component_provenance: Dict[str, str],
        molecules: Optional[Union[List[off.Molecule], off.Molecule]] = None,
        input_file: Optional[str] = None,
    ):
        """Register the list of molecules to process.

        Parameters
        ----------
        component_name : str,
            The name of the component that produced this result.
        component_description : Dict[str, Any],
            The dictionary representation of the component which details the function and running parameters.
        component_provenance : Dict[str, str],
            The dictionary of the modules used and there version number when running the component.
        component_provenance : Dict[str, str],
            The dictionary of the provenance information about the component that was used to generate the data.
        molecules : openforcefield.topology.Molecule or List, optional, default=None
            The list of molecules that have been possessed by a component and returned as a result.
        input_file : str, optional, default=None,
            The name of the input file used to produce the result if not from a component.
        """

        self.molecules: List[off.Molecule] = []
        self.filtered: List[off.Molecule] = []
        self.component_name: str = component_name
        self.component_description: Dict = component_description
        self.component_provenance: Dict = component_provenance

        assert (
            molecules or input_file is None
        ), "Provide either a list of molecules or an input file name."

        # if we have an input file load it
        if input_file is not None:
            molecules = off.Molecule.from_file(
                file_path=input_file, allow_undefined_stereo=True
            )

        # now lets process the molecules and add them to the class
        if molecules is not None:
            for molecule in molecules:
                self.add_molecule(molecule)

    def add_molecule(self, molecule: off.Molecule):
        """
        Add a molecule to the molecule list after checking that it is not present already. If it is de-duplicate the
        record and condense the conformers.
        """

        import numpy as np
        from simtk import unit

        if molecule in self.molecules:
            # we need to align the molecules and transfer the coords and properties
            mol_id = self.molecules.index(molecule)
            # get the mapping
            isomorphic, mapping = off.Molecule.are_isomorphic(
                molecule, self.molecules[mol_id], return_atom_map=True
            )
            assert isomorphic is True
            # transfer any torsion indexs for similar fragments
            if "torsion_index" in molecule.properties:
                for torsion_index, dihedral_range in molecule.properties[
                    "torsion_index"
                ].items():
                    self.molecules[mol_id].properties["torsion_index"][
                        torsion_index
                    ] = dihedral_range

            if molecule.n_conformers != 0:

                # transfer the coordinates
                for conformer in molecule.conformers:
                    new_conformer = np.zeros((molecule.n_atoms, 3))
                    for i in range(molecule.n_atoms):
                        new_conformer[i] = conformer[mapping[i]].value_in_unit(
                            unit.angstrom
                        )

                    new_conf = unit.Quantity(value=new_conformer, unit=unit.angstrom)

                    # check if the conformer is already on the molecule
                    for old_conformer in self.molecules[mol_id].conformers:
                        if old_conformer.tolist() == new_conf.tolist():
                            break
                    else:
                        self.molecules[mol_id].add_conformer(
                            new_conformer * unit.angstrom
                        )
            else:
                # molecule already in list and coords not present so just return
                return

        else:
            self.molecules.append(molecule)

    def filter_molecule(self, molecule: off.Molecule):
        """
        Filter out a molecule that has not passed this workflow component. If the molecule is already in the pass list
        remove it and ensure it is only in the filtered list.
        """

        try:
            self.molecules.remove(molecule)

        except ValueError:
            pass

        finally:
            if molecule not in self.filtered:
                self.filtered.append(molecule)
            else:
                return


class BasicDataSet(BaseModel):
    """
    The general qcfractal dataset class which contains all of the molecules and information about them prior to
    submission.

    The class is a simple holder of the dataset and information about it and can do simple checks on the data before
    submitting it such as ensuring that the molecules have cmiles information
    and a unique index to be identified by.

    Note:
        The molecules in this dataset are all expanded so that different conformers are unique submissions.
    """

    dataset_name: str = "BasicDataSet"
    dataset_tagline: str = "OpenForcefield single point evaluations."
    method: str = "B3LYP-D3BJ"
    basis: Optional[str] = "DZVP"
    program: str = "psi4"
    maxiter: int = 200
    driver: str = "energy"
    scf_properties: List[str] = ["dipole", "qudrupole", "wiberg_lowdin_indices"]
    spec_name: str = "default"
    spec_description: str = "Standard OpenFF optimization quantum chemistry specification."
    priority: str = "normal"
    tag: str = "openff"
    dataset: Dict[str, Dict[str, Union[Dict[str, str], List[ptl.Molecule]]]] = {}
    filtered_molecules: Dict[str, Dict] = {}
    _file_writers = {"json": json.dump}

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {np.ndarray: lambda v: v.flatten().tolist()}

    @property
    def filtered(self) -> off.Molecule:
        """
        A generator for the molecules that have been filtered.

        Returns
        -------
        offmol : openforcefield.topology.Molecule
            A molecule representation created from the filtered molecule lists

        Notes
        -----
            Modifying the molecule will have no effect on the data stored.
        """

        for component, data in self.filtered_molecules.items():
            for smiles in data["molecules"]:
                offmol = off.Molecule.from_smiles(smiles, allow_undefined_stereo=True)
                yield offmol

    @property
    def n_filtered(self) -> int:
        """
        Calculate the total number of molecules filtered by the components used in a workflow to create this dataset.

        Returns
        -------
        filtered : int
                The total number of molecules filtered by components.
        """
        filtered = sum(
            [len(data["molecules"]) for data in self.filtered_molecules.values()]
        )
        return filtered

    @property
    def n_records(self) -> int:
        """
        Return the total number of records that will be created on submission of the dataset.

        Returns
        -------
        n_records : int
            The number of records that will be added to the collection.

        Notes
        -----
            * The number returned will be different depending on the dataset used.
            * The amount of unqiue molecule can be found using `n_molecules`

        See also
        --------
            n_molecules
        """

        n_records = sum(
            [len(data["initial_molecules"]) for data in self.dataset.values()]
        )
        return n_records

    @property
    def n_molecules(self) -> int:
        """
        Calculate the total number of unique molecules which will be submitted as part of this dataset.

        Returns
        -------
        n_molecules : int
            The number of molecules in the dataset.

        Notes
        -----
            The number of molecule records submitted is not always the same as the amount of records created, this can
            also be checked using `n_records`. Here we give the number of unique molecules not excluding conformers.

        See also
        --------
            n_conformers
        """

        n_molecules = len(self.dataset)
        return n_molecules

    @property
    def molecules(self) -> off.Molecule:
        """
        A generator that creates an openforcefield.topology.Molecule one by one from the dataset.

        Returns
        -------
        offmol : openforcefield.topology.Molecule
            The instance of the molecule from the dataset.

        Notes
        -----
            Editing the molecule will not effect the data stored in the dataset as it is immutable.
        """

        from simtk import unit
        import numpy as np

        for index_name, molecule_data in self.dataset.items():
            # create the molecule from the cmiles data
            offmol = off.Molecule.from_mapped_smiles(
                mapped_smiles=molecule_data["attributes"][
                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                ],
                allow_undefined_stereo=True,
            )
            offmol.name = index_name
            for conformer in molecule_data["initial_molecules"]:
                geometry = unit.Quantity(np.array(conformer.geometry), unit=unit.bohr)
                offmol.add_conformer(geometry.in_units_of(unit.angstrom))
            yield offmol

    @property
    def n_components(self) -> int:
        """
        Return the amount of components that have been ran during generating the dataset.

        Returns
        -------
         n_filtered : int
            The number of components that were ran while generating the dataset.
        """

        n_filtered = len(self.filtered_molecules)
        return n_filtered

    @property
    def components(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        """
        Gather the details of the components that were ran during the creation of this dataset.

        Returns
        -------
        components : List[Dict[str, str]]
            A list of dictionaries containing information about the components ran during the generation of the dataset.
        """

        components = []
        for component in self.filtered_molecules.values():
            data = component["component_description"]
            data["component_provenance"] = component["component_provenance"]
            components.append(data)

        return components

    def filter_molecules(
        self,
        molecules: Union[off.Molecule, List[off.Molecule]],
        component_description: Dict[str, str],
        component_provenance: Dict[str, str],
    ) -> None:
        """
        Filter a molecule or list of molecules by the component they failed.

        Parameters
        ----------
        molecules : Union[openforcefield.topology.Molecule, List[openforcefield.topology.Molecule]]
            A molecule or list of molecules to be filtered.
        component_description : Dict[str, str]
            The dictionary representation of the component that filtered this set of molecules.
        component_provenance : Dict[str, str]
            The dictionary representation of the component provenance.
        """

        print(molecules)
        if isinstance(molecules, off.Molecule):
            # make into a list
            molecules = [molecules]

        self.filtered_molecules[component_description["component_name"]] = {
            "component_description": component_description,
            "component_provenance": component_provenance,
            "molecules": [
                molecule.to_smiles(isomeric=True, explicit_hydrogens=True)
                for molecule in molecules
            ],
        }

    def add_molecule(
        self,
        index: str,
        molecule: off.Molecule,
        attributes: Dict[str, str],
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a molecule to the dataset under the given index with the passed cmiles.

        Parameters
        ----------
        index : str
            The molecule index that was generated by the factory.
        molecule : openforcefield.topology.Molecule
            The instance of the molecule which contains its conformer information.
        attributes : Dict[str, str]
            The attributes dictionary containing all of the relevant identifier tags for the molecule and
            extra meta information on the calculation.
        extras : Dict[str, Any], optional, default=None
            The extras that should be supplied into the qcportal.moldels.Molecule.

        Notes
        -----
            Each molecule in this basic dataset should have all of its conformers expanded out into separate entries.
            Thus here we take the general molecule index and increment it.
        """

        if molecule.n_conformers == 0:
            raise DatasetInputError(
                "The input molecule does not have any conformers, make sure to generate them first."
            )
        if "canonical_isomeric_explicit_hydrogen_mapped_smiles" not in attributes:
            raise DatasetInputError(
                "The attributes does not contain valid cmiles identifiers make sure they are"
                "generated by adding the molecule."
            )

        schema_mols = [
            molecule.to_qcschema(conformer=conformer)
            for conformer in range(molecule.n_conformers)
        ]

        self.dataset[index] = {
            "attributes": attributes,
            "initial_molecules": schema_mols,
        }

    def _clean_index(self, index: str) -> Tuple[str, int]:
        """
        Take an index and clean it by checking if it already has an enumerator in it return the core index and any
        numeric tags if no tag is found the tag is set to 0.

        Parameters
        ----------
        index : str
            The index for the entry which should be checked, if no numeric tag can be found return 0.

        Returns
        -------
        core : str
            A tuple of the core index and the numeric tag it starts from.

        Note:
            This function allows the dataset to add more conformers to a molecule set so long as the index the molecule
            is stored under is a new index not in the database for example if 3 conformers for ethane exist then the
            new index should start from 'CC-3'.
        """
        # tags take the form '-no'
        match = re.search("-[0-9]+$", index)
        if match is not None:
            core = index[: match.span()[0]]
            # drop the -
            tag = int(match.group()[1:])
        else:
            core = index
            tag = 0

        return core, tag

    def submit(
        self,
        client: Union[str, ptl.FractalClient],
        await_result: Optional[bool] = False,
    ) -> SingleResult:
        """
        Submit the dataset to the chosen qcarchive address and finish or wait for the results and return the
        corresponding result class.

        Parameters
        ----------
        client : Union[str, qcportal.FractalClient]
            The name of the file containing the client information or an actual client instance.
        await_result : bool, optional, default=False
            If the user wants to wait for the calculation to finish before returning.


        Returns
        -------
        results :
            The collection of the results which have completed.
        """

        target_client = self._activate_client(client)
        # work out if we are extending a collection
        try:
            collection = target_client.get_collection("Dataset", self.dataset_name)
        except KeyError:
            collection = ptl.collections.Dataset(
                name=self.dataset_name,
                client=target_client,
                default_driver=self.driver,
                default_program=self.program,
                tagline=self.dataset_tagline,
            )

        # store the keyword set into the collection
        kw = ptl.models.KeywordSet(
            values=self.dict(include={"maxiter", "scf_properties"})
        )
        try:
            # try and add the keywords if present then continue
            collection.add_keywords(
                alias=self.spec_name, program=self.program, keyword=kw, default=True
            )
            collection.save()
        except (KeyError, AttributeError):
            pass

        i = 0
        # now add the molecules to the database, saving every 30 for speed
        for index, data in self.dataset.items():
            # check if the index we have been supplied has a number tag already if so start from this tag
            index, tag = self._clean_index(index=index)

            for j, molecule in enumerate(data["initial_molecules"]):
                name = index + f"-{tag + j}"
                try:
                    collection.add_entry(name=name, molecule=molecule)
                    i += 1
                except KeyError:
                    continue

                finally:
                    if i % 30 == 0:
                        # save the added entries
                        collection.save()

        # save the final dataset
        collection.save()

        # submit the calculations
        response = collection.compute(
            method=self.method,
            basis=self.basis,
            keywords=self.spec_name,
            program=self.program,
            tag=self.tag,
            priority=self.priority,
        )

        collection.save()

        return response
        # result = BasicResult()
        # while await_result:
        #
        #     pass
        #
        # return result

    def export_dataset(self, file_name: str) -> None:
        """
        Export the dataset to file so that it can be used to make another dataset quickly.

        Parameters:
            file_name: The name of the file the dataset should be wrote to.

        Note:
            The supported file types are:

            - `json`

        Raises:
            UnsupportedFiletypeError: If the requested file type is not supported.
        """

        import copy

        file_type = file_name.split(".")[-1]
        data = self.dict(exclude={"dataset"})
        data["dataset"] = {}

        for index, molecule in self.dataset.items():
            molecules = list(
                mol.dict(encoding="json") for mol in molecule["initial_molecules"]
            )
            mol_data = copy.deepcopy(molecule)
            mol_data["initial_molecules"] = molecules
            data["dataset"][index] = mol_data

        try:
            writer = self._file_writers[file_type]
            with open(file_name, "w") as output:
                if file_type == "json":
                    writer(data, output, indent=2)
        except KeyError:
            raise UnsupportedFiletypeError(
                f"The requested file type {file_type} is not supported please use "
                f"json or yaml"
            )

    def coverage_report(self, forcefields: List[str]) -> Dict:
        """
        Produce a coverage report of all of the parameters that are exercised by the molecules in the dataset.

        Parameters:
            forcefields: The name of the openforcefield force field which should be included in the coverage report.

        Returns:
            A dictionary for each of the force fields which break down which parameters are exercised by their
            parameter type.
        """

        from openforcefield.typing.engines.smirnoff import ForceField
        from openforcefield.utils.structure import get_molecule_parameterIDs

        coverage = {}
        param_types = {
            "a": "Angles",
            "b": "Bonds",
            "c": "Constraints",
            "t": "ProperTorsions",
            "i": "ImproperTorsions",
            "n": "vdW",
        }
        if isinstance(forcefields, str):
            forcefields = [forcefields]

        for forcefield in forcefields:

            result = {}
            ff = ForceField(forcefield)
            parameters_by_molecule, parameters_by_id = get_molecule_parameterIDs(
                list(self.molecules), ff
            )

            # now create the the dict to store the ids used
            for param_id in parameters_by_id.keys():
                result.setdefault(param_types[param_id[0]], []).append(param_id)

            # now store the force field dict into the main result
            coverage[forcefield] = result

        return coverage

    def _activate_client(self, client) -> ptl.FractalClient:
        """
        Make the fractal client and connect to the requested instance.

        Parameters:
            client: The name of the file containing the client information or the client instance.

        Returns:
            A qcportal.FractalClient instance.
        """

        if isinstance(client, ptl.FractalClient):
            return client
        elif client == "public":
            return ptl.FractalClient()
        else:
            return ptl.FractalClient.from_file(client)

    def molecules_to_file(self, file_name: str, file_type: str) -> None:
        """
        Write the molecules to the requested file type.

        Important:
            The supported file types are:

            - SMI
            - INCHI
            - INCKIKEY
        """

        file_writers = {
            "smi": self._molecules_to_smiles,
            "inchi": self._molecules_to_inchi,
            "inchikey": self._molecules_to_inchikey,
        }

        try:
            # get the list of molecules
            molecules = file_writers[file_type.lower()]()

            with open(file_name, "w") as output:
                for molecule in molecules:
                    output.write(f"{molecule}\n")
        except KeyError:
            raise UnsupportedFiletypeError(
                f"The requested file type {file_type} is not supported, supported types are"
                f"{file_writers.keys()}."
            )

    def _molecules_to_smiles(self) -> List[str]:
        """
        Create a list of molecules canonical smiles.
        """

        smiles = [
            data["attributes"]["canonical_smiles"] for data in self.dataset.values()
        ]
        return smiles

    def _molecules_to_inchi(self) -> List[str]:
        """
        Create a list of the molecules standard InChI.
        """

        inchi = [data["attributes"]["standard_inchi"] for data in self.dataset.values()]
        return inchi

    def _molecules_to_inchikey(self) -> List[str]:
        """
        Create a list of the molecules standard InChIKey.
        """

        inchikey = [data["attributes"]["inchi_key"] for data in self.dataset.values()]
        return inchikey


class OptimizationDataset(BasicDataSet):
    """
    An optimisation dataset class which handles submission of settings differently from the basic dataset, and creates
    optimization datasets in the public or local qcarcive instance.
    """

    dataset_name = "OptimizationDataset"
    dataset_tagline = "OpenForcefield optimizations."
    driver = "gradient"
    optimization_program: GeometricProcedure = GeometricProcedure()

    @validator("driver")
    def _check_driver(cls, driver):
        """Make sure that the driver is set to gradient only and not changed."""
        if driver != "gradient":
            driver = "gradient"
        return driver

    def _add_keywords(self, client: ptl.FractalClient) -> str:
        """
        Add the keywords to the client and return the index number of the keyword set.

        Returns
        -------
        kw_id : str
            The keyword index number in the client.
        """

        kw = ptl.models.KeywordSet(
            values=self.dict(include={"maxiter", "scf_properties"})
        )
        kw_id = client.add_keywords([kw])[0]
        return kw_id

    def get_qc_spec(self, keyword_id: str) -> Dict[str, str]:
        """
        Create the QC specification for the computation.

        Parameters:
            keyword_id: The string of the keyword set id number.

        Returns:
            The dictionary representation of the QC specification
        """

        qc_spec = self.dict(include={"driver", "method", "basis", "program"})
        qc_spec["keywords"] = keyword_id

        return qc_spec

    def submit(
        self, client: Union[str, ptl.FractalClient], await_result: bool = False
    ) -> SingleResult:
        """
        Submit the dataset to the chosen qcarchive address and finish or wait for the results and return the
        corresponding result class.

        Parameters:
            await_result: If the user wants to wait for the calculation to finish before returning.
            client: The name of the file containing the client information or the client instance.

        Returns:
            Either `None` if we are not waiting for the results or a BasicResult instance with all of the completed
            calculations.
        """

        target_client = self._activate_client(client)
        # work out if we are extending a collection
        try:
            collection = target_client.get_collection(
                "OptimizationDataset", self.dataset_name
            )
        except KeyError:
            collection = ptl.collections.OptimizationDataset(
                name=self.dataset_name,
                client=target_client,
                tagline=self.dataset_tagline,
            )

        # store the keyword set into the collection
        kw_id = self._add_keywords(target_client)
        # create the optimization specification
        opt_spec = self.optimization_program.get_optimzation_spec()
        # create the qc specification
        qc_spec = self.get_qc_spec(keyword_id=kw_id)
        collection.add_specification(
            name=self.spec_name,
            optimization_spec=opt_spec,
            qc_spec=qc_spec,
            description=self.spec_description,
            overwrite=True,
        )

        i = 0
        # now add the molecules to the database, saving every 30 for speed
        for index, data in self.dataset.items():
            # check if the index we have been supplied has a number tag already if so start from this tag
            index, tag = self._clean_index(index=index)

            for j, molecule in enumerate(data["initial_molecules"]):
                name = index + f"-{tag + j}"
                try:
                    collection.add_entry(
                        name=name,
                        initial_molecule=molecule,
                        attributes=data["attributes"],
                        save=False,
                    )
                    i += 1
                except KeyError:
                    continue

                finally:
                    if i % 30 == 0:
                        # save the added entries
                        collection.save()

        # save the added entries
        collection.save()

        # submit the calculations
        response = collection.compute(
            specification=self.spec_name, tag=self.tag, priority=self.priority
        )

        return response

        # result = BasicResult()
        # while await_result:
        #
        #     pass
        #
        # return result


class TorsiondriveDataset(OptimizationDataset):
    """
    An torsiondrive dataset class which handles submission of settings differently from the basic dataset, and creates
    torsiondrive datasets in the public or local qcarcive instance.

    Important:
        The dihedral_ranges for the whole dataset can be defined here or if different scan ranges are required on a case
         by case basis they can be defined for each torsion in a molecule separately in the properties attribute of the
        molecule. For example `mol.properties['dihedral_range'] = (-165, 180)`
    """

    dataset_name = "TorsionDriveDataset"
    dataset_tagline = "OpenForcefield TorsionDrives."
    # define the types again as they are slightly different for the TorsionDrive data
    dataset: Dict[
        str,
        Dict[str, Union[Dict[str, str], List[ptl.Molecule], Tuple[int, int, int, int]]],
    ] = {}
    optimization_program: GeometricProcedure = GeometricProcedure.parse_obj(
        {"enforce": 0.1, "reset": True, "qccnv": True, "epsilon": 0.0}
    )
    grid_spacings: List[int] = [15]
    energy_upper_limit: float = 0.05
    dihedral_ranges: Optional[List[Tuple[int, int]]] = None
    energy_decrease_thresh: Optional[float] = None

    def submit(
        self, client: Union[str, ptl.FractalClient], await_result: bool = False
    ) -> SingleResult:
        """
        Submit the dataset to the chosen qcarchive address and finish or wait for the results and return the
        corresponding result class.

        Parameters:
            await_result: If the user wants to wait for the calculation to finish before returning.
            client: The name of the file containing the client information or the client instance.


        Returns:
            Either `None` if we are not waiting for the results or a BasicResult instance with all of the completed
            calculations.
        """

        target_client = self._activate_client(client)
        # work out if we are extending a collection
        try:
            collection = target_client.get_collection(
                "TorsionDriveDataset", self.dataset_name
            )
        except KeyError:
            collection = ptl.collections.TorsionDriveDataset(
                name=self.dataset_name,
                client=target_client,
                default_driver=self.driver,
                default_program=self.program,
                tagline=self.dataset_tagline,
            )
        # store the keyword set into the collection
        kw_id = self._add_keywords(target_client)
        # create the optimization specification
        opt_spec = self.optimization_program.get_optimzation_spec()
        # create the qc specification
        qc_spec = self.get_qc_spec(keyword_id=kw_id)
        collection.add_specification(
            name=self.spec_name,
            optimization_spec=opt_spec,
            qc_spec=qc_spec,
            description=self.spec_description,
            overwrite=True,
        )

        # start add the molecule to the dataset, multipule conformers/molecules can be used as the starting geometry
        for i, (index, data) in enumerate(self.dataset.items()):
            try:
                collection.add_entry(
                    name=index,
                    initial_molecules=data["initial_molecules"],
                    dihedrals=data["torsion_index"],
                    grid_spacing=self.grid_spacings,
                    energy_upper_limit=self.energy_upper_limit,
                    attributes=data["attributes"],
                    energy_decrease_thresh=self.energy_decrease_thresh,
                )
            except KeyError:
                continue
            finally:
                if i % 30 == 0:
                    collection.save()

        collection.save()
        # submit the calculations
        response = collection.compute(
            specification=self.spec_name, tag=self.tag, priority=self.priority
        )

        return response

    def add_molecule(
        self,
        index: str,
        molecule: off.Molecule,
        attributes: Dict[str, str],
        atom_indices: Tuple[int, int, int, int],
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a molecule to the dataset under the given index with the passed cmiles.

        Parameters:
            index: The molecule index that was generated by the factory.
            molecule: The instance of the [openforcefield.topology.Molecule][molecule] which contains its conformer
                information.
            attributes: The attributes dictionary containing all of the relevant identifier tags for the molecule.
            atom_indices: The atom indices of the atoms to be restrained during the torsiondrive.
            extras : Dict[str, Any], optional, default=None
                An extras that should be passed into the qcportal.models.Molecule instance.

        Important:
            Each molecule in this basic dataset should have all of its conformers expanded out into separate entries.
            Thus here we take the general molecule index and increment it.
        """

        schema_mols = [
            molecule.to_qcschema(conformer=conformer)
            for conformer in range(molecule.n_conformers)
        ]

        self.dataset[index] = {
            "attributes": attributes,
            "initial_molecules": schema_mols,
            "atom_indices": atom_indices,
        }


# class QCFractalDataset(object):
#     """
#     Abstract base class for QCFractal dataset.
#
#     Attributes
#     ----------
#     name : str
#         The dataset name
#     description : str
#         A detailed description of the dataset
#     input_oemols : list of OEMol
#         Original molecules prior to processing
#     oemols : list of multi-conformer OEMol
#         Unique molecules in the dataset
#
#     """
#
#     def __init__(self, name, description, input_oemols, oemols):
#         """
#         Create a new QCFractalDataset
#
#         Parameters
#         ----------
#         name : str
#             The dataset name
#         description : str
#             A detailed description of the dataset
#         input_oemols : list of OEMol
#             The original molecules provided to the generator for dataset construction.
#         oemols : list of multi-conformer OEMol
#             Molecules that survived after enumeration, fragmentation, deduplication, filtering, and conformer expansion.
#         """
#         self.name = name
#         self.description = description
#
#         # Store copies of molecules
#         from openeye import oechem
#         self.input_oemols = [ oechem.OEMol(oemol) for oemol in input_oemols ]
#         self.oemols = [ oechem.OEMol(oemol) for oemol in oemols ]
#
#     def mol_to_qcschema_dict(self, oemol):
#         """
#         Render a given OEMol as a QCSchema dict.
#
#         {
#             'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ],
#             'cmiles_identifiers' : ...
#         }
#
#         Returns
#         -------
#         qcschema_dict : dict
#             The dict containing all conformations as a list in qcschma_dict['initial_molecules']
#             and CMILES identifiers as qcschema_dict['cmiles_identifiers']
#         """
#         # Generate CMILES ids
#         import cmiles
#         try:
#             cmiles_ids = cmiles.get_molecule_ids(oemol)
#         except:
#             from openeye import oechem
#             smiles = oechem.OEMolToSmiles(oemol)
#             logging.info('cmiles failed to generate molecule ids {}'.format(smiles))
#             self.cmiles_failures.add(smiles)
#             #continue
#
#         # Extract mapped SMILES
#         mapped_smiles = cmiles_ids['canonical_isomeric_explicit_hydrogen_mapped_smiles']
#
#         # Create QCSchema for all conformers defined in the molecule
#         qcschema_molecules = [ cmiles.utils.mol_to_map_ordered_qcschema(conformer, mapped_smiles) for conformer in oemol.GetConfs() ]
#
#         # Create the QCSchema dict that includes both the specified molecules and CMILES ids
#         qcschema_dict = {
#             'initial_molecules': qcschema_molecules,
#             'cmiles_identifiers': cmiles_ids
#             }
#
#         return qcschema_dict
#
#     def render_molecules(self, filename, rows=10, cols=6):
#         """
#         Create a PDF showing all unique molecules in this dataset.
#
#         Parmeters
#         ---------
#         filename : str
#             Name of file to be written (ending in .pdf or .png)
#         rows : int, optional, default=10
#             Number of rows
#         cols : int, optional, default=6
#             Number of columns
#         """
#         from openeye import oedepict
#
#         # Configure display settings
#         itf = oechem.OEInterface()
#         PageByPage = True
#         suppress_h = True
#         ropts = oedepict.OEReportOptions(rows, cols)
#         ropts.SetHeaderHeight(25)
#         ropts.SetFooterHeight(25)
#         ropts.SetCellGap(2)
#         ropts.SetPageMargins(10)
#         report = oedepict.OEReport(ropts)
#         cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
#         opts = oedepict.OE2DMolDisplayOptions(cellwidth, cellheight, oedepict.OEScale_Default * 0.5)
#         opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
#         pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
#         opts.SetDefaultBondPen(pen)
#         oedepict.OESetup2DMolDisplayOptions(opts, itf)
#
#         # Render molecules
#         for oemol in self.oemols:
#             # Render molecule
#             cell = report.NewCell()
#             oemol_copy = oechem.OEMol(oemol)
#             oedepict.OEPrepareDepiction(oemol_copy, False, suppress_h)
#             disp = oedepict.OE2DMolDisplay(oemol_copy, opts)
#             oedepict.OERenderMolecule(cell, disp)
#
#         # Write the report
#         oedepict.OEWriteReport(filename, report)
#
#     def write_smiles(self, filename, mapped=False):
#         """
#         Write canonical isomeric SMILES entries for all unique molecules in this set.
#
#         Parameters
#         ----------
#         filename : str
#             Filename to which SMILES are to be written
#         mapped : bool, optional, default=False
#             If True, will write explicit hydrogen canonical isomeric tagged SMILES
#         """
#         if filename.endswith('.gz'):
#             import gzip
#             open_fun = gzip.open
#         else:
#             open_fun = open
#
#         import cmiles
#         with open_fun(filename, 'w') as outfile:
#             for oemol in self.oemols:
#                 smiles = cmiles.utils.mol_to_smiles(oemol, mapped=mapped)
#                 outfile.write(smiles + '\n')
#
#     def to_json(self, filename):
#         raise Exception('Abstract base class does not implement this method')
#
#     def submit(self,
#                  address: Union[str, 'FractalServer'] = 'api.qcarchive.molssi.org:443',
#                  username: Optional[str] = None,
#                  password: Optional[str] = None,
#                  verify: bool = True):
#         """
#         Submit the dataset to QCFractal server for computation.
#         """
#         raise Exception('Not implemented')
#
# class OptimizationDataset(QCFractalDataset):
#
#     def to_json(self, filename):
#         """
#         Render the OptimizationDataset to QCSchema JSON
#
#         [
#           {
#              'cmiles_identifiers' : ...,
#              'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ]
#           },
#
#           ...
#         ]
#
#         Parameters
#         ----------
#         filename : str
#             Filename (ending in .json or .json.gz) to be written
#
#         """
#         if filename.endswith('.json.gz'):
#             import gzip
#             open_fun = gzip.open
#         else:
#             open_fun = open
#
#         import json
#         with open_fun(filename, 'w') as outfile:
#             outfile.write(json.dumps(self.optimization_input, indent=2, sort_keys=True).encode('utf-8'))
#
# class TorsionDriveDataset(QCFractalDataset):
#
#     def to_json(self, filename):
#         """
#         Render the TorsionDriveDataset to QCSchema JSON
#
#         [
#           "index" : {
#              'atom_indices' : ...,
#              'cmiles_identifiers' : ...,
#              'initial_molecules' : [ qcschema_mol_conf1, qcschema_mol_conf2, ... ]
#           },
#
#           ...
#         ]
#
#         Parameters
#         ----------
#         filename : str
#             Filename (ending in .json or .json.gz) to be written
#
#         """
#         if filename.endswith('.json.gz'):
#             import gzip
#             open_fun = gzip.open
#         else:
#             open_fun = open
#
#         import json
#         with open_fun(filename, 'w') as outfile:
#             outfile.write(json.dumps(self.qcschema_dict, indent=2, sort_keys=True).encode('utf-8'))
#
#     def render_molecules(self, filename, rows=10, cols=6):
#         """
#         Create a PDF showing all unique molecules in this dataset.
#
#         Parmeters
#         ---------
#         filename : str
#             Name of file to be written (ending in .pdf or .png)
#         rows : int, optional, default=10
#             Number of rows
#         cols : int, optional, default=6
#             Number of columns
#         """
#         from openeye import oedepict
#
#         # Configure display settings
#         itf = oechem.OEInterface()
#         PageByPage = True
#         suppress_h = True
#         ropts = oedepict.OEReportOptions(rows, cols)
#         ropts.SetHeaderHeight(25)
#         ropts.SetFooterHeight(25)
#         ropts.SetCellGap(2)
#         ropts.SetPageMargins(10)
#         report = oedepict.OEReport(ropts)
#         cellwidth, cellheight = report.GetCellWidth(), report.GetCellHeight()
#         opts = oedepict.OE2DMolDisplayOptions(cellwidth, cellheight, oedepict.OEScale_Default * 0.5)
#         opts.SetAromaticStyle(oedepict.OEAromaticStyle_Circle)
#         pen = oedepict.OEPen(oechem.OEBlack, oechem.OEBlack, oedepict.OEFill_On, 1.0)
#         opts.SetDefaultBondPen(pen)
#         oedepict.OESetup2DMolDisplayOptions(opts, itf)
#
#         # Render molecules
#         for json_molecule in json_molecules.values():
#             # Create oemol
#             import cmiles
#             oemol = cmiles.utils.load_molecule(json_molecule['initial_molecules'][0])
#
#             # Get atom indices
#             atom_indices = json_molecule['atom_indices'][0]
#
#             # Render molecule
#             cell = report.NewCell()
#             oemol_copy = oechem.OEMol(oemol)
#             oedepict.OEPrepareDepiction(oemol_copy, False, suppress_h)
#             disp = oedepict.OE2DMolDisplay(oemol_copy, opts)
#
#             # Highlight central torsion bond and atoms selected to be driven for torsion
#             class NoAtom(oechem.OEUnaryAtomPred):
#                 def __call__(self, atom):
#                     return False
#             class AtomInTorsion(oechem.OEUnaryAtomPred):
#                 def __call__(self, atom):
#                     return atom.GetIdx() in atom_indices
#             class NoBond(oechem.OEUnaryBondPred):
#                 def __call__(self, bond):
#                     return False
#             class BondInTorsion(oechem.OEUnaryBondPred):
#                 def __call__(self, bond):
#                     return (bond.GetBgn().GetIdx() in atom_indices) and (bond.GetEnd().GetIdx() in atom_indices)
#             class CentralBondInTorsion(oechem.OEUnaryBondPred):
#                 def __call__(self, bond):
#                     return (bond.GetBgn().GetIdx() in atom_indices[1:3]) and (bond.GetEnd().GetIdx() in atom_indices[1:3])
#
#             atoms = mol.GetAtoms(AtomInTorsion())
#             bonds = mol.GetBonds(NoBond())
#             abset = oechem.OEAtomBondSet(atoms, bonds)
#             oedepict.OEAddHighlighting(disp, oechem.OEColor(oechem.OEYellow), oedepict.OEHighlightStyle_BallAndStick, abset)
#
#             atoms = mol.GetAtoms(NoAtom())
#             bonds = mol.GetBonds(CentralBondInTorsion())
#             abset = oechem.OEAtomBondSet(atoms, bonds)
#             oedepict.OEAddHighlighting(disp, oechem.OEColor(oechem.OEOrange), oedepict.OEHighlightStyle_BallAndStick, abset)
#
#             oedepict.OERenderMolecule(cell, disp)
#
#         # Write the report
#         oedepict.OEWriteReport(filename, report)
