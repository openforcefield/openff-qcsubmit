import json
import os
from typing import Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, validator
from qcportal import FractalClient
from qcportal.models.common_models import DriverEnum

import openforcefield.topology as off

from . import workflow_components
from .datasets import (
    BasicDataset,
    ComponentResult,
    OptimizationDataset,
    TorsiondriveDataset,
)
from .exceptions import (
    CompoenentRequirementError,
    DriverError,
    InvalidWorkflowComponentError,
    MissingWorkflowComponentError,
    UnsupportedFiletypeError,
)
from .procedures import GeometricProcedure


class BasicDatasetFactory(BaseModel):
    """
    Basic dataset generator factory used to build work flows using workflow components before executing them to generate
    a dataset.

    The basic dataset is ideal for large collections of single point calculations using any of the energy, gradient or
    hessian drivers.

    The main metadata features here are concerned with the QM settings to be used which includes the driver.

    Attributes:
        method:  The QM theory to use during dataset calculations.
        basis:   The basis set to use during dataset calculations.
        program: The program which will be used during the calculations.
        maxiter: The maximum amount of SCF cycles allowed.
        driver:  The driver that should be used in the calculation ie energy/gradient/hessian.
        scf_properties: A list of QM properties which should be calculated and collected during the driver calculation.
        client:  The name of the client the data will be sent to for privet clusters this should be the file name where
            the data is stored.
        priority: The priority with which the dataset should be calculated.
        tag: The tag name which should be given to the collection.
        workflow: A dictionary which holds the workflow components to be executed in the set order.
    """

    method: str = "B3LYP-D3BJ"
    basis: Optional[str] = "DZVP"
    program: str = "psi4"
    maxiter: int = 200
    driver: DriverEnum = DriverEnum.energy
    scf_properties: List[str] = ["dipole", "qudrupole", "wiberg_lowdin_indices"]
    spec_name: str = "default"
    spec_description: str = "Standard OpenFF optimization quantum chemistry specification."
    priority: str = "normal"
    dataset_tags: List[str] = ["openff"]
    compute_tag: str = "openff"
    workflow: Dict[str, workflow_components.CustomWorkflowComponent] = {}
    _dataset_type: BasicDataset = BasicDataset
    _mm_programs: List[str] = [
        "openmm",
        "rdkit",
    ]  # a list of mm programs which require cmiles in the extras

    # hidden variable not included in the schema
    _file_readers = {"json": json.load, "yaml": yaml.full_load, "yml": yaml.full_load}
    _file_writers = {"json": json.dump, "yaml": yaml.dump, "yml": yaml.dump}

    class Config:
        validate_assignment: bool = True
        arbitrary_types_allowed: bool = True
        title: str = "QCFractalDatasetFactory"

    @validator("scf_properties")
    def _check_scf_props(cls, scf_props):
        """Make sure wiberg_lowdin_indices is always included in the scf props."""

        if "wiberg_lowdin_indices" not in scf_props:
            scf_props.append("wiberg_lowdin_indices")
            return scf_props
        else:
            return scf_props

    def provenance(self) -> Dict[str, str]:
        """
        Create the provenance of qcsubmit that created that molecule input data.
        Returns:
            A dict of the provenance information.

        Important:
            We can not check which toolkit was used to generate the Cmiles data be we know that openeye will always be
            used first when available.
        """

        import openforcefield
        import qcsubmit

        provenance = {
            "qcsubmit": qcsubmit.__version__,
            "openforcefield": openforcefield.__version__,
        }

        return provenance

    def clear_workflow(self) -> None:
        """
        Reset the workflow to by empty.
        """
        self.workflow = {}

    def add_workflow_component(
        self,
        components: Union[
            List[workflow_components.CustomWorkflowComponent],
            workflow_components.CustomWorkflowComponent,
        ],
    ) -> None:
        """
        Take the workflow components validate them then insert them into the workflow.

        Parameters:
            components: A list of or an individual qcsubmit.workflow_compoents.CustomWokflowComponent which are to be
                validated and added to the current workflow.

        Raises:
            InvalidWorkflowComponentError: If an invalid workflow component is attempted to be added to the workflow.

        """

        if not isinstance(components, list):
            # we have one component make it into a list
            components = [components]

        for component in components:
            if isinstance(component, workflow_components.CustomWorkflowComponent):
                if not component.is_available():
                    raise CompoenentRequirementError(
                        f"The component {component.component_name} could not be added to "
                        f"the workflow due to missing requirements"
                    )
                if component.component_name not in self.workflow.keys():
                    self.workflow[component.component_name] = component
                else:
                    # we should increment the name and add it to the workflow
                    if "@" in component.component_name:
                        name, number = component.component_name.split("@")
                    else:
                        name, number = component.component_name, 0
                    # set the new name
                    component.component_name = f"{name}@{int(number) + 1}"
                    self.workflow[component.component_name] = component

            else:
                raise InvalidWorkflowComponentError(
                    f"Component {component} rejected as it is not a sub "
                    f"class of CustomWorkflowComponent."
                )

    def get_workflow_component(
        self, component_name: str
    ) -> workflow_components.CustomWorkflowComponent:
        """
        Find the workflow component by its component_name attribute.

        Parameters:
            component_name: The name of the component to be gathered from the workflow.

        Returns:
            The instance of the requested component from the workflow.

        Raises:
            MissingWorkflowComponentError: If the component could not be found by its component name in the workflow.
        """

        component = self.workflow.get(component_name, None)
        if component is None:
            raise MissingWorkflowComponentError(
                f"The requested component {component_name} "
                f"was not registered into the workflow."
            )

        return component

    def remove_workflow_component(self, component_name: str) -> None:
        """
        Find and remove the component via its component_name attribute.

        Parameters:
            component_name: The name of the component to be gathered from the workflow.

        Raises:
            MissingWorkflowComponentError: If the component could not be found by its component name in the workflow.
        """

        try:
            del self.workflow[component_name]

        except KeyError:
            raise MissingWorkflowComponentError(
                f"The requested component {component_name} "
                f"could not be removed as it was not registered."
            )

    def import_workflow(
        self, workflow: Union[str, Dict], clear_existing: bool = True
    ) -> None:
        """
        Instance the workflow from a workflow object or from an input file.

        Parameters:
            workflow: The name of the file the workflow should be created from or a workflow dictionary.
            clear_existing: If the current workflow should be deleted and replaced or extended.
        """

        if clear_existing:
            self.clear_workflow()

        if isinstance(workflow, str):
            workflow = self._read_file(workflow)

        if isinstance(workflow, dict):
            # this should be a workflow dict that we can just load
            # if this is from the settings file make sure to unpack the dict first.
            workflow = workflow.get("workflow", workflow)

        # load in the workflow
        for key, value in workflow.items():
            # check if this is not the first instance of the component
            if "@" in key:
                name = key.split("@")[0]
            else:
                name = key
            component = getattr(workflow_components, name, None)
            if component is not None:
                self.add_workflow_component(component.parse_obj(value))

    def _read_file(self, file_name: str) -> Dict:
        """
        Method to help with file reading and returning the data object from the file.

        Parameters:
            file_name: The name of the file to be read.

        Returns:
            The dictionary representation of the json or yaml file.

        Raises:
            UnsupportedFiletypeError: The file give was not of the supported types ie json or yaml.
            RuntimeError: The given file could not be found to be opened.
        """

        # check that the file exists
        if os.path.exists(file_name):
            file_type = self._get_file_type(file_name)
            try:
                reader = self._file_readers[file_type]
                with open(file_name) as input_data:
                    data = reader(input_data)

                    return data

            except KeyError:
                raise UnsupportedFiletypeError(
                    f"The requested file type {file_type} is not supported "
                    f"currently we can write to {self._file_writers}."
                )
        else:
            raise RuntimeError(f"The file {file_name} could not be found.")

    def export_workflow(self, file_name: str) -> None:
        """
        Export the workflow components and their settings to file so that they can be loaded latter.

        Parameters:
            file_name: The name of the file the workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: If the file type is not supported.
        """

        file_type = self._get_file_type(file_name=file_name)

        # try and get the file writer
        workflow = self.dict()["workflow"]
        try:
            writer = self._file_writers[file_type]
            with open(file_name, "w") as output:
                if file_type == "json":
                    writer(workflow, output, indent=2)
                else:
                    writer(workflow, output)
        except KeyError:
            raise UnsupportedFiletypeError(
                f"The requested file type {file_type} is not supported, "
                f"currently we can write to {self._file_writers}."
            )

    def _get_file_type(self, file_name: str) -> str:
        """
        Helper function to work out the file type being requested from the file name.

        Parameters:
            file_name: The name of the file from which we should work out the file type.

        Returns:
            The file type extension.
        """

        file_type = file_name.split(".")[-1]
        return file_type

    def export_settings(self, file_name: str) -> None:
        """
        Export the current model to file this will include the workflow as well along with each components settings.

        Parameters:
            file_name: The name of the file the settings and workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: When the file type requested is not supported.
        """

        file_type = self._get_file_type(file_name=file_name)

        # try and get the file writer
        try:
            writer = self._file_writers[file_type]
            with open(file_name, "w") as output:
                if file_type == "json":
                    writer(self.dict(), output, indent=2)
                else:
                    data = self.dict(exclude={"driver"})
                    data["driver"] = self.driver.value
                    writer(data, output)
        except KeyError:
            raise UnsupportedFiletypeError(
                f"The requested file type {file_type} is not supported, "
                f"currently we can write to {self._file_writers}."
            )

    def import_settings(
        self, settings: Union[str, Dict], clear_workflow: bool = True
    ) -> None:
        """
        Import settings and workflow from a file.

        Parameters:
            settings: The name of the file the settings should be extracted from or the reference to a settings
                dictionary.
            clear_workflow: If the current workflow should be extended or replaced.
        """

        if isinstance(settings, str):
            data = self._read_file(settings)

            # take the workflow out and import the settings
            workflow = data.pop("workflow")

        elif isinstance(settings, dict):
            workflow = settings.pop("workflow")
            data = settings

        else:
            raise RuntimeError(
                f"The input type could not be converted into a settings dictionary."
            )

        # now set the factory meta settings
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                continue

        # now we want to add the workflow back in
        self.import_workflow(workflow=workflow, clear_existing=clear_workflow)

    def add_compute(
        self,
        dataset_name: str,
        client: Union[str, FractalClient],
        await_result: bool = False,
    ) -> None:
        """
        A method that can add compute to an existing collection, this involves registering the QM settings and keywords
        and running the compute.

        Parameters:
            dataset_name: The name of the collection in the qcarchive instance that the compute should be added to.
            await_result: If the function should block until the calculations have finished.
            client: The name of the file containing the client information or the client instance.
        """
        import qcportal as ptl

        if isinstance(client, ptl.FractalClient):
            target_client = client
        elif client == "public":
            target_client = ptl.FractalClient()
        else:
            target_client = ptl.FractalClient.from_file(client)

        try:
            collection = target_client.get_collection("Dataset", dataset_name)
        except KeyError:
            raise KeyError(
                f"The collection: {dataset_name} could not be found, you can only add compute to existing"
                f"collections."
            )

        kw = ptl.models.KeywordSet(
            values=self.dict(include={"maxiter", "scf_properties"})
        )
        try:
            # try add the keywords, if we get an error they have already been added.
            collection.add_keywords(
                alias=self.spec_name, program=self.program, keyword=kw, default=False
            )
            # save the keywords
            collection.save()
        except (KeyError, AttributeError):
            pass

        # submit the calculations
        response = collection.compute(
            method=self.method,
            basis=self.basis,
            keywords=self.spec_name,
            program=self.program,
            tag=self.compute_tag,
            priority=self.priority,
        )
        collection.save()

        return response

    def _create_initial_component_result(
        self, molecules: Union[str, off.Molecule, List[off.Molecule]]
    ) -> ComponentResult:
        """
        Create the initial component result which is used for de-duplication.

        Parameters:
            molecules: The input molecules which can be a file name or list of molecule instances

        Returns:
            The initial component result used to start the workflow.
        """

        #  check if we have been given an input file with molecules inside
        if isinstance(molecules, str):
            if os.path.exists(molecules):
                workflow_molecules = ComponentResult(
                    component_name=self.Config.title,
                    component_description={"component_name": self.Config.title},
                    component_provenance=self.provenance(),
                    input_file=molecules,
                )

        elif isinstance(molecules, off.Molecule):
            workflow_molecules = ComponentResult(
                component_name=self.Config.title,
                component_description={"component_name": self.Config.title},
                component_provenance=self.provenance(),
                molecules=[molecules],
            )

        else:
            workflow_molecules = ComponentResult(
                component_name=self.Config.title,
                component_description={"component_name": self.Config.title},
                component_provenance=self.provenance(),
                molecules=molecules,
            )

        return workflow_molecules

    def create_dataset(
        self,
        dataset_name: str,
        molecules: Union[str, List[off.Molecule], off.Molecule],
        description: Optional[str] = None,
    ) -> BasicDataset:
        """
        Process the input molecules through the given workflow then create and populate the dataset class which acts as
        a local representation for the collection in qcarchive and has the ability to submit its self to a local or
        public instance.

        Parameters:
             dataset_name: The name that will be given to the collection on submission to an archive instance.
             molecules: The list of molecules which should be processed by the workflow and added to the dataset, this
                can also be a file name which is to be unpacked by the openforcefield toolkit.
            description: A string describing the dataset.

        Example:
            How to make a dataset from a list of molecules

            ```python
            >>> from qcsubmit.factories import BasicDatasetFactory
            >>> from qcsubmit import workflow_components
            >>> from openforcefield.topology import Molecule
            >>> factory = BasicDatasetFactory()
            >>> gen = workflow_components.StandardConformerGenerator()
            >>> gen.clear_exsiting = True
            >>> gen.max_conformers = 1
            >>> factory.add_workflow_component(gen)
            >>> smiles = ['C', 'CC', 'CCO']
            >>> mols = [Molecule.from_smiles(smile) for smile in smiles]
            >>> dataset = factory.create_dataset(dataset_name='My collection', molecules=mols)
            ```

        Returns:
            A [DataSet][qcsubmit.datasets.DataSet] instance populated with the molecules that have passed through the
                workflow.

        Important:
            The dataset once created does not allow mutation.
        """
        # TODO set up a logging system to report the components

        #  create an initial component result
        workflow_molecules = self._create_initial_component_result(molecules=molecules)

        # create the dataset
        # first we need to instance the dataset and assign the metadata
        object_meta = self.dict(exclude={"workflow"})

        # the only data missing is the collection name so add it here.
        object_meta["dataset_name"] = dataset_name
        object_meta["description"] = description
        object_meta["provenance"] = self.provenance()
        dataset = self._dataset_type.parse_obj(object_meta)

        # if the workflow has components run it
        if self.workflow:
            for component in self.workflow.values():
                workflow_molecules = component.apply(
                    molecules=workflow_molecules.molecules
                )

                dataset.filter_molecules(
                    molecules=workflow_molecules.filtered,
                    component_name=workflow_molecules.component_name,
                    component_description=workflow_molecules.component_description,
                    component_provenance=workflow_molecules.component_provenance,
                )

        # now add the molecules to the correct attributes
        for molecule in workflow_molecules.molecules:
            # order the molecule
            order_mol = molecule.canonical_order_atoms()
            attributes = self.create_cmiles_metadata(molecule=order_mol)
            attributes["provenance"] = self.provenance()

            # if we are using MM we should put the cmiles in the extras
            extras = molecule.properties.get("extras", {})
            if self.program in self._mm_programs:
                extras[
                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                ] = attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]

            keywords = molecule.properties.get("keywords", None)

            # now submit the molecule
            dataset.add_molecule(
                index=self.create_index(molecule=order_mol),
                molecule=order_mol,
                attributes=attributes,
                extras=extras if bool(extras) else None,
                keywords=keywords,
            )

        return dataset

    def create_cmiles_metadata(self, molecule: off.Molecule) -> Dict[str, str]:
        """
        Create the Cmiles metadata for the molecule in this dataset.

        Parameters:
            molecule: The molecule for which the cmiles data will be generated.

        Returns:
            The Cmiles identifiers generated for the input molecule.

        Note:
            The Cmiles identifiers currently include:

            - `canonical_smiles`
            - `canonical_isomeric_smiles`
            - `canonical_explicit_hydrogen_smiles`
            - `canonical_isomeric_explicit_hydrogen_smiles`
            - `canonical_isomeric_explicit_hydrogen_mapped_smiles`
            - `molecular_formula`
            - `standard_inchi`
            - `inchi_key`
        """

        cmiles = {
            "canonical_smiles": molecule.to_smiles(
                isomeric=False, explicit_hydrogens=False, mapped=False
            ),
            "canonical_isomeric_smiles": molecule.to_smiles(
                isomeric=True, explicit_hydrogens=False, mapped=False
            ),
            "canonical_explicit_hydrogen_smiles": molecule.to_smiles(
                isomeric=False, explicit_hydrogens=True, mapped=False
            ),
            "canonical_isomeric_explicit_hydrogen_smiles": molecule.to_smiles(
                isomeric=True, explicit_hydrogens=True, mapped=False
            ),
            "canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(
                isomeric=True, explicit_hydrogens=True, mapped=True
            ),
            "molecular_formula": molecule.hill_formula,
            "standard_inchi": molecule.to_inchi(fixed_hydrogens=False),
            "inchi_key": molecule.to_inchikey(fixed_hydrogens=False),
        }

        return cmiles

    def create_index(self, molecule: off.Molecule) -> str:
        """
        Create an index for the current molecule.

        Parameters:
            molecule: The molecule for which the dataset index will be generated.

        Returns:
            The canonical isomeric smiles for the molecule which is used as the dataset index.

        Important:
            Each dataset can have a different indexing system depending on the data, in this basic dataset each conformer
            of a molecule is expanded into its own entry separately indexed entry. This is handled by the dataset however
            so we just generate a general index for the molecule before adding to the dataset.
        """

        index = molecule.to_smiles(
            isomeric=True, explicit_hydrogens=False, mapped=False
        )
        return index


class OptimizationDatasetFactory(BasicDatasetFactory):
    """
    This factory produces OptimisationDatasets which include settings associated with geometric which is used to run the
    optimisation.
    """

    # set the driver to be gradient this should not be changed when running
    driver = DriverEnum.gradient
    _dataset_type = OptimizationDataset

    # use the default geometric settings during optimisation
    optimization_program: GeometricProcedure = GeometricProcedure()

    class Config:
        title = "OptimizationDatasetFactory"

    @validator("driver")
    def _check_driver(cls, driver):
        """Make sure that the driver is set to gradient only and not changed."""
        available_drivers = ["gradient"]
        if driver not in available_drivers:
            raise DriverError(
                f"The requested driver ({driver}) is not in the list of available "
                f"drivers: {available_drivers}"
            )
        return driver

    def add_compute(
        self,
        dataset_name: str,
        client: Union[str, FractalClient],
        await_result: bool = False,
    ) -> None:
        """
        Add compute to an exsiting collection of molecules.

        Parameters:
            dataset_name:
            client:
            await_result:
        """

        raise NotImplementedError()


class TorsiondriveDatasetFactory(OptimizationDatasetFactory):
    """
    This factory produces TorsiondriveDatasets which include settings associated with geometric which is used to run
    the optimisation.
    """

    grid_spacings: List[int] = [15]
    energy_upper_limit: float = 0.05
    dihedral_ranges: Optional[List[Tuple[int, int]]] = None
    energy_decrease_thresh: Optional[float] = None
    _dataset_type = TorsiondriveDataset

    # set the default settings for a torsiondrive calculation.
    optimization_program = GeometricProcedure.parse_obj(
        {"enforce": 0.1, "reset": True, "qccnv": True, "epsilon": 0.0}
    )

    class Config:
        title = "TorsiondriveDatasetFactory"

    def create_dataset(
        self,
        dataset_name: str,
        molecules: Union[str, List[off.Molecule], off.Molecule],
        description: str = None,
    ) -> TorsiondriveDataset:
        """
        Process the input molecules through the given workflow then create and populate the torsiondrive
        dataset class which acts as a local representation for the collection in qcarchive and has the ability to
        submit its self to a local or public instance.

        Note:
            The torsiondrive dataset allows for multiple starting geometries.

        Important:
            Any molecules with linear torsions identified for torsion driving will be removed and failed from the
            workflow.

        Important:
            If fragmentation is used each molecule in the dataset will have the torsion indexes already set else indexes
            are generated for each rotatable torsion in the molecule.

        Parameters:
             dataset_name: The name that will be given to the collection on submission to an archive instance.
             molecules: The list of molecules which should be processed by the workflow and added to the dataset, this
                can also be a file name which is to be unpacked by the openforcefield toolkit.

        Returns:
            A [DataSet][qcsubmit.datasets.TorsiondriveDataset] instance populated with the molecules that have passed
            through the workflow.
        """

        # create the initial component result
        workflow_molecules = self._create_initial_component_result(molecules=molecules)

        # cach any linear torsions here
        linear_torsions = {
            "component_name": "LinearTorsionRemoval",
            "component_description": {
                "component_description": "Remove any molecules with a linear torsions selected to drive.",
            },
            "component_provenance": self.provenance(),
            "molecules": [],
        }

        # first we need to instance the dataset and assign the metadata
        object_meta = self.dict(exclude={"workflow"})

        # the only data missing is the collection name so add it here.
        object_meta["dataset_name"] = dataset_name
        object_meta["description"] = description
        object_meta["provenance"] = self.provenance()
        dataset = self._dataset_type(**object_meta)

        # if the workflow has components run it
        if self.workflow:
            for component_name, component in self.workflow.items():
                workflow_molecules = component.apply(
                    molecules=workflow_molecules.molecules
                )

                dataset.filter_molecules(
                    molecules=workflow_molecules.filtered,
                    component_name=workflow_molecules.component_name,
                    component_description=workflow_molecules.component_description,
                    component_provenance=workflow_molecules.component_provenance,
                )

        # now add the molecules to the correct attributes
        for molecule in workflow_molecules.molecules:

            # check if there are any linear torsions in the molecule
            linear_bonds = self._detect_linear_torsions(molecule)

            # check for extras and keywords
            extras = molecule.properties.get("extras", {})
            keywords = molecule.properties.get("keywords", {})

            # check if the molecule has an atom map or dihedrals defined
            if "atom_map" in molecule.properties:
                # we need to check the map and convert it to use the dihedrals method
                if len(molecule.properties["atom_map"]) == 4:
                    # the map is for the correct amount of atoms
                    atom_map = molecule.properties.pop("atom_map")
                    molecule.properties["dihedrals"] = {tuple(atom_map.keys()): None}

            # make the general attributes
            attributes = self.create_cmiles_metadata(molecule=molecule)

            # now check for the dihedrals
            if "dihedrals" in molecule.properties:
                for dihedral, dihedral_range in molecule.properties[
                    "dihedrals"
                ].items():
                    # check for a 2d torsion scan
                    if len(dihedral) == 8:
                        # create the dihedrals list of tuples
                        dihedrals = [tuple(dihedral[0:4]), tuple(dihedral[4:8])]
                    elif len(dihedral) == 4:
                        dihedrals = [dihedral]
                    else:
                        continue

                    for torsion in dihedrals:
                        if (
                            torsion[1:3] in linear_bonds
                            or torsion[2:0:-1] in linear_bonds
                        ):
                            linear_torsions["molecules"].append(molecule)
                            break
                    else:
                        # create the index
                        molecule.properties["atom_map"] = dict(
                            (atom, i) for i, atom in enumerate(dihedral)
                        )
                        index = self.create_index(molecule=molecule)
                        del molecule.properties["atom_map"]

                        keywords["dihedral_ranges"] = dihedral_range
                        dataset.add_molecule(
                            index=index,
                            molecule=molecule,
                            attributes=attributes,
                            dihedrals=dihedrals,
                            keywords=keywords,
                            extras=extras,
                        )

            else:
                # the molecule has not had its atoms identified yet so process them here
                # order the molecule
                order_mol = molecule.canonical_order_atoms()
                rotatble_bonds = order_mol.find_rotatable_bonds()
                attributes = self.create_cmiles_metadata(molecule=order_mol)
                for bond in rotatble_bonds:
                    # create a torsion to hold as fixed using non-hydrogen atoms
                    torsion_index = self._get_torsion_string(bond)
                    order_mol.properties["atom_map"] = dict(
                        (atom, index) for index, atom in enumerate(torsion_index)
                    )
                    dataset.add_molecule(
                        index=self.create_index(molecule=order_mol),
                        molecule=order_mol,
                        attributes=attributes,
                        dihedrals=[torsion_index],
                        extras=extras,
                    )

        # now we need to filter the linear molecules
        dataset.filter_molecules(**linear_torsions)

        return dataset

    def _get_torsion_string(self, bond: off.Bond) -> Tuple[int, int, int, int]:
        """
        Create a torsion tuple which will be restrained in the torsiondrive.

        Parameters:
            bond: The tuple of the atom indexes for the central bond.

        Returns:
            The tuple of the four atom indices which should be restrained.

        Note:
            If there is more than one possible combination of atoms the heaviest set are selected to be restrained.
        """

        atoms = [bond.atom1, bond.atom2]
        terminal_atoms = {}

        for atom in atoms:
            for neighbour in atom.bonded_atoms:
                if neighbour not in atoms:
                    if (
                        neighbour.atomic_number
                        > terminal_atoms.get(atom, off.Atom(0, 0, False)).atomic_number
                    ):
                        terminal_atoms[atom] = neighbour
        # build out the torsion
        torsion = [atom.molecule_atom_index for atom in terminal_atoms.values()]
        for i, atom in enumerate(atoms, 1):
            torsion.insert(i, atom.molecule_atom_index)

        return tuple(torsion)

    def create_index(self, molecule: off.Molecule) -> str:
        """
        Create a specific torsion index for the molecule, this will use the atom map on the molecule.

        Parameters:
            molecule:  The molecule for which the dataset index will be generated.

        Returns:
            The canonical mapped isomeric smiles, where the mapped indices are on the atoms in the torsion.

        Important:
            This dataset uses a non-standard indexing with 4 atom mapped indices representing the atoms in the torsion
            to be rotated.
        """

        assert "atom_map" in molecule.properties.keys()
        assert (
            len(molecule.properties["atom_map"]) == 4
            or len(molecule.properties["atom_map"]) == 8
        )

        index = molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        return index

    def _detect_linear_torsions(self, molecule: off.Molecule) -> List:
        """
        Try and find any linear bonds in the molecule with torsions that should not be driven.

        Parameters:
            molecule: An openforcefield molecule instance

        Returns:
            A list of the central bond tuples in the molecule which should not be driven, this can then be compared
            against the torsions which have been selected.
        """

        # this is based on the past submissions to QCarchive which have failed
        # highlight the central bond of a linear torsion
        linear_smarts = "[*!D1:1]-,#[$(*#*)&D2:2]"

        matches = molecule.chemical_environment_matches(linear_smarts)

        return matches

    def add_compute(
        self,
        dataset_name: str,
        client: Union[str, FractalClient],
        await_result: bool = False,
    ) -> None:
        """

        """

        raise NotImplementedError()
