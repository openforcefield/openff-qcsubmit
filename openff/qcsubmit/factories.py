import os
from typing import Any, Dict, List, Optional, Tuple, Union

import tqdm
from openff.toolkit import topology as off
from pydantic import Field, validator
from qcportal import FractalClient
from qcportal.models.common_models import DriverEnum
from typing_extensions import Literal

from openff.qcsubmit.common_structures import CommonBase, Metadata, MoleculeAttributes
from openff.qcsubmit.datasets import (
    BasicDataset,
    ComponentResult,
    OptimizationDataset,
    TorsiondriveDataset,
)
from openff.qcsubmit.exceptions import (
    ComponentRequirementError,
    DihedralConnectionError,
    DriverError,
    InvalidWorkflowComponentError,
    LinearTorsionError,
    MissingWorkflowComponentError,
    MolecularComplexError,
)
from openff.qcsubmit.procedures import GeometricProcedure
from openff.qcsubmit.serializers import deserialize, serialize
from openff.qcsubmit.workflow_components import CustomWorkflowComponent, get_component


class BasicDatasetFactory(CommonBase):
    """
    Basic dataset generator factory used to build work flows using workflow components before executing them to generate
    a dataset.

    The basic dataset is ideal for large collections of single point calculations using any of the energy, gradient or
    hessian drivers.
    """

    factory_type: Literal["BasicDatasetFactory"] = Field(
        "BasicDatasetFactory",
        description="The type of dataset factory which corresponds to the dataset made.",
    )
    workflow: Dict[str, CustomWorkflowComponent] = Field(
        {},
        description="The set of workflow components and their settings which will be executed in order on the input molecules to make the dataset.",
    )
    _dataset_type: BasicDataset = BasicDataset

    def _get_molecular_complex_info(self) -> Dict[str, Any]:
        """
        Make a molecular complex dummy filter

        Returns:
            A dictionary to collect any molecules which are molecular complexes and should be removed.
        """
        return {
            "component_name": "MolecularComplexRemoval",
            "component_description": {
                "component_description": "Remove any molecules which are complexes.",
            },
            "component_provenance": self.provenance(),
            "molecules": [],
        }

    def provenance(self) -> Dict[str, str]:
        """
        Create the provenance of qcsubmit that created that molecule input data.
        Returns:
            A dict of the provenance information.

        Important:
            We can not check which toolkit was used to generate the Cmiles data be we know that openeye will always be
            used first when available.
        """

        from openff import qcsubmit, toolkit

        provenance = {
            "openff-qcsubmit": qcsubmit.__version__,
            "openff-toolkit": toolkit.__version__,
        }
        try:
            import openeye

            provenance["openeye"] = openeye.__version__
        except ImportError:
            import rdkit

            provenance["rdkit"] = rdkit.__version__

        return provenance

    def clear_workflow(self) -> None:
        """
        Reset the workflow to by empty.
        """
        self.workflow = {}

    def add_workflow_component(
        self,
        components: Union[
            List[CustomWorkflowComponent],
            CustomWorkflowComponent,
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
            if issubclass(type(component), CustomWorkflowComponent):
                try:
                    component.is_available()
                except ModuleNotFoundError as e:
                    raise ComponentRequirementError(
                        f"The component {component.component_name} could not be added to "
                        f"the workflow due to missing requirements"
                    ) from e
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

    def get_workflow_component(self, component_name: str) -> CustomWorkflowComponent:
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
            workflow = deserialize(workflow)

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
            component = get_component(name)
            self.add_workflow_component(component.parse_obj(value))

    def export_workflow(self, file_name: str) -> None:
        """
        Export the workflow components and their settings to file so that they can be loaded latter.

        Parameters:
            file_name: The name of the file the workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: If the file type is not supported.
        """

        # grab only the workflow
        workflow = self.dict(include={"workflow"})
        serialize(serializable=workflow, file_name=file_name)

    def export_settings(self, file_name: str) -> None:
        """
        Export the current model to file this will include the workflow as well along with each components settings.

        Parameters:
            file_name: The name of the file the settings and workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: When the file type requested is not supported.
        """
        serialize(serializable=self, file_name=file_name)

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
            data = deserialize(settings)

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

    def _get_collection(
        self, dataset_type: str, dataset_name: str, client: Union[str, FractalClient]
    ) -> "Collection":
        """
        Try and get the requested collection from the archive.
        """

        try:
            collection = client.get_collection(dataset_type, dataset_name)
            return collection
        except KeyError:
            raise KeyError(
                f"The collection: {dataset_name} could not be found, you can only add compute to existing"
                f" collections."
            )

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

        target_client = self._activate_client(client)

        collection = self._get_collection(
            dataset_type="Dataset", dataset_name=dataset_name, client=target_client
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
            if os.path.isfile(molecules):
                workflow_molecules = ComponentResult(
                    component_name=self.factory_type,
                    component_description={"component_name": self.factory_type},
                    component_provenance=self.provenance(),
                    input_file=molecules,
                )

            elif os.path.isdir(molecules):
                workflow_molecules = ComponentResult(
                    component_name=self.factory_type,
                    component_description={"component_name": self.factory_type},
                    component_provenance=self.provenance(),
                    input_directory=molecules,
                )

        elif isinstance(molecules, off.Molecule):
            workflow_molecules = ComponentResult(
                component_name=self.factory_type,
                component_description={"component_name": self.factory_type},
                component_provenance=self.provenance(),
                molecules=[
                    molecules,
                ],
            )

        else:
            workflow_molecules = ComponentResult(
                component_name=self.factory_type,
                component_description={"component_name": self.factory_type},
                component_provenance=self.provenance(),
                molecules=molecules,
            )

        return workflow_molecules

    def create_dataset(
        self,
        dataset_name: str,
        molecules: Union[str, List[off.Molecule], off.Molecule],
        description: str,
        tagline: str,
        metadata: Optional[Metadata] = None,
        processors: Optional[int] = None,
        verbose: bool = True,
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
            tagline: A tagline displayed with collection name in the QCArchive.
            metadata: Any metadata which should be associated with this dataset this can be changed from the default
                after making the dataset.
            processors: The number of processors avilable to the workflow, note None will use all avilable processors.
            verbose: If True a progress bar for each workflow component will be shown.

        Example:
            How to make a dataset from a list of molecules

            ```python
            >>> from openff.qcsubmit.factories import BasicDatasetFactory
            >>> from openff.qcsubmit.workflow_components import get_component
            >>> from openff.toolkit.topology import Molecule
            >>> factory = BasicDatasetFactory()
            >>> gen = get_component("StandardConformerGenerator")
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
        object_meta["dataset_tagline"] = tagline
        if metadata is not None:
            object_meta["metadata"] = metadata.dict()
        dataset = self._dataset_type.parse_obj(object_meta)

        # if the workflow has components run it
        if self.workflow:
            for component in self.workflow.values():
                workflow_molecules = component.apply(
                    molecules=workflow_molecules.molecules,
                    processors=processors,
                    verbose=verbose,
                )

                dataset.filter_molecules(
                    molecules=workflow_molecules.filtered,
                    component_name=workflow_molecules.component_name,
                    component_description=workflow_molecules.component_description,
                    component_provenance=workflow_molecules.component_provenance,
                )

        # get a molecular complex filter
        molecular_complex = self._get_molecular_complex_info()

        # now add the molecules to the correct attributes
        for molecule in tqdm.tqdm(
            workflow_molecules.molecules,
            total=len(workflow_molecules.molecules),
            ncols=80,
            desc="{:30s}".format("Preparation"),
            disable=not verbose,
        ):
            # order the molecule
            order_mol = molecule.canonical_order_atoms()
            attributes = self.create_cmiles_metadata(molecule=order_mol)
            attributes.provenance = self.provenance()

            # always put the cmiles in the extras from what we have just calculated to ensure correct order
            extras = molecule.properties.get("extras", {})

            keywords = molecule.properties.get("keywords", None)

            # now submit the molecule
            try:
                dataset.add_molecule(
                    index=self.create_index(molecule=order_mol),
                    molecule=order_mol,
                    attributes=attributes,
                    extras=extras if bool(extras) else None,
                    keywords=keywords,
                )
            except MolecularComplexError:
                molecular_complex["molecules"].append(molecule)

        # add the complexes if there are any
        if molecular_complex["molecules"]:
            dataset.filter_molecules(**molecular_complex)

        return dataset

    def create_cmiles_metadata(self, molecule: off.Molecule) -> MoleculeAttributes:
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

        return MoleculeAttributes(**cmiles)

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

    factory_type: Literal["OptimizationDatasetFactory"] = Field(
        "OptimizationDatasetFactory",
        description="The type of dataset factory which corresponds to the dataset made.",
    )
    # set the driver to be gradient this should not be changed when running
    driver = DriverEnum.gradient
    _dataset_type = OptimizationDataset

    # use the default geometric settings during optimisation
    optimization_program: GeometricProcedure = Field(
        GeometricProcedure(),
        description="The optimization program and settings that should be used.",
    )

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
        A method that can add compute to an existing collection, this involves registering the QM settings and keywords
        and running the compute.

        Parameters:
            dataset_name: The name of the collection in the qcarchive instance that the compute should be added to.
            await_result: If the function should block until the calculations have finished.
            client: The name of the file containing the client information or the client instance.
        """

        import qcportal as ptl

        target_client = self._activate_client(client)

        # try and get the collection.
        collection = self._get_collection(
            dataset_type=self._dataset_type.__name__,
            dataset_name=dataset_name,
            client=target_client,
        )

        # create the keywords
        kw = ptl.models.KeywordSet(
            values=self.dict(include={"maxiter", "scf_properties"})
        )

        kw_id = target_client.add_keywords([kw])[0]
        # create the spec
        opt_spec = self.optimization_program.get_optimzation_spec()
        qc_spec = ptl.models.common_models.QCSpecification(
            driver=self.driver,
            method=self.method,
            basis=self.basis,
            keywords=kw_id,
            program=self.program,
        )

        # now add the compute tasks
        collection.add_specification(
            name=self.spec_name,
            optimization_spec=opt_spec,
            qc_spec=qc_spec,
            description=self.spec_description,
            overwrite=False,
        )

        response = collection.compute(
            specification=self.spec_name, tag=self.compute_tag, priority=self.priority
        )

        return response


class TorsiondriveDatasetFactory(OptimizationDatasetFactory):
    """
    This factory produces TorsiondriveDatasets which include settings associated with geometric which is used to run
    the optimisation.
    """

    factory_type: Literal["TorsiondriveDatasetFactory"] = Field(
        "TorsiondriveDatasetFactory",
        description="The type of dataset factory which corresponds to the dataset made.",
    )
    grid_spacing: List[int] = Field(
        [15],
        description="The grid spcaing that should be used for all torsiondrives, this can be overwriten on a per entry basis.",
    )
    energy_upper_limit: float = Field(
        0.05,
        description="The upper energy limit to spawn new optimizations in the torsiondrive.",
    )
    dihedral_ranges: Optional[List[Tuple[int, int]]] = Field(
        None,
        description="The scan range that should be used for each torsiondrive, this can be overwriten on a per entry basis.",
    )
    energy_decrease_thresh: Optional[float] = Field(
        None,
        description="The energy lower threshold to trigger new optimizations in the torsiondrive.",
    )
    _dataset_type = TorsiondriveDataset

    # set the default settings for a torsiondrive calculation.
    optimization_program = GeometricProcedure.parse_obj(
        {"enforce": 0.1, "reset": True, "qccnv": True, "epsilon": 0.0}
    )

    def create_dataset(
        self,
        dataset_name: str,
        molecules: Union[str, List[off.Molecule], off.Molecule],
        description: str,
        tagline: str,
        metadata: Optional[Metadata] = None,
        processors: Optional[int] = None,
        verbose: bool = True,
    ) -> TorsiondriveDataset:
        """
        Process the input molecules through the given workflow then create and populate the torsiondrive
        dataset class which acts as a local representation for the collection in qcarchive and has the ability to
        submit its self to a local or public instance.

        Note:
            The torsiondrive dataset allows for multiple starting geometries.
            If fragmentation is used each molecule in the dataset will have the torsion indexes already set else indexes
            are generated for each rotatable torsion in the molecule.

        Important:
            Any molecules with linear torsions identified for torsion driving will be removed and failed from the
            workflow.

        Parameters:
             dataset_name: The name that will be given to the collection on submission to an archive instance.
             molecules: The list of molecules which should be processed by the workflow and added to the dataset, this
                can also be a file name which is to be unpacked by the openforcefield toolkit.
            description: A short string describing the dataset.
            tagline: A short string overview of the collection displayed on the QCArchive.
            metadata: Any metadata which should be associated with this dataset this can be changed from the default
                after making the dataset.
            processors: The number of processors avilable to the workflow, note None will use all avilable processors.
            verbose: If True a progress bar for each workflow component will be shown.

        Returns:
            A [DataSet][qcsubmit.datasets.TorsiondriveDataset] instance populated with the molecules that have passed
            through the workflow.
        """

        # create the initial component result
        workflow_molecules = self._create_initial_component_result(molecules=molecules)

        # catch any linear torsions here
        linear_torsions = {
            "component_name": "LinearTorsionRemoval",
            "component_description": {
                "component_description": "Remove any molecules with a linear torsions selected to drive.",
            },
            "component_provenance": self.provenance(),
            "molecules": [],
        }

        unconnected_torsions = {
            "component_name": "UnconnectedTorsionRemoval",
            "component_description": {
                "component_description": "Remove any molecules with unconnected torsion indices highlighted to drive.",
            },
            "component_provenance": self.provenance(),
            "molecules": [],
        }

        molecular_complex = self._get_molecular_complex_info()

        # first we need to instance the dataset and assign the metadata
        object_meta = self.dict(exclude={"workflow"})

        # the only data missing is the collection name so add it here.
        object_meta["dataset_name"] = dataset_name
        object_meta["description"] = description
        object_meta["provenance"] = self.provenance()
        object_meta["dataset_tagline"] = tagline
        if metadata is not None:
            object_meta["metadata"] = metadata.dict()
        dataset = self._dataset_type(**object_meta)

        # if the workflow has components run it
        if self.workflow:
            for component_name, component in self.workflow.items():
                workflow_molecules = component.apply(
                    molecules=workflow_molecules.molecules,
                    processors=processors,
                    verbose=verbose,
                )

                dataset.filter_molecules(
                    molecules=workflow_molecules.filtered,
                    component_name=workflow_molecules.component_name,
                    component_description=workflow_molecules.component_description,
                    component_provenance=workflow_molecules.component_provenance,
                )

        # now add the molecules to the correct attributes
        for molecule in tqdm.tqdm(
            workflow_molecules.molecules,
            total=len(workflow_molecules.molecules),
            ncols=80,
            desc="{:30s}".format("Preparation"),
            disable=not verbose,
        ):
            # check for extras and keywords
            extras = molecule.properties.get("extras", {})
            keywords = molecule.properties.get("keywords", {})

            # make the general attributes
            attributes = self.create_cmiles_metadata(molecule=molecule)
            # attributes = cmiles.get_molecule_ids(molecule)

            # now check for the dihedrals
            if "dihedrals" in molecule.properties:
                # first do 1-D torsions
                for dihedral in molecule.properties["dihedrals"].get_dihedrals:
                    # create the index
                    molecule.properties["atom_map"] = dihedral.get_atom_map
                    index = self.create_index(molecule=molecule)
                    del molecule.properties["atom_map"]
                    # get the dihedrals to scan
                    dihedrals = dihedral.get_dihedrals

                    keywords["dihedral_ranges"] = dihedral.get_scan_range
                    try:
                        dataset.add_molecule(
                            index=index,
                            molecule=molecule,
                            attributes=attributes,
                            dihedrals=dihedrals,
                            keywords=keywords,
                            extras=extras,
                        )
                    except DihedralConnectionError:
                        unconnected_torsions["molecules"].append(molecule)
                    except LinearTorsionError:
                        linear_torsions["molecules"].append(molecule)
                    except MolecularComplexError:
                        molecular_complex["molecules"].append(molecule)

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
                    try:
                        dataset.add_molecule(
                            index=self.create_index(molecule=order_mol),
                            molecule=order_mol,
                            attributes=attributes,
                            dihedrals=[torsion_index],
                            extras=extras,
                            keywords=keywords,
                        )
                    except DihedralConnectionError:
                        unconnected_torsions["molecules"].append(molecule)
                    except LinearTorsionError:
                        linear_torsions["molecules"].append(molecule)
                    except MolecularComplexError:
                        molecular_complex["molecules"].append(molecule)

        # now we need to filter the linear molecules
        dataset.filter_molecules(**linear_torsions)
        # and we need to filter any molecules with unconnected torsions
        dataset.filter_molecules(**unconnected_torsions)
        # add molecular complex errors
        if molecular_complex["molecules"]:
            dataset.filter_molecules(**molecular_complex)

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
        linear_smarts = "[*!D1:1]~[$(*#*)&D2,$(C=*)&D2:2]"

        matches = molecule.chemical_environment_matches(linear_smarts)

        return matches
