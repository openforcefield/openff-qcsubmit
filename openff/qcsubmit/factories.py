import abc
import os
from typing import Literal, Type, TypeVar

import tqdm
from openff.toolkit import topology as off
from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY, ToolkitRegistry
from qcportal.singlepoint import SinglepointDriver

from openff.qcsubmit._pydantic import Field, validator
from openff.qcsubmit.common_structures import CommonBase, Metadata
from openff.qcsubmit.datasets import (
    BasicDataset,
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
from openff.qcsubmit.workflow_components import (
    ComponentResult,
    Components,
    CustomWorkflowComponent,
    get_component,
)

T = TypeVar("T", bound=BasicDataset)


class BaseDatasetFactory(CommonBase, abc.ABC):
    """
    The Base factory which all other dataset factories should inherit from.
    """

    type: Literal["BaseDatasetFactory"] = Field(
        "BaseDatasetFactory",
        description="The type of dataset factory which corresponds to the dataset made.",
    )
    workflow: list[Components] = Field(
        [],
        description=(
            "The set of workflow components and their settings which will be executed in order on the input "
            "molecules to make the dataset.",
        ),
    )

    @classmethod
    def from_file(cls, file_name: str):
        """Create a factory from the serialised model file."""
        data = deserialize(file_name=file_name)
        return cls(**data)

    @classmethod
    @abc.abstractmethod
    def _dataset_type(cls) -> Type[BasicDataset]:
        """Get the type of dataset made by this factory."""
        ...

    def _molecular_complex_filter(self, dataset: T, molecule: off.Molecule, toolkit_registry: ToolkitRegistry) -> None:
        """
        Make a molecular complex dummy filter and filter a molecule.
        """
        try:
            dataset.filtered_molecules["MolecularComplexRemoval"].add_molecule(molecule=molecule)
        except KeyError:
            dataset.filter_molecules(
                molecules=[
                    molecule,
                ],
                component="MolecularComplexRemoval",
                component_settings={},
                component_provenance=self.provenance(toolkit_registry=toolkit_registry),
            )

    def _no_dihedrals_filter(self, dataset: T, molecule: off.Molecule, toolkit_registry: ToolkitRegistry) -> None:
        """
        Fail a molecule for having no tagged torsions.
        """
        try:
            dataset.filtered_molecules["NoDihedralRemoval"].add_molecule(molecule=molecule)
        except KeyError:
            dataset.filter_molecules(
                molecules=[
                    molecule,
                ],
                component="NoDihedralRemoval",
                component_settings={},
                component_provenance=self.provenance(toolkit_registry=toolkit_registry),
            )

    def provenance(self, toolkit_registry: ToolkitRegistry) -> dict[str, str]:
        """
        Create the provenance of openff-qcsubmit that created the molecule input data.

        Returns:
            A dict of the provenance information.

        Important:
            We can not check which toolkit was used to generate the Cmiles data but we know that openeye will always be
            used first when available.
        """

        from openff import qcsubmit, toolkit

        provenance = {
            "openff-qcsubmit": qcsubmit.__version__,
            "openff-toolkit": toolkit.__version__,
        }
        # add all toolkit wrappers in the same order they are registered
        for tk in toolkit_registry.registered_toolkits:
            if tk.__class__.__name__ != "BuiltInToolkitWrapper":
                provenance[tk.__class__.__name__] = tk.toolkit_version

        return provenance

    def clear_workflow(self) -> None:
        """
        Reset the workflow to be empty.
        """
        self.workflow = []

    def add_workflow_components(self, *components: Components) -> None:
        """
        Take the workflow components validate them then insert them into the workflow.

        Args:
            components:
                A list of or an individual workflow component which is to be validated
                and added to the current workflow.

        Raises:
            InvalidWorkflowComponentError: If an invalid workflow component is attempted to be added to the workflow.

        """

        for component in components:
            if issubclass(type(component), CustomWorkflowComponent):
                try:
                    component.is_available()
                    self.workflow.append(component)
                except ModuleNotFoundError as e:
                    raise ComponentRequirementError(
                        f"The component {component.type} could not be added to "
                        f"the workflow due to missing requirements",
                    ) from e

            else:
                raise InvalidWorkflowComponentError(
                    f"Component {component} rejected as it is not a sub " f"class of CustomWorkflowComponent.",
                )

    def get_workflow_components(self, component_name: str) -> list[Components]:
        """
        Find any workflow components with this component name.

        Args:
            component_name:
                The name of the component to be gathered from the workflow.

        Returns:
            A list of instances of the requested component from the workflow.

        Raises:
            MissingWorkflowComponentError: If the component could not be found by its component name in the workflow.
        """

        components = [component for component in self.workflow if component.type.lower() == component_name.lower()]
        if not components:
            raise MissingWorkflowComponentError(
                f"The requested component {component_name} " f"was not registered into the workflow.",
            )

        return components

    def remove_workflow_component(self, component_name: str) -> None:
        """
        Find and remove any components via its type attribute.

        Args:
            component_name:
                The name of the component to be gathered from the workflow.

        Raises:
            MissingWorkflowComponentError: If the component could not be found by its component name in the workflow.
        """

        components = self.get_workflow_components(component_name=component_name)
        if not components:
            raise MissingWorkflowComponentError(
                f"The requested component {component_name} " f"could not be removed as it was not registered.",
            )

        for component in components:
            self.workflow.remove(component)

    def import_workflow(self, workflow: str | dict, clear_existing: bool = True) -> None:
        """
        Instance the workflow from a workflow object or from an input file.

        Args:
            workflow:
                The name of the file the workflow should be created from or a workflow dictionary.
            clear_existing:
                If the current workflow should be deleted and replaced or extended.
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
        for value in workflow.values():
            # check if this is not the first instance of the component
            component = get_component(value["type"])
            self.add_workflow_components(component.parse_obj(value))

    def export_workflow(self, file_name: str) -> None:
        """
        Export the workflow components and their settings to file so that they can be loaded later.

        Args:
            file_name: The name of the file the workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: If the file type is not supported.
        """

        # grab only the workflow
        workflow = self.dict(include={"workflow"})
        serialize(serializable=workflow, file_name=file_name)

    def export(self, file_name: str) -> None:
        """
        Export the whole factory to file including settings and workflow.

        Args:
            file_name:
                The name of the file the factory should be exported to.
        """
        serialize(serializable=self.dict(), file_name=file_name)

    def export_settings(self, file_name: str) -> None:
        """
        Export the current model to file this will include the workflow as well along with each components settings.

        Args:
            file_name:
                The name of the file the settings and workflow should be exported to.

        Raises:
            UnsupportedFiletypeError: When the file type requested is not supported.
        """
        serialize(serializable=self, file_name=file_name)

    def import_settings(self, settings: str | dict, clear_workflow: bool = True) -> None:
        """
        Import settings and workflow from a file.

        Args:
            settings:
                The name of the file the settings should be extracted from or the reference to a settings dictionary.
            clear_workflow:
                If the current workflow should be extended or replaced.
        """

        if isinstance(settings, str):
            data = deserialize(settings)

            # take the workflow out and import the settings
            workflow = data.pop("workflow")

        elif isinstance(settings, dict):
            workflow = settings.pop("workflow")
            data = settings

        else:
            raise RuntimeError("The input type could not be converted into a settings dictionary.")

        # now set the factory meta settings
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                continue

        # now we want to add the workflow back in
        self.import_workflow(workflow=workflow, clear_existing=clear_workflow)

    def _create_initial_component_result(
        self,
        molecules: str | off.Molecule | list[off.Molecule],
        toolkit_registry: ToolkitRegistry,
    ) -> ComponentResult:
        """
        Create the initial component result which is used for de-duplication.

        Args:
            molecules:
                The input molecules which can be a file name or list of molecule instances

        Returns:
            The initial component result used to start the workflow.
        """

        #  check if we have been given an input file with molecules inside
        if isinstance(molecules, str):
            if os.path.isfile(molecules):
                workflow_molecules = ComponentResult(
                    component_name=self.type,
                    component_description={"type": self.type},
                    component_provenance=self.provenance(toolkit_registry=toolkit_registry),
                    input_file=molecules,
                )

            elif os.path.isdir(molecules):
                workflow_molecules = ComponentResult(
                    component_name=self.type,
                    component_description={"type": self.type},
                    component_provenance=self.provenance(toolkit_registry=toolkit_registry),
                    input_directory=molecules,
                )
            else:
                raise FileNotFoundError(f"The input {molecules} could not be found.")

        elif isinstance(molecules, off.Molecule):
            workflow_molecules = ComponentResult(
                component_name=self.type,
                component_description={"type": self.type},
                component_provenance=self.provenance(toolkit_registry=toolkit_registry),
                molecules=[
                    molecules,
                ],
            )

        else:
            workflow_molecules = ComponentResult(
                component_name=self.type,
                component_description={"type": self.type},
                component_provenance=self.provenance(toolkit_registry=toolkit_registry),
                molecules=molecules,
            )

        return workflow_molecules

    @abc.abstractmethod
    def _process_molecule(self, dataset: T, molecule: off.Molecule, toolkit_registry: ToolkitRegistry) -> None:
        """
        Process the molecules returned from running the workflow into a new dataset.
        """
        ...

    def create_dataset(
        self,
        dataset_name: str,
        molecules: str | list[off.Molecule] | off.Molecule,
        description: str,
        tagline: str,
        metadata: Metadata | None = None,
        processors: int | None = None,
        toolkit_registry: ToolkitRegistry | None = None,
        verbose: bool = True,
    ) -> T:
        """
        Process the input molecules through the given workflow then create and populate the corresponding dataset class
        which acts as a local representation for the collection and tasks to be performed in qcarchive.

        Args:
            dataset_name:
                The name that will be given to the collection on submission to an archive instance.
            molecules:
                The list of molecules which should be processed by the workflow and added to the dataset, this
                can also be a file name which is to be unpacked by the openforcefield toolkit.
            description:
                A string describing the dataset this should be detail the purpose of the dataset and outline the
                selection method of the molecules.
            tagline:
                A short tagline description which will be displayed with collection name in the QCArchive.
            metadata:
                Any metadata which should be associated with this dataset this can be changed from the default
                after making the dataset.
            processors:
                The number of processors available to the workflow, note None will use all available processors.
            toolkit_registry:
                The openff.toolkit.utils.ToolkitRegistry which declares the available toolkits and the order in which
                they should be queried for functionality.If ``None`` is passed the default global registry will be used
                with all installed toolkits.
            verbose:
                If True a progress bar for each workflow component will be shown.


        Returns:
            A dataset instance populated with the molecules that have passed through the
            workflow.
        """
        # TODO set up a logging system to report the components
        if toolkit_registry is None:
            toolkit_registry = GLOBAL_TOOLKIT_REGISTRY

        #  create an initial component result
        workflow_molecules = self._create_initial_component_result(
            molecules=molecules,
            toolkit_registry=toolkit_registry,
        )

        # create the dataset
        # first we need to instance the dataset and assign the metadata
        object_meta = self.dict(exclude={"workflow", "type"})

        # the only data missing is the collection name so add it here.
        object_meta["dataset_name"] = dataset_name
        object_meta["description"] = description
        object_meta["provenance"] = self.provenance(toolkit_registry=toolkit_registry)
        object_meta["dataset_tagline"] = tagline
        if metadata is not None:
            object_meta["metadata"] = metadata.dict()
        dataset = self._dataset_type().parse_obj(object_meta)

        # if the workflow has components run it
        if self.workflow:
            for component in self.workflow:
                workflow_molecules = component.apply(
                    molecules=workflow_molecules.molecules,
                    processors=processors,
                    toolkit_registry=toolkit_registry,
                    verbose=verbose,
                )

                dataset.filter_molecules(
                    molecules=workflow_molecules.filtered,
                    component=workflow_molecules.component_name,
                    component_settings=workflow_molecules.component_description,
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
            self._process_molecule(dataset=dataset, molecule=molecule, toolkit_registry=toolkit_registry)

        return dataset

    def create_index(self, molecule: off.Molecule) -> str:
        """
        Create an index for the current molecule.

        Args:
            molecule:
                The molecule for which the dataset index will be generated.

        Returns:
            The molecule name or the canonical isomeric smiles for the molecule if the name is not assigned or is
            blank.

        Important:
            Each dataset can have a different indexing system depending on the data, in this basic dataset each
            conformer of a molecule is expanded into its own entry separately indexed entry. This is handled by the
            dataset however so we just generate a general index for the molecule before adding to the dataset.
        """
        name = molecule.name.lower()
        if name and name != "unnamed":
            return name
        else:
            return molecule.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False)


class BasicDatasetFactory(BaseDatasetFactory):
    """
    Basic dataset generator factory used to build work flows using workflow components before executing them to
    generate a dataset.

    The basic dataset is ideal for large collections of single point calculations using any of the energy, gradient or
    hessian drivers.
    """

    type: Literal["BasicDatasetFactory"] = "BasicDatasetFactory"

    @classmethod
    def _dataset_type(cls) -> Type[BasicDataset]:
        return BasicDataset

    def _process_molecule(self, dataset: T, molecule: off.Molecule, toolkit_registry: ToolkitRegistry) -> None:
        # always put the cmiles in the extras from what we have just calculated to ensure correct order
        extras = molecule.properties.get("extras", {})

        keywords = molecule.properties.get("keywords", None)

        # now submit the molecule
        try:
            dataset.add_molecule(
                index=self.create_index(molecule=molecule),
                molecule=molecule,
                extras=extras if bool(extras) else None,
                keywords=keywords,
            )
        except MolecularComplexError:
            self._molecular_complex_filter(dataset=dataset, molecule=molecule, toolkit_registry=toolkit_registry)


class OptimizationDatasetFactory(BasicDatasetFactory):
    """
    This factory produces OptimisationDatasets which include settings associated with geometric which is used to run
    the optimisation.
    """

    type: Literal["OptimizationDatasetFactory"] = Field(
        "OptimizationDatasetFactory",
        description="The type of dataset factory which corresponds to the dataset made.",
    )
    # set the driver to be gradient this should not be changed when running
    driver: SinglepointDriver = SinglepointDriver.gradient

    # use the default geometric settings during optimisation
    optimization_program: GeometricProcedure = Field(
        GeometricProcedure(),
        description="The optimization program and settings that should be used.",
    )

    @classmethod
    def _dataset_type(cls) -> Type[OptimizationDataset]:
        return OptimizationDataset

    @validator("driver")
    def _check_driver(cls, driver):
        """Make sure that the driver is set to gradient only and not changed."""
        available_drivers = ["gradient"]
        if driver not in available_drivers:
            raise DriverError(
                f"The requested driver ({driver}) is not in the list of available " f"drivers: {available_drivers}",
            )
        return driver


class TorsiondriveDatasetFactory(OptimizationDatasetFactory):
    """
    This factory produces TorsiondriveDatasets which include settings associated with geometric which is used to run
    the optimisation.
    """

    type: Literal["TorsiondriveDatasetFactory"] = Field(
        "TorsiondriveDatasetFactory",
        description="The type of dataset factory which corresponds to the dataset made.",
    )
    grid_spacing: list[int] = Field(
        [15],
        description=(
            "The grid spcaing that should be used for all torsiondrives, this can be overwriten on a per entry basis.",
        ),
    )
    energy_upper_limit: float = Field(
        0.05,
        description="The upper energy limit to spawn new optimizations in the torsiondrive.",
    )
    dihedral_ranges: list[tuple[int, int]] | None = Field(
        None,
        description=(
            "The scan range that should be used for each torsiondrive, this can be overwriten on a per entry basis.",
        ),
    )
    energy_decrease_thresh: float | None = Field(
        None,
        description="The energy lower threshold to trigger new optimizations in the torsiondrive.",
    )

    # set the default settings for a torsiondrive calculation.
    optimization_program = GeometricProcedure.parse_obj({"enforce": 0.1, "reset": True, "qccnv": True, "epsilon": 0.0})

    @classmethod
    def _dataset_type(cls) -> Type[TorsiondriveDataset]:
        return TorsiondriveDataset

    def _linear_torsion_filter(self, dataset: T, molecule: off.Molecule, toolkit_registry: ToolkitRegistry) -> None:
        """
        Mock a linear torsion filtering component and filter the molecule.
        """
        try:
            dataset.filtered_molecules["LinearTorsionRemoval"].add_molecule(molecule=molecule)
        except KeyError:
            dataset.filter_molecules(
                molecules=[
                    molecule,
                ],
                component="LinearTorsionRemoval",
                component_settings={},
                component_provenance=self.provenance(toolkit_registry=toolkit_registry),
            )

    def _unconnected_torsion_filter(
        self,
        dataset: T,
        molecule: off.Molecule,
        toolkit_registry: ToolkitRegistry,
    ) -> None:
        """Mock a unconnected torsion filtering component and filter the molecule."""

        try:
            dataset.filtered_molecules["UnconnectedTorsionRemoval"].add_molecule(molecule=molecule)
        except KeyError:
            dataset.filter_molecules(
                molecules=[
                    molecule,
                ],
                component="UnconnectedTorsionRemoval",
                component_settings={},
                component_provenance=self.provenance(toolkit_registry=toolkit_registry),
            )

    def _process_molecule(self, dataset: T, molecule: off.Molecule, toolkit_registry: ToolkitRegistry) -> None:
        # check for extras and keywords
        extras = molecule.properties.get("extras", {})
        keywords = molecule.properties.get("keywords", {})

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
                keywords["grid_spacing"] = dihedral.scan_increment
                try:
                    dataset.add_molecule(
                        index=index,
                        molecule=molecule,
                        dihedrals=dihedrals,
                        keywords=keywords,
                        extras=extras,
                    )
                except DihedralConnectionError:
                    self._unconnected_torsion_filter(
                        dataset=dataset,
                        molecule=molecule,
                        toolkit_registry=toolkit_registry,
                    )
                except LinearTorsionError:
                    self._linear_torsion_filter(
                        dataset=dataset,
                        molecule=molecule,
                        toolkit_registry=toolkit_registry,
                    )
                except MolecularComplexError:
                    self._molecular_complex_filter(
                        dataset=dataset,
                        molecule=molecule,
                        toolkit_registry=toolkit_registry,
                    )
        else:
            # if no dihedrals have been tagged we need to fail the molecule
            self._no_dihedrals_filter(dataset=dataset, molecule=molecule, toolkit_registry=toolkit_registry)

    def create_index(self, molecule: off.Molecule) -> str:
        """
        Create a specific torsion index for the molecule, this will use the atom map on the molecule.

        Args:
            molecule:
                The molecule for which the dataset index will be generated.

        Returns:
            The canonical mapped isomeric smiles, where the mapped indices are on the atoms in the torsion.

        Important:
            This dataset uses a non-standard indexing with 4 atom mapped indices representing the atoms in the torsion
            to be rotated.
        """

        assert "atom_map" in molecule.properties.keys()
        assert len(molecule.properties["atom_map"]) == 4 or len(molecule.properties["atom_map"]) == 8

        index = molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        return index

    def _detect_linear_torsions(self, molecule: off.Molecule) -> list:
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
