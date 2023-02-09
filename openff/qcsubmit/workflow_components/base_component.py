import abc
from typing import Dict, List, Optional

import tqdm
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import ToolkitRegistry
from pydantic import BaseModel, Field, PrivateAttr
from qcelemental.util import which_import
from typing_extensions import Literal

from openff.qcsubmit.common_structures import ComponentProperties
from openff.qcsubmit.workflow_components.utils import ComponentResult


class CustomWorkflowComponent(BaseModel, abc.ABC):
    """
    This is an abstract base class which should be used to create all workflow components, following the design of this
    class should allow users to easily create new work flow components with out needing to change any of the dataset
    factory code.
    """

    class Config:
        allow_mutation = True
        validate_assignment = True

    type: Literal["CustomWorkflowComponent"] = Field(
        "CustomWorkflowComponent",
        description="The name of the component which should match the class name.",
    )
    # new pydantic private attr is loaded into slots
    _cache: Dict = PrivateAttr(default={})

    @classmethod
    @abc.abstractmethod
    def description(cls) -> str:
        """Returns a friendly description of the workflow component."""
        ...

    @classmethod
    @abc.abstractmethod
    def fail_reason(cls) -> str:
        """Returns a friendly description of why a molecule would fail to pass the component."""
        ...

    @classmethod
    @abc.abstractmethod
    def properties(cls) -> ComponentProperties:
        """Returns the runtime properties of the component such as parallel safe."""
        ...

    @classmethod
    def info(cls) -> Dict[str, str]:
        """Returns a dictionary of the friendly descriptions of the class."""
        return dict(
            name=cls.__name__,
            description=cls.description(),
            fail_reason=cls.fail_reason(),
        )

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """
        This method should identify if the component can be used by checking if the requirements are available.

        Returns:
            `True` if the component can be used else `False`.
        """
        ...

    @abc.abstractmethod
    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return the result.

        Args:
            molecules: The list of molecules to be processed by this component.
            toolkit_registry: The openff.toolkit.utils.ToolkitRegistry which declares the available toolkits.

        Returns:
            A component result class which handles collecting together molecules that pass and fail
            the component
        """
        ...

    def _apply_init(self, result: ComponentResult) -> None:
        """
        Any actions that should be performed before running the main apply method should set up such as setting up the _cache for multiprocessing.
        Here we clear out the _cache in case something has been set.
        """
        self._cache.clear()

    def _apply_finalize(self, result: ComponentResult) -> None:
        """
        Any clean up actions should be added here, by default the _cache is cleaned.
        """
        self._cache.clear()

    def apply(
        self,
        molecules: List[Molecule],
        toolkit_registry: ToolkitRegistry,
        processors: Optional[int] = None,
        verbose: bool = True,
    ) -> ComponentResult:
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return any resulting molecules.

        Args:
            molecules:
                The list of molecules to be processed by this component.
            toolkit_registry:
                The openff.toolkit.utils.ToolkitRegistry which declares the available backend toolkits to be used.
            processors:
                The number of processor the component can use to run the job in parallel across molecules,
                None will default to all cores.
            verbose:
                If true a progress bar should be shown on screen.

        Returns:
            A component result class which handles collecting together molecules that pass and fail
            the component
        """
        result: ComponentResult = self._create_result(toolkit_registry=toolkit_registry)

        self._apply_init(result)

        # Use a Pool to get around the GIL. As long as self does not contain
        # too much data, this should be efficient.

        if (
            processors is None or processors > 1
        ) and self.properties().process_parallel:
            from multiprocessing.pool import Pool

            with Pool(processes=processors) as pool:
                # Assumes to process in batches of 1 for now
                work_list = [
                    pool.apply_async(self._apply, ([molecule], toolkit_registry))
                    for molecule in molecules
                ]
                for work in tqdm.tqdm(
                    work_list,
                    total=len(work_list),
                    ncols=80,
                    desc="{:30s}".format(self.type),
                    disable=not verbose,
                ):
                    work = work.get()
                    for success in work.molecules:
                        result.add_molecule(success)
                    for fail in work.filtered:
                        result.filter_molecule(fail)

        else:
            for molecule in tqdm.tqdm(
                molecules,
                total=len(molecules),
                ncols=80,
                desc="{:30s}".format(self.type),
                disable=not verbose,
            ):
                work = self._apply([molecule], toolkit_registry)
                for success in work.molecules:
                    result.add_molecule(success)
                for fail in work.filtered:
                    result.filter_molecule(fail)

        self._apply_finalize(result)

        return result

    @abc.abstractmethod
    def provenance(self, toolkit_registry: ToolkitRegistry) -> Dict:
        """
        This function should detail the programs with version information and procedures called during activation
        of the workflow component.

        Returns:
            A dictionary containing the information about the component and the functions called.
        """
        ...

    def _create_result(
        self, toolkit_registry: ToolkitRegistry, **kwargs
    ) -> ComponentResult:
        """
        A helpful method to build to create the component result with the required information.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instantiated with the required information.
        """

        result = ComponentResult(
            component_name=self.type,
            component_description=self.dict(),
            component_provenance=self.provenance(toolkit_registry=toolkit_registry),
            skip_unique_check=not self.properties().produces_duplicates,
            **kwargs,
        )

        return result


class ToolkitValidator(BaseModel):
    """
    A pydantic mixin class that adds toolkit settings and validation along with provenance information.

    Note:
        The provenance information and toolkit settings are handled by the
        [ToolkitValidator][qcsubmit.workflow_components.base_component.ToolkitValidator] mixin.
    """

    def provenance(self, toolkit_registry: ToolkitRegistry) -> Dict:
        """
        This component calls the OFFTK to perform the task and logs information on the backend toolkit used.

        Args:
            toolkit_registry: The openff.toolkit.utils.ToolkitRegistry which declares the available toolkits for the component.

        Returns:
            A dictionary containing the version information about the backend toolkit called to perform the task.
        """

        from openff import qcsubmit, toolkit

        provenance = {
            "openff-toolkit": toolkit.__version__,
            "openff-qcsubmit": qcsubmit.__version__,
        }
        for tk in toolkit_registry.registered_toolkits:
            if tk.__class__.__name__ != "BuiltInToolkitWrapper":
                provenance[tk.__class__.__name__] = tk.toolkit_version

        return provenance

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if any of the requested backend toolkits can be used.
        """
        return which_import(
            ".toolkit",
            package="openff",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install openff-toolkit -c conda-forge`.",
        )
