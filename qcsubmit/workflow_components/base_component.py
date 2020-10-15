import abc
from typing import Any, Dict, List, Optional

import tqdm
from openforcefield.topology import Molecule
from openforcefield.utils.toolkits import OpenEyeToolkitWrapper, RDKitToolkitWrapper
from pydantic import BaseModel, validator
from pydantic.main import ModelMetaclass
from qcelemental.util import which_import

from ..common_structures import ComponentProperties
from ..datasets import ComponentResult


class InheritSlots(ModelMetaclass):

    # This allows subclasses of CustomWorkFlowComponentto inherit __slots__
    def __new__(mcs, name, bases, namespace):

        slots = set(namespace.pop("__slots__", tuple()))

        for base in bases:
            if hasattr(base, "__slots__"):
                slots.update(base.__slots__)

        if "__dict__" in slots:
            slots.remove("__dict__")

        namespace["__slots__"] = tuple(slots)

        return ModelMetaclass.__new__(mcs, name, bases, namespace)


class CustomWorkflowComponent(BaseModel, abc.ABC, metaclass=InheritSlots):
    """
    This is an abstract base class which should be used to create all workflow components, following the design of this
    class should allow users to easily create new work flow components with out needing to change much of the dataset
    factory code
    """

    component_name: str
    component_description: str
    component_fail_message: str
    _properties: ComponentProperties
    _cache: Dict

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    # this is a pydantic workaround to add private variables taken from
    # https://github.com/samuelcolvin/pydantic/issues/655
    __slots__ = [
        "_cache",
    ]

    def __init__(self, *args, **kwargs):
        super(CustomWorkflowComponent, self).__init__(*args, **kwargs)
        self._cache = {}

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Overwrite the Pydantic setattr to configure the handling of our __slots___
        """
        if attr in self.__slots__:
            object.__setattr__(self, attr, value)
        else:
            super(CustomWorkflowComponent, self).__setattr__(attr, value)

    # getstate and setstate are needed since private instance members (_*)
    # are not included in pickles by Pydantic at the current moment. Force them
    # to be added here. This is needed for multiprocessing support.
    def __getstate__(self):
        return (
            super().__getstate__(),
            {slot: getattr(self, slot) for slot in self.__slots__},
        )

    def __setstate__(self, state):
        super().__setstate__(state[0])
        d = state[1]
        for slot in d:
            setattr(self, slot, d[slot])

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """
        This method should identify if the component can be used by checking if the requirements are available.

        Returns:
            `True` if the component can be used else `False`
        """
        ...

    @abc.abstractmethod
    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return the result.

        Parameters:
            molecules: The list of molecules to be processed by this component.

        Returns:
            An instance of the [ComponentResult][qcsubmit.datasets.ComponentResult]
            class which handles collecting together molecules that pass and fail
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
        processors: Optional[int] = None,
        verbose: bool = True,
    ) -> ComponentResult:
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return the

        Parameters:
            molecules: The list of molecules to be processed by this component.
            processors: The number of processor the component can use to run the job in parallel across molecules, None will default to all cores.
            verbose: If true a progress bar will be shown on screen.

        Returns:
            An instance of the [ComponentResult][qcsubmit.datasets.ComponentResult]
            class which handles collecting together molecules that pass and fail
            the component
        """
        result: ComponentResult = self._create_result()

        self._apply_init(result)

        # Use a Pool to get around the GIL. As long as self does not contain
        # too much data, this should be efficient.

        if (processors is None or processors > 1) and self._properties.process_parallel:

            from multiprocessing.pool import Pool

            with Pool(processes=processors) as pool:

                # Assumes to process in batches of 1 for now
                work_list = [
                    pool.apply_async(self._apply, ([molecule],))
                    for molecule in molecules
                ]
                for work in tqdm.tqdm(
                    work_list,
                    total=len(work_list),
                    ncols=80,
                    desc="{:30s}".format(self.component_name),
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
                desc="{:30s}".format(self.component_name),
                disable=not verbose,
            ):
                work = self._apply([molecule])
                for success in work.molecules:
                    result.add_molecule(success)
                for fail in work.filtered:
                    result.filter_molecule(fail)

        self._apply_finalize(result)

        return result

    @abc.abstractmethod
    def provenance(self) -> Dict:
        """
        This function should detail the programs with version information and procedures called during activation
        of the workflow component.

        Returns:
            A dictionary containing the information about the component and the functions called.
        """
        ...

    def _create_result(self, **kwargs) -> ComponentResult:
        """
        A helpful method to build to create the component result with the required information.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instantiated with the required information.
        """

        result = ComponentResult(
            component_name=self.component_name,
            component_description=self.dict(),
            component_provenance=self.provenance(),
            skip_unique_check=not self._properties.produces_duplicates,
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

    toolkit: str = "openeye"
    _toolkits: Dict = {"rdkit": RDKitToolkitWrapper, "openeye": OpenEyeToolkitWrapper}

    @validator("toolkit")
    def _check_toolkit(cls, toolkit):
        """
        Make sure that toolkit is one of the supported types in the OFFTK.
        """
        if toolkit not in cls._toolkits.keys():
            raise ValueError(
                f"The requested toolkit ({toolkit}) is not support by the OFFTK. "
                f"Please chose from {cls._toolkits.keys()}."
            )
        else:
            return toolkit

    def provenance(self) -> Dict:
        """
        This component calls the OFFTK to perform the task and logs information on the backend toolkit used.

        Returns:
            A dictionary containing the version information about the backend toolkit called to perform the task.
        """

        import openforcefield

        import qcsubmit

        provenance = {
            "OpenforcefieldToolkit": openforcefield.__version__,
            "QCSubmit": qcsubmit.__version__,
        }

        if self.toolkit == "rdkit":
            import rdkit

            provenance["rdkit"] = rdkit.__version__

        elif self.toolkit == "openeye":
            import openeye

            provenance["openeye"] = openeye.__version__

        return provenance

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if any of the requested backend toolkits can be used.
        """
        if len(cls._toolkits) == 1:
            # the package needs a specific toolkit so raise the error
            raise_error = True
        else:
            raise_error = False

        for toolkit in cls._toolkits:
            if toolkit == "openeye":
                oe = which_import(
                    ".oechem",
                    package="openeye",
                    return_bool=True,
                    raise_error=raise_error,
                    raise_msg="Please install via `conda install openeye-toolkits -c openeye`.",
                )
                if oe:
                    return True
            elif toolkit == "rdkit":
                rdkit = which_import(
                    "rdkit",
                    return_bool=True,
                    raise_error=raise_error,
                    raise_msg="Please install via `conda install rdkit -c conda-forge`.",
                )
                if rdkit:
                    return True
        # if we are here both toolkits are missing
        raise ModuleNotFoundError(
            f"Openeye or RDKit is required to use this component please install via `conda install openeye-toolkits -c openeye` or `conda install rdkit -c conda-forge`."
        )


class BasicSettings(BaseModel):
    """
    This mixin identifies the class as being basic and always being available as it only requires basic packages.
    """

    @classmethod
    def is_available(cls) -> bool:
        """
        This component is basic if it requires no extra dependencies.
        """

        return True

    def provenance(self) -> Dict:
        """
        The basic settings provenance generator.
        """

        import openforcefield

        import qcsubmit

        provenance = {
            "OpenforcefieldToolkit": openforcefield.__version__,
            "QCSubmit": qcsubmit.__version__,
        }

        return provenance
