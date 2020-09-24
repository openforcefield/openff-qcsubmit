import abc
from typing import Dict, List, Tuple, Union

import tqdm
from openforcefield.topology import Molecule
from openforcefield.utils.toolkits import OpenEyeToolkitWrapper, RDKitToolkitWrapper
from pydantic import BaseModel, validator

from ..datasets import ComponentResult


class CustomWorkflowComponent(BaseModel, abc.ABC):
    """
    This is an abstract base class which should be used to create all workflow components, following the design of this
    class should allow users to easily create new work flow components with out needing to change much of the dataset
    factory code
    """

    component_name: str
    component_description: str
    component_fail_message: str
    _processes: Union[None, int] = 1  # By default everything is done in serial
    skip_unique_check: bool = False

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    @property
    def processes(self):
        return self._processes

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
        and then return the

        Parameters:
            molecules: The list of molecules to be processed by this component.

        Returns:
            An instance of the [ComponentResult][qcsubmit.datasets.ComponentResult]
            class which handles collecting together molecules that pass and fail
            the component
        """
        ...

    def _apply_init(self, result: ComponentResult) -> None:
        """"""
        ...

    def _apply_finalize(self, result: ComponentResult) -> None:
        """"""
        ...

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        This is the main feature of the workflow component which should accept a molecule, perform the component action
        and then return the

        Parameters:
            molecules: The list of molecules to be processed by this component.

        Returns:
            An instance of the [ComponentResult][qcsubmit.datasets.ComponentResult]
            class which handles collecting together molecules that pass and fail
            the component
        """
        result: ComponentResult = self._create_result(
            skip_unique_check=self.skip_unique_check
        )

        self._apply_init(result)

        # Use a Pool to get around the GIL. As long as self does not contain
        # too much data, this should be efficient.

        if self.processes is None or self.processes > 1:

            from multiprocessing.pool import Pool

            with Pool(processes=self.processes) as pool:

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
                ):
                    work = work.get()
                    for success in work.molecules:
                        result.add_molecule(success)
                    for molecule in work.filtered:
                        self.fail_molecule(molecule, component_result=result)

        else:
            for molecule in tqdm.tqdm(
                molecules,
                total=len(molecules),
                ncols=80,
                desc="{:30s}".format(self.component_name),
            ):
                work = self._apply([molecule])
                for success in work.molecules:
                    result.add_molecule(success)
                for fail in work.filtered:
                    self.fail_molecule(fail, component_result=result)

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
            **kwargs,
        )

        return result

    def fail_molecule(
        self, molecule: Molecule, component_result: ComponentResult
    ) -> None:
        """
        A method to fail a molecule.

        Parameters:
            molecule: The instance of the molecule to be failed.
            component_result: The [ComponentResult][qcsubmit.datasets.ComponentResult] instance that the molecule should
                be added to.
        """

        component_result.filter_molecule(molecule)


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

        for toolkit in cls._toolkits.values():
            if toolkit.is_available():
                return True

        return False


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
