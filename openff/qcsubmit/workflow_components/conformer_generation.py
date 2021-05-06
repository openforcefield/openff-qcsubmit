from typing import List, Optional

import simtk.unit as unit
from openff.toolkit.topology import Molecule
from pydantic import Field
from typing_extensions import Literal

from openff.qcsubmit.common_structures import ComponentProperties
from openff.qcsubmit.datasets import ComponentResult
from openff.qcsubmit.workflow_components.base_component import (
    CustomWorkflowComponent,
    ToolkitValidator,
)


class StandardConformerGenerator(ToolkitValidator, CustomWorkflowComponent):
    """
    Standard conformer generator using the OFFTK and the back end toolkits.

    Note:
        The provenance information and toolkit settings are handled by the
        [ToolkitValidator][qcsubmit.workflow_components.base_component.ToolkitValidator] mixin.
    """

    component_name: Literal["StandardConformerGenerator"] = "StandardConformerGenerator"

    rms_cutoff: Optional[float] = Field(
        None,
        description="The rms cut off in angstroms to be used when generating the conformers. Passing None will use the default in toolkit of 1.",
    )
    max_conformers: int = Field(
        10, description="The maximum number of conformers to be generated per molecule."
    )
    clear_existing: bool = Field(
        True, description="If any pre-existing conformers should be kept."
    )

    @classmethod
    def description(cls) -> str:
        return "Generate conformations for the given molecules."

    @classmethod
    def fail_reason(cls) -> str:
        return "Conformers could not be generated."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def _apply_init(self, result: ComponentResult) -> None:
        """
        Set up the standard conformer filter
        """
        if self.rms_cutoff is not None:
            self._cache["cutoff"] = self.rms_cutoff * unit.angstrom
        else:
            self._cache["cutoff"] = None

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Generate conformers for the molecules using the selected toolkit backend.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            An instance of the [ComponentResult][qcsubmit.datasets.ComponentResult]
            class which handles collecting together molecules that pass and fail
            the component
        """

        # create the toolkit
        toolkit = self._toolkits[self.toolkit]()

        result = self._create_result()

        rms_cutoff = self._cache["cutoff"]

        for molecule in molecules:
            try:
                # assume input is angstrom until Quantity can be serialized
                molecule.generate_conformers(
                    n_conformers=self.max_conformers,
                    clear_existing=self.clear_existing,
                    rms_cutoff=rms_cutoff,
                    toolkit_registry=toolkit,
                )

            # need to catch more specific exceptions here.
            except Exception:
                result.filter_molecule(molecule)

            finally:
                # if we could not produce a conformer then fail the molecule
                if molecule.n_conformers == 0:
                    result.filter_molecule(molecule)
                else:
                    result.add_molecule(molecule)

        return result
