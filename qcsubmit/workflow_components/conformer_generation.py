from typing import List, Union

import simtk.unit as unit
from openforcefield.topology import Molecule
from qcsubmit.datasets import ComponentResult

from .base_component import CustomWorkflowComponent, ToolkitValidator


class StandardConformerGenerator(ToolkitValidator, CustomWorkflowComponent):
    """
    Standard conformer generator using the OFFTK and the back end toolkits.

    Note:
        The provenance information and toolkit settings are handled by the
        [ToolkitValidator][qcsubmit.workflow_components.base_component.ToolkitValidator] mixin.
    """

    # standard components which must be defined
    component_name = "StandardConformerGenerator"
    component_description = "Generate conformations for the given molecules"
    component_fail_message = "Conformers could not be generated"

    # custom components for this class
    rms_cutoff: Union[float, None] = None
    max_conformers: int = 10
    clear_existing: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skip_unique_check: bool = (
            True  # This component does not create new molecules
        )
        self._processes: Union[
            int, None
        ] = None  # This component uses an expensive calculation
        self._cache: Union[unit.Quantity, None] = None

    def _apply_init(self, result: ComponentResult):

        self._cache = None

        if self.rms_cutoff is not None:
            self._cache = self.rms_cutoff * unit.angstrom

    def _apply_finalize(self, result: ComponentResult):

        self._cache = None

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

        rms_cutoff = self._cache

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
