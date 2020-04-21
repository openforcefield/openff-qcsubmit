from typing import List

from openforcefield.topology import Molecule

from .base_component import CustomWorkflowComponent, ToolkitValidator
from qcsubmit.datasets import ComponentResult


class StandardConformerGenerator(ToolkitValidator, CustomWorkflowComponent):
    """
    Standard conformer generator using the OFFTK and the back end toolkits.

    Notes
    -----
        The provenance information and toolkit settings are handled by the
        [ToolkitValidator][qcsubmit.workflow_components.base_component.ToolkitValidator] mixin.
    """

    # standard components which must be defined
    component_name = "StandardConformerGenerator"
    component_description = "Generate conformations for the given molecules"
    component_fail_message = "Conformers could not be generated"

    # custom components for this class
    max_conformers: int = 10
    clear_existing: bool = True

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
         Generate conformers for the molecules using the selected toolkit backend.

         Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = ComponentResult(component_name=self.component_name, component_description=self.dict())

        # create the toolkit
        toolkit = self._toolkits[self.toolkit]()

        for molecule in molecules:
            try:
                molecule.generate_conformers(
                    n_conformers=self.max_conformers, clear_existing=self.clear_existing, toolkit_registry=toolkit
                )

            # need to catch more specific exceptions here.
            except Exception:
                self.fail_molecule(molecule=molecule, component_result=result)

            finally:
                # if we could not produce a conformer then fail the molecule
                if molecule.n_conformers == 0:
                    self.fail_molecule(molecule=molecule, component_result=result)
                else:
                    result.add_molecule(molecule)

        return result
