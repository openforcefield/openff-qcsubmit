from typing import List, Dict
from pydantic import validator

from openforcefield.topology import Molecule
from openforcefield.utils.toolkits import RDKitToolkitWrapper, OpenEyeToolkitWrapper

from .base_component import CustomWorkflowComponet
from qcsubmit.datasets import ComponentResult


class StandardConformerGenerator(CustomWorkflowComponet):
    # standard components which must be defined
    componet_name = 'StandardConformerGenerator'
    componet_descripton = "Generate conformations for the given molecules"
    componet_fail_message = "Conformers could not be generated"

    # custom components for this class
    max_conformers: int = 20
    clear_exsiting: bool = True
    toolkit: str = 'rdkit'
    _toolkits: Dict = {'rdkit': RDKitToolkitWrapper, 'openeye': OpenEyeToolkitWrapper}

    @validator('toolkit')
    def _check_toolkit(cls, toolkit):
        """
        Make sure that toolkit is one of the supported types in the OFFTK.
        """
        if toolkit not in cls._toolkits.keys():
            raise ValueError(f'The requested toolkit ({toolkit}) is not support by the OFFTK to generate conformers. '
                             f'Please chose from {cls._toolkits.keys()}.')
        else:
            return toolkit

    def apply(self, molecules: List[Molecule]) -> ComponentResult:
        "test apply the conformers"

        result = ComponentResult(component_name=self.componet_name,
                                 component_description=self.componet_descripton,
                                 component_fail_reason=self.componet_fail_message)

        # create the toolkit
        toolkit = self._toolkits[self.toolkit]()

        for molecule in molecules:
            try:
                molecule.generate_conformers(n_conformers=self.max_conformers,
                                             clear_existing=self.clear_exsiting,
                                             toolkit_registry=toolkit)

                result.add_molecule(molecule)
            # need to catch more specific exceptions here.
            except Exception:
                self.fail_molecule(molecule=molecule, component_result=result)

        return result

    def provenance(self) -> Dict:
        """
        This component calls the OFFTK to do conformer generation.
        """

        import openforcefield

        provenance = {'OpenforcefieldToolkit': openforcefield.__version__}
        if self.toolkit == 'rdkit':
            import rdkit
            provenance['rdkit'] = rdkit.__version__

        elif self.toolkit == 'openeye':
            import openeye
            provenance['openeye'] = openeye.__version__

        return provenance

