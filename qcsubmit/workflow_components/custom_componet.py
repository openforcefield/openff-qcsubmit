from typing import List
import abc
from pydantic import BaseModel, validator, ValidationError
from openforcefield.topology import Molecule

class ComponentMissingError(Exception):
    pass

class CustomWorkflowComponet(BaseModel, abc.ABC):
    """
    This is an abstract base class which should be used to create all workflow componets, following the desgin of this
    class should allow users to easily create new work flow components with out needing to change much of the dataset
    factory code
    """

    componet_name: str
    componet_descripton: str
    componet_fail_message: str

    class Config:
        validate_assignment = True

    @abc.abstractmethod
    def apply(self, molecules: List[Molecule]) -> List[List[Molecule]]:
        """This is the main feature of the workflow component which should accept a list of molecules and return two
        lists those that passed this step and those that did not.

        Prameters
        ----------

        molecules: List[openforcefield.topology.Molecules]:
            The list fof molecules to be processed by this component.

        Returns
        -------
        pass_molecules:

        fail_molecules:
        """
        ...


class StandardConformerGenerator(CustomWorkflowComponet):

    # standard componets which must be defined
    componet_name = 'StandardConformerGenerator'
    componet_descripton = "Generate conformations for the given molecules"
    componet_fail_message = "Conformers could not be generated"

    # custom components for this class
    max_conformers: int = 20
    clear_exsiting: bool = True

    def apply(self, molecules: List[Molecule]) -> List[List[Molecule]]:
        "test apply the conformers"

        pass_molecules = []
        fail_molecues = []
        for molecule in molecules:
            try:
                molecule.generate_conformers(n_conformers=self.max_conformers, clear_existing=self.clear_exsiting)
                pass_molecules.append(molecule)
            except Exception:
                fail_molecues.append(molecule)

        return [pass_molecules, fail_molecues]

