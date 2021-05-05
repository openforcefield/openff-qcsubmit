"""
File containing the filters workflow components.
"""
import re
from typing import Dict, List, Optional, Set, Union

from openff.toolkit.topology import Molecule
from openff.toolkit.typing.chemistry.environment import (
    ChemicalEnvironment,
    SMIRKSParsingError,
)
from openff.toolkit.typing.engines.smirnoff import ForceField
from pydantic import Field, validator
from rdkit.Chem.rdMolAlign import AlignMol
from typing_extensions import Literal

from openff.qcsubmit.common_structures import ComponentProperties, TorsionIndexer
from openff.qcsubmit.datasets import ComponentResult
from openff.qcsubmit.validators import check_allowed_elements
from openff.qcsubmit.workflow_components.base_component import (
    BasicSettings,
    CustomWorkflowComponent,
)


class MolecularWeightFilter(BasicSettings, CustomWorkflowComponent):
    """
    Filters molecules based on the minimum and maximum allowed molecular weights.
    """

    component_name: Literal["MolecularWeightFilter"] = "MolecularWeightFilter"
    minimum_weight: int = Field(
        130,
        description="The minimum allowed molecule weight  default value taken from the openeye blockbuster filter",
    )
    maximum_weight: int = Field(
        781,
        description="The maximum allow molecule weight, default taken from the openeye blockbuster filter.",
    )

    @classmethod
    def description(cls) -> str:
        return "Molecules are filtered based on the allowed molecular weights."

    @classmethod
    def fail_reason(cls) -> str:
        return "Molecule weight was not in the specified region."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        The common entry point of all workflow components which applies the workflow component to the given list of
        molecules.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        from rdkit.Chem import Descriptors

        result = self._create_result()

        for molecule in molecules:
            total_weight = Descriptors.ExactMolWt(molecule.to_rdkit())

            if self.minimum_weight < total_weight < self.maximum_weight:
                result.add_molecule(molecule)
            else:
                result.filter_molecule(molecule)

        return result

    def provenance(self) -> Dict:
        """
        Generate version information for all of the software used during the running of this component.

        Returns:
            A dictionary of all of the software used in the component along wither their version numbers.

        Important:
            The simtk unit module has no version information so the version of OpenMM is given instead.
        """

        from simtk import openmm

        provenance = super().provenance()
        provenance["openmm_units"] = openmm.__version__

        return provenance


class ElementFilter(BasicSettings, CustomWorkflowComponent):
    """
    Filter the molecules based on a list of allowed elements.

    Note:
        The `allowed_elements` attribute can take a list of either symbols or atomic numbers and will resolve them to a
        common internal format as required.

    Example:
        Using atomic symbols or atomic numbers in components.

        ```python
        >>> from openff.qcsubmit.workflow_components import ElementFilter
        >>> efil = ElementFilter()
        # set the allowed elements to H,C,N,O
        >>> efil.allowed_elements = ['H', 'C', 'N', 'O']
        >>> efil.allowed_elements = [1, 6, 7, 8]
        ```
    """

    component_name: Literal["ElementFilter"] = "ElementFilter"
    allowed_elements: List[Union[int, str]] = Field(
        [
            "H",
            "C",
            "N",
            "O",
            "F",
            "P",
            "S",
            "Cl",
            "Br",
            "I",
        ],
        description="The list of allowed elements as symbols or atomic number ints.",
    )
    _check_elements = validator("allowed_elements", each_item=True, allow_reuse=True)(
        check_allowed_elements
    )

    @classmethod
    def description(cls) -> str:
        return (
            "Filter out molecules who contain elements not in the allowed element list."
        )

    @classmethod
    def fail_reason(cls) -> str:
        return "Molecules contained elements not in the allowed elements list."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def _apply_init(self, result: ComponentResult) -> None:

        from simtk.openmm.app import Element

        self._cache["elements"] = [
            Element.getBySymbol(ele).atomic_number if isinstance(ele, str) else ele
            for ele in self.allowed_elements
        ]

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        The common entry point of all workflow components which applies the workflow component to the given list of
        molecules.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = self._create_result()

        # First lets convert the allowed_elements list to ints as this is what is stored in the atom object
        _allowed_elements = self._cache["elements"]

        # now apply the filter
        for molecule in molecules:
            for atom in molecule.atoms:
                if atom.atomic_number not in _allowed_elements:
                    result.filter_molecule(molecule)
                    break
            else:
                result.add_molecule(molecule)

        return result

    def provenance(self) -> Dict:
        """
        Generate version information for all of the software used during the running of this component.

        Returns:
            A dictionary of all of the software used in the component along wither their version numbers.

        Note:
            The element class in OpenMM is used to match the elements so the OpenMM version is given.
        """

        from simtk import openmm

        provenance = super().provenance()
        provenance["openmm_elements"] = openmm.__version__

        return provenance


class CoverageFilter(BasicSettings, CustomWorkflowComponent):
    """
    Filters molecules based on the requested forcefield coverage.

    Important:
        The ids supplied to the respective group are the ids that are allowed, if `None` is passed all ids are allowed.

    Note:
        * If a molecule has any id in the allowed_ids and not in the filtered ids it is passed. Any molecule with a
            parameter in both sets is failed.
        * If None is passed to allowed IDs and tag_dihedrals will have no effect as all dihedrals are scanned by default.

    Important:
        A value of None in a list will let all molecules through.
    """

    component_name: Literal["CoverageFilter"] = "CoverageFilter"
    allowed_ids: Optional[Set[str]] = Field(
        None,
        description="The SMIRKS parameter ids of the parameters which are allowed to be exercised by the molecules. Molecules should use atleast one of these ids to be passed by the component.",
    )
    filtered_ids: Optional[Set[str]] = Field(
        None,
        description="The SMIRKS parameter ids of the parameters which are not allowed to be exercised by the molecules.",
    )
    forcefield: str = Field(
        "openff_unconstrained-1.0.0.offxml",
        description="The name of the forcefield which we want to filter against.",
    )
    tag_dihedrals: bool = Field(
        False,
        description="If we should tag any dihedral ids exercised for torsion driving.",
    )

    @classmethod
    def description(cls) -> str:
        return "Filter the molecules based on the requested FF allowed parameters."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule was typed with disallowed parameters."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def _apply_init(self, result: ComponentResult) -> None:

        self._cache["forcefield"] = ForceField(self.forcefield)

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Apply the filter to the list of molecules to remove any molecules typed by an id that is not allowed, i.e. not
        included in the allowed list.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = self._create_result()

        forcefield: ForceField = self._cache["forcefield"]

        # type the molecules
        for molecule in molecules:
            labels = forcefield.label_molecules(molecule.to_topology())[0]
            # format the labels into a set
            covered_types = set(
                [label.id for types in labels.values() for label in types.values()]
            )
            # use set intersection to check coverage for unwanted types
            # if filtered is None change to an empty set.
            unwanted_types = covered_types.intersection(self.filtered_ids or set())
            if unwanted_types:
                # fail the molecule for any unwanted matches
                result.filter_molecule(molecule)
                continue

            # now check for wanted common types
            # if the allowed option is None change to have overlap
            common_types = covered_types.intersection(self.allowed_ids or covered_types)
            if common_types:
                # here we have to find improper and proper dihedrals to tag
                if self.tag_dihedrals:
                    torsion_indexer = TorsionIndexer()
                    # combine a full torsion list
                    torsion_labels = labels["ProperTorsions"]
                    torsion_labels.update(labels["ImproperTorsions"])
                    for type_label in common_types:
                        if "t" in type_label or "i" in type_label:
                            for torsion, parameter in torsion_labels.items():
                                if type_label == parameter.id:
                                    if "Improper" in parameter.__class__.__name__:
                                        torsion_indexer.add_improper(
                                            central_atom=torsion[1],
                                            improper=torsion,
                                            scan_range=None,
                                        )
                                    elif "Proper" in parameter.__class__.__name__:
                                        torsion_indexer.add_torsion(
                                            torsion=torsion, scan_range=None
                                        )

                    molecule.properties["dihedrals"] = torsion_indexer

                result.add_molecule(molecule)

            else:
                result.filter_molecule(molecule)

        return result

    def provenance(self) -> Dict:
        """
        Generate version information for all of the software used during the running of this component.

        Returns:
            A dictionary of all of the software used in the component along wither their version numbers.
        """
        import openforcefields

        provenance = super().provenance()
        provenance["openforcefields"] = openforcefields.__version__

        return provenance


class RotorFilter(BasicSettings, CustomWorkflowComponent):
    """
    Filters molecules based on the maximum allowed number of rotatable bonds.

    Note:
        Rotatable bonds are torsions found using the `find_rotatable_bonds` method of the
        openforcefield.topology.Molecule class.
    """

    component_name: Literal["RotorFilter"] = "RotorFilter"
    maximum_rotors: Optional[int] = Field(
        4,
        description="The maximum number of rotatable bonds allowed in the molecule, if `None` the molecule has no maximum limit on rotatable bonds.",
    )
    minimum_rotors: Optional[int] = Field(
        None,
        description="The minimum number of rotatble bonds allowed in the molecule, if `None` the molecule has no limit to the minimum number of rotatble bonds.",
    )

    @classmethod
    def description(cls) -> str:
        return "Filter the molecules based on the maximum number of allowed rotatable bonds."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule has too many rotatable bonds."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def _apply_init(self, result: ComponentResult) -> None:
        """
        Validate the choice of minimum and maximum rotators.
        """
        if self.maximum_rotors and self.minimum_rotors:
            if self.maximum_rotors < self.minimum_rotors:
                raise ValueError(
                    "The maximum number of rotors should >= the minimum to ensure some molecules pass."
                )

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Apply the filter to the list of molecules to remove any molecules with more rotors then the maximum allowed
        number.

        Args:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        # create the return
        result = self._create_result()

        for molecule in molecules:
            # cache the rotatable bonds calc and only check fail conditions
            rotatable_bonds = molecule.find_rotatable_bonds()
            if self.maximum_rotors and len(rotatable_bonds) > self.maximum_rotors:
                result.filter_molecule(molecule)
            elif self.minimum_rotors and len(rotatable_bonds) < self.minimum_rotors:
                result.filter_molecule(molecule)
            else:
                result.add_molecule(molecule)

        return result


class SmartsFilter(BasicSettings, CustomWorkflowComponent):
    """
    Filters molecules based on if they contain certain smarts substructures.

    Note:
        * The smarts tags used for filtering should be numerically tagged in order to work with the toolkit.
        * If None is passed to the allowed list all molecules that dont match a filter pattern will be passed.
        * If tag_dihedrals is set to true any smarts pattern tagging 4 atoms in a torsion will be prepared for a torsiondrive.
    """

    component_name: Literal["SmartsFilter"] = "SmartsFilter"
    allowed_substructures: Optional[List[str]] = Field(
        None,
        description="The list of allowed substructures which should be tagged with indicies.",
    )
    filtered_substructures: Optional[List[str]] = Field(
        None, description="The list of substructures which should be filtered."
    )
    tag_dihedrals: bool = Field(
        False,
        description="If any dihedrals included in the allowed smarts should also be tagged for torsion driving.",
    )

    @classmethod
    def description(cls) -> str:
        return "Filter molecules based on the given smarts patterns."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule did/didn't contain the given smarts patterns."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    @validator("allowed_substructures", "filtered_substructures", each_item=True)
    def _check_environments(cls, environment):
        """
        Check the the string passed is valid by trying to create a ChemicalEnvironment in the toolkit.
        """

        # try and make a new chemical environment checking for parse errors
        _ = ChemicalEnvironment(smirks=environment)

        # check for numeric tags in the environment
        if re.search(":[0-9]]+", environment) is not None:
            return environment

        else:
            raise SMIRKSParsingError(
                "The smarts pattern passed had no tagged atoms please tag the atoms in the "
                "substructure you wish to include/exclude."
            )

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Apply the filter to the input list of molecules removing those that match the filtered set or do not contain an
        allowed substructure.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = self._create_result()

        if self.allowed_substructures is None:
            # pass all of the molecules
            for molecule in molecules:
                result.add_molecule(molecule=molecule)

        else:
            for molecule in molecules:
                # keep all dihedral matches here
                dihedrals = TorsionIndexer()
                for substructure in self.allowed_substructures:
                    matches = molecule.chemical_environment_matches(query=substructure)
                    if matches and not self.tag_dihedrals:
                        result.add_molecule(molecule=molecule)
                        break
                    elif matches and self.tag_dihedrals:
                        # add the dihedral for tagging if valid
                        for match in matches:
                            # this will handle deduplication
                            dihedrals.add_torsion(torsion=match, scan_range=None)
                    else:
                        continue
                else:
                    # if we have dihedrals then add the molecule else fail it as we didn't break
                    if dihedrals.n_torsions >= 1:
                        molecule.properties["dihedrals"] = dihedrals
                        result.add_molecule(molecule)
                    else:
                        result.filter_molecule(molecule=molecule)

        if self.filtered_substructures is not None:
            # now we only want to check the molecules in the pass list
            molecules_to_remove = []
            for molecule in result.molecules:
                for substructure in self.filtered_substructures:
                    if molecule.chemical_environment_matches(query=substructure):
                        molecules_to_remove.append(molecule)
                        break

            # Failing a molecule automatically removes it from the successes
            for molecule in molecules_to_remove:
                result.filter_molecule(molecule)

        return result


class RMSDCutoffConformerFilter(BasicSettings, CustomWorkflowComponent):
    """
    Prunes conformers from a molecule that are less than a specified RMSD from
    all other conformers
    """

    # standard components which must be defined
    component_name: Literal["RMSDCutoffConformerFilter"] = "RMSDCutoffConformerFilter"
    # custom components for this class
    cutoff: float = Field(-1.0, description="The RMSD cut off in angstroms.")

    @classmethod
    def description(cls) -> str:
        return "Filter conformations for the given molecules using a RMSD cutoff."

    @classmethod
    def fail_reason(cls) -> str:
        return "Could not filter the conformers using RMSD."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def _prune_conformers(self, molecule: Molecule) -> None:

        no_conformers: int = molecule.n_conformers

        # This will be used to determined whether it should be pruned
        # from the RMSD calculations. If we find it should be pruned
        # just once, it is sufficient to avoid it later in the pairwise
        # processing.
        uniq: List = list([True] * no_conformers)

        # Needed to get the aligned best-fit RMSD
        rdmol = molecule.to_rdkit()

        rmsd = []
        # This begins the pairwise RMSD pruner
        if no_conformers > 1 and self.cutoff >= 0.0:

            # The reference conformer for RMSD calculation
            for j in range(no_conformers - 1):

                # A previous loop has determine this specific conformer
                # is too close to another, so we can entirely skip it
                if not uniq[j]:
                    continue

                # since k starts from j+1, we are only looking at the
                # upper triangle of the comparisons (j < k)
                for k in range(j + 1, no_conformers):

                    rmsd_i = AlignMol(rdmol, rdmol, k, j)
                    rmsd.append(rmsd_i)

                    # Flag this conformer for pruning, and also
                    # prevent it from being used as a reference in the
                    # future comparisons
                    if rmsd_i < self.cutoff:
                        uniq[k] = False

            confs = [
                molecule.conformers[j] for j, add_bool in enumerate(uniq) if add_bool
            ]

            molecule._conformers = confs.copy()

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Prunes conformers from a molecule that are less than a specified RMSD from
        all other conformers

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = self._create_result()

        for molecule in molecules:

            if molecule.n_conformers == 0:
                result.filter_molecule(molecule)
            else:
                self._prune_conformers(molecule)
                result.add_molecule(molecule)

        return result
