"""
File containing the filters workflow components.
"""
from typing import Dict, List, Optional, Set, Union

from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from pydantic import Field, root_validator, validator
from rdkit.Chem.rdMolAlign import AlignMol
from simtk import unit
from typing_extensions import Literal

from openff.qcsubmit.common_structures import ComponentProperties
from openff.qcsubmit.validators import check_allowed_elements, check_environments
from openff.qcsubmit.workflow_components.base_component import (
    BasicSettings,
    CustomWorkflowComponent,
)
from openff.qcsubmit.workflow_components.utils import ComponentResult


class MolecularWeightFilter(BasicSettings, CustomWorkflowComponent):
    """
    Filters molecules based on the minimum and maximum allowed molecular weights.
    """

    type: Literal["MolecularWeightFilter"] = "MolecularWeightFilter"
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

    type: Literal["ElementFilter"] = "ElementFilter"
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
    Filters molecules based on the requested force field parameter ids.

    Note:
        * The options ``allowed_ids`` and ``filtered_ids`` are mutually exclusive.

    """

    type: Literal["CoverageFilter"] = "CoverageFilter"
    allowed_ids: Optional[Set[str]] = Field(
        None,
        description="The SMIRKS parameter ids of the parameters which are allowed to be exercised by the molecules. "
        "Molecules should use at least one of these ids to be passed by the component.",
    )
    filtered_ids: Optional[Set[str]] = Field(
        None,
        description="The SMIRKS parameter ids of the parameters which are not allowed to be exercised by the molecules.",
    )
    forcefield: str = Field(
        "openff_unconstrained-1.0.0.offxml",
        description="The name of the force field which we want to filter against.",
    )

    @classmethod
    def description(cls) -> str:
        return "Filter the molecules based on the requested FF allowed parameters."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule was typed with disallowed parameters."

    @root_validator
    def _validate_mutually_exclusive(cls, values):
        ids_to_include = values.get("allowed_ids")
        ids_to_exclude = values.get("filtered_ids")

        message = "exactly one of ``allowed_ids` and `filtered_ids` must specified."

        assert ids_to_include is not None or ids_to_exclude is not None, message
        assert ids_to_include is None or ids_to_exclude is None, message

        return values

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def _apply_init(self, result: ComponentResult) -> None:

        self._cache["forcefield"] = ForceField(self.forcefield)

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Apply the filter to the list of molecules to remove any molecules typed by an id that is not allowed, i.e. not
        included in the allowed list.

        Args:
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

            # use set intersection to check coverage for unwanted and wanted types
            unwanted_types = covered_types.intersection(self.filtered_ids or set())
            common_types = covered_types.intersection(self.allowed_ids or set())

            if self.filtered_ids is not None and unwanted_types:
                # the molecule has an unwanted parameter id
                result.filter_molecule(molecule=molecule)
            elif self.allowed_ids is not None and not common_types:
                # the molecule does not contain the wanted parameter id
                result.filter_molecule(molecule=molecule)
            else:
                # the molecule contains a wanted or does not contain a filtered parameter id
                result.add_molecule(molecule=molecule)

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
    Filters molecules based on the maximum and or minimum allowed number of rotatable bonds.

    Note:
        Rotatable bonds are torsions found using the `find_rotatable_bonds` method of the
        openforcefield.topology.Molecule class.
    """

    type: Literal["RotorFilter"] = "RotorFilter"
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
        * The options ``allowed_substructures`` and ``filtered_substructures`` are mutually exclusive.
    """

    type: Literal["SmartsFilter"] = "SmartsFilter"
    allowed_substructures: Optional[List[str]] = Field(
        None,
        description="The list of allowed substructures which should be tagged with indices.",
    )
    filtered_substructures: Optional[List[str]] = Field(
        None, description="The list of substructures which should be filtered."
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

    _check_smarts = validator(
        "allowed_substructures",
        "filtered_substructures",
        each_item=True,
        allow_reuse=True,
    )(check_environments)

    @root_validator
    def _validate_mutually_exclusive(cls, values):
        allowed_substructures = values.get("allowed_substructures")
        filtered_substructures = values.get("filtered_substructures")

        message = "exactly one of ``allowed_substructures` and `filtered_substructures` must specified."

        assert (
            allowed_substructures is not None or filtered_substructures is not None
        ), message
        assert allowed_substructures is None or filtered_substructures is None, message

        return values

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Apply the filter to the input list of molecules removing those that match the filtered set or do not contain an
        allowed substructure.

        Args:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = self._create_result()

        for molecule in molecules:

            if self.allowed_substructures is not None:
                for substructure in self.allowed_substructures:
                    if molecule.chemical_environment_matches(query=substructure):
                        result.add_molecule(molecule=molecule)
                        break
                else:
                    # the molecule does not contain the allowed substructure so remove it
                    result.filter_molecule(molecule=molecule)

            elif self.filtered_substructures is not None:
                for substructure in self.filtered_substructures:
                    if molecule.chemical_environment_matches(query=substructure):
                        result.filter_molecule(molecule=molecule)
                        break
                else:
                    # there was no filtered substructure so keep the molecule
                    result.add_molecule(molecule=molecule)

        return result


class RMSDCutoffConformerFilter(BasicSettings, CustomWorkflowComponent):
    """
    Prunes conformers from a molecule that are less than a specified RMSD from
    all other conformers
    """

    # standard components which must be defined
    type: Literal["RMSDCutoffConformerFilter"] = "RMSDCutoffConformerFilter"
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

        Args:
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


class ScanFilter(BasicSettings, CustomWorkflowComponent):
    """
    A filter to remove/include molecules from the workflow who have scans targeting the specified SMARTS.

    Important:
        Currently only checks against 1D scans.
    """

    type: Literal["ScanFilter"] = "ScanFilter"
    scans_to_include: Optional[List[str]] = Field(
        None,
        description="Only molecules with SCANs covering these SMARTs"
        "patterns should be kept. This option is mutually"
        "exclusive with ``scans_to_exclude``.",
    )

    scans_to_exclude: Optional[List[str]] = Field(
        None,
        description="Any molecules with scans covering these SMARTs will"
        "be removed from the dataset. This option is mutally"
        "exclusive with ``scans_to_include``.",
    )

    _check_smarts = validator(
        "scans_to_include", "scans_to_exclude", each_item=True, allow_reuse=True
    )(check_environments)

    @classmethod
    def description(cls) -> str:
        return "Filter molecules who have the desired/unwanted scans."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule contained an unwanted or did not contain a desired dihedral/improper scan."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    @root_validator
    def _validate_mutally_exclusive(cls, values):
        scans_to_include = values.get("scans_to_include")
        scans_to_exclude = values.get("scans_to_exclude")

        message = (
            "exactly one of `scans_to_include` and `scans_to_exclude` must be specified"
        )

        assert scans_to_include is not None or scans_to_exclude is not None, message
        assert scans_to_include is None or scans_to_exclude is None, message
        return values

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Keep or remove scans based on the list of torsions to include or remove.
        """

        result = self._create_result()

        target_environments = self.scans_to_exclude or self.scans_to_include

        for molecule in molecules:
            torsion_indexer = molecule.properties.get("dihedrals", None)
            # if no dihedrals are tagged remove the molecule
            if torsion_indexer is None or torsion_indexer.n_torsions == 0:
                result.filter_molecule(molecule=molecule)
                continue

            all_matches = set()
            for env in target_environments:
                # get all matches as a list of sorted central bonds as they are stored this way
                matches = molecule.chemical_environment_matches(query=env)
                for match in matches:
                    match = match if len(match) == 2 else match[1:3]
                    all_matches.add(tuple(sorted(match)))

            # now we either remove any torsions in this list or any missing from it based on include/exclude
            to_remove = []
            if self.scans_to_include is not None:
                for center_bond in torsion_indexer.torsions.keys():
                    if center_bond not in all_matches:
                        to_remove.append(center_bond)
            else:
                for center_bond in all_matches:
                    if center_bond in torsion_indexer.torsions.keys():
                        to_remove.append(center_bond)

            # now remove
            for bond in to_remove:
                del torsion_indexer.torsions[bond]

            # if we have no torsions left filter the molecule
            if not torsion_indexer.get_dihedrals:
                result.filter_molecule(molecule)

            result.add_molecule(molecule)

        return result


class ChargeFilter(BasicSettings, CustomWorkflowComponent):
    """
    Filter molecules if their formal charge is not in the `charges_to_include` list or is in the `charges_to_exclude` list.
    """

    type: Literal["ChargeFilter"] = "ChargeFilter"

    charges_to_include: Optional[List[int]] = Field(
        None,
        description="The list of net molecule formal charges which are allowed in the dataset."
        "This option is mutually exclusive with ``charges_to_exclude``.",
    )
    charges_to_exclude: Optional[List[int]] = Field(
        None,
        description="The list of net molecule formal charges which are to be removed from the dataset."
        "This option is mutually exclusive with ``charges_to_include``.",
    )

    @classmethod
    def description(cls) -> str:
        return "Filter molecules by net formal charge."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecules net formal charge was not requested or was in the `charges_to_exclude`."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    @root_validator
    def _validate_mutually_exclusive(cls, values):
        charges_to_include = values.get("charges_to_include")
        charges_to_exclude = values.get("charges_to_exclude")

        message = "exactly one of ``charges_to_include` and `charges_to_exclude` must specified."

        assert charges_to_include is not None or charges_to_exclude is not None, message
        assert charges_to_include is None or charges_to_exclude is None, message

        return values

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Filter molecules based on their net formal charge
        """

        result = self._create_result()

        for molecule in molecules:
            total_charge = molecule.total_charge.value_in_unit(unit.elementary_charge)

            if (
                self.charges_to_include is not None
                and total_charge not in self.charges_to_include
            ) or (
                self.charges_to_exclude is not None
                and total_charge in self.charges_to_exclude
            ):
                result.filter_molecule(molecule=molecule)

            else:
                result.add_molecule(molecule)

        return result
