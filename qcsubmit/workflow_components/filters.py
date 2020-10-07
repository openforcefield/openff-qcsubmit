"""
File containing the filters workflow components.
"""
import re
from typing import Dict, List, Optional, Set, Union

import numpy as np

from openforcefield.topology import Molecule
from openforcefield.typing.chemistry.environment import (
    ChemicalEnvironment,
    SMIRKSParsingError,
)
from openforcefield.typing.engines.smirnoff import ForceField
from pydantic import validator
from qcsubmit.common_structures import TorsionIndexer
from qcsubmit.datasets import ComponentResult
from rdkit.Chem.rdMolAlign import AlignMol

from .base_component import BasicSettings, CustomWorkflowComponent


class FilterComponent(CustomWorkflowComponent):
    """
    A base class for molecule filtering components, which define common settings
    and configuration.
    All filters are expected to take a list of molecules, and pass on a subset
    of them.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skip_unique_check: bool = True  # Filters do not create molecules


class MolecularWeightFilter(BasicSettings, FilterComponent):
    """
    Filters molecules based on the minimum and maximum allowed molecular weights.

    Attributes:
        fields.component_name: The name of component.
        fields.component_description: A short description of the component.
        fields.component_fail_message: The message logged when a molecule fails this component.
        fields.minimum_weight: The minimum allowed molecular weight of a molecule.
        fields.maximum_weight: The maximum allowed molecular weight of a molecule.
    """

    component_name = "MolecularWeightFilter"
    component_description = (
        "Molecules are filtered based on the allowed molecular weights."
    )
    component_fail_message = "Molecule weight was not in the specified region."

    minimum_weight: int = (
        130  # values taken from the base settings of the openeye blockbuster filter
    )
    maximum_weight: int = 781

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


class ElementFilter(BasicSettings, FilterComponent):
    """
    Filter the molecules based on a list of allowed elements.

    Attributes:
        component_name: The name of component.
        component_description: A short desciption of the component.
        component_fail_message: The message logged when a molecule fails this component.
        allowed_elements: A list of atomic symbols or atomic numbers which are allowed passed the filter.

    Note:
        The `allowed_elements` attribute can take a list of either symbols or atomic numbers and will resolve them to a
        common internal format as required.

    Example:
        Using atomic symbols or atomic numbers in components.

        ```python
        >>> from qcsubmit import workflow_components
        >>> efil = workflow_components.ElementFilter()
        # set the allowed elements to H,C,N,O
        >>> efil.allowed_elements = ['H', 'C', 'N', 'O']
        >>> efil.allowed_elements = [1, 6, 7, 8]
        ```
    """

    component_name = "ElementFilter"
    component_description = (
        "Filter out molecules who contain elements not in the allowed element list"
    )
    component_fail_message = (
        "Molecule contained elements not in the allowed elements list"
    )

    allowed_elements: List[Union[int, str]] = [
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
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache: Union[List[int], None] = None

    @validator("allowed_elements", each_item=True)
    def check_allowed_elements(cls, element: Union[str, int]) -> Union[str, int]:
        """
        Check that each item can be cast to a valid element.

        Parameters:
            element: The element that should be checked.

        Raises:
            ValueError: If the element number or symbol passed could not be converted into a valid element.
        """
        from simtk.openmm.app import Element

        if isinstance(element, int):
            return element
        else:
            try:
                _ = Element.getBySymbol(element)
                return element
            except KeyError:
                raise KeyError(
                    f"An element could not be determined from symbol {element}, please enter symbols only."
                )

    def _apply_init(self, result: ComponentResult) -> None:

        from simtk.openmm.app import Element

        self._cache = [
            Element.getBySymbol(ele).atomic_number if isinstance(ele, str) else ele
            for ele in self.allowed_elements
        ]

    def _apply_finalize(self, result: ComponentResult) -> None:

        self._cache = None

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
        _allowed_elements = self._cache

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


class CoverageFilter(BasicSettings, FilterComponent):
    """
    Filters molecules based on the requested forcefield coverage.

    Important:
        The ids supplied to the respective group are the ids that are allowed, if `None` is passed all ids are allowed.

    Attributes:
        allowed_ids: The list of parameter ids that we want to actively pass the filter.
        filtered_ids: The list of parameter ids that we want to actively filter out and fail the filter.
        forcefield: The name of the force field we are checking against.
        tag_dihedrals: If any dihedral terms are in the allowed IDs they will be tagged as well as being passed, note
            only one dihedral per rotatable bond will be tagged for driving.

    Note:
        * If a molecule has any id in the allowed_ids and not in the filtered ids it is passed. Any molecule with a
            parameter in both sets is failed.
        * If None is passed to allowed IDs and tag_dihedrals will have no effect as all dihedrals are scanned by default.

    Important:
        A value of None in a list will let all molecules through.
    """

    component_name = "CoverageFilter"
    component_description = (
        "Filter the molecules based on the requested FF allowed parameters."
    )
    component_fail_message = "The molecule was typed with disallowed parameters."

    allowed_ids: Optional[Set[str]] = None
    filtered_ids: Optional[Set[str]] = None
    forcefield: str = "openff_unconstrained-1.0.0.offxml"
    tag_dihedrals: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._processes = None  # This component uses an expensive calculation
        self._cache: Union[ForceField, None] = None

    def _apply_init(self, result: ComponentResult) -> None:

        self._cache = ForceField(self.forcefield)

    def _apply_finalize(self, result: ComponentResult) -> None:

        self._cache = None

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

        forcefield: ForceField = self._cache

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


class RotorFilter(BasicSettings, FilterComponent):
    """
    Filters molecules based on the maximum allowed number of rotatable bonds.

    Note:
        Rotatable bonds are non terminal torsions found using the `find_rotatable_bonds` method of the
        openforcefield.topology.Molecule class.
    """

    component_name = "RotorFilter"
    component_description = (
        "Filter the molecules based on the maximum number of allowed rotatable bonds."
    )
    component_fail_message = "The molecule has too many rotatable bonds."

    maximum_rotors: int = 4

    def _apply(self, molecules: List[Molecule]) -> ComponentResult:
        """
        Apply the filter to the list of molecules to remove any molecules with more rotors then the maximum allowed
        number.

        Parameters:
            molecules: The list of molecules the component should be applied on.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        # create the return
        result = self._create_result()

        # run the the molecules and calculate the number of rotatable bonds
        for molecule in molecules:
            if len(molecule.find_rotatable_bonds()) > self.maximum_rotors:
                result.filter_molecule(molecule)

            else:
                result.add_molecule(molecule)

        return result


class SmartsFilter(BasicSettings, FilterComponent):
    """
    Filters molecules based on if they contain certain smarts substructures.

    Note:
        * The smarts tags used for filtering should be numerically tagged in order to work with the toolkit.
        * If None is passed to the allowed list all molecules that dont match a filter pattern will be passed.
    """

    component_name = "SmartsFilter"
    component_description = "Filter molecules based on the given smarts patterns."
    component_fail_message = (
        "The molecule did/didn't contain the given smarts patterns."
    )

    allowed_substructures: Optional[List[str]] = None
    filtered_substructures: Optional[List[str]] = None

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
                for substructure in self.allowed_substructures:
                    if molecule.chemical_environment_matches(query=substructure):
                        result.add_molecule(molecule)
                        break
                else:
                    result.filter_molecule(molecule)

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


class RMSDCutoffConformerFilter(BasicSettings, FilterComponent):
    """
    Prunes conformers from a molecule that are less than a specified RMSD from
    all other conformers
    """

    # standard components which must be defined
    component_name = "RMSDCutoffConformerFilter"
    component_description = (
        "Filter conformations for the given molecules using a RMSD cutoff"
    )
    component_fail_message = "Could not filter the conformers using RMSD"

    # custom components for this class
    rms_cutoff: float = -1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._processes = None  # This component uses an expensive calculation

    def _prune_conformers(self, molecule: Molecule) -> None:

        L: int = len(molecule.conformers)

        # This will be used to determined whether it should be pruned
        # from the RMSD calculations. If we find it should be pruned
        # just once, it is sufficient to avoid it later in the pairwise
        # processing.
        uniq: List = list([True] * L)

        # Needed to get the aligned best-fit RMSD
        rdmol = molecule.to_rdkit()

        rmsd = []
        # This begins the pairwise RMSD pruner
        if L > 1 and self.rms_cutoff >= 0.0:

            # The reference conformer for RMSD calculation
            for j in range(L - 1):

                # A previous loop has determine this specific conformer
                # is too close to another, so we can entirely skip it
                if not uniq[j]:
                    continue

                # since k starts from j+1, we are only looking at the
                # upper triangle of the comparisons (j < k)
                for k in range(j + 1, L):

                    # r = np.linalg.norm(
                    #     molecule.conformers[k] - molecule.conformers[j], axis=1
                    # )
                    # rmsd_i = r.mean()
                    rmsd_i = AlignMol(rdmol, rdmol, k, j)
                    rmsd.append(rmsd_i)

                    # Flag this conformer for pruning, and also
                    # prevent it from being used as a reference in the
                    # future comparisons
                    if rmsd_i < self.rms_cutoff:
                        uniq[k] = False

            confs = [
                molecule.conformers[j] for j, add_bool in enumerate(uniq) if add_bool
            ]

            # TODO: use a logger
            # rmsd = np.array(rmsd)
            # print(
            #     "Pruned conformers {}/{} min={} mean={} max={}".format(
            #         len(confs), L, rmsd.min(), rmsd.mean(), rmsd.max()
            #     )
            # )
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
