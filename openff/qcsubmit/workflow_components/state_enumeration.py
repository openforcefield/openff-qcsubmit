"""
Components to expand stereochemistry and tautomeric states of molecules.
"""

from typing import List, Optional, Tuple

from openff.toolkit.topology import Molecule
from openff.toolkit.utils import ToolkitRegistry
from qcelemental.util import which_import
from typing_extensions import Literal

from openff.qcsubmit._pydantic import Field
from openff.qcsubmit.common_structures import ComponentProperties
from openff.qcsubmit.utils import (
    check_missing_stereo,
    get_symmetry_classes,
    get_symmetry_group,
)
from openff.qcsubmit.workflow_components.base_component import (
    CustomWorkflowComponent,
    ToolkitValidator,
)
from openff.qcsubmit.workflow_components.utils import (
    ComponentResult,
    ImproperScan,
    Scan1D,
    Scan2D,
    TorsionIndexer,
)


class EnumerateTautomers(ToolkitValidator, CustomWorkflowComponent):
    """
    Enumerate the tautomers of a molecule using the backend toolkits through the OFFTK.
    """

    type: Literal["EnumerateTautomers"] = "EnumerateTautomers"
    # custom settings for the class
    max_tautomers: int = Field(
        20, description="The maximum number of tautomers that should be generated."
    )

    @classmethod
    def description(cls) -> str:
        return "Enumerate the tautomers of a molecule if possible, returning the input plus any new molecules."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule tautomers could not be enumerated."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=True)

    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        Enumerate tautomers of the input molecule.

        The input molecules tautomers are enumerated using the desired backend toolkit and are returned along with the input molecule.

         Parameters:
            molecules: The list of molecules the component should be applied on.
            toolkit_registry: The openff.toolkit.utils.ToolkitRegistry which declares the available toolkits.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = self._create_result(toolkit_registry=toolkit_registry)

        for molecule in molecules:
            try:
                tautomers = molecule.enumerate_tautomers(
                    max_states=self.max_tautomers, toolkit_registry=toolkit_registry
                )
                for tautomer in tautomers:
                    result.add_molecule(tautomer)
                # add the input molecule
                result.add_molecule(molecule=molecule)

            except Exception:
                result.filter_molecule(molecule)

        return result


class EnumerateStereoisomers(ToolkitValidator, CustomWorkflowComponent):
    """
    Enumerate the stereo centers and bonds of a molecule using the backend toolkits through the OFFTK, only well defined
    molecules are returned by this component, this is check via a OFFTK round trip.
    """

    type: Literal["EnumerateStereoisomers"] = "EnumerateStereoisomers"
    undefined_only: bool = Field(
        False,
        description="If we should only enumerate parts of the molecule with undefined stereochemistry or all stereochemistry.",
    )
    max_isomers: int = Field(
        20, description="The maximum number of stereoisomers to be generated."
    )
    rationalise: bool = Field(
        True,
        description="If we should check that the resulting molecules are physically possible by attempting to generate conformers for them.",
    )

    @classmethod
    def description(cls) -> str:
        return "Enumerate the stereo centers and bonds of the molecule, returing the input molecule if valid and any new molecules."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecules stereo centers or bonds could not be enumerated."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=True)

    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        Enumerate stereo centers and bonds of the input molecule

        Parameters:
            molecules: The list of molecules the component should be applied on.
            toolkit_registry: The openff.toolkit.utils.ToolkitRegistry which declares the available toolkits.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.
        """

        result = self._create_result(toolkit_registry=toolkit_registry)

        for molecule in molecules:
            try:
                isomers = molecule.enumerate_stereoisomers(
                    undefined_only=self.undefined_only,
                    max_isomers=self.max_isomers,
                    rationalise=self.rationalise,
                    toolkit_registry=toolkit_registry,
                )

                # now check that each molecule is well defined
                for isomer in isomers:
                    if not check_missing_stereo(isomer):
                        result.add_molecule(isomer)

                # now check the input
                # rationalise if needed
                if self.rationalise:
                    molecule.generate_conformers(n_conformers=1)
                if not check_missing_stereo(molecule):
                    result.add_molecule(molecule)

            except Exception:
                result.filter_molecule(molecule)

        return result


class EnumerateProtomers(ToolkitValidator, CustomWorkflowComponent):
    """
    Enumerate the formal charges of the input molecule using the backend toolkits through the OFFTK.

    Important:
        Only Openeye is supported so far.
    """

    type: Literal["EnumerateProtomers"] = "EnumerateProtomers"
    max_states: int = Field(
        10, description="The maximum number of states that should be generated."
    )

    @classmethod
    def is_available(cls) -> bool:
        tk = super().is_available()
        oe = which_import(
            ".oechem",
            return_bool=True,
            raise_error=True,
            package="openeye",
            raise_msg="Please install via `conda install openeye-toolkits -c openeye`",
        )
        return tk and oe

    @classmethod
    def description(cls) -> str:
        return "Enumerate the protomers of the molecule, returning the input molecule and any new molecules."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecules formal charges could not be enumerated possibly due to a missing toolkit."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=True)

    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        Enumerate the formal charges of the molecule.

        Parameters:
            molecules: The list of molecules the component should be applied on.
            toolkit_registry: The openff.toolkit.utils.ToolkitRegistry which declares the avilable toolkits.

        Returns:
            A [ComponentResult][qcsubmit.datasets.ComponentResult] instance containing information about the molecules
            that passed and were filtered by the component and details about the component which generated the result.

        Important:
            This is only possible using Openeye so far, if openeye is not available this step will fail.
        """

        result = self._create_result(toolkit_registry=toolkit_registry)

        for molecule in molecules:
            try:
                protomers = molecule.enumerate_protomers(max_states=self.max_states)

                for protomer in protomers:
                    result.add_molecule(protomer)
                result.add_molecule(molecule)

            except Exception:
                # if openeye is not available this will fail
                result.filter_molecule(molecule)

        return result


class ScanEnumerator(ToolkitValidator, CustomWorkflowComponent):
    """
    This module will tag any matching substructures for scanning, useful for torsiondrive datasets.
    """

    type: Literal["ScanEnumerator"] = "ScanEnumerator"

    torsion_scans: List[Scan1D] = Field(
        [],
        description="A list of scan objects which describes the scan range and scan increment"
        "that should be used with the associated smarts pattern.",
    )
    double_torsion_scans: List[Scan2D] = Field(
        [],
        description="A list of double scan objects which describes the scan ranges and scan increments,"
        "that should be used with each of the smarts patterns.",
    )
    improper_scans: List[ImproperScan] = Field(
        [],
        description="A list of improper scan objects which describes the scan range and scan increment"
        "that should be used with the smarts pattern.",
    )

    @classmethod
    def description(cls) -> str:
        return "Tag any matched substructures for scanning."

    @classmethod
    def fail_reason(cls) -> str:
        return "The molecule contained no substructure matches."

    @classmethod
    def properties(cls) -> ComponentProperties:
        return ComponentProperties(process_parallel=True, produces_duplicates=False)

    def add_torsion_scan(
        self,
        smarts: str,
        scan_rage: Optional[Tuple[int, int]] = None,
        scan_increment: int = 15,
    ) -> None:
        """
        Add a targeted 1D torsion scan to the scan enumerator.

        Args:
            smarts:
                The numerically tagged SMARTs pattern that should be used to identify the torsion atoms.
            scan_rage:
                The angle in degrees the torsion should be scanned between, from low to high
            scan_increment:
                The value in degrees between each grid point in the scan.
        """
        self.torsion_scans.append(
            Scan1D(
                smarts1=smarts, scan_range1=scan_rage, scan_increment=[scan_increment]
            )
        )

    def add_double_torsion(
        self,
        smarts1: str,
        smarts2: str,
        scan_range1: Optional[Tuple[int, int]] = None,
        scan_range2: Optional[Tuple[int, int]] = None,
        scan_increments: List[int] = (15, 15),
    ) -> None:
        """
        Add a targeted 2D torsion scan to the scan enumerator.

        Args:
            smarts1:
                The numerically tagged SMARTs pattern that should be used to identify the first torsion atoms.
            smarts2:
                The numerically tagged SMARTs pattern that should be used to identify the second torsion atoms.
            scan_range1:
                The angle in degrees the first torsion should be scanned between, from low to high
            scan_range2:
                The angle in degrees the second torsion should be scanned between, from low to high
            scan_increments:
                A list of the values in degrees between each grid point in the scans.
        """
        self.double_torsion_scans.append(
            Scan2D(
                smarts1=smarts1,
                smarts2=smarts2,
                scan_range1=scan_range1,
                scan_range2=scan_range2,
                scan_increment=scan_increments,
            )
        )

    def add_improper_torsion(
        self,
        smarts: str,
        central_smarts: str,
        scan_range: Optional[Tuple[int, int]] = None,
        scan_increment: int = 15,
    ) -> None:
        """
        Add a targeted Improper torsion to the scan enumerator.

        Args:
            smarts:
                The numerically tagged SMARTs pattern which describes the entire improper.
            central_smarts:
                The numerically tagged SMARTs pattern which identifies the central atom in the improper.
            scan_range:
                The angles in degrees the improper should be scanned between, from low to high.
            scan_increment:
                The value in degrees between each grid point in the scan.
        """
        self.improper_scans.append(
            ImproperScan(
                smarts=smarts,
                central_smarts=central_smarts,
                scan_range=scan_range,
                scan_increment=[scan_increment],
            )
        )

    def _get_unique_torsions(
        self, matches: List[Tuple[int, int, int, int]], symmetry_classes: List[int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Use the symmetry classes to condense the matches into unique torsions in the molecule by symmetry.
        """
        torsions_by_symmetry = {
            tuple(sorted(symmetry_classes[idx] for idx in match[1:3])): match
            for match in matches
        }
        unique_torsions = [*torsions_by_symmetry.values()]
        return unique_torsions

    def _apply(
        self, molecules: List[Molecule], toolkit_registry: ToolkitRegistry
    ) -> ComponentResult:
        """
        Tag any dihedrals which match the defined set of targets in the enumerator.
        """

        result = self._create_result(toolkit_registry=toolkit_registry)

        for molecule in molecules:
            symmetry_classes = get_symmetry_classes(molecule)
            molecule.properties["dihedrals"] = TorsionIndexer()
            self._tag_torsions(
                molecule, symmetry_classes, toolkit_registry=toolkit_registry
            )
            self._tag_double_torsions(
                molecule, symmetry_classes, toolkit_registry=toolkit_registry
            )
            self._tag_improper_torsions(
                molecule, symmetry_classes, toolkit_registry=toolkit_registry
            )

            indexer = molecule.properties["dihedrals"]
            if len(indexer.get_dihedrals) == 0:
                result.filter_molecule(molecule)

            result.add_molecule(molecule)

        return result

    def _tag_torsions(
        self,
        molecule: Molecule,
        symmetry_classes: List[int],
        toolkit_registry: ToolkitRegistry,
    ) -> None:
        """
        For each of the torsions in the torsion list try and tag only one in the molecule.
        """

        indexer: TorsionIndexer = molecule.properties["dihedrals"]

        for torsion in self.torsion_scans:
            matches = molecule.chemical_environment_matches(
                query=torsion.smarts1, toolkit_registry=toolkit_registry
            )
            unique_torsions = self._get_unique_torsions(
                matches=matches, symmetry_classes=symmetry_classes
            )
            for tagged_torsion in unique_torsions:
                indexer.add_torsion(
                    torsion=tagged_torsion,
                    scan_range=torsion.scan_range1,
                    scan_increment=torsion.scan_increment,
                    symmetry_group=get_symmetry_group(
                        atom_group=tagged_torsion[1:3],
                        symmetry_classes=symmetry_classes,
                    ),
                )

    def _tag_double_torsions(
        self,
        molecule: Molecule,
        symmetry_classes: List[int],
        toolkit_registry: ToolkitRegistry,
    ) -> None:
        """
        For each double torsion in the list try and tag the combination in the molecule.
        """

        indexer: TorsionIndexer = molecule.properties["dihedrals"]

        for double_torsion in self.double_torsion_scans:
            matches1 = molecule.chemical_environment_matches(
                query=double_torsion.smarts1, toolkit_registry=toolkit_registry
            )
            matches2 = molecule.chemical_environment_matches(
                query=double_torsion.smarts2, toolkit_registry=toolkit_registry
            )
            unique_torsions1 = self._get_unique_torsions(
                matches=matches1, symmetry_classes=symmetry_classes
            )
            unique_torsions2 = self._get_unique_torsions(
                matches=matches2, symmetry_classes=symmetry_classes
            )
            for tagged_torsion1 in unique_torsions1:
                symmetry_group1 = get_symmetry_group(
                    atom_group=tagged_torsion1[1:3], symmetry_classes=symmetry_classes
                )
                for tagged_torsion2 in unique_torsions2:
                    symmetry_group2 = get_symmetry_group(
                        atom_group=tagged_torsion2[1:3],
                        symmetry_classes=symmetry_classes,
                    )
                    indexer.add_double_torsion(
                        torsion1=tagged_torsion1,
                        torsion2=tagged_torsion2,
                        symmetry_group1=symmetry_group1,
                        symmetry_group2=symmetry_group2,
                        scan_range1=double_torsion.scan_range1,
                        scan_range2=double_torsion.scan_range2,
                        scan_increment=double_torsion.scan_increment,
                    )

    def _tag_improper_torsions(
        self,
        molecule: Molecule,
        symmetry_classes: List[int],
        toolkit_registry: ToolkitRegistry,
    ) -> None:
        """
        For each improper torsion in the list try and tag the combination in the molecule.
        """

        indexer: TorsionIndexer = molecule.properties["dihedrals"]

        for improper in self.improper_scans:
            matches = molecule.chemical_environment_matches(
                query=improper.smarts, toolkit_registry=toolkit_registry
            )
            unique_torsions = self._get_unique_torsions(
                matches=matches, symmetry_classes=symmetry_classes
            )
            central_atoms = molecule.chemical_environment_matches(
                improper.central_smarts
            )
            for tagged_torsion in unique_torsions:
                symmetry_group = get_symmetry_group(
                    atom_group=tagged_torsion, symmetry_classes=symmetry_classes
                )
                for atom in central_atoms:
                    if atom[0] in tagged_torsion:
                        indexer.add_improper(
                            central_atom=atom[0],
                            improper=tagged_torsion,
                            symmetry_group=symmetry_group,
                            scan_range=improper.scan_range,
                            scan_increment=improper.scan_increment,
                        )
                        break
