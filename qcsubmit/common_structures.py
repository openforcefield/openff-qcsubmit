"""
This file contains common starting structures which can be mixed into datasets, results and factories.
"""
import getpass
import re
from datetime import date, datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import qcportal as ptl
from pydantic import BaseModel, Field, HttpUrl, PositiveInt, constr, validator
from qcelemental import constants
from qcelemental.models.results import WavefunctionProtocolEnum
from qcfractal.interface import FractalClient
from qcportal.models.common_models import DriverEnum

from .exceptions import DatasetInputError, PCMSettingError, QCSpecificationError


class DatasetConfig(BaseModel):
    """
    The basic configurations for all datasets.
    """

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = True
        validate_assignment: bool = True
        json_encoders: Dict[str, Any] = {
            np.ndarray: lambda v: v.flatten().tolist(),
            Enum: lambda v: v.value,
        }
        extra: True


class ResultsConfig(BaseModel):
    """
    A basic config class for results structures.
    """

    class Config:
        arbitrary_types_allowed: bool = True
        allow_mutation: bool = False
        json_encoders: Dict[str, Any] = {
            np.ndarray: lambda v: v.flatten().tolist(),
            Enum: lambda v: v.value,
        }


class ComponentProperties(BaseModel):
    """
    The workflow properties class which controls if the component can be used in multiprocessing or if the component
    produces duplicates.
    """

    process_parallel: bool = True
    produces_duplicates: bool = True

    class Config:
        allow_mutation: bool = False
        extra: "forbid"


class TDSettings(DatasetConfig):
    """
    A replacement of the TDKeywords class in the QCFractal which drops the dihedrals field as this is moved up the model.
    The settings here overwrite the global dataset and allow the user to have control over the individual scans.
    """

    grid_spacing: Optional[List[int]] = Field(
        None, description="List of grid spacings for the dihedral scan in degrees."
    )
    dihedral_ranges: Optional[List[Tuple[int, int]]] = Field(
        None,
        description="A list of the dihedral scan limits of the form (lower, upper)",
    )
    energy_decrease_thresh: Optional[float] = Field(
        None,
        description="The threshold of the smallest energy decrease amount to trigger activating optimizations from "
        "grid point.",
    )
    energy_upper_limit: Optional[float] = Field(
        None,
        description="The threshold if the energy of a grid point that is higher than the current global minimum, to "
        "start new optimizations, in unit of a.u. I.e. if energy_upper_limit = 0.05, current global "
        "minimum energy is -9.9 , then a new task starting with energy -9.8 will be skipped.",
    )


class PCMSettings(ResultsConfig):
    """
    A class to handle PCM settings which can be used with PSi4.
    """

    units: str = Field(
        ...,
        description="The units used in the input options atomic units are used by default.",
    )
    codata: int = Field(
        2010,
        description="The set of fundamental physical constants to be used in the module.",
    )
    cavity_Type: str = Field(
        "GePol",
        description="Completely specifies type of molecular surface and its discretization.",
    )
    cavity_Area: float = Field(
        0.3,
        description="Average area (weight) of the surface partition for the GePol cavity in the specified units. By default this is in AU.",
    )
    cavity_Scaling: bool = Field(
        True,
        description="If true, the radii for the spheres will be scaled by 1.2. For finer control on the scaling factor for each sphere, select explicit creation mode.",
    )
    cavity_RadiiSet: str = Field(
        "Bondi",
        description="Select set of atomic radii to be used. Currently Bondi-Mantina Bondi, UFF  and Allinger’s MM3 sets available. Radii in Allinger’s MM3 set are obtained by dividing the value in the original paper by 1.2, as done in the ADF COSMO implementation We advise to turn off scaling of the radii by 1.2 when using this set.",
    )
    cavity_MinRadius: float = Field(
        100,
        description="Minimal radius for additional spheres not centered on atoms. An arbitrarily big value is equivalent to switching off the use of added spheres, which is the default in AU.",
    )
    cavity_Mode: str = Field(
        "Implicit",
        description="How to create the list of spheres for the generation of the molecular surface.",
    )
    medium_SolverType: str = Field(
        "IEFPCM",
        description="Type of solver to be used. All solvers are based on the Integral Equation Formulation of the Polarizable Continuum Model.",
    )
    medium_Nonequilibrium: bool = Field(
        False,
        description="Initializes an additional solver using the dynamic permittivity. To be used in response calculations.",
    )
    medium_Solvent: str = Field(
        ...,
        description="Specification of the dielectric medium outside the cavity. Note this will always be converted to the molecular formula to aid parsing via PCM.",
    )
    medium_MatrixSymm: bool = Field(
        True,
        description="If True, the PCM matrix obtained by the IEFPCM collocation solver is symmetrized.",
    )
    medium_Correction: float = Field(
        0.0,
        description="Correction, k for the apparent surface charge scaling factor in the CPCM solver.",
        ge=0,
    )
    medium_DiagonalScaling: float = Field(
        1.07,
        description="Scaling factor for diagonal of collocation matrices, values commonly used in the literature are 1.07 and 1.0694.",
        ge=0,
    )
    medium_ProbeRadius: float = Field(
        1.0,
        description="Radius of the spherical probe approximating a solvent molecule. Used for generating the solvent-excluded surface (SES) or an approximation of it. Overridden by the built-in value for the chosen solvent. Default in AU.",
    )
    _solvents: ClassVar[Dict[str, str]] = {
        "water": "H2O",
        "dimethylsulfoxide": "DMSO",
        "nitromethane": "CH3NO2",
        "acetonitrile": "CH3CN",
        "methanol": "CH3OH",
        "ethanol": "CH3CH2OH",
        "1,2-dichloroethane": "C2H4CL2",
        "methylenechloride": "CH2CL2",
        "tetrahydrofurane": "THF",
        "aniline": "C6H5NH2",
        "chlorobenzene": "C6H5CL",
        "chloroform": "CHCL3",
        "toluene": "C6H5CH3",
        "1,4-dioxane": "C4H8O2",
        "carbon tetrachloride": "CCL4",
        "cyclohexane": "C6H12",
        "n-heptane": "C7H16",
    }

    @validator("units")
    def _check_units(cls, unit: str) -> str:
        """
        Make sure the units are a valid option.
        """
        units = ["au", "angstrom"]
        if unit.lower() not in units:
            raise PCMSettingError(f"{unit} is not valid only {units} are supported.")
        return unit

    @validator("codata")
    def _check_codata(cls, codata: int) -> int:
        """
        Make sure the codata is a valid option in PCM.
        """
        datasets = [2010, 2006, 2002, 1998]
        if codata not in datasets:
            raise PCMSettingError(
                f"{codata} is not valid only {datasets} are supported."
            )
        return codata

    @validator("cavity_Type")
    def _check_cavity_type(cls, cavity: str) -> str:
        """
        Make sure the cavity type is GePol as this is the only kind supported.
        """
        if cavity.lower() != "gepol":
            raise PCMSettingError(
                f"{cavity} is not a supported type only GePol is available."
            )
        return "GePol"

    @validator("cavity_RadiiSet")
    def _check_radii_set(cls, radiiset: str) -> str:
        """
        Make sure a valid radii set is passed.
        """
        radiisets = ["bondi", "uff", "allinger"]
        if radiiset.lower() not in radiisets:
            raise PCMSettingError(
                f"{radiiset} is not a supported set please chose from {radiisets}"
            )
        return radiiset

    @validator("cavity_Mode")
    def _check_cavity_mode(cls, cavity: str) -> str:
        """
        Make sure that a valid cavity mode is passed.
        """
        if cavity.lower() != "implicit":
            raise PCMSettingError(
                f"{cavity} is not supported via QCSubmit only implicit can be used for collection based calculations."
            )
        return "Implicit"

    @validator("medium_SolverType")
    def _check_solver(cls, solver: str) -> str:
        """
        Make sure valid solver is passed.
        """
        solvers = ["IEFPCM", "CPCM"]
        if solver.upper() not in solvers:
            raise PCMSettingError(f"{solver} not supported please chose from {solvers}")
        return solver.upper()

    @validator("medium_Solvent")
    def _check_solvent(cls, solvent: str) -> str:
        """
        Make sure that a valid solvent from the list of supported values is passed.
        """

        solvent_formula = cls._solvents.get(solvent.lower(), solvent.upper())
        if solvent_formula not in cls._solvents.values():
            raise PCMSettingError(
                f"The solvent {solvent} is not supported please chose from the following solvents or formulas {cls._solvents.items()}"
            )
        return solvent_formula

    def __init__(self, **kwargs):
        """
        Fully validate the model making sure options are compatible and convert any defaults to the give unit system.
        """
        # convert all inputs to the correct units
        units = kwargs.get("units", None)
        if units is not None and units.lower() == "angstrom":
            # we need to convert the default values only which have length scales
            if "medium_ProbeRadius" not in kwargs:
                medium_ProbeRadius = (
                    self.__fields__["medium_ProbeRadius"].default
                    * constants.bohr2angstroms
                )
                kwargs["medium_ProbeRadius"] = medium_ProbeRadius
            if "cavity_MinRadius" not in kwargs:
                cavity_MinRadius = (
                    self.__fields__["cavity_MinRadius"].default
                    * constants.bohr2angstroms
                )
                kwargs["cavity_MinRadius"] = cavity_MinRadius
            if "cavity_Area" not in kwargs:
                cavity_Area = (
                    self.__fields__["cavity_Area"].default
                    * constants.bohr2angstroms ** 2
                )
                kwargs["cavity_Area"] = cavity_Area
        super(PCMSettings, self).__init__(**kwargs)

    def to_string(self) -> str:
        """
        Generate the formated PCM settings string which can be ingested by psi4 via the qcschema interface.
        """
        # format the medium keywords
        medium_str, cavity_str = "", ""
        for prop in self.__fields__.keys():
            if "medium" in prop:
                medium_str += f"\n     {prop[7:]} = {getattr(self, prop)}"
            elif "cavity" in prop:
                cavity_str += f"\n     {prop[7:]} = {getattr(self, prop)}"
        # format the cavity keywords
        pcm_string = f"""
        Units = {self.units}
        CODATA = {self.codata}
        Medium {{{medium_str
        }}}
        Cavity {{{cavity_str}}}"""
        return pcm_string


class QCSpec(ResultsConfig):

    method: constr(strip_whitespace=True) = Field(
        "B3LYP-D3BJ",
        description="The name of the computational model used to execute the calculation. This could be the QC method or the forcefield name.",
    )
    basis: Optional[constr(strip_whitespace=True)] = Field(
        "DZVP",
        description="The name of the basis that should be used with the given method, outside of QC this can be the parameterization ie antechamber or None.",
    )
    program: constr(strip_whitespace=True) = Field(
        "psi4",
        description="The name of the program that will be used to perform the calculation.",
    )
    spec_name: constr(strip_whitespace=True) = Field(
        "default",
        description="The name the specification will be stored under in QCArchive.",
    )
    spec_description: str = Field(
        "Standard OpenFF optimization quantum chemistry specification.",
        description="The description of the specification which will be stored in QCArchive.",
    )
    store_wavefunction: WavefunctionProtocolEnum = Field(
        WavefunctionProtocolEnum.none,
        description="The level of wavefunction detail that should be saved in QCArchive. Note that this is done for every calculation and should not be used with optimizations.",
    )
    implicit_solvent: Optional[PCMSettings] = Field(
        None,
        description="If PCM is to be used with psi4 this is the full description of the settings that should be used.",
    )

    def __init__(
        self,
        method: constr(strip_whitespace=True) = "B3LYP-D3BJ",
        basis: Optional[constr(strip_whitespace=True)] = "DZVP",
        program: constr(strip_whitespace=True) = "psi4",
        spec_name: constr(strip_whitespace=True) = "default",
        spec_description: str = "Standard OpenFF optimization quantum chemistry specification.",
        store_wavefunction: WavefunctionProtocolEnum = WavefunctionProtocolEnum.none,
        implicit_solvent: Optional[PCMSettings] = None,
    ):
        """
        Validate the combination of method, basis and program.
        """
        from openforcefield.typing.engines.smirnoff import get_available_force_fields
        from openmmforcefields.generators.template_generators import (
            GAFFTemplateGenerator,
        )

        # set up the valid method basis and program combinations
        ani_methods = {"ani1x", "ani1ccx", "ani2x"}
        openff_forcefields = list(
            ff.split(".offxml")[0] for ff in get_available_force_fields()
        )
        gaff_forcefields = GAFFTemplateGenerator.INSTALLED_FORCEFIELDS
        xtb_methods = {
            "gfn0-xtb",
            "gfn0xtb",
            "gfn1-xtb",
            "gfn1xtb",
            "gfn2-xtb",
            "gfn2xtb",
            "gfn-ff",
            "gfnff",
        }
        rdkit_methods = {"uff", "mmff94", "mmff94s"}
        settings = {
            "openmm": {"antechamber": gaff_forcefields, "smirnoff": openff_forcefields},
            "torchani": {None: ani_methods},
            "xtb": {None: xtb_methods},
            "rdkit": {None: rdkit_methods},
            "psi4": {},
        }

        if program.lower() != "psi4":
            # make sure PCM is not set
            if implicit_solvent is not None:
                raise QCSpecificationError(
                    "PCM can only be used with PSI4 please set implicit solvent to None."
                )
            # we need to make sure it is valid in the above list
            program_settings = settings.get(program.lower(), None)
            if program_settings is None:
                raise QCSpecificationError(
                    f"The program {program.lower()} is not supported please use one of the following {settings.keys()}"
                )
            allowed_methods = program_settings.get(basis, None)
            if allowed_methods is None:
                raise QCSpecificationError(
                    f"The Basis {basis} is not supported for the program {program}, chose from {program_settings.keys()}"
                )
            # now we need to check the methods
            # strip the offxml if present
            method = method.split(".offxml")[0].lower()
            # if we have a smirnoff99Frosst force field we need to get the capitalization correct
            method = re.sub("frosst", "Frosst", method)
            if method not in allowed_methods:
                raise QCSpecificationError(
                    f"The method {method} is not supported for the program {program} with basis {basis}, please chose from {allowed_methods}"
                )

        super().__init__(
            method=method,
            basis=basis,
            program=program.lower(),
            spec_name=spec_name,
            spec_description=spec_description,
            store_wavefunction=store_wavefunction,
            implicit_solvent=implicit_solvent,
        )

    def dict(
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> "DictStrAny":

        data = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
        )
        if "store_wavefunction" in data:
            data["store_wavefunction"] = self.store_wavefunction.value
        return data


class QCSpecificationHandler(BaseModel):
    """
    A mixin class for handling the QCSpecification
    """

    qc_specifications: Dict[str, QCSpec] = Field(
        {"default": QCSpec()},
        description="The QCSpecifications which will be computed for this dataset.",
    )

    def clear_qcspecs(self) -> None:
        """
        Clear out any current QCSpecs.
        """
        self.qc_specifications = {}

    def remove_qcspec(self, spec_name: str) -> None:
        """
        Remove a QCSpec from the dataset.

        Parameters:
            spec_name: The name of the spec that should be removed.

        Note:
            The QCSpec settings are not mutable and so they must be removed and a new one added to ensure they are fully validated.
        """
        if spec_name in self.qc_specifications.keys():
            del self.qc_specifications[spec_name]

    @property
    def n_qc_specs(self) -> int:
        """
        Return the number of QCSpecs on this dataset.
        """
        return len(self.qc_specifications)

    def _check_qc_specs(self) -> None:
        if self.n_qc_specs == 0:
            raise QCSpecificationError(
                f"There are no QCSpecifications for this dataset please add some using `add_qc_spec`"
            )

    def add_qc_spec(
        self,
        method: str,
        basis: Optional[str],
        program: str,
        spec_name: str,
        spec_description: str,
        store_wavefunction: str = "none",
        overwrite: bool = False,
        implicit_solvent: Optional[PCMSettings] = None,
    ) -> None:
        """
        Add a new qcspecification to the factory which will be applied to the dataset.

        Parameters:
            method: The name of the method to use eg B3LYP-D3BJ
            basis: The name of the basis to use can also be `None`
            program: The name of the program to execute the computation
            spec_name: The name the spec should be stored under
            spec_description: The description of the spec
            store_wavefunction: what parts of the wavefunction that should be saved
            overwrite: If there is a spec under this name already overwrite it
            implicit_solvent: The implicit solvent settings if it is to be used.
        """
        spec = QCSpec(
            method=method,
            basis=basis,
            program=program,
            spec_name=spec_name,
            spec_description=spec_description,
            store_wavefunction=store_wavefunction,
            implicit_solvent=implicit_solvent,
        )

        if spec_name not in self.qc_specifications:
            self.qc_specifications[spec.spec_name] = spec
        elif overwrite:
            self.qc_specifications[spec.spec_name] = spec
        else:
            raise QCSpecificationError(
                f"A specification is already stored under {spec_name} to replace it set `overwrite=True`."
            )


def order_torsion(torsion: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Order a torsion string to ensure the central bond is ordered.
    """
    if torsion[1:3] == tuple(sorted(torsion[1:3])):
        return torsion
    else:
        return torsion[::-1]


def order_scan_range(scan_range: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """
    Order the scan range.
    """
    if scan_range is not None:
        return tuple(sorted(scan_range))
    else:
        return None


class SingleTorsion(ResultsConfig):
    """
    A class used to mark torsions that will be driven for torsiondrive datasets.

    Note:
        This is only for 1D torsiondrives.
    """

    torsion1: Tuple[int, int, int, int] = Field(
        ..., description="The torsion which is to be driven."
    )
    scan_range1: Optional[Tuple[int, int]] = Field(
        None, description="The scan range used in the torsion drive"
    )

    _order_torsion1 = validator("torsion1", allow_reuse=True)(order_torsion)
    _order_scan_range1 = validator("scan_range1", allow_reuse=True)(order_scan_range)

    @property
    def central_bond(self) -> Tuple[int, int]:
        """Get the sorted index of the central bond."""

        return tuple(sorted(self.torsion1[1:3]))

    @property
    def get_dihedrals(self) -> List[Tuple[int, int, int, int]]:
        """
        Get the formatted representation of the dihedrals to scan over.
        """
        return [
            self.torsion1,
        ]

    @property
    def get_scan_range(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get the formatted representation of the dihedral scan ranges.
        """
        if self.scan_range1 is not None:
            return [
                self.scan_range1,
            ]
        else:
            return self.scan_range1

    @property
    def get_atom_map(self) -> Dict[int, int]:
        """
        Create an atom map which will tag the correct dihedral atoms.
        """
        return dict((atom, i) for i, atom in enumerate(self.torsion1))


class DoubleTorsion(SingleTorsion):
    """A class used to mark coupled torsions which should be scanned."""

    torsion2: Tuple[int, int, int, int] = Field(
        ...,
        description="The torsion tuple of the second dihedral to be drive at the same time as the first.",
    )
    scan_range2: Optional[Tuple[int, int]] = Field(
        None,
        description="The separate scan range that should be used for the second dihedral.",
    )

    _order_torsion2 = validator("torsion2", allow_reuse=True)(order_torsion)
    _order_scan_range2 = validator("scan_range2", allow_reuse=True)(order_scan_range)

    @property
    def central_bond(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get the 4 integer tuple of the two central bonds."""

        central_bond = tuple(
            sorted(
                [
                    tuple(sorted(self.torsion1[1:3])),
                    tuple(sorted(self.torsion2[1:3])),
                ]
            )
        )
        return central_bond

    @property
    def get_dihedrals(self) -> List[Tuple[int, int, int, int]]:
        """
        Get the formatted representation of the dihedrals to scan over.
        """
        return [
            self.torsion1,
            self.torsion2,
        ]

    @property
    def get_scan_range(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get the formatted representation of the dihedral scan ranges.
        """
        if self.scan_range1 is not None and self.scan_range2 is not None:
            return [
                self.scan_range1,
                self.scan_range2,
            ]
        else:
            return None

    @property
    def get_atom_map(self) -> Dict[int, int]:
        """
        Create an atom map which will tag the correct dihedral atoms.
        """
        atom_map = {}
        i = 0
        for torsion in self.get_dihedrals:
            for atom in torsion:
                atom_map[atom] = i
                i += 1
        return atom_map


class ImproperTorsion(ResultsConfig):
    """
    A class to keep track of improper torsions being scanned.
    """

    central_atom: int = Field(
        ..., description="The index of the central atom of an improper torsion."
    )
    improper: Tuple[int, int, int, int] = Field(
        ..., description="The tuple of the atoms in the improper torsion."
    )
    scan_range: Optional[Tuple[int, int]] = Field(
        None,
        description="The scan range of the improper dihedral which should normally be limited.",
    )

    _order_scan_range = validator("scan_range", allow_reuse=True)(order_scan_range)

    @property
    def get_dihedrals(self) -> List[Tuple[int, int, int, int]]:
        """
        Get the formatted representation of the dihedrals to scan over.
        """
        return [
            self.improper,
        ]

    @property
    def get_scan_range(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get the formatted representation of the dihedral scan ranges.
        """
        if self.scan_range is not None:
            return [
                self.scan_range,
            ]
        else:
            return self.scan_range

    @property
    def get_atom_map(self) -> Dict[int, int]:
        """
        Create an atom map which will tag the correct dihedral atoms.
        """
        return dict((atom, i) for i, atom in enumerate(self.improper))


class TorsionIndexer(DatasetConfig):
    """
    A class to keep track of the torsions highlighted for scanning, with methods for combining and deduplication.
    """

    torsions: Dict[Tuple[int, int], SingleTorsion] = Field(
        {},
        description="A dictionary of the torsions to be scanned grouped by the central bond in the torsion.",
    )
    double_torsions: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]], DoubleTorsion
    ] = Field(
        {},
        description="A dictionary of the 2D torsions to be scanned grouped by the sorted combination of the central bonds.",
    )
    imporpers: Dict[int, ImproperTorsion] = Field(
        {},
        description="A dictionary of the improper torsions to be scanned grouped by the central atom in the torsion.",
    )

    @property
    def get_dihedrals(
        self,
    ) -> List[Union[SingleTorsion, DoubleTorsion, ImproperTorsion]]:
        """
        Return a list of all of the dihedrals tagged making it easy to loop over.
        """
        all_torsions = []
        all_torsions.extend(list(self.torsions.values()))
        all_torsions.extend(list(self.double_torsions.values()))
        all_torsions.extend(list(self.imporpers.values()))
        return all_torsions

    def add_torsion(
        self,
        torsion: Tuple[int, int, int, int],
        scan_range: Optional[Tuple[int, int]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add a single torsion to the torsion indexer if this central bond has not already been tagged.
        """
        torsion = SingleTorsion(torsion1=torsion, scan_range1=scan_range)

        if torsion.central_bond not in self.torsions:
            self.torsions[torsion.central_bond] = torsion
        elif overwrite:
            self.torsions[torsion.central_bond] = torsion

    def add_double_torsion(
        self,
        torsion1: Tuple[int, int, int, int],
        torsion2: Tuple[int, int, int, int],
        scan_range1: Optional[Tuple[int, int]] = None,
        scan_range2: Optional[Tuple[int, int]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add a double torsion to the indexer if this central bond combination has not been tagged.
        """

        double_torsion = DoubleTorsion(
            torsion1=torsion1,
            torsion2=torsion2,
            scan_range1=scan_range1,
            scan_range2=scan_range2,
        )

        if double_torsion.central_bond not in self.double_torsions:
            self.double_torsions[double_torsion.central_bond] = double_torsion
        elif overwrite:
            self.double_torsions[double_torsion.central_bond] = double_torsion

    def add_improper(
        self,
        central_atom: int,
        improper: Tuple[int, int, int, int],
        scan_range: Optional[Tuple[int, int]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add an improper torsion to the indexer if its central atom is not already covered.
        """

        improper_torsion = ImproperTorsion(
            central_atom=central_atom, improper=improper, scan_range=scan_range
        )

        if improper_torsion.central_atom not in self.imporpers:
            self.imporpers[improper_torsion.central_atom] = improper_torsion
        elif overwrite:
            self.imporpers[improper_torsion.central_atom] = improper_torsion

    def update(
        self,
        torsion_indexer: "TorsionIndexer",
        reorder_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Update the current torsion indexer with another.

        Parameters:
            torsion_indexer: The other torsionindxer that should be used to update the current object.
            reorder_mapping: The mapping between the other and current molecule should the order need updating.
        """

        if reorder_mapping is None:
            self.torsions.update(torsion_indexer.torsions)
            self.double_torsions.update(torsion_indexer.double_torsions)
            self.imporpers.update(torsion_indexer.imporpers)

        else:
            # we need to use the reorder_mapping to change the objects before adding them
            for torsion in torsion_indexer.torsions.values():
                new_torsion = self._reorder_torsion(torsion.torsion1, reorder_mapping)
                self.add_torsion(torsion=new_torsion, scan_range=torsion.scan_range1)

            for double_torsion in torsion_indexer.double_torsions.values():
                new_torsion1 = self._reorder_torsion(
                    double_torsion.torsion1, reorder_mapping
                )
                new_torsion2 = self._reorder_torsion(
                    double_torsion.torsion2, reorder_mapping
                )
                self.add_double_torsion(
                    torsion1=new_torsion1,
                    torsion2=new_torsion2,
                    scan_range1=double_torsion.scan_range1,
                    scan_range2=double_torsion.scan_range2,
                )

            for improper in torsion_indexer.imporpers.values():
                new_improper = self._reorder_torsion(improper.improper, reorder_mapping)
                new_central = reorder_mapping[improper.central_atom]
                self.add_improper(
                    new_central, new_improper, scan_range=improper.scan_range
                )

    @staticmethod
    def _reorder_torsion(
        torsion: Tuple[int, int, int, int], mapping: Dict[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Reorder the given torsion based on the mapping.

        Parameters:
            torsion: The other molecules torsion that should be remapped.
            mapping: The mapping between the other molecule and the current.
        """

        return tuple([mapping[index] for index in torsion])

    @property
    def n_torsions(self) -> int:
        """Return the number of torsions highlighted."""

        return len(self.torsions)

    @property
    def n_double_torsions(self) -> int:
        """
        Return the number of double torsions highlighted.
        """

        return len(self.double_torsions)

    @property
    def n_impropers(self) -> int:
        """
        Return the number of imporpers highlighted.
        """

        return len(self.imporpers)


class IndexCleaner:
    """
    This class offers the ability to clean a molecule index that already has a numeric tag useful for datasets and
    results.
    """

    @staticmethod
    def _clean_index(index: str) -> Tuple[str, int]:
        """
        Take an index and clean it by checking if it already has an enumerator in it return the core index and any
        numeric tags if no tag is found the tag is set to 0.

        Parameters:
            index: The index for the entry which should be checked, if no numeric tag can be found return 0.

        Returns:
            A tuple of the core index and the numeric tag it starts from.

        Note:
            This function allows the dataset to add more conformers to a molecule set so long as the index the molecule
            is stored under is a new index not in the database for example if 3 conformers for ethane exist then the
            new index should start from 'CC-3'.
        """
        # tags take the form '-no'
        match = re.search("-[0-9]+$", index)
        if match is not None:
            core = index[: match.span()[0]]
            # drop the -
            tag = int(match.group()[1:])
        else:
            core = index
            tag = 0

        return core, tag


class ClientHandler:
    """
    This mixin class offers the ability to handle activating qcportal Fractal client instances.
    """

    @staticmethod
    def _activate_client(client) -> ptl.FractalClient:
        """
        Make the fractal client and connect to the requested instance.

        Parameters:
            client: The name of the file containing the client information or the client instance.

        Returns:
            A qcportal.FractalClient instance.
        """

        if isinstance(client, ptl.FractalClient):
            return client
        elif isinstance(client, FractalClient):
            return client
        elif client == "public":
            return ptl.FractalClient()
        else:
            return ptl.FractalClient.from_file(client)


class Metadata(DatasetConfig):
    """
    A general metadata class which is required to be filled in before submitting a dataset to the qcarchive.
    """

    submitter: str = Field(
        getpass.getuser(),
        description="The name of the submitter/creator of the dataset, this is automatically generated but can be changed.",
    )
    creation_date: date = Field(
        datetime.today().date(),
        description="The date the dataset was created on, this is automatically generated.",
    )
    collection_type: Optional[str] = Field(
        None,
        description="The type of collection that will be created in QCArchive this is automatically updated when attached to a dataset.",
    )
    dataset_name: Optional[str] = Field(
        None,
        description="The name that will be given to the collection once it is put into QCArchive, this is updated when attached to a dataset.",
    )
    short_description: Optional[constr(min_length=8, regex="[a-zA-Z]")] = Field(
        None, description="A short informative description of the dataset."
    )
    long_description_url: Optional[HttpUrl] = Field(
        None,
        description="The url which links to more information about the submission normally a github repo with scripts showing how the dataset was created.",
    )
    long_description: Optional[constr(min_length=8, regex="[a-zA-Z]")] = Field(
        None,
        description="A long description of the purpose of the dataset and the molecules within.",
    )
    elements: Set[str] = Field(
        set(), description="The unique set of elements present in the dataset"
    )

    def validate_metadata(self, raise_errors: bool = False) -> Optional[List[str]]:
        """
        Before submitting this function should be called to highlight any incomplete fields.
        """

        empty_fields = []
        for field in self.__fields__:
            attr = getattr(self, field)
            if attr is None:
                empty_fields.append(field)

        if empty_fields and raise_errors:
            raise DatasetInputError(
                f"The metadata has the following incomplete fields {empty_fields}"
            )
        else:
            return empty_fields


class MoleculeAttributes(DatasetConfig):
    """
    A class to hold and validate the molecule attributes associated with a QCArchive entry, All attributes are required
    to be entered into a dataset.

    Note:
        The attributes here are not exhaustive but are based on those given by cmiles and can all be obtain through the openforcefield toolkit Molecule class.
    """

    class Config:
        extra = "allow"

    canonical_smiles: str
    canonical_isomeric_smiles: str
    canonical_explicit_hydrogen_smiles: str
    canonical_isomeric_explicit_hydrogen_smiles: str
    canonical_isomeric_explicit_hydrogen_mapped_smiles: str = Field(
        ...,
        description="The fully mapped smiles where every atom should have a numerical tag so that the molecule can be rebuilt to match the order of the coordinates.",
    )
    molecular_formula: str = Field(
        ...,
        description="The hill formula of the molecule as given by the openfftoolkit.",
    )
    standard_inchi: str = Field(
        ...,
        description="The standard inchi given by the inchi program ie not fixed hydrogen layer.",
    )
    inchi_key: str = Field(
        ..., description="The standard inchi key given by the inchi program."
    )


class SCFProperties(str, Enum):
    """
    The type of SCF property that should be extracted from a single point calculation.
    """

    Dipole = "dipole"
    Quadrupole = "quadrupole"
    MullikenCharges = "mulliken_charges"
    LowdinCharges = "lowdin_charges"
    WibergLowdinIndices = "wiberg_lowdin_indices"
    MayerIndices = "mayer_indices"
    MBISCharges = "mbis_charges"

    @classmethod
    def _missing_(cls, value):
        """
        overwrite the missing method to handle properties with incorrect capitalization.
        """
        for member in cls.__members__.values():
            if member._value_ == value.lower():
                return member
        raise DatasetInputError(
            f"{value} is not a valid {cls.__name__} please chose from {cls.__members__.values()}"
        )


class CommonBase(DatasetConfig, IndexCleaner, ClientHandler, QCSpecificationHandler):
    """
    A common base structure which the dataset and factory classes derive from.
    """

    maxiter: PositiveInt = Field(
        200,
        description="The maximum number of SCF iterations in QM calculations this will be ignored by programs where this does not make sense.",
    )
    driver: DriverEnum = Field(
        DriverEnum.energy,
        description="The type of single point calculations which will be computed. Note some services require certain calculations for example optimizations require graident calculations.",
    )
    scf_properties: List[SCFProperties] = Field(
        [
            SCFProperties.Dipole,
            SCFProperties.Quadrupole,
            SCFProperties.WibergLowdinIndices,
            SCFProperties.MayerIndices,
        ],
        description="The SCF properties which should be extracted after every single point calculation.",
    )
    priority: str = Field(
        "normal",
        description="The priority the dataset should be computed at compared to other datasets currently running.",
    )
    dataset_tags: List[str] = Field(
        ["openff"], description="The dataset tags which help identify the dataset."
    )
    compute_tag: str = Field(
        "openff",
        description="The tag the computes tasks will be assigned to, managers wishing to execute these tasks should use this compute tag.",
    )

    def add_scf_property(self, scf_property: SCFProperties) -> None:
        """
        Add an scf_property to the list of scf_properties requested during a calculation.

        Parameters:
            scf_property: The name of the property which should be entered into the list.

        Raises:
            DatasetInputError: If the scf_property is not valid.
        """
        # make sure the property is valid first
        try:
            validated_property = SCFProperties(scf_property)
        except ValueError:
            raise DatasetInputError(
                f"{scf_property} is not a valid SCF property please chose from {SCFProperties.__members__}"
            )
        if validated_property not in self.scf_properties:
            self.scf_properties.append(validated_property)

    def remove_scf_property(self, scf_property: SCFProperties) -> None:
        """
        Remove an scf_property from the validated list.

        Parameters:
            scf_property: The name of the property which should be removed.
        """
        if scf_property.lower() in self.scf_properties:
            self.scf_properties.remove(scf_property.lower())

    def dict(self, *args, **kwargs):
        """
        Overwrite the dict method to handle any enums when saving to yaml/json via a dict call.
        """
        data = super(CommonBase, self).dict(*args, **kwargs)
        # now add the enum values
        if "driver" in data:
            data["driver"] = self.driver.value
        if "scf_properties" in data:
            data["scf_properties"] = [prop.value for prop in self.scf_properties]
        return data
