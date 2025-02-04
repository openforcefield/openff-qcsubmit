"""
The procedure settings controllers
"""

from typing import Any, Dict, List

from qcportal.optimization import OptimizationSpecification
from typing_extensions import Literal

from openff.qcsubmit._pydantic import BaseModel, Field, root_validator, validator
from openff.qcsubmit.validators import (
    check_custom_converge,
    literal_lower,
    literal_upper,
)
from openff.qcsubmit.exceptions import ConflictingConvergeSettingsError


class GeometricProcedure(BaseModel):
    """
    This is a settings class controlling the various runtime options that can be used when running geometric.

    Note:
        The coordinate systems supported by geometric are:

            - `cart` Cartesian
            - `prim` Primitive a.k.a redundant
            - `dlc` Delocalised Internal Coordinates
            - `hdlc` Hybrid Delocalised Internal Coordinates
            - `tric` Translation-Rotation-Internal Coordinates, this is the default default

    Important:
        Geometric currently accepts the following convergence criteria sets:

            +-------------------+---------------------+--------+---------+--------+--------+-------+
            |  Set              | Set Name            | Energy |  GRMS   | GMAX   | DRMS   | DMAX  |
            +-------------------+---------------------+--------+---------+--------+--------+-------+
            | `GAU`             | Gaussian default    | 1e-6   | 3e-4    | 4.5e-4 | 1.2e-3 | 1.8e-3|
            +-------------------+---------------------+--------+---------+--------+--------+-------+
            | `NWCHEM_LOOSE`    | NW-Chem loose       | 1e-6   | 3e-3    | 4.5e-3 | 3.6e-3 | 5.4e-3|
            +-------------------+---------------------+--------+---------+--------+--------+-------+
            | `GAU_LOOSE`       | Gaussian loose      | 1e-6   | 1.7e-3  | 2.5e-3 | 6.7e-3 | 1e-2  |
            +-------------------+---------------------+--------+---------+--------+--------+-------+
            | `TURBOMOLE`       | Turbomole default   | 1e-6   | 5e-4    | 1e-3   | 5.0e-4 | 1e-3  |
            +-------------------+---------------------+--------+---------+--------+--------+-------+
            | `INTERFRAG_TIGHT` | Interfrag tight     | 1e-6   | 1e-5    | 1.5e-5 | 4.0e-4 | 6.0e-4|
            +-------------------+---------------------+--------+---------+--------+--------+-------+
            | `GAU_TIGHT`       | Gaussian tight      | 1e-6   | 1e-5    | 1.5e-5 | 4e-5   | 6e-5  |
            +-------------------+---------------------+--------+---------+--------+--------+-------+
            | `GAU_VERYTIGHT`   | Gaussian very tight | 1e-6   | 1e-6    | 2e-6   | 4e-6   | 6e-6  |
            +-------------------+---------------------+--------+---------+--------+--------+-------+

            The recommended use case is to provide the name of one of these sets with the `convergence_set` keyword.

        Alternatively, you can provide a custom convergence criteria set by providing a list of strings to the `converge` keyword.
        These should be provided in the following format, with the `convergence_set` keyword set to 'CUSTOM':

            ```
            convergence_set = 'CUSTOM',
            converge = ['energy', '1e-6', 'grms', '3e-4', 'gmax', '4.5e-4', 'drms', '1.2e-3', 'dmax', '1.8e-3']
            ```

            Not all the flags are required, please see the GeomeTRIC documentation for more information on custom convergence criteria sets.
            Note that the units are are Hartree for energies and Bohr for distances.

            The `maxiter` flag can also be passed to the `converge` keyword to tell the program to exit gracefully upon reaching the maximum number of iterations.
            This can be used to run a few optimization steps to relax excessively high forces.
            It can be passed with a list of custom criteria:

            ```
            convergence_set = 'CUSTOM',
            converge = ['energy', '1e-6', 'grms', '3e-4', 'gmax', '4.5e-4', 'drms', '1.2e-3', 'dmax', '1.8e-3', 'maxiter']
            ```

            It can also be used in conjunction with one of the `convergence_set` options:

            ```
            convergence_set = 'GAU',
            converge = ['maxiter']
            ```

            The `maxiter` flag to the `converge` keyword should not be followed by anything; to set the maximum number of iterations, please use the separate `maxiter` keyword:

            ```
            convergence_set = 'GAU',
            converge = ['maxiter']
            maxiter = 5
            ```

    """

    program: Literal["geometric"] = Field(
        "geometric", description="The name of the program executing the procedure."
    )
    coordsys: Literal["tric", "prim", "dlc", "hdlc", "cart"] = Field(
        "dlc",
        description="The type of coordinate system which should be used during the optimization. Choices are tric, prim, dlc, hdlc, and cart.",
    )
    enforce: float = Field(
        0.0,
        description="The threshold( in a.u / rad) to activate precise constraint satisfaction.",
    )
    epsilon: float = Field(1e-5, description="Small eigenvalue threshold.")
    reset: bool = Field(
        True, description="Reset the Hessian when the eigenvalues are under epsilon."
    )
    qccnv: bool = Field(
        False,
        description="Activate Q-Chem style convergence criteria(i.e.gradient and either energy or displacement).",
    )
    molcnv: bool = Field(
        False,
        description="Activate Molpro style convergence criteria (i.e.gradient and either energy or displacement, with different defaults).",
    )
    check: int = Field(
        0, description="The interval for checking the coordinate system for changes."
    )
    trust: float = Field(0.1, description="Starting value of the trust radius.")
    tmax: float = Field(0.3, description="Maximum value of trust radius.")
    maxiter: int = Field(300, description="Maximum number of optimization cycles.")
    convergence_set: Literal[
        "GAU",
        "NWCHEM_LOOSE",
        "GAU_LOOSE",
        "TURBOMOLE",
        "INTERFRAG_TIGHT",
        "GAU_TIGHT",
        "GAU_VERYTIGHT",
        "CUSTOM",
    ] = Field(
        "GAU",
        description="The set of convergence criteria to be used for the optimisation.",
    )
    constraints: Dict = Field(
        {},
        description="The list of constraints orginsed by set and freeze that should be used in the optimization",
    )
    converge: List = Field(
        [],
        description="(Optional): The custom-specified convergence criteria to be used for the optimization. If none provided, will fall back to the option provided in convergence_set.",
    )

    class Config:
        validate_assignment = True
        title = "GeometricProcedure"

    _convergence_set_check = validator("convergence_set", pre=True, allow_reuse=True)(
        literal_upper
    )
    _converge_check = validator("converge", pre=True, allow_reuse=True)(
        check_custom_converge
    )
    _coordsys_check = validator("coordsys", pre=True, allow_reuse=True)(literal_lower)

    @root_validator()
    def check_convergence_all(cls, values):
        convergence_set = values.get("convergence_set")
        convergence_keywords = values.get("converge")

        # Make sure that if a custom convergence set is provided via the converge keyword, the convergence_set keyword is set to 'CUSTOM'
        if len(convergence_keywords) > 0 and convergence_set != "CUSTOM":
            # It is okay to provide only maxiter to converge, and use a regular convergence_set
            if (
                len(convergence_keywords) == 1
                and convergence_keywords[0].lower() == "maxiter"
            ):
                pass
            else:
                raise ConflictingConvergeSettingsError(
                    f"Received convergence_set = {convergence_set} and converge = {convergence_keywords}. If a custom convergence criteria set is provided via the converge keyword, the convergence_set keyword must be set to 'CUSTOM'."
                )

        # Make sure that if convergence_set = CUSTOM, the converge keyword is not empty
        elif convergence_set == "CUSTOM" and len(convergence_keywords) < 2:
            raise ConflictingConvergeSettingsError(
                f"Received convergence_set = {convergence_set} and converge = {convergence_keywords}. If convergence_set = 'CUSTOM', the convergence criteria must be specified by converge = ['energy','1e-6',...]."
            )
        return values

    def get_optimzation_keywords(self) -> Dict[str, Any]:
        """
        Create the optimization specification to be used in qcarchive.

        Returns:
            A dictionary representation of the optimization specification.
        """
        exclude = {"program"}
        if self.constraints is not None:
            exclude.add("constraints")
        if self.convergence_set == "CUSTOM":
            exclude.add("convergence_set")
        if len(self.converge) == 0:
            exclude.add("converge")

        return self.dict(exclude=exclude)

    @classmethod
    def from_opt_spec(
        cls, optimization_specification: OptimizationSpecification
    ) -> "GeometricProcedure":
        """
        Create a geometric procedure from an Optimization spec.
        """

        if optimization_specification.keywords is None:
            return GeometricProcedure()

        else:
            data = optimization_specification.dict(exclude={"program"})
            return GeometricProcedure(**data["keywords"])
