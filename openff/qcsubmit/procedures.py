"""
The procedure settings controllers
"""

from typing import Any, Dict, Union

from qcportal.optimization import OptimizationSpecification
from typing_extensions import Literal

from openff.qcsubmit._pydantic import BaseModel, Field, validator
from openff.qcsubmit.validators import literal_lower, literal_upper, check_geometric_convergence


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

        One can also request custom convergence criteria using a dictionary with the following format:

           {'energy': 1e-6,
            'grms': 3e-4,
            'gmax': 4.5e-4,
            'drms': 1.2e-3,
            'dmax': 1.8e-3 }

            Note that the units are Hartrees for energy and Bohr for distances. 

            It is also possible to request that the optimization exit 
            gracefully after hitting the maximum number of iterations by including 
            an additional `maxiter` key to the `convergence_set` dictionary as follows:
            
            {'energy': 1e-6,
             'grms': 3e-4,
             'gmax': 4.5e-4,
             'drms': 1.2e-3,
             'dmax': 1.8e-3
             'maxiter': '' }

            Note that the entry corresponding to the `maxiter` key will be ignored. 
            To specify the maximum number of iterations, please use the separate `maxiter` _keyword_.
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
    convergence_set: Union[Literal[
        "GAU",
        "NWCHEM_LOOSE",
        "GAU_LOOSE",
        "TURBOMOLE",
        "INTERFRAG_TIGHT",
        "GAU_TIGHT",
        "GAU_VERYTIGHT",
        "TEST"
    ],Dict] = Field(
        "GAU",
        description="The set of convergence criteria to be used for the optimisation.",
    )
    constraints: Dict = Field(
        {},
        description="The list of constraints orginsed by set and freeze that should be used in the optimization",
    )

    class Config:
        validate_assignment = True
        title = "GeometricProcedure"

    _convergence_set_check = validator("convergence_set", pre=True, allow_reuse=True)(
        check_geometric_convergence
    )
    _coordsys_check = validator("coordsys", pre=True, allow_reuse=True)(literal_lower)

    def get_optimzation_keywords(self) -> Dict[str, Any]:
        """
        Create the optimization specification to be used in qcarchive.

        Returns:
            A dictionary representation of the optimization specification.
        """
        exclude = {"program"}
        if self.constraints is not None:
            exclude.add("constraints")

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
