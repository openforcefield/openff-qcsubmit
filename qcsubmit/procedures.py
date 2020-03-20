"""
The procedure settings controllers
"""

from pydantic import BaseModel, validator


class GeometricProcedure(BaseModel):
    """
    This is a settings class controlling the varius runtime options that can be used when running geometric.

    Attributes
    ----------
    program: str
        The name of the procedure.
    coordsys: str
        The name of the coordinate system which should be used during the optimisation.
    enforce: str
        The threshold (in a.u /rad) to activate precise constraint satisfaction.
    epsilon: float
        Small eigenvalue threshold.
    reset: bool
        Reset the Hessian when the eigenvalues are under epsilon.
    qccnv: bool
         Q-Chem style convergence criteria (i.e. gradient and either energy or displacement).
    molcnv: bool
        Molpro style convergence criteria (i.e. gradient and either energy or displacement, with different defaults).
    check: int
        The interval for checking the coordinate system for changes.
    trust: float
        Starting value of the trust radius.
    tmax: float
        Maximum value of trust radius.
    maxiter: int
        Maximum number of optimization cycles.
    """

    program: str = 'geometric'
    coordsys: str = 'tric'
    enforce: float = 0.0
    epsilon: float = 1e-5
    reset: bool = False
    qccnv: bool = False
    molcnv: bool = False
    check: int = 0
    trust: float = 0.1

    class Config:
        validate_assignment = True
        title = 'GeometricProcedure'

    





