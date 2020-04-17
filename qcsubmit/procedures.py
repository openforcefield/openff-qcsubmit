"""
The procedure settings controllers
"""

from pydantic import BaseModel, validator
from typing import Dict


class GeometricProcedure(BaseModel):
    """
    This is a settings class controlling the various runtime options that can be used when running geometric.

    Attributes:
        program: The name of the procedure.
        coordsys: The name of the coordinate system which should be used during the optimisation.
        enforce: The threshold (in a.u /rad) to activate precise constraint satisfaction.
        epsilon: Small eigenvalue threshold.
        reset: Reset the Hessian when the eigenvalues are under epsilon.
        qccnv: Q-Chem style convergence criteria (i.e. gradient and either energy or displacement).
        molcnv: Molpro style convergence criteria (i.e. gradient and either energy or displacement, with different defaults).
        check: The interval for checking the coordinate system for changes.
        trust: Starting value of the trust radius.
        tmax: Maximum value of trust radius.
        maxiter: Maximum number of optimization cycles.
        convergence_set: The set of convergence criteria to be used for the optimisation.
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
    tmax: float = 0.3
    maxiter: int = 300
    convergence_set: str = 'GAU'
    constraints: Dict = {}

    class Config:
        validate_assignment = True
        title = 'GeometricProcedure'

    @validator('coordsys')
    def check_coordsys(cls, coordsys: str):
        """
        Make sure the user is assigning a valid geometric coordinate system.

        Parameters:
            coordsys: The coordinate system to be used during optimisation.

        Raises:
            ValueError: If the coordinate system is not supported by geometric.

        Important:
            The coordinate systems supported by geometric are:

            - `cart` Cartesian
            - `prim` Primitive a.k.a redundant
            - `dlc` Delocalised Internal Coordinates
            - `hdlc` Hybrid Delocalised Internal Coordinates
            - `tric` Translation-Rotation-Internal Coordinates, this is the default default
        """
        allowed_coordsys = ['cart', 'prim', 'dlc', 'hdlc', 'tric']
        if coordsys.lower() in allowed_coordsys:
            return coordsys.lower()
        else:
            raise ValueError(f'{coordsys} is not supported by geometric please pass a valid coordinate system.')

    @validator('convergence_set')
    def check_convergence_set(cls, convergence: str):
        """
        Ensure a valid convergence set of criteria has been passed.

        Parameters:
            convergence: The convergence criteria set, see below for allowed values.

        Raises:
            ValueError: If the convergence set is not supported.

        Important:
            Geometric currently accepts the following convergence criteria sets:

            | Set  | Set Name  | Energy  | GRMS  | GMAX  | DRMS  | DMAX  |
            |---|---|---|---|---|---|---|
            | `GAU` | Gaussian default  | 1e-6  | 3e-4  | 4.5e-4  | 1.2e-3  | 1.8e-3  |
            | `NWCHEM_LOOSE` | NW-Chem loose  | 1e-6  | 3e-3  | 4.5e-3  | 3.6e-3  | 5.4e-3  |
            | `GAU_LOOSE` | Gaussian loose  | 1e-6  | 1.7e-3  | 2.5e-3  | 6.7e-3  | 1e-2  |
            | `TURBOMOLE` | Turbomole default | 1e-6 | 5e-4 | 1e-3 | 5.0e-4 | 1e-3 |
            | `INTERFRAG_TIGHT` | Interfrag tight | 1e-6 | 1e-5 | 1.5e-5 | 4.0e-4 | 6.0e-4 |
            | `GAU_TIGHT` | Gaussian tight | 1e-6 | 1e-5 | 1.5e-5 | 4e-5 | 6e-5 |
            | `GAU_VERYTIGHT` | Gaussian very tight | 1e-6 | 1e-6 | 2e-6 | 4e-6 | 6e-6 |
        """

        allowed_convergence = ['GAU', 'NWCHEM_LOOSE', 'GAU_LOOSE', 'TURBOMOLE',
                               'INTERFRAG_TIGHT', 'GAU_TIGHT', 'GAU_VERYTIGHT']
        if convergence.upper() in allowed_convergence:
            return convergence.upper()
        else:
            raise ValueError(f'The requested convergence set {convergence} is not supported.')

    def get_optimzation_spec(self) -> Dict:
        """
        Create the optimization specification to be used in qcarchive.

        Returns:
            A dictionary representation of the optimization specification.
        """
        exclude = {'program'}
        if self.constraints is not None:
            exclude.add('constraints')

        opt_spec = {'program': self.program,
                    'keywords': self.dict(exclude=exclude)}

        return opt_spec
