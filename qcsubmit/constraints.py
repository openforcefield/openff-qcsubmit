"""
Constraint base classes and methods.
"""
from typing import List, Tuple, Union

from pydantic import ValidationError, constr, validator

from .common_structures import ResultsConfig
from .exceptions import ConstraintError


class DistanceConstraint(ResultsConfig):
    type: constr(regex="distance") = "distance"
    indices: Tuple[int, int]

    @validator("indices")
    def _order_and_check_indices(cls, indices: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Check all indices are unique and then order them to make comparisons between constraints easier.
        """
        if len(indices) != len(set(indices)):
            raise ConstraintError(f"The constraint indices {indices} are not unique.")

        if 1 < len(indices) <= 4:
            if indices[0] < indices[-1]:
                return indices
            else:
                return tuple(reversed(indices))
        elif len(indices) > 4:
            return tuple(sorted(indices))
        else:
            return indices


class DistanceConstraintSet(DistanceConstraint):
    value: float


class AngleConstraint(DistanceConstraint):
    type: constr(regex="angle") = "angle"
    indices: Tuple[int, int, int]


class AngleConstraintSet(AngleConstraint):
    value: float


class DihedralConstraint(DistanceConstraint):
    type: constr(regex="dihedral") = "dihedral"
    indices: Tuple[int, int, int, int]


class DihedralConstraintSet(DihedralConstraint):
    value: float


class PositionConstraint(DistanceConstraint):
    type: constr(regex="xyz") = "xyz"
    indices: Tuple[int, ...]


class PositionConstraintSet(PositionConstraint):
    indices: Tuple[int]
    value: Union[str, Tuple[float, float, float]]

    @validator("value")
    def _format_position(cls, value: Union[str, List[float]]) -> str:
        """
        The position must be a space separated string so we do conversion here.
        """
        from .utils import clean_strings

        split_value = None
        if isinstance(value, str):
            # split the string and check the length
            if len(value.split()) == 3:
                split_value = clean_strings(value.split())
            elif len(value.split(",")) == 3:
                split_value = clean_strings(value.split(","))

        elif isinstance(value, tuple):
            if len(value) == 3:
                split_value = value
        if split_value is None:
            raise ConstraintError(
                f"Position constraints require a valid 3 number position as a string or list/tuple."
            )

        # now make sure each value is a valid float and convert to the correct string
        try:
            str_value = " ".join(str(float(x)) for x in split_value)
            return str_value

        except ValueError as e:
            raise ConstraintError(
                f"Position constraints require a valid 3 float position"
            ) from e


class Constraints(ResultsConfig):
    """
    A constraints holder which validates the constraints type and data structure however the indices are not checked for connection as this is not required.
    """

    freeze: List[
        Union[
            DihedralConstraint, AngleConstraint, DistanceConstraint, PositionConstraint
        ]
    ] = []
    set: List[
        Union[
            DihedralConstraintSet,
            AngleConstraintSet,
            DistanceConstraintSet,
            PositionConstraintSet,
        ]
    ] = []
    _constraint_types_freeze = {
        "distance": DistanceConstraint,
        "angle": AngleConstraint,
        "dihedral": DihedralConstraint,
        "xyz": PositionConstraint,
    }
    _constraint_types_set = {
        "distance": DistanceConstraintSet,
        "angle": AngleConstraintSet,
        "dihedral": DihedralConstraintSet,
        "xyz": PositionConstraintSet,
    }

    def add_freeze_constraint(self, constraint_type: str, indices: List[int]) -> None:
        """
        Add a new freeze constraint to the constraint holder after validating it and making sure it is not already present.
        """
        try:
            constraint = self._constraint_types_freeze[constraint_type.lower()](
                indices=indices
            )
            if constraint not in self.freeze:
                self.freeze.append(constraint)
        except KeyError:
            raise ConstraintError(
                f"The constraint type {constraint_type} is not supported please chose from {self._constraint_types_freeze.keys()}"
            )
        except ValidationError as e:
            raise ConstraintError(
                f"A valid constraint could not be built due to the above validation error."
            ) from e

    def add_set_constraint(
        self,
        constraint_type: str,
        indices: List[int],
        value: Union[float, List[float], str],
    ) -> None:
        """
        Add a new set constraint to the constraint holder after validating it and making sure it is not already present.
        """
        try:
            constraint = self._constraint_types_set[constraint_type.lower()](
                indices=indices, value=value
            )
            if constraint not in self.set:
                self.set.append(constraint)
        except KeyError:
            raise ConstraintError(
                f"The constraint type {constraint_type} is not supported please chose from {self._constraint_types_set.keys()}"
            )
        except ValidationError as e:
            raise ConstraintError(
                f"A valid constraint could not be built due to the above validation error."
            ) from e

    @property
    def has_constraints(self) -> bool:
        """
        Quickly check if the constraint holder has any valid constraints.
        """
        if self.freeze or self.set:
            return True
        else:
            return False

    def dict(self, *args, **kwargs):
        """
        Overwrite the default to only include constraints which are present.
        """
        drop_constraints = set()
        for constraint in ["freeze", "set"]:
            constraints = getattr(self, constraint)
            if not constraints:
                drop_constraints.add(constraint)
        return super().dict(exclude=drop_constraints)
