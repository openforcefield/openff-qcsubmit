"""
Constraint base classes and methods.
"""

from typing import Literal

from openff.qcsubmit._pydantic import Field, ValidationError, validator
from openff.qcsubmit.common_structures import ResultsConfig
from openff.qcsubmit.exceptions import ConstraintError

ConstraintType = Literal["distance", "angle", "dihedral", "xyz"]


class Constraint(ResultsConfig):
    type: Literal["basic_constraint"] = "basic_constraint"

    indices: tuple[int, ...] = Field(..., description="The indices of the atoms which are to be constrained.")

    @validator("indices")
    def _order_and_check_indices(cls, indices: tuple[int, ...]) -> tuple[int, ...]:
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

    def dict(self, *args, **kwargs):
        """
        Overwrite the dict method to make sure the bonded flag is removed and not passed to qcsubmit.
        """
        exclude = kwargs.get("exclude", set()) or set()
        exclude.add("bonded")
        kwargs["exclude"] = exclude
        return super().dict(*args, **kwargs)


class DistanceConstraint(Constraint):
    type: Literal["distance"] = "distance"
    indices: tuple[int, int]
    bonded: bool = Field(
        True,
        description=(
            "If this is a bonded constraint, this will trigger a validation step to ensure all of the atoms are "
            "bonded.",
        ),
    )


class DistanceConstraintSet(DistanceConstraint):
    value: float


class AngleConstraint(DistanceConstraint):
    type: Literal["angle"] = "angle"
    indices: tuple[int, int, int]


class AngleConstraintSet(AngleConstraint):
    value: float


class DihedralConstraint(DistanceConstraint):
    type: Literal["dihedral"] = "dihedral"
    indices: tuple[int, int, int, int]


class DihedralConstraintSet(DihedralConstraint):
    value: float


class PositionConstraint(Constraint):
    type: Literal["xyz"] = "xyz"
    indices: tuple[int, ...]


class PositionConstraintSet(PositionConstraint):
    indices: tuple[int]
    value: str | tuple[float, float, float] = Field(
        ...,
        description="The value the constraint should be set to, a value or possition.",
    )

    @validator("value")
    def _format_position(cls, value: str | list[float]) -> str:
        """
        The position must be a space separated string so we do conversion here.
        """
        from openff.qcsubmit.utils import clean_strings

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
            raise ConstraintError("Position constraints require a valid 3 number position as a string or list/tuple.")

        # now make sure each value is a valid float and convert to the correct string
        try:
            str_value = " ".join(str(float(x)) for x in split_value)
            return str_value

        except ValueError as e:
            raise ConstraintError("Position constraints require a valid 3 float position") from e


class Constraints(ResultsConfig):
    """
    A constraints holder which validates the constraints type and data structure however the indices are not checked
    for connection as this is not required.
    """

    freeze: list[DihedralConstraint | AngleConstraint | DistanceConstraint | PositionConstraint] = Field(
        [],
        description="The list of freeze type constraints.",
    )
    set: list[(DihedralConstraintSet | AngleConstraintSet | DistanceConstraintSet | PositionConstraintSet)] = Field(
        [],
        description="The list of set type constraints.",
    )
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

    def add_freeze_constraint(self, constraint_type: ConstraintType, indices: list[int], bonded: bool = True) -> None:
        """
        Add a new freeze constraint to the constraint holder after validating it and making sure it is not already
        present.

        Parameters:
            constraint_type: The type of frozen constraint to be generated
            indices: The indices of the atoms which will be constrained
            bonded: If the atoms in the constraint are bonded, this will trigger a connection check when added to a
                dataset.
        """
        kwargs = {"bonded": bonded, "indices": indices}
        try:
            constraint = self._constraint_types_freeze[constraint_type.lower()](**kwargs)
            if constraint not in self.freeze:
                self.freeze.append(constraint)
        except KeyError:
            raise ConstraintError(
                f"The constraint type {constraint_type} is not supported please chose from "
                f"{self._constraint_types_freeze.keys()}",
            )
        except ValidationError as e:
            raise ConstraintError("A valid constraint could not be built due to the above validation error.") from e

    def add_set_constraint(
        self,
        constraint_type: ConstraintType,
        indices: list[int],
        value: float | list[float] | str,
        bonded: bool = True,
    ) -> None:
        """
        Add a new set constraint to the constraint holder after validating it and making sure it is not already
        present.

        Parameters:
            constraint_type: The type of constraint to be generated
            indices: The indices of the atoms which will be constrained
            value: The value the constraint should be set to
            bonded: If the atoms in the constraint are bonded, this will trigger a connection check when added to a
                dataset.
        """
        kwargs = {"bonded": bonded, "indices": indices, "value": value}
        try:
            constraint = self._constraint_types_set[constraint_type.lower()](**kwargs)
            if constraint not in self.set:
                self.set.append(constraint)
        except KeyError:
            raise ConstraintError(
                f"The constraint type {constraint_type} is not supported please chose from "
                f"{self._constraint_types_set.keys()}",
            )
        except ValidationError as e:
            raise ConstraintError("A valid constraint could not be built due to the above validation error.") from e

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

    def __eq__(self, other: "Constraints") -> bool:
        """
        Check that all constraints are the same before returning.
        """
        for set_con in self.set:
            if set_con not in other.set:
                return False
        for freeze_con in self.freeze:
            if freeze_con not in other.freeze:
                return False
        return True
