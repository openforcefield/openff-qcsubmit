"""Centralized shim for Pydantic v1/v2 compatible import."""

from pydantic.v1 import (
    BaseModel,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    constr,
    root_validator,
    validator,
)
