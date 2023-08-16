"""Centralized shim for Pydantic v1/v2 compatible import."""
try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        PrivateAttr,
        ValidationError,
        constr,
        root_validator,
        validator,
    )
except ImportError:
    from pydantic import (
        BaseModel,
        Field,
        PrivateAttr,
        ValidationError,
        constr,
        root_validator,
        validator,
    )
