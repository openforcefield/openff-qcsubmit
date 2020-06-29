"""
Serialization methods.
"""

import abc
import json
import os
from enum import Enum
from typing import Dict, Union

import yaml
from pydantic import BaseModel

from .exceptions import UnsupportedFiletypeError

__all__ = [
    "Serializer",
    "DeSerializer",
    "register_serializer",
    "register_deserializer",
    "serialize",
    "deserialize",
]

serializers = {}
deserializers = {}


class DataType(str, Enum):
    """
    The type of data the de/serializers deal with, which helps with file loading.
    """

    TEXT = ""
    BYTES = "b"


class GeneralSerializer(BaseModel):

    data_type: DataType

    class Config:
        allow_mutation = False
        extra = "forbid"


class Serializer(GeneralSerializer, abc.ABC):
    @abc.abstractmethod
    def serialize(self, serializable: Dict) -> Union[str, bytes]:
        """
        The method should give the string representation of the serialization ready for dumping to file.
        """
        raise NotImplementedError()


class DeSerializer(GeneralSerializer, abc.ABC):
    @abc.abstractmethod
    def deserialize(self, file_object) -> Dict:
        """
        The method should return a dict representation that the pydantic models can be built from.
        """
        raise NotImplementedError()


class JsonSerializer(Serializer):
    data_type = DataType.TEXT

    def serialize(self, serializable) -> str:
        return json.dumps(serializable, indent=2)


class YamlSerializer(Serializer):
    data_type = DataType.TEXT

    def serialize(self, serializable) -> str:
        return yaml.dump(serializable)


class JsonDeSerializer(DeSerializer):
    data_type = DataType.TEXT

    def deserialize(self, file_object) -> Dict:
        return json.load(file_object)


class YamlDeSerializer(DeSerializer):
    data_type = DataType.TEXT

    def deserialize(self, file_object) -> Dict:
        return yaml.full_load(file_object)


def register_serializer(format_name: str, serializer: "Serializer") -> None:
    """
    Register a new serializer method with qcsubmit.
    """
    if format_name.lower() in serializers.keys():
        raise ValueError(f"{format_name} already has a serializer registered.")

    serializers[format_name.lower()] = serializer


def register_deserializer(format_name: str, deserializer: DeSerializer) -> None:
    """
    Register a new deserializer method with qcsubmit.
    """
    if format_name.lower() in deserializers.keys():
        raise ValueError(f"{format_name} already has a deserializer registered.")

    deserializers[format_name] = deserializer


def unregister_serializer(format_name: str) -> None:
    """
    Remove one of the registered serializers with qcsubmit.
    """
    method = serializers.pop(format_name.lower(), None)
    if method is None:
        raise KeyError(f"The serializer {format_name} is not registered")


def unregister_deserializer(format_name: str) -> None:
    """
    Remove one of the registered deserializers with qcsubmit.
    """
    method = deserializers.pop(format_name.lower(), None)
    if method is None:
        raise KeyError(f"The deserializer {format_name} is not registered.")


def get_serializer(format_name: str) -> "Serializer":
    """
    Return the requested serializer class.
    """
    serializer = serializers.get(format_name.lower(), None)
    if serializer is None:
        raise UnsupportedFiletypeError(
            f"The specified serialization format {format_name} is not supported; "
            f"supported formats are {serializers.keys()}"
        )
    return serializer()


def get_deserializer(format_name: str) -> "DeSerializer":
    """
    Return the requested deserializer class.
    """
    deserailizer = deserializers.get(format_name.lower(), None)
    if deserailizer is None:
        raise UnsupportedFiletypeError(
            f"The specified deserialization format {format_name} is not supported; "
            f"supported formats are {deserializers.keys()}"
        )
    return deserailizer()


def get_format_name(file_name: str) -> str:
    """
    Get the format name by splitting on the .

    Parameters:
        file_name: The name of the file from which we should work out the format.
    """
    return file_name.split(".")[-1].lower()


def serialize(serializable: Dict, file_name: str) -> None:
    """
    The main serialization method used to serialize objects.

    Parameters:
        serializable: The object to be serialized
        file_name: The name of the file the object will be serialized to.
    """
    format_name = get_format_name(file_name)
    serializer = get_serializer(format_name)
    with open(file_name, "w" + serializer.data_type.value) as output:
        output.write(serializer.serialize(serializable))


def deserialize(file_name: str) -> Dict:
    """
    The main method used to deserialize the objects from file.

    Parameters:
        file_name: The file from which the object should be extracted.
    """
    format_name = get_format_name(file_name)
    deserializer = get_deserializer(format_name)
    if os.path.exists(file_name):
        with open(file_name, "r" + deserializer.data_type.value) as input_data:
            return deserializer.deserialize(input_data)
    else:
        raise RuntimeError(f"The file {file_name} could not be found.")


# Register the json and yaml de/serializers
register_serializer(format_name="json", serializer=JsonSerializer)
register_serializer(format_name="yaml", serializer=YamlSerializer)
register_serializer(format_name="yml", serializer=YamlSerializer)
register_deserializer(format_name="json", deserializer=JsonDeSerializer)
register_deserializer(format_name="yaml", deserializer=YamlDeSerializer)
register_deserializer(format_name="yml", deserializer=YamlDeSerializer)
