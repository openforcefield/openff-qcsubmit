"""
Serialization methods.
"""

import abc
import functools
import json
import os
from enum import Enum
from typing import IO, Dict, Union

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

# We know how to compress/decompress these automatically
compression_algorithms = ["bz2", "xz", "gz"]


class DataType(str, Enum):
    """
    The type of data the de/serializers deal with, which helps with file loading.
    """

    TEXT = "t"
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
        if hasattr(serializable, "json"):
            return serializable.json(indent=2)
        else:
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
    format_name = format_name.lower()

    for compression in compression_algorithms:

        compressed_format = ".".join([format_name, compression])

        if compressed_format in serializers:
            raise ValueError(f"{format_name} already has a serializer registered.")

        serializers[compressed_format] = serializer

    if format_name in serializers:
        raise ValueError(f"{format_name} already has a serializer registered.")
    serializers[format_name] = serializer


def register_deserializer(format_name: str, deserializer: DeSerializer) -> None:
    """
    Register a new deserializer method with qcsubmit.
    """
    format_name = format_name.lower()

    for compression in compression_algorithms:
        compressed_format = ".".join([format_name, compression])

        if compressed_format in deserializers:
            raise ValueError(f"{format_name} already has a deserializer registered.")

        deserializers[compressed_format] = deserializer

    if format_name in deserializers:
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

    def is_compression_extension(ext):
        return ext in compression_algorithms

    if is_compression_extension(file_name.split(".")[-1].lower()):
        return ".".join(file_name.split(".")[-2:]).lower()
    else:
        return file_name.split(".")[-1].lower()


def _compress_using_suffix(file_name: str, mode) -> IO:

    file_type = file_name.split(".")[-1]

    opener = open
    open_kwargs = dict()

    if file_type == "xz":
        import lzma

        opener = lzma.open
        open_kwargs = dict(preset=9)

    elif file_type == "gz":

        import gzip

        opener = gzip.open
        open_kwargs = dict(compresslevel=9)

    elif file_type == "bz2":
        import bz2

        opener = bz2.open
        open_kwargs = dict(compresslevel=9)

    elif file_type == "zip":
        raise UnsupportedFiletypeError("Compression using zip not supported")

    return opener(file_name, mode, **open_kwargs)


def _decompress_using_suffix(file_name: str, mode) -> IO:

    file_type = file_name.split(".")[-1]

    opener = open

    if file_type == "xz":
        import lzma

        opener = lzma.open

    elif file_type == "gz":

        import gzip

        opener = gzip.open

    elif file_type == "bz2":
        import bz2

        opener = bz2.open

    elif file_type == "zip":
        raise UnsupportedFiletypeError("Compression using zip not supported")

    return opener(file_name, mode)


def serialize(serializable: Dict, file_name: str) -> None:
    """
    The main serialization method used to serialize objects.

    Parameters:
        serializable: The object to be serialized
        file_name: The name of the file the object will be serialized to.
    """
    format_name = get_format_name(file_name)
    serializer = get_serializer(format_name)

    with _compress_using_suffix(file_name, "w" + serializer.data_type.value) as output:
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
        with _decompress_using_suffix(
            file_name, "r" + deserializer.data_type.value
        ) as input_data:
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
