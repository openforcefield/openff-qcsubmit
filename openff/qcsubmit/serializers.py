"""
Serialization methods.
"""

import abc
import json
import os
from enum import Enum
from typing import IO, Literal

import yaml

from openff.qcsubmit._pydantic import BaseModel
from openff.qcsubmit.exceptions import UnsupportedFiletypeError

__all__ = [
    "Compressor",
    "DeSerializer",
    "Serializer",
    "deserialize",
    "register_compressor",
    "register_deserializer",
    "register_serializer",
    "serialize",
]

Compression = Literal["xz", "bz2", "gz", ""]
serializers = {}
deserializers = {}
# We know how to compress/decompress these automatically
compression_algorithms = {}


class DataType(str, Enum):
    """
    The type of data the de/serializers deal with, which helps with file loading.
    """

    TEXT = "t"
    BYTES = "b"


class BaseSerializer(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class GeneralSerializer(BaseSerializer):
    data_type: DataType


class Compressor(BaseSerializer, abc.ABC):
    @abc.abstractmethod
    def compress(self, file_name: str, mode: str) -> IO:
        """
        Return the compression IO object.

        Parameters:
            file_name: The name of the file that will be created.
            mode: The writing mode used to make file ie text or bytes

        Notes:
            This will modify the file name to include the extension if not already provided.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decompress(self, file_name: str, mode: str) -> IO:
        """
        Return the decompression IO object.

        Parameters:
            file_name: The name of the file that should be opened.
            mode: The IO mode that should be used to read the file.
        """
        raise NotImplementedError()


class NoneCompressor(Compressor):
    def compress(self, file_name: str, mode: str) -> IO:
        return open(file_name, mode)

    def decompress(self, file_name: str, mode: str) -> IO:
        return open(file_name, mode)


class LZMACompressor(Compressor):
    def compress(self, file_name: str, mode: str) -> IO:
        import lzma

        return lzma.open(file_name, mode, **dict(preset=9))

    def decompress(self, file_name: str, mode: str) -> IO:
        import lzma

        return lzma.open(file_name, mode)


class GzipCompressor(Compressor):
    def compress(self, file_name: str, mode: str) -> IO:
        import gzip

        return gzip.open(file_name, mode, **dict(compresslevel=9))

    def decompress(self, file_name: str, mode: str) -> IO:
        import gzip

        return gzip.open(file_name, mode)


class BZ2Compressor(Compressor):
    def compress(self, file_name: str, mode: str) -> IO:
        import bz2

        return bz2.open(file_name, mode, **dict(compresslevel=9))

    def decompress(self, file_name: str, mode: str) -> IO:
        import bz2

        return bz2.open(file_name, mode)


class Serializer(GeneralSerializer, abc.ABC):
    @abc.abstractmethod
    def serialize(self, serializable: dict) -> str | bytes:
        """
        The method should give the string representation of the serialization ready for dumping to file.
        """
        raise NotImplementedError()


class DeSerializer(GeneralSerializer, abc.ABC):
    @abc.abstractmethod
    def deserialize(self, file_object) -> dict:
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

    def deserialize(self, file_object) -> dict:
        return json.load(file_object)


class YamlDeSerializer(DeSerializer):
    data_type = DataType.TEXT

    def deserialize(self, file_object) -> dict:
        return yaml.full_load(file_object)


def register_compressor(format_name: str, compressor: "Compressor") -> None:
    """
    Register a new compression method with qcsubmit.
    """
    format_name = format_name.lower()
    if format_name in compression_algorithms:
        raise ValueError(f"{format_name} already has a compressor registered.")

    compression_algorithms[format_name] = compressor


def unregister_compressor(format_name: str) -> None:
    """
    Remove one of the registered compression methods with qcsubmit.
    """
    compressor = compression_algorithms.pop(format_name.lower(), None)
    if compressor is None:
        raise KeyError(f"The compression method {format_name} is not registered with qcsubmit.")


def get_compressor(format_name: str) -> "Compressor":
    """
    Return the requested compressor class.
    """
    compressor = compression_algorithms.get(format_name.lower(), None)
    if compressor is None:
        raise UnsupportedFiletypeError(
            f"The specified compression format {format_name} is not supported; supported formats are "
            f"{compression_algorithms.keys()}",
        )
    return compressor()


def register_serializer(format_name: str, serializer: "Serializer") -> None:
    """
    Register a new serializer method with qcsubmit.
    """
    format_name = format_name.lower()
    if format_name in serializers:
        raise ValueError(f"{format_name} already has a serializer registered.")

    serializers[format_name] = serializer


def register_deserializer(format_name: str, deserializer: DeSerializer) -> None:
    """
    Register a new deserializer method with qcsubmit.
    """
    format_name = format_name.lower()

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
            f"supported formats are {serializers.keys()}",
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
            f"supported formats are {deserializers.keys()}",
        )
    return deserailizer()


def get_format_name(file_name: str) -> tuple[str, str]:
    """
    Get the format name by splitting on the . also looks for compression.

    Parameters:
        file_name: The name of the file from which we should work out the format.


    Returns:
        Tuple[str, str]
        The format name and compression type if also supplied.
    """
    split_name = [x.lower() for x in file_name.split(".")]

    def is_compression_extension(ext):
        return ext in compression_algorithms

    if is_compression_extension(split_name[-1]):
        return split_name[-2], split_name[-1]
    else:
        return split_name[-1], ""


def serialize(serializable: dict, file_name: str, compression: Compression | None = None) -> None:
    """
    The main serialization method used to serialize objects.

    Parameters:
        serializable: The object to be serialized
        file_name: The name of the file the object will be serialized to.
        compression: The form of compression to be used, where None will not add any compression.
    """
    format_name, compression_name = get_format_name(file_name)
    compression = compression or compression_name
    serializer = get_serializer(format_name)
    compressor = get_compressor(compression)
    if compression not in file_name and compression != "":
        new_name = "".join([file_name, ".", compression])

    else:
        new_name = file_name

    with compressor.compress(new_name, "w" + serializer.data_type.value) as output:
        output.write(serializer.serialize(serializable))


def deserialize(file_name: str) -> dict:
    """
    The main method used to deserialize the objects from file.

    Parameters:
        file_name: The file from which the object should be extracted.
    """
    format_name, compression_name = get_format_name(file_name)
    deserializer = get_deserializer(format_name)
    decompressor = get_compressor(compression_name)

    if os.path.exists(file_name):
        with decompressor.decompress(file_name, "r" + deserializer.data_type.value) as input_data:
            return deserializer.deserialize(input_data)
    else:
        raise FileNotFoundError(f"The file {file_name} could not be found.")


# Register the json and yaml de/serializers
register_serializer(format_name="json", serializer=JsonSerializer)
register_serializer(format_name="yaml", serializer=YamlSerializer)
register_serializer(format_name="yml", serializer=YamlSerializer)
register_deserializer(format_name="json", deserializer=JsonDeSerializer)
register_deserializer(format_name="yaml", deserializer=YamlDeSerializer)
register_deserializer(format_name="yml", deserializer=YamlDeSerializer)
# register the compression methods
register_compressor(format_name="xz", compressor=LZMACompressor)
register_compressor(format_name="bz2", compressor=BZ2Compressor)
register_compressor(format_name="gz", compressor=GzipCompressor)
register_compressor(format_name="", compressor=NoneCompressor)
