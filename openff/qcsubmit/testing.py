"""
Useful functions for testing the package.
"""
import contextlib
import os
import shutil
import tempfile


@contextlib.contextmanager
def temp_directory():
    """
    Create and enter a temporary directory, used as a context manager.
    Taken from https://github.com/mdtraj/mdtraj/blob/master/mdtraj/utils/contextmanagers.py#L39
    """
    temp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(temp_dir)
    yield
    os.chdir(cwd)
    shutil.rmtree(temp_dir)
