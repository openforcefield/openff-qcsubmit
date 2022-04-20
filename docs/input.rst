.. |component_result|       replace:: :py:class:`~openff.qcsubmit.workflow_components.ComponentResult`
.. |create_dataset|         replace:: :py:func:`~openff.qcsubmit.factories.BaseDatasetFactory.create_dataset`

============
Inputs
============

Powered by the fantastic `OpenFF Toolkit <https://github.com/openforcefield/openff-toolkit/>`_,  QCSubmit can consume input molecules from a wide range of sources, including:

.. rst-class:: spaced-list

    - :ref:`Standard file formats`
    - :ref:`Molecule objects`
    - :ref:`HDF5 files` via a custom specification


You can read more about each of these input paths below, but in general you can simply pass the input to your chosen QCSubmit :ref:`dataset generation factory <factories>` via the ``molecules`` keyword argument in the |create_dataset| function:

.. code-block:: python

    dataset = factory.create_dataset(
        dataset_name="My exotic dataset",
        # pass the single/multiple molecule sdf here
        molecules="my_exotic_sdf.sdf",
        ...
    )

QCSubmit will then determine the type of input and process it accordingly using the |component_result| class, which deduplicates the molecules while preserving unique conformations.



Standard file formats
----------------------

QCSubmit supports the following individual file formats, as well as directories containing a mix of formats. Simply provide the path to the target directory and QCSubmit will search through the directory and read in molecules for each file.

.. rst-class:: spaced-list

    - MOL/SDF
    - PDB, provided the `openeye-toolkits <https://docs.eyesopen.com/toolkits/python/intro.html>`_ package is available
    - SMILES file



Molecule objects
----------------

In some cases you may want to pre-process the molecules using a custom workflow not yet supported by ``QCSubmit``, and thus will have some collection of molecule objects from RDKit, OpenEye or the OpenFF Toolkit. As QCSubmit uses the OpenFF Toolkit `Molecule <https://open-forcefield-toolkit.readthedocs.io/en/latest/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule>`_ class internally when processing datasets, the objects need to be first converted to this type. To ensure the correctness of the conversion, convenience methods are provided by the molecule class between `RDKit <https://open-forcefield-toolkit.readthedocs.io/en/latest/users/molecule_cookbook.html#from-rdkit-mol>`_ and `OpenEye <https://open-forcefield-toolkit.readthedocs.io/en/latest/users/molecule_cookbook.html#from-openeye-oemol>`_ objects:

.. code-block:: python

    from openff.toolkit.topology import Molecule

    # a list of OE and RDKit molecules
    processed_mols = [oemol1, oemol2, rdmol1, rdmol2]

    # convert to openff.toolkit.topology.Molecule instances
    molecules = [Molecule(ref_mol) for ref_mol in processed_mols]

    dataset = factory.create_dataset(
        dataset_name="My exotic dataset",
        # pass the list of molecules
        molecules=molecules,
        ...
    )


HDF5 files
-----------

.. warning:: ``HDF5`` support is still pre-alpha and so the specification is still evolving.

QCSubmit also supports the `HDF5 <http://www.h5py.org/>`_ file format, which is well suited to inputs containing many conformations per molecule. The format consists of one named `group <https://docs.h5py.org/en/stable/high/group.html#groups>`_ per molecule. Two `datasets <https://docs.h5py.org/en/stable/high/dataset.html#datasets>`_ should then be made under this group with the following naming and information

.. rst-class:: spaced-list

    - ``conformations``: A numpy ndarray with shape (n, n_atoms, 3) containing all of the molecule conformations in Cartesian coordinates, where ``n`` is the number of conformations and ``n_atoms`` is the number of atoms in the molecule.
    - ``smiles``: A length 1 list containing a single mapped SMILES string representing the molecule. Every atom in the molecule should be mapped to an index from 1 to ``n``.

.. note::
    If the "molecule" contains multiple components, this format still uses a single SMILES string; individual components may be distinguished using the ``.`` separator.

Finally, the units of the molecule conformation should be recorded as an `attribute <https://docs.h5py.org/en/stable/high/attr.html#attributes>`_ of the ``conformations`` dataset under the key ``units``. Recognised units include:

.. rst-class:: spaced-list

    - nanometer(s)
    - angstrom(s)
    - bohr(s)


Demonstration
"""""""""""""

``HDF5`` files following this format can then be constructed using the OpenFF Toolkit:

.. code-block:: python

    import h5py
    import numpy as np
    from simtk import unit

    output_file = h5py.File("my_exotic_molecules.hdf5", "w")

    for molecule in target_molecules: # a list of openff.toolkit.topology.Molecule instances with conformations
        smiles = molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
        conformations = [c.value_in_unit(unit.nanometers) for c in molecule.conformers]
        group = output_file.create_group(molecule.name)
        group.create_dataset('smiles', data=[smiles], dtype=h5py.string_dtype())
        ds = group.create_dataset('conformations', data=np.array(conformations), dtype=np.float32)
        ds.attrs['units'] = 'nanometers'

    output_file.close()
