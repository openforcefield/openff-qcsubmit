.. |component_result|       replace:: :py:class:`~openff.qcsubmit.workflow_components.ComponentResult`
.. |create_dataset|         replace:: :py:func:`~openff.qcsubmit.factories.BaseDatasetFactory.create_dataset`

============
Inputs
============

Powered by the fantastic `openff-toolkit <https://github.com/openforcefield/openff-toolkit/>`_  ``QCSubmit`` can consume input molecules from a wide range of sources including:

.. rst-class:: spaced-list

    - :ref:`Standard file formats`
    - :ref:`Molecule objects`
    - :ref:`HDF5 files` via a custom specification


You can read more about each of these inputs below, but in general getting started simply requires you to pass the input
to your chosen ``QCSubmit`` :ref:`dataset generation factory <factories>` via the ``molecules`` keyword argument in the |create_dataset| function as shown here:

.. code-block:: python

    dataset = factory.create_dataset(
        dataset_name="My exotic dataset",
        # pass the single/multiple molecule sdf here
        molecules="my_exotic_sdf.sdf",
        ...
    )

``QCSubmit`` will then determine the type of input and process it accordingly using the |component_result| class which will deduplicate the molecules while preserving unique conformations.



Standard file formats
----------------------

``QCSubmit`` supports the following individual file formats as well as directories containing a mix of formats, simply provide the path to the target directory and ``QCSubmit`` will search through the directory trying to read in molecules for each file.

.. rst-class:: spaced-list

    - MOL/SDF
    - PDB provided you have `openeye-toolkits <https://docs.eyesopen.com/toolkits/python/intro.html>`_ available
    - SMILES file



Molecule objects
----------------

In some cases you may want to pre-process the molecules using a custom workflow not yet supported by ``QCSubmit`` and thus will have some collection of molecule objects from ``RDKit``, ``OpenEye`` or the ``openff-toolkit``.
As ``QCSubmit`` uses the ``openff-toolkit`` `Molecule <https://open-forcefield-toolkit.readthedocs.io/en/latest/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule>`_ class internally when processing datasets the objects need to be first converted to this type. To ensure the correctness of the conversion convince methods are provided by the molecule class between `RDKit <https://open-forcefield-toolkit.readthedocs.io/en/latest/users/molecule_cookbook.html#from-rdkit-mol>`_ and `OpenEye <https://open-forcefield-toolkit.readthedocs.io/en/latest/users/molecule_cookbook.html#from-openeye-oemol>`_ objects.

.. code-block:: python

    from openff.toolkit.topology import Molecule

    # a list of OE and RDKit molecules
    processed_mols = [oemol1, oemol2 rdmol1, rdmol2]

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

``QCSubmit`` also supports `HDF5 Files <http://www.h5py.org/>`_ following a simple format which is well suited to inputs containing many
conformations per molecule. The format consists of one `group <https://docs.h5py.org/en/stable/high/group.html#groups>`_ per molecule stored
under the index which should be assigned to the molecule. Two `datasets <https://docs.h5py.org/en/stable/high/dataset.html#datasets>`_ should then
be made under this group with the following naming and information

.. rst-class:: spaced-list

    - ``conformations``: A numpy ndarray containing all of the molecule conformations with shape (n, n_atoms, 3), where ``n`` is the number of conformations and ``n_atoms`` is the number of atoms in the molecule.
    - ``smiles``: A length 1 list of mapped smiles strings which represents the topology of the entire system.

.. note::
    If the system contains multiple components we should have a single smiles
    string indexed from 1 to m where m is the total number of atoms, distinguishing individual components using the ``.`` separator.

Finally the units of the molecule conformation should be set as an `attribute <https://docs.h5py.org/en/stable/high/attr.html#attributes>`_ of the ``conformations`` dataset under the key ``units``,
recognised units are as follows:

.. rst-class:: spaced-list

    - nanometer(s)
    - angstrom(s)
    - bohr(s)


Demonstration
"""""""""""""

``HDF5`` files following this format can then be readily made using the ``openff-toolkit``:

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
