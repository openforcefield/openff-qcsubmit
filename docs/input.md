# Inputs

Powered by the fantastic [OpenFF Toolkit],  QCSubmit can consume input molecules from a wide range of sources, including:

* [](std-formats)
* [](mol-objects)
* [HDF5 files](hdf5-files) via a custom specification


You can read more about each of these input paths below, but in general you can simply pass the input to your chosen QCSubmit [dataset generation factory](factories) via the `molecules` keyword argument in the [`create_dataset`] function:

```python
dataset = factory.create_dataset(
    dataset_name="My exotic dataset",
    # pass the single/multiple molecule sdf here
    molecules="my_exotic_sdf.sdf",
    ...
)
```

QCSubmit will then determine the type of input and process it accordingly using the [`component_result`] class, which deduplicates the molecules while preserving unique conformations.

[OpenFF Toolkit]: https://github.com/openforcefield/openff-toolkit/
[`create_dataset`]: openff.qcsubmit.factories.BaseDatasetFactory.create_dataset
[`component_result`]: openff.qcsubmit.workflow_components.ComponentResult

(std-formats)=
## Standard file formats

QCSubmit supports the following individual file formats, as well as directories containing a mix of formats. Simply provide the path to the target directory and QCSubmit will search through the directory and read in molecules for each file.

* MOL/SDF
* PDB, provided the [`openeye-toolkits`] package is available
* SMILES file

[`openeye-toolkits`]: https://docs.eyesopen.com/toolkits/python/intro.html

(mol-objects)=
## Molecule objects

In some cases you may want to pre-process the molecules using a custom workflow not yet supported by QCSubmit, and thus will have some collection of molecule objects from RDKit, OpenEye or the OpenFF Toolkit. As QCSubmit uses the OpenFF Toolkit [`Molecule`] class internally when processing datasets, the objects need to be first converted to this type. To ensure the correctness of the conversion, convenience methods are provided by the molecule class between [RDKit] and [OpenEye] molecule objects:

```python
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
```

[`Molecule`]: https://open-forcefield-toolkit.readthedocs.io/en/latest/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule
[RDKit]: https://open-forcefield-toolkit.readthedocs.io/en/latest/users/molecule_cookbook.html#from-rdkit-mol
[OpenEye]: https://open-forcefield-toolkit.readthedocs.io/en/latest/users/molecule_cookbook.html#from-openeye-oemol

(hdf5-files)=
## HDF5 files

:::{warning}
`HDF5` support is still pre-alpha and so the specification is still evolving.
:::

QCSubmit also supports the [HDF5] file format, which is well suited to inputs containing many conformations per molecule. The format consists of one named [group] per molecule. Two [datasets] should then be made under this group with the following naming and information:

- `conformations`: A Numpy array with shape `(n, n_atoms, 3)` containing all of the molecule conformations in Cartesian coordinates, where `n` is the number of conformations and `n_atoms` is the number of atoms in the molecule.
- `smiles`: A length 1 list containing a single mapped SMILES string representing the molecule. Every atom in the molecule should be mapped to an index from 1 to `n`.

:::{note}
If the "molecule" contains multiple components, this format still uses a single SMILES string; individual components may be distinguished using the `.` separator.
:::

Finally, the units of the molecule conformation should be recorded as an [attribute] of the `conformations` dataset under the key `units`. Recognized units include:

* `nanometers`
* `angstroms`
* `bohrs`

[HDF5]: http://www.h5py.org/
[group]: https://docs.h5py.org/en/stable/high/group.html#groups
[datasets]: https://docs.h5py.org/en/stable/high/dataset.html#datasets
[attribute]: https://docs.h5py.org/en/stable/high/attr.html#attributes

### Demonstration

`HDF5` files following this format can be constructed using the OpenFF Toolkit:

```python
import h5py
import numpy as np
from openff.units import Quantity, unit

output_file = h5py.File("my_exotic_molecules.hdf5", "w")

# Create a list of OpenFF Toolkit Molecule instances with conformations
target_molecules = [...]

for mol in target_molecules: 
    smiles = mol.to_smiles(
        isomeric=True, 
        explicit_hydrogens=True, 
        mapped=True,
    )
    conformations = [c.m_as(unit.nanometers) for c in mol.conformers]

    group = output_file.create_group(mol.name)
    group.create_dataset(
        'smiles', 
        data=[smiles], 
        dtype=h5py.string_dtype(),
    )
    ds = group.create_dataset(
        'conformations', 
        data=np.array(conformations), 
        dtype=np.float32,
    )
    ds.attrs['units'] = 'nanometers'

output_file.close()
```
