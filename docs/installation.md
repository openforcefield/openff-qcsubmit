# Installation

## Installing using conda

The recommended way to install QCSubmit is via the `conda` package manger:

```bash
conda install -c conda-forge openff-qcsubmit
```

If you do not have Conda installed, see the [OpenFF installation guide](openff.docs:install)

### Optional dependencies

If you have access to the OpenEye toolkits (namely `oechem`, `oequacpac`, `oeomega` and `oedepict`), we recommend installing these also as these can speed up various operations performed by this framework significantly:

```bash
conda install -c openeye openeye-toolkits
```

## Installing from source

To install `openff-qcsubmit` from source, begin by cloning the repository from [GitHub](https://github.com/openforcefield/openff-qcsubmit):

```bash
git clone https://github.com/openforcefield/openff-qcsubmit.git
cd openff-qcsubmit
```

Create a custom conda environment which contains the required dependencies and activate it:

```bash
conda env create --name openff-qcsubmit --file devtools/conda-envs/basic.yaml
conda activate openff-qcsubmit
```

Finally, install the `openff-qcsubmit` package into the current environment:

```bash
python setup.py develop
```
