Installation
============

Installing using conda
----------------------

The recommended way to install QCSubmit is via the ``conda`` package manger:

.. code-block:: bash

    conda install -c conda-forge openff-qcsubmit

Optional dependencies
"""""""""""""""""""""

If you have access to the OpenEye toolkits (namely ``oechem``, ``oequacpac``, ``oeomega`` and ``oedepict``), we recommend installing these also as these can speed up various operations performed by this framework significantly:

.. code-block:: bash

    conda install -c openeye openeye-toolkits

Installing from source
----------------------

To install ``openff-qcsubmit`` from source, begin by cloning the repository from `github <https://github.com/openforcefield/openff-qcsubmit>`_:

.. code-block:: bash

    git clone https://github.com/openforcefield/openff-qcsubmit.git
    cd openff-qcsubmit

Create a custom conda environment which contains the required dependencies and activate it:

.. code-block:: bash

    conda env create --name openff-qcsubmit --file devtools/conda-envs/meta.yaml
    conda activate openff-qcsubmit

Finally, install the ``openff-qcsubmit`` package into the current environment:

.. code-block:: bash

    python setup.py develop
