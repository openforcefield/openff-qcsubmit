# QCSubmit

[//]: # (Badges)
[![CI](https://github.com/openforcefield/qcsubmit/workflows/CI/badge.svg?branch=master)](https://github.com/openforcefield/qcsubmit/actions)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/openforcefield/qcsubmit.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/openforcefield/qcsubmit/context:python)
[![codecov](https://codecov.io/gh/openforcefield/qcsubmit/branch/master/graph/badge.svg)](https://codecov.io/gh/openforcefield/qcsubmit/branch/master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Automated tools for submitting molecules to QCArchive instances.


## Installation

A development version of QCSubmit can be installed via conda:

    conda install -c conda-forge -c omnia/label/rc -c omnia -c qcsubmit
    pip install basis_set_exchange

If you want the OpenEye Toolkit and Fragmenter, do instead:

    conda install -c conda-forge -c omnia/label/rc -c omnia -c openeye qcsubmit fragmenter
    pip install basis_set_exchange

Note that the OpenEye Toolkit is required to use fragmenter and protomer enumeration.
All other features can be used with RDKit.


### Copyright

Copyright (c) 2019-2020, Open Force Field Initiative


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
