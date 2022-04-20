Building the Docs
=================

Although documentation for the OpenFF QCSubmit is `readily available online
<https://openff-qcsubmit.readthedocs.io/en/latest/>`_, it is sometimes useful
to build a local version such as when

- developing new pages which you wish to preview without having to wait
  for ReadTheDocs to finish building.

- debugging errors which occur when building on ReadTheDocs.

In these cases, the docs can be built locally by doing the following::

    git clone https://github.com/openforcefield/openff-qcsubmit.git
    cd openff-qcsubmit
    conda env create --name openff-qcsubmit-docs --file devtools/conda-envs/docs.yml
    conda activate openff-qcsubmit-docs
    rm -rf docs/api/generated docs/_build/html && sphinx-build -b html -j auto docs docs/_build/html

The above will yield a new directory named `docs/_build/html` which will
contain the built html files which can be viewed in your local browser.