# Release History

Releases follow the ``major.minor.micro`` scheme recommended by
[PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

<!--## Version / Date DD-MM-YYYY -->

## 0.3.1 / 08-03-2022

### Bugfixes

* [PR #183:] Fixed a bug which stopped molecules from being added to datasets via attributes passed to `Dataset.add_molecule`. [@dotsdl]
* [PR #184:] Correctly create results from collections with missing `inchi_key` fields [@jthorton]
* [PR #187:] Now able to make results from single point datasets with `Nan` records [@jthorton]
* [PR #192:] now keep all conformers for molecules. [@jthorton]

### New Features

* [PR #186:] Improved performance for submissions of large datasets by reducing the number of save calls [@dotsdl]
* [PR #193:] Fragmentation components now support user supplied torsion target SMARTS [@jthorton]

[PR #183:]: https://github.com/openforcefield/openff-qcsubmit/pull/183
[PR #184:]: https://github.com/openforcefield/openff-qcsubmit/pull/184
[PR #186:]: https://github.com/openforcefield/openff-qcsubmit/pull/186
[PR #187:]: https://github.com/openforcefield/openff-qcsubmit/pull/187
[PR #192:]: https://github.com/openforcefield/openff-qcsubmit/pull/192
[PR #193:]: https://github.com/openforcefield/openff-qcsubmit/pull/193

[@jthorton]: https://github.com/jthorton
[@dotsdl]: https://github.com/dotsdl