# Release History

Releases follow the ``major.minor.micro`` scheme recommended by
[PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

<!--## Version / Date DD-MM-YYYY -->
## Current development

### Bugfixes

* [PR #237:] Correctly use `openff.units` in `TorsionDriveResultCollection.to_records()` and the same method of other classes. (#237) [@chapincavender]

### Tests updated

* [PR #252:] Update constrained torsiondrive test to use a smaller molecule to avoid CI runs timing out.


## 0.4.0 / 11-15-2022

### New Features

* [PR #204:] Compatibility OpenFF Toolkit 0.11.x (#204) [@Yoshanuikabundi]


## 0.3.2 / 08-11-2022

### New Features

* [PR #198:] Updated documentation, including migration to use Markdown files and `openff-sphinx-theme` [@Yoshanuikabundi]
* [PR #202:] Support multi-dimensional TorsionDrive grid indices [@chapincavender]
* [PR #206:] Support more QC programs provided by QCEngine [@mattwthompson]


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
[PR #198:]: https://github.com/openforcefield/openff-qcsubmit/pull/198
[PR #202:]: https://github.com/openforcefield/openff-qcsubmit/pull/202
[PR #204:]: https://github.com/openforcefield/openff-qcsubmit/pull/204
[PR #206:]: https://github.com/openforcefield/openff-qcsubmit/pull/206
[PR #237:]: https://github.com/openforcefield/openff-qcsubmit/pull/237

[@jthorton]: https://github.com/jthorton
[@dotsdl]: https://github.com/dotsdl
[@Yoshanuikabundi]: https://github.com/Yoshanuikabundi
[@mattwthompson]: https://github.com/mattwthompson
[@chapincavender]: https://github.com/chapincavender
