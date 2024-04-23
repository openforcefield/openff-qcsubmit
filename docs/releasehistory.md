# Release History

Releases follow the ``major.minor.micro`` scheme recommended by
[PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

Releases are given with dates in DD-MM-YYYY format.

<!--## Version / Date DD-MM-YYYY -->
## 0.51.0 / 23-04-2024

### Behaviors changed

* [PR #277:] Changes the behavior of the `max_states` named argument to `workflow_components.EnumerateProtomers`. Previously this could return anywhere from `1` to `max_states+2`, but now it can return `1` to `max_states+1` (depending on whether the backend includes the input in the protomers that are generated). 

### Bugfixes

* [PR #277:] Updates for QCPortal 0.54 (#275) and OpenFF Toolkit 0.16 (#278) [@bennybp @mattwthompson @j-wags]


## 0.50.3 / 24-03-2024

### Bugfixes

* [PR #257:] Fixes dataset visualization with the RDKit backend (#257) [@pavankum]
* [PR #260:] Fixes a bug where adding different tautomers of the same molecule to a ComponentResult would raise an error. Also fixes a case where ComponentResult.add_molecule would fail to return a bool  (#255) [@mattwthompson]
* [PR #268:] Fix broken star imports (#268) [@mattwthompson]

### Performance Improvements

* [PR #270:] Speed up `TorsionDriveResultCollection.to_records` by batching requests [@ntBre]

## 0.50.2 / 24-01-2024

### New Features

* [PR #232:] Introduce runtime compatibility with Pydantic v1 and v2. (#232) [@mattwthompson]
* [PR #251:] Use Psi4 1.9 by default. (#251) [@mattwthompson]

### Behavior changes

* [PR #238:] Removes dependency on OpenMM by using `openff-units` to map between atomic numbers and symbols, and other small internal changes. (#238) [@mattwthompson]

### API-breaking changes

* [PR #242:] Make tests private (`openff/qcsubmit/tests` --> `openff/qcsubmit/_tests`) (#242) [@mattwthompson]

### Tests updated

* [PR #252:] Update constrained torsiondrive test to use a smaller molecule to avoid CI runs timing out. (#252) [@j-wags]

## 0.50.1 / 10-11-2023

### Bugfixes

* [PR #237:] Correctly use `openff.units` in `TorsionDriveResultCollection.to_records()` and the same method of other classes. (#237) [@chapincavender]

## 0.50.0 / 31-10-2023

For more information on this release, see https://github.com/openforcefield/openff-qcsubmit/releases/tag/0.50.0

### API-breaking changes

* [PR #195:] Support QCFractal 0.50+ (#195) [@j-wags]

### Bugfixes

* [PR #235:] Update `versioneer` for Python 3.12 compatibility (#235) [@Yoshanuikabundi]

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
[PR #195:]: https://github.com/openforcefield/openff-qcsubmit/pull/195
[PR #198:]: https://github.com/openforcefield/openff-qcsubmit/pull/198
[PR #202:]: https://github.com/openforcefield/openff-qcsubmit/pull/202
[PR #204:]: https://github.com/openforcefield/openff-qcsubmit/pull/204
[PR #206:]: https://github.com/openforcefield/openff-qcsubmit/pull/206
[PR #232:]: https://github.com/openforcefield/openff-qcsubmit/pull/232
[PR #235:]: https://github.com/openforcefield/openff-qcsubmit/pull/235
[PR #237:]: https://github.com/openforcefield/openff-qcsubmit/pull/237
[PR #238:]: https://github.com/openforcefield/openff-qcsubmit/pull/238
[PR #242:]: https://github.com/openforcefield/openff-qcsubmit/pull/242
[PR #251:]: https://github.com/openforcefield/openff-qcsubmit/pull/251
[PR #252:]: https://github.com/openforcefield/openff-qcsubmit/pull/252
[PR #257:]: https://github.com/openforcefield/openff-qcsubmit/pull/257
[PR #260:]: https://github.com/openforcefield/openff-qcsubmit/pull/260
[PR #268:]: https://github.com/openforcefield/openff-qcsubmit/pull/268
[PR #270:]: https://github.com/openforcefield/openff-qcsubmit/pull/270
[PR #277:]: https://github.com/openforcefield/openff-qcsubmit/pull/277

[@jthorton]: https://github.com/jthorton
[@dotsdl]: https://github.com/dotsdl
[@Yoshanuikabundi]: https://github.com/Yoshanuikabundi
[@mattwthompson]: https://github.com/mattwthompson
[@chapincavender]: https://github.com/chapincavender
[@j-wags]: https://github.com/j-wags
[@pavankum]: https://github.com/pavankum
[@ntBre]: https://github.com/ntBre
