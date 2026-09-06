# Third-Party Notices

Proteia is licensed under the Apache License 2.0. It depends on third-party
open-source packages, each distributed under its own license.

A complete, generated inventory will be added before the first release (for
example via `pip-licenses`). The principal runtime dependencies are:

| Package | License |
| --- | --- |
| napari | BSD-3-Clause |
| PySide6 (Qt for Python) | LGPL-3.0 |
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| scikit-image | BSD-3-Clause |
| tifffile | BSD-3-Clause |

**License hygiene note.** The Qt binding is **PySide6 (LGPL)**, not PyQt
(GPL), to keep Proteia's dependency graph compatible with permissive
redistribution. Do not switch the Qt backend to PyQt without a license review.
