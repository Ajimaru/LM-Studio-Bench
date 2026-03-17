# Third-Party Licenses

This project depends on third-party software at development time and runtime.

This file is a practical reference for maintainers and users. It is not legal
advice.

## Runtime / System Dependencies

- **Python**
  - Website: <https://www.python.org/>
  - License: Python Software Foundation License

- **LM Studio / lmstudio SDK / lms CLI**
  - Website: <https://lmstudio.ai/>
  - License: See vendor terms and licensing documentation

- **FastAPI**
  - Website: <https://fastapi.tiangolo.com/>
  - License: MIT

- **Uvicorn**
  - Website: <https://www.uvicorn.org/>
  - License: BSD-3-Clause

- **Jinja2**
  - Website: <https://jinja.palletsprojects.com/>
  - License: BSD-3-Clause

- **Plotly**
  - Website: <https://plotly.com/python/>
  - License: MIT

- **ReportLab**
  - Website: <https://www.reportlab.com/>
  - License: BSD-3-Clause

- **psutil**
  - Website: <https://github.com/giampaolo/psutil>
  - License: BSD-3-Clause

- **tqdm**
  - Website: <https://tqdm.github.io/>
  - License: MPL-2.0

- **httpx**
  - Website: <https://www.python-httpx.org/>
  - License: BSD-3-Clause

- **distro**
  - Website: <https://github.com/python-distro/distro>
  - License: Apache-2.0

- **py-cpuinfo**
  - Website: <https://github.com/workhorsy/py-cpuinfo>
  - License: MIT

- **SciPy**
  - Website: <https://scipy.org/>
  - License: BSD-3-Clause

- **PyGObject**
  - Website: <https://pygobject.gnome.org/>
  - License: LGPL-2.1-or-later

- **GPU tooling (optional)**
  - Tools: `nvidia-smi`, `rocm-smi`, `lspci`
  - License: See vendor terms and licensing documentation

## Development / CI Dependencies

The repository may use GitHub Actions workflows and community actions under
`.github/workflows/*`. Their licenses and terms are governed by each upstream
action repository and GitHub Terms.

Please review each action repository for exact license details.

## Test Data Assets

- **Vision test images (`tests/data/images/*.jpg`)**
  - Source: Wikimedia Commons
  - Selected licenses: CC0 1.0 and Public Domain (as reported by Wikimedia)
  - Attribution and per-file provenance:
    `tests/data/images/LICENSES.md`
  - Raw metadata capture for reproducibility:
    `tests/data/images/wikimedia_manifest.json`

## Notes

- System package licenses may vary by distribution packaging.
- If you redistribute binaries or bundled dependencies, ensure full license
  text and notice requirements are met.
- Revisit this file when adding new runtime dependencies or workflow actions.
- Revalidate third-party asset license status before release distribution.
