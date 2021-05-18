# minterpy

Classical interpolation schemes in arbitrary dimensions.

## Description

:construction: :construction: Here will follow a longer description of the project :construction: :construction:


## Installation using venv/pip

1. use the virtual environment of your choice, e.g. [virtualenv]:

   ```bash
   python -m venv <your_venv_name>
   ```
   and activate it:
   ```bash
   source <your_venv_name>/bin/activate
   ```
   so you use the python verion of your virtual environment.    

2. install `minterpy` via pip (in the activated venv):
   ```bash
   pip install [-e] .
   ```
   After installation, you might restart your virtual environment:
   ```bash
   deactivate && source <your_venv_name>/bin/activate
   ```
   or just run
   ```bash
   hash -r
   ```
   instead.  
3. run the unit_tests using [pytest].
   ```bash
   pytest
   ```
   if all tests are passed, the tested functions shall run properly. Here the restart of your venv mentioned above might be necessary.


## Installation using conda/pip

In order to set up the necessary environment:

1. create an environment `minterpy` with the help of [conda],
   ```bash
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```bash
   conda activate minterpy
   ```
3. install `minterpy` with:
   ```bash
   pip install -e .
   ```
4. run the unit_tests using `pytest`
    ```bash
    pytest
    ```
    if all tests are passed, the tested functions shall run properly.

Take a quick look into `CONTRIBUTING.md` for more details.

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE                 <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── minterpy            <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── pyproject.toml          <- Specification build requirements
├── MANIFEST.in             <- Keep track of (minimal) source distribution files
├── CONTRIBUTING.md         <- Contribution guidelines.
├── .readthedocs.yml        <- Configuration of readthedocs support
├── .gitignore              <- ignored files/directories if `git add/commit`
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## :construction: :construction:  Useful badges:

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[virtualenv]: https://virtualenv.pypa.io/en/latest/index.html
[pytest]: https://docs.pytest.org/en/6.2.x/

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![Code style: black][black-badge]][black-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]




[actions-badge]:            https://gitlab.hzdr.de/interpol/minterpy/workflows/CI/badge.svg
[actions-link]:             https://gitlab.hzdr.de/interpol/minterpy/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/minterpy
[conda-link]:               https://github.com/conda-forge/minterpy-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://gitlab.hzdr.de/interpol/minterpy/discussions
[gitter-badge]:             https://badges.gitter.im/https://gitlab.hzdr.de/interpol/minterpy/community.svg
[gitter-link]:              https://gitter.im/https://gitlab.hzdr.de/interpol/minterpy/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/minterpy/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/minterpy
[pypi-version]:             https://badge.fury.io/py/minterpy.svg
[rtd-badge]:                https://readthedocs.org/projects/minterpy/badge/?version=latest
[rtd-link]:                 https://minterpy.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
