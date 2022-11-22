# Contribution Guide

Thanks a lot for your interest and taking the time to contribute to the `minterpy` project!

This document provides guidelines for contributing to the `minterpy` project.

## Installation

This installation guide is focused on development.
For installing `minterpy` in production runs check out the [README.md](./README.md).

In order to get the source of latest release,
clone the `minterpy` repository from the [HZDR GitLab]:

```bash
git clone https://gitlab.hzdr.de/interpol/minterpy.git
```

By default, the cloned branch is the `main` branch.

To get the latest development version, checkout to the `dev` branch:

```bash
git checkout dev
```

We recommend to always pull the latest commit:

```bash
git pull origin dev
```

You are not allowed to directly push to `dev` or `master` branch.
Please follow the instructions under [Branching workflow](#branching-workflow).

### Virtual environments

Following a best practice in Python development,
we strongly encourage you to create and use virtual environments for development and production runs.
A virtual environment encapsulates the package and all dependencies without messing up your other Python installations.

The following instructions should be executed from the `minterpy` source directory.

#### Using [venv](https://docs.python.org/3/tutorial/venv.html) from the python standard library:

1. Build a virtual environment:

    ```bash
    python -m venv <your_venv_name>
    ```

   Replace `<you_venv_name>` with an environment name of your choice.

2. Activate the environment you just created:

    ```bash
    source <your_venv_name>/bin/activate
    ```

    as before replace `<you_venv_name>` with the environment name.


3. To deactivate the virtual environment, type:


    ```bash
    deactivate
    ```

#### Using [virtualenv](https://virtualenv.pypa.io/en/latest/):

1. Building a virtual environment:

    ```bash
    virtualenv <your_venv_name>
    ```

   Replace `<you_venv_name>` with an environment name of your choice.

2. Activate the environment:

    ```bash
    source <your_venv_name>/bin/activate
    ```

    As before replace `<you_venv_name>` with the environment name.

3. To deactivate the virtual environment, type:


    ```bash
    deactivate
    ```

#### Using [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv):

1. Building the virtual environment:

    ```bash
    pyenv virtualenv 3.8 <your_venv_name>
    ```

   Replace `<you_venv_name>` with an environment name of your choice.

2. Activate the newly created environment in the current local directory:

    ```bash
    pyenv local <your_venv_name>
    ```

    As before replace `<you_venv_name>` with the environment name.

    The above command creates a hidden file `.python_version` containing a "link" to the actual virtual environment managed by `pyenv`.

3. To "deactivate" the virtual environment just remove this hidden file:

    ```bash
    rm .python_version
    ```

#### Using [conda](https://conda.io/projects/conda/en/latest/index.html):

1. Create an environment `minterpy` with the help of [conda]https://conda.io/projects/conda/en/latest/index.html)
   and the file `environment.yaml`
   (included in the source distribution of `minterpy`):

   ```bash
   conda env create -f environment.yaml
   ```
   The command creates a new conda environment called `minterpy`.

2. Activate the new environment with:

   ```bash
   conda activate minterpy
   ```

   You may need to initialize conda env;
   follow the instructions printed out or read the conda docs.

3. To deactivate the conda environment, type:

    ```bash
    conda deactivate
    ```

### Installation

We recommend using [pip](https://pip.pypa.io/en/stable/) from within a virtual environment (see above)
to install `minterpy`.

To install `minterpy`, type:

```bash
pip install [-e] .[all,dev,docs]
```

where the flag `-e` means the package is directly linked into the Python site-packages.
The options `[all,dev,docs]` refer to the requirements defined in the `options.extras_require` section in `setup.cfg`.

You **must not** use `python setup.py install`,
since the file `setup.py` will not be present for every build of the package.

### Troubleshooting: pytest with venv (*not* conda)

After installation, you might need to restart your virtual environment
since the `pytest` command uses the `PYTHONPATH` environment variable which not automatically change to your virtual environment.

To restart your virtual environment created by `venv`, type:

```bash
deactivate && source <your_venv_name>/bin/activate
```

or run `hash -r` instead.

This problem does not seem to appear for virtual environments created by conda.

### Dependency Management & Reproducibility (conda)

Here are a few recommendations for managing dependency and maintaining the reproducibility of your `minterpy` development environment:

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.

2. Create concrete dependencies as `environment.lock.yaml`
   for the exact reproduction of your environment with:

   ```bash
   conda env export -n minterpy -f environment.lock.yaml
   ```

   For multi-OS development, consider using `--no-builds` during the export.

3. Update your current environment with respect to a new `environment.lock.yaml` using:

   ```bash
   conda env update -f environment.lock.yaml --prune
   ```

## Testing

:construction: :construction: :construction: :construction: :construction: :construction: :construction: :construction:
Since the whole test environment needs a refactoring, we shall update this section with more detailed informations.
:construction: :construction: :construction: :construction:  :construction: :construction: :construction: :construction:

### Running the unit tests

We use [pytest](https://docs.pytest.org/en/6.2.x/) to run the unit tests of `minterpy`.
The unit tests themselves must always be placed into the `tests` directory.
To run all tests, type:

```bash
pytest
```

from within the `minterpy` source directory.

If you want to run the tests of a particular module,
for instance the `multi_index_utils.py` module, execute:

```bash
pytest tests/test_multi_index_utils.py
```

When you run `pytest`, the coverage test is also done automatically.
A summary of the coverage test is printed out in the terminal.
Furthermore, you can find an HTML version of the coverage test results
in `htmlcov/index.html`.

### Writing new tests

We strongly encourage you to use the capabilities of `pytest` for writing the unit tests

It is highly recommended to use the capabilities of `pytest` for writing unittests.

Be aware of the following points:

- the developer of the code should write the tests
- test the behavior you expect from your code, not breaking points
- use as small samples as possible
- unit tests do *not* test if the code works, they test if the code *still* works
- the coverage shall always be as high as possible
- BUT, even 100% coverage does not mean, there is nothing missed (buzz: edge case!)

For additional reference on how to write tests, have a look at the following resources:

- [Pytest: Examples and customization tricks](https://docs.pytest.org/en/6.2.x/example/index.html)
- [Effective Python Testing with Pytest](https://realpython.com/pytest-python-testing/)
- [Testing best practices for ML libraries](https://towardsdatascience.com/testing-best-practices-for-machine-learning-libraries-41b7d0362c95)

## Documentation

This section provides some information about contributing to the docs.

### Install dependencies

Building the docs requires additional dependencies.
If you follow the above installation steps, the dependencies are satisfied.
Otherwise you need to install them separately via:

```bash
pip install .[docs]
```

from the `minterpy` source directory.

### Building the documentation

We use [sphinx](https://www.sphinx-doc.org/en/master/) to build the `minterpy` docs.
To build the docs in HTML format, run the following command:

```bash
sphinx-build -M html docs docs/build
```

Alternatively you can build the docs using the supplied Makefile.
For that, you need to navigate to the `docs` directory and run the `make` command in Linux/mac OS or `make.bat` in Windows:

```bash
cd docs
make html
```

The command builds the docs and stores it in in `docs/build`.
You may open the docs using a web browser of your choice by opening `docs/build/html/index.html`.

You can also generate the docs in PDF format using `pdflatex` (requires a LaTeX distribution installed in your system):

```bash
cd docs
make latexpdf
```

The command builds the docs as a PDF document and stores it along with all the LaTeX source files in `docs/build/latex`.

### Design of the docs

The source files for the docs are stored in the `docs` directory.
The Sphinx configuration file is `docs/conf.py`
and the main index file of the docs is `docs/index.rst`.

The docs itself contains five different main sections:

- The Getting Started Guide or tutorials (`docs/getting-started`) contains all the tutorials of `minterpy`.
- The How-to Guides (`docs/how-to`) contains the Jupyter notebooks of instructions on how to achieve common tasks with `minterpy`.
- The Fundamentals (`docs/fundamentals`) contains all the explanations on the mathematical background that underlies `minterpy`.
- The Contributors Guide (`docs/contributors`) contains the information on how to contribute to the `minterpy` project,
  be it to its development or to its docs.
- The API Reference (`docs/api`) contains the reference to all exposed components of `minterpy` (functions, classes, etc.).

You can find more information about the `minterpy` docs in the Contributors Guide.

## Code Style

To ensure the readability of the codebase, we are following a common code style for `minterpy`.
Our long-term goal is to fulfill the [PEP8](https://www.python.org/dev/peps/pep-0008/) regulations.
For the build system, it is recommended to follow [PEP517](https://www.python.org/dev/peps/pep-0517/)
and [PEP518](https://www.python.org/dev/peps/pep-0518/).
However, since these requirements are very challenging, we use [black](https://github.com/psf/black) to enforce the code style of `minterpy`.

During the development process,
you can check the format using [pre-commit](https://pre-commit.com) (see below) and

In the development process, one can check the format using  and the hooks defined in `.pre-commit-config.yaml`. For instance running `black` for the whole `minterpy` code, just run

```bash
pre-commit run black --all-files
```

For now, it is recommended to run single hooks.

## Pre-commit

For further developments, it is recommended to run all pre-commit-hooks every time
before committing some changes to your branch.

To enable this, type:

```bash
pre-commit install
```

If you want to disable the pre-commit script, type:

```bash
pre-commit uninstall
```

To run all hooks defined in `.pre-commit-config.yaml`, type:

```bash
pre-commit run --all-files # DON'T DO THIS IF YOU DON'T KNOW WHAT HAPPENS
```

In the current state of the code, you should use this with caution
since it might change code in the manner that it breaks (see below).

Down the road, we shall try to fulfill the full set of pre-commit hoos.
However, further developments shall try to fulfil the full set of pre-commit-hooks.

### Currently defined hooks

The following hooks are defined:

- [black](https://github.com/psf/black): a straightforward code formatter;
  it modifies the code in order to fulfill the format requirement.
- [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks): A collection of widely used hooks;
  see their repository for more informations.
- [isort](https://github.com/PyCQA/isort): sorts the import statements;
  changes the code !!!DO NOT RUN THIS: IT WILL BREAK THE CURRENT VERSION!!!
- [pyupgrade](https://github.com/asottile/pyupgrade): convert the syntax from Python2 to Python3.
  It's nice if you use code from an old post in stackoverflow ;-)
- [setup-cfg-fmt](https://github.com/asottile/setup-cfg-fmt): formats the `setup.cfg` file for consistency.
- [flake8](https://github.com/pycqa/flake8): a collection of hooks to ensure most of the PEP8 guidelines are satisfied.
  The concrete checks are defined in the `setup.cfg[flake8]`.
- [mypy](https://github.com/python/mypy): a static type checker;
  `mypy` itself is configured in the `setup.cfg[mypy-*]`.
- [check-manifest](https://github.com/mgedmin/check-manifest):
  checks if the `MANIFEST.in` is in a proper state.
  This ensures proper builds for uploading the package to [PyPI](https://pypi.org).
  This is configured in `setup.cfg[check-manifest]`.

## Code development

### Version control

We only use [git](https://git-scm.com/) to version control `minterpy`.
The main repository for development is place on [HZDR GitLab](https://gitlab.hzdr.de/interpol/minterpy).
Moreover, the releases and the development branch are also mirrored into the [CASUS GitHub](https://github.com/casus/) repository.

We are currently considering to upload the builds of `minterpy` to [PyPI](https://pypi.org) and [conda-forge](https://conda-forge.org)
to make the code more accessible.

### Branching workflow <a name="branching workflow"></a>

We loosely follow the structure of [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
for our branching workflow.
There are three types of branches in this workflow:

1. `master`  branch:
    On this branch, only the releases are stored.
    This means, on this branch, one has only fully tested, documented and cleaned up code.
2. `dev` branch:
    On this branch, the development version are stored.
    At any given time, the branch must pass all the tests.
    This also means that on this branch, there is always a running version of `minterpy`
    even if the code and the docs are not in a "release state."

3. `feature` branches:
    On these branches, all the features and code developments happen.
    These branches must always be branched from the `dev` branch (not from `master`).

Based on this workflow, you can freely push, change, and merge *only* on the `feature` branches.
Furthermore, your feature branch is open to every developers in the `minterpy` project.

Once the implementation of a feature is finished,
you can merge the `feature` branch to the `dev` branch via a merge request.
The project maintainers will merge your merge request once the request is reviewed.
In general, you cannot merge your `feature` branch directly to the `dev` branch.

Furthermore, as a contributor, you cannot merge directly to the `master` branch and you cannot make a merge request for that.
Only the project maintainers can merge the `dev` to the `master` branch following the release procedure
of [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

We manage the bug fixes on every branch separately with the relevant developers, usually via `hotfix` branches to implement the patches.

In the future, we may set up a continuous integration and development (CI/CD) on [HZDR GitLab](https://gitlab.hzdr.de/interpol/minterpy).

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
