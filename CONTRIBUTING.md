# :construction: :construction: This document is under construction :construction: :construction:
# Contribution guide

# 1. Installation
This installation guide is focused on development. For productive runs of `minterpy` check out the `README.md`.  
In order to get the source of latest release, just clone the master branch from hzdr.gitlab:
```bash
git clone https://gitlab.hzdr.de/interpol/minterpy.git
```
To get the latest development version, just checkout to the dev branch:
```bash
git checkout dev
```
and it is recommended to always pull the latest commit:
```bash
git pull origin dev
```
Notice: in future it will **not** be allowed to push to dev or master. Please follow the instructions under [Working with git](#git)

## Virtual environments
It is stongly recommended to use virtual environments for development and productive runs. This encapsulates the package and all dependencies without messing up other python installations.  
#### Using [venv](https://docs.python.org/3/tutorial/venv.html) from the python standard library:  
1. Building the virtual environment:

    ```bash
    python -m venv <your_venv_name>
    ```

2. Activate the environment:

    ```bash
    source <your_venv_name>/bin/activate
    ```

3. To deactivate the virtual environment, just type  

    ```bash
    deactivate
    ```


#### Using [virtualenv](https://virtualenv.pypa.io/en/latest/):
1. Building the virtual environment:

    ```bash
    virtualenv <your_venv_name>
    ```

2. Activate the environment:

    ```bash
    source <your_venv_name>/bin/activate
    ```

3. To deactivate the virtual environment, just type  

    ```bash
    deactivate
    ```


#### Using [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv):
1. Building the virtual environment:

    ```bash
    pyenv virtualenv 3.8 <your_venv_name>
    ```

2. Activate the environment in the local directory:

    ```bash
    pyenv local <your_venv_name>
    ```

    This will create a hidden file `.python_version` containing a 'link' to the actual virtual environment managed by pyenv.


3. To 'deactivate' the virtual environment just remove this hidden file:

    ```bash
    rm .python_version
    ```


#### Using [conda](https://conda.io/projects/conda/en/latest/index.html):
1. create an environment `minterpy` with the help of [conda],
   ```bash
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```bash
   conda activate minterpy
   ```
   Maybe you need to `init` the conda env; just follow the instructions printed out or read the conda docs.  

3. To deactivate the conda environment, just hit:
    ```bash
    conda deactivate
    ```

## Dependency Management & Reproducibility (conda)

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.  

2. Create concrete dependencies as `environment.lock.yaml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n minterpy -f environment.lock.yaml
   ```
   For multi-OS development, consider using `--no-builds` during the export.  

3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```bash
   conda env update -f environment.lock.yaml --prune
   ```


## Installation
Installing `minterpy` it is recommended to use [pip](https://pip.pypa.io/en/stable/) (in the activated environment, see above):  
```bash
pip install [-e] .[all,dev,docs]
```
where the flag `-e` means the package is directly linked into the python site-packages.
The options `[all,dev,docs]` refer to the requirements defined in the `options.extras_require` section in `setup.cfg`.
One shall **not** use `python setup.py install`, since the file `setup.py` will not be present for every build of the package.  

### Troubleshooting: pytest with venv (not conda)
After installation, the restart of your virtual environment might be necessary, since the `pytest` command uses the `PYTHONPATH` which is not automatically changed to your venv.  

```bash
deactivate && source <your_venv_name>/bin/activate
```

or just run `hash -r` instead. This seems **not** to be an issuse for environments build with conda.

# 2. Testing
We use [pytest](https://docs.pytest.org/en/6.2.x/) to run the unit tests of `minterpy`. Unit tests itself shall always be placed into the directory `tests`. To run all tests, just type
```bash
pytest
```
into the terminal.  
:construction: :construction: Since the whole test environment needs a refactoring, we shall update this section with more detailed informations. :construction: :construction:

# 3. Documentation
## Install dependencies
Since building the docs has additional dependencies, we need to separately install them:

```bash
pip install .[docs]
```


## Building the documentation
We use [sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation of `minterpy`. To set the documentation in html, run the command
```bash
sphinx-build -M html docs docs/_build
```
It is also possible to build the documentation directly from the `make` file. For that, you need to navigate to the `docs` directory and run the `make` command:

```bash
cd docs
 make html
```

This generates the documentation, which is stored in `docs/_build` can be accessed through your favorite browser  
```bash
firefox docs/_build/html/index.html
```
It is also possible to automatically generate a PDF of the whole documentation, where `pdflatex` is used:
```bash
cd docs
 make latexpdf
```
The generated PDF will be stored (along with the latex files) in `docs/latex`.

## Design of the documentation
The files for generating the documentation are stored in the directory `docs` and will be configured with the file `docs/conf.py`. The main file of the documentation is `docs/index.rst`.
The documentation itself contains two different parts  
1. References section:  
    The reference to all exposed functions/classes will be generated exclusively from their docstrings, stored in the variable `func.__doc__`. Here we use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) docstring format (see [here](https://numpydoc.readthedocs.io/en/latest/example.html#example) for an example function, or [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) for a more complete example including classes). This shall be the only part of the documentation in the [reST](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). Be aware of the fact, that the docstring of an object might be used elsewhere. The files which are imported to the documentation (containing the generated docstring part) are stored in `docs/api`.      

2. Usage section:  
    Constains all tutorials/guides/explanations usually in the [markdown](https://www.markdownguide.org/getting-started/) format. The files for this section are stored in `docs/usage`. Each subsection shall be stored into a separate file, which can than be linked into the intro page (`docs/usage/intro`) of somewhere else.

# 4. Code Style
In order to ensure readability of the code, we shall follow a common code style for `minterpy`. The long-term goal shall be fulfilling the [PEP8](https://www.python.org/dev/peps/pep-0008/) regulations. For the build system, it is recommended to follow [PEP517](https://www.python.org/dev/peps/pep-0517/) and [PEP518](https://www.python.org/dev/peps/pep-0518/). However, since these requirements are very challenging, we shall at least agree on the code style enforced from [black](https://github.com/psf/black).  

## Pre-commit
In the development process, one can check the format using [pre-commit](https://pre-commit.com) and the hooks defined in `.pre-commit-config.yaml`. For instance running `black` for the whole `minterpy` code, just run
```bash
pre-commit run black --all-files
```
in the terminal. For now, it is recommended to run single hooks. However, to run all hooks defined in `.pre-commit-config.yaml`, just type
```bash
pre-commit run --all-files #DON'T DO THIS IF YOU DON'T KNOW WHAT HAPPENS
```
In the current state of the code, this shall be used with caution since it might change code in the manner, that it breaks (see below). However, further developments shall try to fulfil the full set of pre-commit-hooks.  
It is also possible, and also recommended for further developments, to run all pre-commit-hooks every time if you commit some changes to your branch. To enable this, just type:
```bash
pre-commit install
```
If you want to disable the pre-commit script, just type
```bash
pre-commit uninstall
```

### Current defined hooks
The following hooks are defined for now:
- [black](https://github.com/psf/black): straightforward code formatter; changes the code in order to fulfil the format.
- [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) a collection of widely used hooks, see their repo for more informations.
- [isort](https://github.com/PyCQA/isort): sorter for imports; changes the code !!!DO NOT RUN THIS: IT WILL BREAK THE CURRENT VERSION!!!
- [pyupgrade](https://github.com/asottile/pyupgrade) Automatically changes syntax from python2 to python3. It's nice if you use code from old post from stackoverflow ;-)
- [setup-cfg-fmt](https://github.com/asottile/setup-cfg-fmt): ensures a consistant format for the setup.cfg file.
- [flake8](https://github.com/pycqa/flake8) Runs a collection of hooks to ensure most of PEP8. The concrete checks are defined in the `setup.cfg[flake8]`.
- [mypy](https://github.com/python/mypy) Checks the static typing of `minterpy`. `mypy` itself is configured in the `setup.cfg[mypy-*]`.
- [check-manifest](https://github.com/mgedmin/check-manifest) Checks if the `MANIFEST.in` is in a proper state. This ensures proper builds for uploading the package to [PyPI](https://pypi.org). This is configured in `setup.cfg[check-manifest]`.

# 5. Code Development
For the code versioning of `minterpy` we only use [git](). The main repository for development is placed on the [hzdr.gitlab](https://gitlab.hzdr.de/interpol/minterpy), but releases will be mirrored into [github.com/casus](https://github.com/casus/). Furthermore, we shall consider uploading builds of `minterpy` to [PyPI](https://pypi.org) and [conda-forge](https://conda-forge.org) to make it easy to access.

## Versioning with git <span id="git"><span>
At least, up to the point we installed CI/CD on [hzdr.gitlab](https://gitlab.hzdr.de/interpol/minterpy), we loosely follow the structure of [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). This means for us, that we have three different types of branches:

1. `master` branch:  
    On this branch, only releases will be stored. This means, on this branch, one has only fully tested, documented and cleaned up code.
2. `dev` branch:  
    On this branch, the development version are stored. These shall pass all of the tests, which means on this branch, there is always a running version even if documentation and code is not in the 'release state'.  
3. `feature` branches:
    Shall always be build from the `dev` branch (not from `master`). Here all the features and actual code developments shall be happen.

Within this system, only on `feature` branches, one can freely push, change and merge. These are free for every developer. If the implementation of the 'feature' is finished, the `feature` branch can only be merged to the `dev` branch via a merge request. It is not possible to merge to `master` or even setup a request for that. The merging of `dev` to `master` will be done by the package maintainer following the release procedure of [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). Bugfixes on every branch will be managed separately with the relevant developer and the package maintainer (usually using `hotfix` branches, where patches will be implemented on).
