# DFTToolkit
[![Coverage Status](https://coveralls.io/repos/github/maurergroup/dfttoolkit/badge.svg?branch=main)](https://coveralls.io/github/maurergroup/dfttoolkit?branch=main)

## Installation

This package is available to download on PyPi. To install, simply execute 

``` shell
pip install dfttoolkit 
```

in a python environment, and modules will be available to install and use.

## Usage

The full API is documented at <https://maurergroup.github.io/dfttoolkit/>

Tutorials are currently available as Jupyter notebooks in [tutorials](tutorials/), however in future this will move into the wiki.

## Contributing

In order to contribute to the code, please create a new branch, setup a draft pull request, and commit to that branch. Add your commits, rebase to the main branch, and once the CI has passed, request a review, and upon approval, perform a squash merge with the main branch. If there are only several small commits, a fast-forward merge may be performed.

### Specification 

This repository is almost completely written in Python in an object-oriented style. The source code is contained in [dfttoolkit](dfttoolkit/), which consists of a variety of modules. Each module refers to the types of jobs it does, with one class in each module containing generic routines for all classes in the module. Generally, all classes in the module inherit this class. For example, [output.py](dfttoolkit/output.py) currently contains the `Output`, `AimsOutput`, and `ELSIOutput` classes. `Output` contains generic routines for both `AimsOutput` and `ELSIOutput`. `AimsOutput` is then only responsible for parsing `aims.out` files. If a new parser is needed for a different DFT code, then a new class should be created which inherits `Output`. All parsers across different modules inherit `BaseParser` in [base_parser.py](dffttoolkit/base_parser.py), which contains generic routines for all parsers. 

For other routines not specific to particular classes, there exists the [dfttoolkit/utils/](dfttoolkit/utils/) directory. Some useful examples of modules here are 
- [exceptions.py](dfttoolkit/utils/exceptions.py)
  Custom exceptions.
- [math_utils.py](dfttoolkit/utils/math_utils.py)
  General maths functions.
- [periodic_table.py](dfttoolkit/utils/periodic_table.py)
  Functions to retrieve physical properties of elements. The entire periodic table is contained within a yaml file.
- [units.py](dfttoolkit/utils/units.py)
  Commonly used physical units
  
#### Type Hints

[PEP 484](https://peps.python.org/pep-0484/) introduced the standard for type hinting in python, which was implemented in [version 3.5](https://docs.python.org/3/library/typing.html). It has been made extensive use of in this project, and we also ask that further contributions also include type hinting for function arguments and return values.

#### Docstrings 

It was decided to use the numpydoc style for function, class, and module docstrings. We also ask that you include docstrings in this style for any new added routines. This is because Sphinx automatically parses these docstrings to build the API documentation. Please refer to this [style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for how to write docstrings in this format.

### Testing 

[Pytest](https://docs.pytest.org/en/stable/) has been used to setup unit tests. If contributing new functionality, please also write a unit test. Tests should be grouped in an analogous manner to how the routines that they test are represented in the source code. For instance, for the `AimsOutput` class referenced earlier, the tests are grouped as functions within a class called `TestAimsOutput`. Fixtures should be defined as either class- or module-scoped depending if the original routine is contained within a class or just as functions in a module. Ideally, fixtures should be used for all tests in a module.

Currently, there exist 10 FHI-aims calculations consisting of `control.in`, `geometry.in`, and `aims.out` files, which are used as fixtures for tests. If new fixtures are needed to be added, they will also run for all other existing tests, so ensure that they pass for the existing tests too. It is important that any tests added are not computationally intensive and only operate on the smallest systems, otherwise they will take prohibitively long to run.

#### Running Standard Tests

To run the tests, simply call `pytest`, using either the `poetry run` or `poetry shell` invocations defined in [Installing](#installing). Running Pytest can be done with a variety of options and arguments, which can be accessed by calling `pytest --help` in the poetry shell.

#### Running Tests with a Custom FHI-aims Binary

It is also possible to generate `aims.out` files using an FHI-aims binary, which the tests can be run on. This is useful to ensure that dfttoolkit works correctly with output files generated with different versions of FHI-aims, which may have different output formats and keywords. 

To do this, call

``` shell
pytest -s --run-aims
```

This will automatically prompt the user for the location of the FHI-aims binary using `$EDITOR`. Then, it will create a copy of the aims fixture directory but without the `aims.out` files, and run them using the custom FHI-aims binary. The tests will then run on these new generated files. As `dfttoolkit` stores the FHI-aims binary location so the user isn't prompted for its location every time `pytest` is run like this, `--run-aims` takes an optional argument `change_bin`, which will prompt the user again for the binary location and re-run the custom tests, regardless of whether they have been run before or not. 

Currently, it is necessary to also run this with `-s` in order to capture the STDOUT to show the prompt. Also note that this will likely take some time to run this calculation, and it runs with 4 threads by default using `mpirun`. Finally, it is almost certain that several tests will fail, especially those relating to timing of calculations

### Installing

Clone this repository by executing 

``` shell
git clone https://github.com/maurergroup/dfttoolkit
```

The build processes is then automated using Poetry. To install poetry, it is possible to set up a virtual environment and install everything in that virtual environment. However, the recommended installation method is using [pipx](https://pipx.pypa.io/stable/), which automatically sets up a virtual environment for each installed library.

```shell
pipx install poetry
```

Before installing dfttoolkit with Poetry, specify the python environment. For example,

```shell
poetry env use 3.12
```

If you only wish to install the base dependencies, use the command

```shell
poetry install
```

However, there are also two other dependency groups, which install additional dependencies for development and building the documentation. Use the above command but with the additional flags `--with=dev` and `--with=docs` respectively. These dependency groups can each be installed independently or together.

### Adding Dependencies 

To add a general dependency, specify it through Poetry. For example, to add a version of numpy less than v2.1:

``` shell
poetry add numpy@<2.1
```

To add a dependency to one of the aforementioned dependency groups

``` shell
poetry add interrogate --group=dev
```

### Running

In order to use the installed package, it needs to be run in the poetry environment. There are two ways of doing this: 
1. Run your command as usual whilst prefixing `poetry run`, like so
    ``` shell
    poetry run <args> -- <command> <args>
    ```

2. Enter the Poetry shell environment using the command `poetry shell`. 
   This opens a subshell with the poetry environment loaded. Commands can then be executed as normal. The advantages of this are that changes can be made to the code, which are then automatically loaded into the subshell, so it is unnecessary to reload the subshell every time a change is made to the code. It can be exited as you would exit a normal subshell.
   
### Further Information

For more information on how to use Poetry, please refer to its [documentation](https://python-poetry.org/docs/).
