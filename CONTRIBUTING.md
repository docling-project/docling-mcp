## Contributing In General

Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

For more details on the contributing guidelines head to the Docling Project [community repository](https://github.com/docling-project/community).

## Developing

### Clone the project

Clone this project on your local machine with `git`. For instance, if using an SSH key, run:

```bash
git clone git@github.com:docling-project/docling-mcp.git
```

Ensure that your user name and email are properly set:

```bash
git config list
```

### Usage of uv

We use [uv](https://docs.astral.sh/uv/) as package and project manager.

#### Installation

To install `uv`, check the documentation on [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

#### Create an environment and sync it

You can use the `uv sync` to create a project virtual environment (if it does not already exist) and sync
the project's dependencies with the environment.

```bash
uv sync
```

#### Use a specific Python version (optional)

If you need to work with a specific version of Python, you can create a new virtual environment for that version
and run the sync command:

```bash
uv venv --python 3.12
uv sync
```

More detailed options are described on the [Using Python environments](https://docs.astral.sh/uv/pip/environments/) documentation.

#### Add a new dependency

Simply use the `uv add` command. The `pyproject.toml` and `uv.lock` files will be updated.

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```

### Code sytle guidelines

We use the following tools to enforce code style:

- [Ruff](https://docs.astral.sh/ruff/), as linter and code formatter
- [MyPy](https://mypy.readthedocs.io), as static type checker

A set of styling checks, as well as regression tests, are defined and managed through the [pre-commit](https://pre-commit.com/) framework. To ensure that those scripts run automatically before a commit is finalized, install `pre-commit` on your local repository:

```bash
uv run pre-commit install
```

To run the checks on-demand, type:

```bash
uv run pre-commit run --all-files
```