## Contributing in general

Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a [pull request](https://github.com/docling-project/docling-mcp/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue tracker](https://github.com/docling-project/docling-mcp/issues).
Before embarking on a more ambitious contribution, please quickly [get in touch](#communication) with us.

For general questions or support requests, please refer to the [discussion section](https://github.com/docling-project/docling/discussions).

**Note: We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in backlog for a long time, or
cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please [raise an issue](https://github.com/docling-project/docling-mcp/issues)
before sending a pull request so the feature can be discussed. This is to avoid
you wasting your valuable time working on a feature that the project developers
are not interested in accepting into the code base.

### Fixing bugs

If you would like to fix a bug, please [raise an issue](https://github.com/docling-project/docling-mcp/issues) before sending a
pull request so it can be tracked.

### Merge approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. A change requires LGTMs from two of the
maintainers of each component affected.

For a list of the maintainers, see the [MAINTAINERS.md](MAINTAINERS.md) page.

## Legal

Each source file must include a license header for the MIT
Software. Using the SPDX format is the simplest approach.
e.g.

```text
/*
Copyright IBM Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/
```

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) - that the Linux® Kernel [community](https://elinux.org/Developer_Certificate_Of_Origin)
uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the
submitter accepts the DCO:

```text
Signed-off-by: John Doe <john.doe@example.com>
```

You can include this automatically when you commit a change to your
local git repository using the following command:

```text
git commit -s
```

## Communication

Please feel free to connect with us using the [discussion section](https://github.com/docling-project/docling/discussions).

## Developing

### Clone the project

Clone this project on your local machine with `git`. For instance, if using an SSH key, run:

```bash
git clone git@github.com:docling-project/docling-sdg.git
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

## Code sytle guidelines

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