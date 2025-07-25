# Contributing to This Project

## How to Contribute

1. **Fork the repository**, if you are not a collaborator, and create your branch from `main`.
2. **Install the project**  by running `poetry install`.
3. **Set up pre-commit hooks** (see below).
4. **Make your changes** with clear, descriptive commit messages.
5. **Document your changes** by adding docstrings to the new functions and classes
following the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format.
5. **Test your changes** to ensure nothing is broken.
6. **Open a Pull Request** with a clear description of your changes.

## Code Style
- Follow existing code style and conventions.
- Write clear, concise code and comments.
- The code style is enforced by [ruff](https://github.com/astral-sh/ruff).
See `pyproject.toml` for the configuration.
- The code is staticcaly type checked by [mypy](https://mypy.readthedocs.io/en/stable/).
See `pyproject.toml` for the configuration.

## Pre-commit Hooks
We use [pre-commit](https://pre-commit.com/) to maintain code quality.
Please install and run the hooks before committing:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-commit
```

This will automatically check your code for formatting and linting issues before each commit.

Run the hooks manually by running `pre-commit run --all-files`.


## commitlint

The commit messages are being linted using the [commitlint](https://commitlint.js.org/) hook
at the [commit-msg](https://pre-commit.com/#commit-msg) stage.
The schema for the commit strings is taken from the
[conventional commits](https://www.conventionalcommits.org) specification.
The rules of this schema can be found
[here](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional).


## Issues
- Search for existing issues before opening a new one.
- Provide detailed information when reporting bugs or suggesting features.

Thank you for helping improve this project!
