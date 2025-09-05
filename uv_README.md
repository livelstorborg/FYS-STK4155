# Project Name

This project uses [**uv**](https://astral.sh/uv), a next-generation Python package and project manager by Astral,
for dependency management, builds, and running scripts.

---

## 🚀 Getting Started

### 1. Install **uv**

If you don’t already have **uv**, install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with Homebrew (macOS/Linux):

```bash
brew install uv
```

Verify installation:

```bash
uv --version
```

---

### 2. Set up the project

In order to set up the project, enter the project directory after cloning it:

```bash
cd project1
```

Installing the dependencies is then as simple as:

```bash
uv sync
```

This will create (or update) a `.venv/` inside your project directory with all dependencies defined in `pyproject.toml` / `uv.lock`. Make sure to open VSCode in the project directory, so that you are able to get correct code highlighting and linting.

---

### 3. Run the project

The projects are written as installable packages, such that development happens within `src/`, which can be imported as a package in separate files for analysis and plotting. These files are then ran as


```bash
uv run uv_example.py
```

We follow the same pattern when running tests, or other scripts:

```bash
uv run pytest
uv run ruff check --fix
```

---

### 4. Adding dependencies

To add a new package:

```bash
uv add requests
```

To remove one:

```bash
uv remove requests
```

All changes will update your `uv.lock` file to ensure reproducible environments.

If you wish to add packages which the code is not directly dependent on, e.g., `pytest` or `ipython`, please use

```bash
uv add --dev pytest ipython
```

---

### 6. Updating dependencies

To upgrade everything:

```bash
uv sync --upgrade
```

To upgrade a single package:

```bash
uv add --upgrade requests
```

---

## 📂 Project Structure

```
.
├── pyproject.toml   # Project metadata and dependencies
├── uv.lock          # Lockfile for reproducible builds
├── src/             # Source code
├── tests/           # Tests
└── README.md        # This file
```

---

## 📖 Documentation

- [uv documentation](https://docs.astral.sh/uv/)  
- [PEP 621](https://peps.python.org/pep-0621/) – project metadata standard used by `pyproject.toml`

---

⚡ With **uv**, you don’t need `pip`, `pip-tools`, or `venv` directly — everything is managed in one tool.
