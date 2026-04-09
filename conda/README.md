# Conda Environment Exports

These files capture the local `distserve` Conda environment in Git-friendly form.

- `distserve.full.yml`: full Conda environment export
- `distserve.from-history.yml`: minimal environment history export
- `distserve.explicit.txt`: explicit Conda package URLs for exact recreation
- `distserve.pip-freeze.txt`: pip-installed packages in the same environment

Recommended restore order:

1. Try `conda env create -f conda/distserve.full.yml`
2. If exact Conda recreation is required, use `conda create -n distserve --file conda/distserve.explicit.txt`
3. Use `conda/distserve.pip-freeze.txt` as the pip fallback/reference

The full environment directory itself is not stored in Git.
