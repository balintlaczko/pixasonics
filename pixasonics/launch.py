import os
import subprocess
import importlib.util

def get_package_path():
    """Dynamically finds the installed package directory without direct import."""
    spec = importlib.util.find_spec("pixasonics")
    if spec and spec.origin:
        return os.path.dirname(spec.origin)
    return None

def launch_notebook(notebook_name="pixasonics_tutorial.ipynb"):
    """Finds and opens a Jupyter Notebook from within the package."""
    package_dir = get_package_path()
    if not package_dir:
        print("Could not locate package installation.")
        return

    notebook_path = os.path.join(package_dir, notebook_name)

    if not os.path.exists(notebook_path):
        print(f"Notebook '{notebook_name}' not found in {package_dir}")
        return

    print(f"Launching notebook: {notebook_path}")
    subprocess.run(["jupyter", "notebook", notebook_path], check=True)
