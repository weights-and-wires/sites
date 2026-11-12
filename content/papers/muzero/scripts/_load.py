"""Load an exercise solution by folder name, so each number has one source."""

import importlib.util
import pathlib

EXERCISES = pathlib.Path(__file__).resolve().parents[1] / "exercises"


def load(folder):
    spec = importlib.util.spec_from_file_location("sol_" + folder.replace("-", "_"), EXERCISES / folder / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
