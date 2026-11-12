"""Run the toy cell and print its numbers, so the page's toy figures can be checked."""
import pathlib, runpy
runpy.run_path(str(pathlib.Path(__file__).resolve().parent.parent / "cells" / "toy_pyramid.py"))
