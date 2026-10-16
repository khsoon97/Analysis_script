# Opal software and availability

Opal analysis code and data are not included in this repository. Further methodological details and code may be requested from the corresponding author, subject to the applicable sharing conditions.

The reviewed Opal analysis/plotting scripts directly import the following third-party packages:

- NumPy
- pandas
- SciPy
- scikit-learn (`sklearn`)
- statsmodels
- Matplotlib

This list was derived from imports in the reviewed Opal scripts. It is not a list of every package installed in the workstation environment and is not a historical version lockfile. In particular, no `lifelines` or `python-pptx` usage is asserted: survival plotting used project-local implementation code and Matplotlib, and presentation assembly is distinct from statistical analysis. Python standard-library modules are omitted.
