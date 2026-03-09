# Bundled checkpoints — ship with the package

Put the four model files **here** (in this directory) so they are included when you build and publish to PyPI:

- `best_model_phys_model0.pt`
- `best_model_phot_model0.pt`
- `scaler_phys_model0.pkl`
- `scaler_phot_model0.pkl`

Then run `python -m build` and `twine upload dist/*` (or use GitHub Release). Users who `pip install cluster-completeness-pipeline` will get these files inside the package and can run `predict()` with no extra download.
