'''DINGO -- joint inference of galaxy kinematics from R+C grism observations.

The working three-file model in ``dingo.three_file`` uses one arctangent
velocity field to rectify coadded R and C emission-line images and compare
them. Intrinsic dispersion is fixed to zero; no gas-light profile is imposed.
Imaging can supply an inclination prior, and ``dingo.manual_start`` supports
sampled or fixed inclination. ``dingo.workflow`` provides explicit per-source
run handles for searching, sampling, review and export. Existing single-source
notebooks are preserved under ``notebooks/legacy``; ``workflow_demo.ipynb`` in
the repository root demonstrates the Python workflow interface.

Complementary dispersion directions help distinguish morphology and velocity,
but do not guarantee identifiable or prior-independent parameters. Inspect
residuals and chain diagnostics before interpreting conditional uncertainties.

``dingo.rotation`` and the native-exposure modules retain earlier experiments
with a fitted gas-emission map and instrumental forward model. Those modules,
the original ``fitting.KinematicsFitter`` and ``dingo.run`` remain available
for reproduction. The README and dated reports describe their validation limits.
'''

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # setuptools_scm is a BUILD dependency and is not importable at runtime, so
    # calling get_version() here always failed and silently reported "unknown".
    # The version baked in at install time is what we want.
    __version__ = _version('DINGO')
except PackageNotFoundError:          # running from a source tree, not installed
    __version__ = '0.0.0+unknown'
