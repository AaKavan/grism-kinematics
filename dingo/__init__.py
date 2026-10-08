'''DINGO -- Dispersion-based INference for Galaxy Observations.

Fits galaxy kinematics to JWST/NIRCam WFSS data taken in BOTH grism dispersion
directions at once. The R and C grisms disperse along perpendicular detector
axes, so the morphology-kinematics degeneracy along the dispersion axis -- the
one a single-direction fit has to break with a light-profile prior -- is broken
by the data instead.

The statistic is unusual and worth stating plainly: DINGO does not forward-model
a galaxy. It rectifies each observed grism back to the direct-image frame through
the velocity field and asks the two reconstructions to agree
(`KinematicsFitter.loss`). There is no model image and no Sersic prior, so every
parameter it reports is a measurement rather than a restatement of a prior.

Module layout, in dependency order:

    grism       NIRCam WFSS trace/dispersion calibration I/O (numpy)
    utils       FFT convolution, and the differentiable grism trace closure
    galaxy      Sersic and point-source surface-brightness models (torch)
    kinematics  the physics: velocity field, Doppler mapping, rectification
    fitting     config loading, staged Adam, emcee sampling, noise model
    plot        matplotlib helpers
    catalogs    catalogue lookup and config generation
'''

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # setuptools_scm is a BUILD dependency and is not importable at runtime, so
    # calling get_version() here always failed and silently reported "unknown".
    # The version baked in at install time is what we want.
    __version__ = _version('DINGO')
except PackageNotFoundError:          # running from a source tree, not installed
    __version__ = '0.0.0+unknown'
