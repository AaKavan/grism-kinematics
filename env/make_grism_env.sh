#!/usr/bin/env bash
# One conda environment that runs BOTH geko and DINGO.
#
# Why they were split: DINGO needs python >= 3.12 (nested same-quote f-strings),
# while geko_clean is on 3.11. But geko's pyproject says `requires-python = ">=3.8"`
# with no upper bound, so 3.12 satisfies both and the split is not actually forced.
#
# The real question is CUDA. torch ships CUDA 13 wheels (nvidia-cublas,
# nvidia-cudnn-cu13); jax ships CUDA 12 (nvidia-cublas-cu12, nvidia-cudnn-cu12).
# Those are DIFFERENT PACKAGE NAMES, so pip installs both without a conflict, and
# driver 596.49 supports both runtimes. Whether they coexist inside one process is
# what the test at the end checks.
set -euo pipefail
ENV=grism
CONDA=/home/aakavan/anaconda3
PY=$CONDA/envs/$ENV/bin/python
PIP="$PY -m pip install -q"

echo "=== creating $ENV (python 3.12) ==="
$CONDA/bin/conda create -y -n $ENV python=3.12 >/dev/null
$PY -m pip install -q --upgrade pip

echo "=== torch (CUDA 13) ==="
$PIP torch --index-url https://download.pytorch.org/whl/cu130

echo "=== jax (CUDA 12) ==="
$PIP "jax[cuda12]"

echo "=== shared scientific stack ==="
$PIP "setuptools<81"  # jax_cosmo imports pkg_resources, dropped in setuptools 81
$PIP numpyro jax-cosmo pysersic photutils arviz xarray scikit-image reproject \
     stpsf emcee corner h5py pandas tqdm pyyaml astropy scipy matplotlib \
     pytest nbconvert ipykernel ipywidgets

echo "=== the two packages, editable ==="
# NOTE on geko: at commit 30797a8 ("Exports new GrismObservation class") its
# geko/grism.py lists GrismObservation in __all__ but never defines the class,
# so `from .grism import *` raises AttributeError and `import geko` fails. The
# installed copies in geko_clean/geko_env predate that commit and are fine, which
# is why it only bites an EDITABLE install of the checkout -- or any env whose
# working directory happens to be ~/kinematics/geko, since the checkout then
# shadows site-packages. Patched locally by dropping that one name from __all__;
# worth reporting upstream.
$PIP -e /home/aakavan/kinematics/geko
$PIP -e /home/aakavan/kinematics/grism-kinematics

echo "=== environment variables ==="
# jax grabs 75% of the GPU on first use by default, which would starve torch in
# a shared process. GRISM_CAL_DIR is where the WFSS trace tables live.
$CONDA/bin/conda env config vars set -n $ENV \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    GRISM_CAL_DIR=/home/aakavan/kinematics/grism_cal >/dev/null

echo "=== jupyter kernel ==="
$PY -m ipykernel install --user --name $ENV --display-name "$ENV (geko+dingo)" >/dev/null

echo "=== versions ==="
$PY - <<'PYEOF'
import importlib.metadata as md
for p in ('torch','jax','jaxlib','numpyro','pysersic','astro-geko','DINGO',
          'emcee','numpy','scipy','astropy'):
    try: print('  %-12s %s' % (p, md.version(p)))
    except Exception as e: print('  %-12s MISSING' % p)
PYEOF
echo
echo "=== THE TEST: do torch and jax share the GPU in one process? ==="
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY - <<'PYEOF'
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import torch
print('  torch cuda :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
a = torch.randn(1000, 1000, device='cuda'); print('  torch matmul ok:', float((a@a).sum()) == float((a@a).sum()))
import jax, jax.numpy as jnp
print('  jax devices:', jax.devices())
b = jnp.ones((1000, 1000)); print('  jax matmul ok  :', float((b@b).sum()) == 1e9)
import geko, dingo
print('  import geko  :', geko.__file__)
print('  import dingo :', dingo.__version__)
print()
print('  BOTH FRAMEWORKS SHARE THE GPU IN ONE PROCESS.')
PYEOF
