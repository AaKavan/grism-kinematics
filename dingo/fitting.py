import torch
import logging
import json
import yaml
from dataclasses import dataclass
from astropy.io import fits
import numpy as np
from typing import Dict, Any, Optional, AnyStr
from abc import ABC, abstractmethod
from pprint import pprint
import os

from . import kinematics, utils, grism, galaxy

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    LOG.addHandler(console_handler)

def log_call(func):
    def wrapper(*args, **kwargs):
        # print(f'Calling "{func.__name__}"')
        LOG.info(f'Calling "{func.__name__}"')
        return func(*args, **kwargs)
    return wrapper

#%% --------------------------------------------------------------------------
# helper classes and functions
# ----------------------------------------------------------------------------

def ensure_float_like(x):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float) if x.numel() == 1 else x.to(dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif isinstance(x, (float, int)):
        return float(x)
    elif isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            raise ValueError(f"Cannot convert string '{x}' to float.")
    else:
        raise TypeError(f"Unsupported type {type(x)} for float-like conversion.")

@dataclass
class FitParamConfig:
    name: str
    tensor: torch.Tensor
    lr: float
    min: float
    max: float

    def __init__(
        self,
        name: str,
        value: Any,
        lr: float,
        min: Any,
        max: Any,
        fit: bool,
        device: torch.device = 'cpu'
    ):
        self.name = name
        try:
            self.tensor = torch.tensor(ensure_float_like(value), requires_grad=fit, device=device)
            self.min = torch.tensor(ensure_float_like(min), dtype=self.tensor.dtype, device=device)
            self.max = torch.tensor(ensure_float_like(max), dtype=self.tensor.dtype, device=device)
        except TypeError:
            self.tensor = value
            self.min = min
            self.max = max
        self.lr = lr
        # Store the “real” fit‐flag in a private variable:
        self._fit = bool(fit)

    def __repr__(self):
        message = (
            f'FitParamConfig('
            f'name={self.name!r}, value={self.value}, lr={self.lr}, '
            f'fit={self._fit}'
        )
        if isinstance(self.min, float) and isinstance(self.max, float):
            message += 'min={self.min:.2e}, max={self.max:.2e}'
        return message

    def __str__(self):
        return self.__repr__()


    @property
    def value(self):
        '''
        Always return a NumPy array (or scalar) extracted from self.tensor.
        '''
        try:
            return self.tensor.cpu().detach().numpy()
        except AttributeError:
            return repr(self.tensor)

    @property
    def fit(self) -> bool:
        return self._fit

    @fit.setter
    def fit(self, _fit: bool):
        '''
        Whenever someone does "obj.fit = True/False", we also
        update tensor.requires_grad automatically.
        '''
        self._fit = bool(_fit)
        self.tensor.requires_grad = self._fit
    
def build_param_config_dict_with_alias(
    raw_dict: Dict[str, Any],
    default_cfgs: Any,
    overrides_list: Any,
    prefix: str, 
    all_cfgs: dict = None,
    allowed_keys_extra: dict = None,
    device: torch.device = 'cpu'
) -> list:
    '''
    Build a list of parameter configuration dictionaries for each fitting strategy.
    raw_dict: parameter values from config.
    default_cfgs: list of default cfg dicts for each stage.
    overrides_list: list of override dicts for each stage.
    prefix: key prefix (e.g., 'velocity', 'image.R').
    '''

    if all_cfgs is None:
        all_cfgs = [{}]*len(default_cfgs)  # Create a fresh dict if not provided

    # --- ensure inputs are lists ----------------------------------------------
    if not isinstance(default_cfgs, list):
        default_cfgs = [default_cfgs]
    if not isinstance(overrides_list, list):
        overrides_list = [overrides_list]

    # --- EXPAND grouped override keys -----------------------------------------
    # allow keys like 'image.R.dx, image.R.dy, image.C.dx' → apply same settings
    for i, ov in enumerate(overrides_list):
        exp = {}
        for key, settings in ov.items():
            for sub in [k.strip() for k in key.split(',')]:
                exp[sub] = settings
        overrides_list[i] = exp  # in-place update!

    # --- allowed keys & build configs -----------------------------------------
    allowed_keys = {
        'velocity': {'V_rot', 'R_v', 'x0_v', 'y0_v', 'theta_v', 'inc_v'},
        'image': {'dx', 'dy'},
        'sersic': {'I_e', 'R_e', 'n', 'x0', 'y0', 'q', 'theta'}, 
        'psf': {'I_psf', 'x_psf', 'y_psf'}, 
        'direct': {'dx', 'dy', 'wt', 'zp'}, # TODO: change image to grism and alas to direct?? TODO: change 'wt' to 'zp'
        'psfs': {'scale', 'zp'} # TODO: change image to grism and alas to direct??
        # TODO: move this part out
    }
    if allowed_keys_extra: 
        allowed_keys.update(allowed_keys_extra)

    cfgs_list = []
    # process each batch of fitting
    for i, (default_cfg, all_cfg) in enumerate(zip(default_cfgs, all_cfgs)):
        overrides = overrides_list[i]  # still references the outer list element
        cfgs = {}
        alias_map = {}
        for name, val in raw_dict.items():
            full_key = f'{prefix}.{name}'
            # NOTE: full_key may be prefix.name or prefix.cid.name. In the latter case prefix.cid is treated as a prefix
            if name not in allowed_keys.get(prefix.split('.')[0], set()):
                continue
            if isinstance(val, str):
                alias_map[full_key] = val
            else:
                oc = overrides.pop(full_key, {})  # now catches any split keys; mutate overrides to make a overrides counter
                cfgs[full_key] = FitParamConfig(
                    name=full_key,
                    value=val,
                    lr=oc.get('lr', default_cfg['lr']),
                    min=oc.get('min', default_cfg['min']),
                    max=oc.get('max', default_cfg['max']),
                    fit=oc.get('fit', default_cfg['fit']),
                    device=device
                )
        # apply any string aliases
        all_cfg.update(cfgs) #joint known keys as alias candidates
        for key, alias in alias_map.items():
            if alias in all_cfg:
                cfgs[key] = all_cfg[alias]
            else:
                raise ValueError(f'Alias {alias} not found for {key}')
        cfgs_list.append(cfgs)

    return cfgs_list

def _extract_tensors(cfgs: Dict[str, FitParamConfig], device: torch.device = 'cpu'):
    seen = set()
    tensors = []
    clamp_list = []
    for cfg in cfgs.values():
        if not cfg.fit:
            continue
        if id(cfg.tensor) in seen:
            continue
        seen.add(id(cfg.tensor))
        tensors.append({'params': [cfg.tensor], 'lr': cfg.lr})
        clamp_list.append((cfg.tensor, cfg.min, cfg.max))
    return tensors, clamp_list

def load_fits_data(path_hdu):
    '''the slice syntax is capable for both hdu index and hdu name'''
    path, hdu = path_hdu
    with fits.open(path) as hdulist:
        return np.array(hdulist[hdu].data, dtype=np.float32)

#%% --------------------------------------------------------------------------
# MCMC helpers (used by BaseFitter.fit_MCMC)
# ----------------------------------------------------------------------------

def _import_emcee():
    '''
    emcee is an optional dependency: the gradient-fitting path must keep working
    without it, so it is imported lazily rather than at module scope.
    '''
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            'fit_MCMC() needs the optional dependency `emcee`. Install it with '
            '`pip install emcee` or `conda install -c conda-forge emcee`.'
        ) from exc
    return emcee

def collect_free_scalar_params(cfgs: Dict[str, FitParamConfig]):
    '''
    Name-preserving companion to `_extract_tensors`, for samplers that need a flat
    theta vector instead of optimizer parameter groups.

    Keeps only *scalar* fittable tensors. Dropped, each with a log line:
      - cfg.fit is False
      - cfg.tensor is not a Tensor (e.g. 'image.R.forward_model' holds a callable)
      - cfg.tensor.numel() != 1 (e.g. 'result.emline_model' is 81x81)
    Aliased keys share one FitParamConfig, hence one tensor, so they collapse onto
    their first occurrence -- deduplicated by id(cfg.tensor) exactly as
    `_extract_tensors` does. Writing the shared tensor keeps every alias consistent.

    Returns:
        names:   list of full dotted keys, names[i] <-> cfgs[i]
        cfgs:    list of the corresponding FitParamConfig objects
        aliases: {kept_key: [dropped alias keys]}
    '''
    seen = {}
    names = []
    out = []
    aliases = {}
    for key, cfg in cfgs.items():
        if not cfg.fit:
            continue
        if not isinstance(cfg.tensor, torch.Tensor):
            LOG.debug(f'[MCMC] skipping non-tensor parameter {key}')
            continue
        if cfg.tensor.numel() != 1:
            LOG.info(f'[MCMC] skipping array parameter {key} ({cfg.tensor.numel()} elements)')
            continue
        tid = id(cfg.tensor)
        if tid in seen:
            aliases.setdefault(seen[tid], []).append(key)
            continue
        seen[tid] = key
        names.append(key)
        out.append(cfg)
    return names, out, aliases

def estimate_background_sigma(image, source_mask=None, sigma=3.0, maxiters=10):
    '''
    Sigma-clipped rms of the off-source background of a 2D image.

    In a DINGO grism cutout the emission line covers a small minority of pixels, so
    iterative sigma clipping rejects the trace on its own and no explicit mask is
    needed. Pass `source_mask` (True = exclude) when the trace fills a larger
    fraction of the cutout.

    Accepts a torch.Tensor or an ndarray; returns a float.
    '''
    from astropy.stats import sigma_clipped_stats
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image, dtype=np.float64)
    good = np.isfinite(image)
    if source_mask is not None:
        if isinstance(source_mask, torch.Tensor):
            source_mask = source_mask.detach().cpu().numpy()
        good &= ~np.asarray(source_mask, dtype=bool)
    _, _, std = sigma_clipped_stats(image[good], sigma=sigma, maxiters=maxiters)
    return float(std)

#%% --------------------------------------------------------------------------
# generic base class for all fitters
# ----------------------------------------------------------------------------

class BaseFitter(ABC):

    def __init__(self, config_path: AnyStr, device=None):
        # ─────────────────────────────
        # Load configuration & metadata
        # ─────────────────────────────
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config_path = config_path
        self.config = config
        if not device: 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # ─────────────────────────────
        # Summary / Metadata
        # ─────────────────────────────
        s = config['summary']
        self.name   = s['name']
        self.ra     = s['ra']
        self.dec    = s['dec']
        self.z      = s.get('z', None) # z 对 ImagesFitter 来说不再必须；KinematicsFitter 仍然会用到
        self.filter = s['filter']
        self.mode   = s['mode']
        self.r_fit = s.get('r_fit', None)
        if self.r_fit is not None:
            LOG.warning(
                f"[{self.__class__.__name__}] summary.r_fit is deprecated under (x,y) origin=(0,0). "
                "Prefer explicit cutouts/centers in the config."
            )


        # ─────────────────────────────
        # 坐标单位：pix / arcsec / kpc
        # ─────────────────────────────
        self.unit = s.get('unit', 'pix')
        if self.unit not in ('pix', 'arcsec', 'kpc'):
            LOG.warning(
                f"[{self.__class__.__name__}] Unknown summary.unit={self.unit!r}; "
                "fallback to 'pix'."
            )
            self.unit = 'pix'
        LOG.info(f"[{self.__class__.__name__}] using coordinate unit = {self.unit}")

        # ─────────────────────────────
        # Fitting stage state
        # ─────────────────────────────
        self.current_stage = 0
        # container for subclass-defined param configs
        self.param_config_lists: Dict[str, list] = {}
        '''
        NOTE: Flat list of fitting parameters. It should looks like:
            {'im0': [{'direct.f356w.im0.dx': FitParamConfig(...),
                      'direct.f356w.im0.dy': FitParamConfig(...)},
                     {'direct.f356w.im0.dx': FitParamConfig(...),
                      'direct.f356w.im0.dy': FitParamConfig(...)}],
             'p0': [{'psf.p0.I_psf': FitParamConfig(...),
                     'psf.p0.x_psf': FitParamConfig(...),
                     'psf.p0.y_psf': FitParamConfig(...)},
                    {'psf.p0.I_psf': FitParamConfig(...),
                     'psf.p0.x_psf': FitParamConfig(...),
                     'psf.p0.y_psf': FitParamConfig(...)}], 
             ...}
        '''

        # ─────────────────────────────
        # Subclass-specific setup:
        # load images, models, build param_config_lists, grid, etc.
        # ─────────────────────────────
        self._setup_data()

        # setup completeness check
        if not self.param_config_lists:
            LOG.warning(
                f'[{self.__class__.__name__}] ⚠️ `param_config_lists` is empty. '
                'You should assign parameter config lists in `_setup_data()` or subclass `__init__`.'
            )

        # ─────────────────────────────
        # Assign stage-0 configs to attributes
        # ─────────────────────────────
        self._assign_cfgs_for_stage(0)

    @abstractmethod
    def _setup_data(self):
        '''
        Subclasses must:
          - load any image data into attributes
          - load forward models if needed
          - build self.param_config_lists, a dict mapping names
            to lists of cfg-dicts (one per fitting stage)
        '''
        pass

    def _assign_cfgs_for_stage(self, stage: int):
        '''
        Shortcut attributes for the current stage's config dicts.
        E.g. self.velocity_cfg = self.param_config_lists['velocity'][stage]
        Example generated attributes: self.im0_cfg, self.s0_cfg, self.p0_cfg
        '''
        for name, cfg_list in self.param_config_lists.items():
            setattr(self, f'{name}_cfg', cfg_list[stage])

    def _get_model_params(self, key): 
        params = {
            k.split('.')[-1]: v.tensor 
            for k,v in getattr(self, f'{key}_cfg').items()
        }
        return params

    def _reset_state(self):
        '''
        Subclasses may override to reset any per-stage state
        (e.g. starting positions) before each fit_gradient run.
        By default, do nothing.
        '''
        pass

    @abstractmethod
    def loss(self):
        '''
        Compute and return a scalar loss Tensor.
        Subclasses implement, using self.*_cfg and any loaded data.
        '''
        pass

    def _log(self, i: int, loss: torch.Tensor, cadence: int=500):
        '''
        Default logging hook: logs stage, step, loss, and lr.
        Subclasses can override to include extra info (e.g. xy_iters).
        '''
        if i == 0 or (i+1) % cadence == 0:
            LOG.info(
                f'Stage {self.current_stage+1}, '
                f'Step {i+1}, loss={loss.item():.5g}, '
                f'lr={self.optimizer.param_groups[0]['lr']:.5f}'
            )

    def fit_gradient(self):
        '''
        Generic gradient-based fitting loop, using:
          - self.loss() for loss computation
          - self.param_config_lists for parameter groups
          - schedulers as specified in config['fitting'][stage]['scheduler']
        '''
        fs = self.config['fitting'][self.current_stage]
        self.maxiter = fs['maxiter']

        # reset any subclass-specific state
        self._reset_state()

        # collect params & build optimizer
        all_cfg = {}
        for cfg_list in self.param_config_lists.values():
            all_cfg.update(cfg_list[self.current_stage])
        param_groups, clamp_list = _extract_tensors(all_cfg, device=self.device)
        # if there’s nothing to fit, warn and bail out

        if len(param_groups)==0:
            LOG.warning(
                f'[{self.__class__.__name__}] ⚠️ '
                f'no trainable parameters in stage {self.current_stage+1}, skipping.'
            )
            return [], []

        if fs['method']=='Adam':
            self.optimizer = torch.optim.Adam(param_groups)
        elif fs['method']=='LBFGS':
            lr = fs['default']['lr']
            flat_params = [p for g in param_groups for p in g['params']]
            self.optimizer = torch.optim.LBFGS(flat_params, lr=lr, history_size=100)
        else: 
            raise ValueError(f'unknown fitting method: {fs['method']}')
        
        self.clamp_list = clamp_list

        # scheduler setup
        # main scueduler
        sched_type = fs['scheduler']
        if sched_type == 'ReduceLROnPlateau':
            patience = fs['_patience'] if '_patience' in fs else 100    
            sched_main = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.707, patience=patience, min_lr=1e-6
            )
            # TODO: set patience in cfg file
        elif sched_type == 'StepLR':
            step_size = fs['_step_size'] if '_step_size' in fs else 1000
            sched_main = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=0.707
            )
        else:
            raise ValueError(f'unknown scheduler: {sched_type}')
        # warmup scheduler and connection
        if '_warmup_size' in fs and fs['_warmup_size']>0:
            warmup_iters = fs['_warmup_size']
            sched_warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1, total_iters=warmup_iters
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[sched_warmup, sched_main], milestones=[warmup_iters]
            )
        else: 
            self.scheduler = sched_main

        self.losses = []
        self.lrs    = []

        try: 

            last_loss = torch.tensor(0, device=self.device) # placeholder

            def closure():
                self.optimizer.zero_grad()
                loss = self.loss()
                loss.backward()
                return loss

            for i in range(self.maxiter):
                if isinstance(self.optimizer, torch.optim.LBFGS):
                    # For LBFGS, step() needs a closure and returns the loss
                    loss = self.optimizer.step(closure)
                else:
                    # For other optimizers, closure() returns the loss
                    loss = closure()
                    self.optimizer.step()

                # Log and record
                cadence = fs['_log_cadence'] if '_log_cadence' in fs else 100
                self._log(i, loss, cadence=cadence)

                self.losses.append(loss.item())
                self.lrs.append(self.optimizer.param_groups[0]['lr'])

                # Step scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss.item())
                    else:
                        self.scheduler.step()

                # Clamp values
                for p, min, max in self.clamp_list:
                    p.data.clamp_(min=min, max=max)

                last_loss = loss

        except KeyboardInterrupt: 
            LOG.info(f'KeyboadrInterrupt, last step: {i}, last loss: {last_loss}')

        # write back final tensor values
        # for cfg in all_cfg.values():
        #     if hasattr(cfg, 'tensor') and isinstance(cfg.tensor, torch.Tensor) and cfg.tensor.ndim == 0:
        #         cfg.value = cfg.tensor.item()

        return self.losses, self.lrs

    def fit_all(self):
        '''
        Run sequential fitting for all configured stages.
        After stage 0, propagate updated values to next-stage cfg
        so that each subsequent fit starts from the last-fitted values.
        Returns:
          all_losses: list of loss histories per stage
          all_lrs:    list of lr histories per stage
        '''
        all_losses = []
        all_lrs    = []

        nstages = len(self.config['fitting'])
        for idx in range(nstages):
            LOG.info(f'Starting fitting stage {idx+1}/{nstages}: method={self.config['fitting'][idx]['method']}')
            self.current_stage = idx

            if idx > 0:
                # propagate values from stage idx-1 -> idx
                for cfg_list in self.param_config_lists.values():
                    prev_cfg = cfg_list[idx-1]
                    curr_cfg = cfg_list[idx]
                    for key, cfg in curr_cfg.items():
                        # TODO: move forward_model entries outside of param group
                        if key.endswith('forward_model'):
                            continue
                        if key in prev_cfg:
                            cfg.tensor = prev_cfg[key].tensor

            # assign the up-to-date cfgs for this stage
            self._assign_cfgs_for_stage(idx)

            losses, lrs = self.fit_gradient()
            all_losses.append(losses)
            all_lrs.append(lrs)

        return all_losses, all_lrs

    def get_losses_and_lrs(self):
        return self.losses, self.lrs

    # MCMC --------------------------------------------------------------------
    # Ensemble sampling over the same stage parameters fit_gradient() optimises.
    # Everything below is additive: fit_gradient/fit_all are untouched, and no
    # sampling state leaks into them.

    _mcmc_loss_is_chi2 = False   # subclasses whose loss() IS a normalised chi2 set True

    def _mcmc_likelihood_mode(self):
        '''Canonical likelihood label used for chain provenance.'''
        return 'gaussian'

    # fit_MCMC() overwrites these from the config. They are class defaults so that
    # log_likelihood() can also be called on its own -- e.g. to scan or profile the
    # objective -- without having to start a chain first.
    mcmc_xy_iters = 40
    mcmc_xy_tol = 5e-3
    mcmc_ln_f_fixed = 0.0
    mcmc_extra_param_names = ()  # nuisance dims appended to theta after the cfg params

    def _mcmc_prepare(self):
        '''Hook run once at the start of fit_MCMC, after the config-driven MCMC
        settings are in place. Subclasses may override to set up a noise model.'''
        pass

    def _mcmc_stage_cfg(self):
        '''
        The same cfg merge fit_gradient does, but tolerant of groups registered
        with fewer stages than config['fitting'] has.
        '''
        all_cfg = {}
        for group, cfg_list in self.param_config_lists.items():
            idx = self.current_stage
            if idx >= len(cfg_list):
                LOG.warning(
                    f'[MCMC] group {group!r} only has {len(cfg_list)} stage(s); '
                    f'reusing the last one for stage {idx+1}.'
                )
                idx = len(cfg_list) - 1
            all_cfg.update(cfg_list[idx])
        return all_cfg

    def _mcmc_set_theta(self, theta):
        '''
        Write the parameter block of theta into the cfg tensors and return the
        trailing nuisance dims as a dict.

        FitParamConfig.value is read-only, so this writes through the tensor's
        storage, as fit_gradient's own `p.data.clamp_` does. Filling in place
        preserves tensor identity (so aliases stay aliased) and leaves
        requires_grad alone, so fit_gradient still works afterwards.
        '''
        n = len(self._mcmc_cfgs)
        with torch.no_grad():
            for cfg, v in zip(self._mcmc_cfgs, theta[:n]):
                cfg.tensor.data.fill_(float(v))
        return {name: float(v) for name, v in zip(self.mcmc_extra_param_names, theta[n:])}

    def _build_mcmc_priors(self):
        '''
        Prior spec per sampled parameter, as a list of (name, dict) in theta order.
        Defaults to a uniform prior over the cfg's own min/max clamp; the optional
        `mcmc.priors` YAML block overrides any of them.
        '''
        spec = (self.config.get('mcmc') or {}).get('priors') or {}
        priors = []
        for name, cfg in zip(self._mcmc_names, self._mcmc_cfgs):
            s = dict(spec.get(name, {}))
            s.setdefault('type', 'uniform')
            s.setdefault('min', float(cfg.min))
            s.setdefault('max', float(cfg.max))
            if s['type'] == 'uniform' and max(abs(s['min']), abs(s['max'])) > 1e6:
                LOG.warning(
                    f'[MCMC] {name}: prior fell back to the YAML clamp '
                    f'({s["min"]:.3g}, {s["max"]:.3g}), which is effectively unbounded. '
                    f'Set mcmc.priors["{name}"] explicitly.'
                )
            priors.append((name, s))
        for name in self.mcmc_extra_param_names:
            priors.append((name, dict(spec.get(name, {'type': 'uniform', 'min': -3.0, 'max': 3.0}))))
        return priors

    def log_prior(self, theta):
        '''
        Sum of the per-parameter log priors, or -inf outside the support.
        Constant normalisations are dropped; they cancel in the sampler.
        '''
        lp = 0.0
        for (name, s), v in zip(self._mcmc_priors, theta):
            t = s['type']
            if t == 'uniform':
                if not (s['min'] <= v <= s['max']):
                    return -np.inf
            elif t == 'log_uniform':
                if v <= 0 or not (s['min'] <= v <= s['max']):
                    return -np.inf
                lp -= np.log(v)
            elif t == 'gaussian':
                if not (s.get('min', -np.inf) <= v <= s.get('max', np.inf)):
                    return -np.inf
                lp += -0.5*((v - s['mu'])/s['sigma'])**2
            elif t == 'sin_i':
                # isotropic-orientation prior p(i) ~ sin(i), for an inclination in radians
                if not (s.get('min', 0.0) <= v <= s.get('max', 0.5*np.pi)):
                    return -np.inf
                if not (0.0 < v < 0.5*np.pi):
                    return -np.inf
                lp += np.log(np.sin(v))
            else:
                raise ValueError(f'unknown prior type for {name}: {t}')
        return lp

    def _draw_mcmc_prior(self, rng):
        '''Draw one point from the independent configured prior distributions.'''
        values = []
        for name, s in self._mcmc_priors:
            t = s['type']
            lo = float(s.get('min', -np.inf))
            hi = float(s.get('max', np.inf))
            if t == 'uniform':
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    raise ValueError(f'prior initialization needs finite bounds for {name}')
                values.append(rng.uniform(lo, hi))
            elif t == 'log_uniform':
                if lo <= 0 or not (np.isfinite(lo) and np.isfinite(hi)):
                    raise ValueError(f'log_uniform prior needs finite positive bounds for {name}')
                values.append(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            elif t == 'gaussian':
                for _ in range(10000):
                    v = rng.normal(float(s['mu']), float(s['sigma']))
                    if lo <= v <= hi:
                        values.append(v)
                        break
                else:
                    raise RuntimeError(f'could not draw a bounded gaussian prior for {name}')
            elif t == 'sin_i':
                # p(i) proportional to sin(i), so cos(i) is uniform.
                values.append(np.arccos(rng.uniform(np.cos(hi), np.cos(lo))))
            else:
                raise ValueError(f'unknown prior type for {name}: {t}')
        return np.asarray(values, dtype=float)

    def log_likelihood(self, **extras):
        '''
        Default log-likelihood: -0.5*loss().

        Only correct when loss() is a properly normalised chi2 up to an additive
        constant. That holds for ImagesFitter, whose loss is
        nansum((res/err)**2), but NOT for KinematicsFitter, whose loss is an
        unweighted, unnormalised sum of squares. Subclasses with an unnormalised
        loss must override this.
        '''
        if not self._mcmc_loss_is_chi2 and not getattr(self, '_mcmc_ll_warned', False):
            LOG.warning(
                f'[{self.__class__.__name__}] log_likelihood() is falling back to '
                '-0.5*loss(); verify that loss() is a normalised chi2.'
            )
            self._mcmc_ll_warned = True
        with torch.no_grad():
            return -0.5*float(self.loss())

    def log_probability(self, theta):
        '''Log posterior, up to a constant. -inf anywhere the model is invalid.'''
        theta = np.asarray(theta, dtype=float)
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        extras = self._mcmc_set_theta(theta)
        try:
            ll = self.log_likelihood(**extras)
        except Exception as exc:
            LOG.debug(f'[MCMC] likelihood failed at {theta}: {exc}')
            return -np.inf
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def fit_MCMC(self, nwalkers=None, nsteps=None, nburn=None,
                 init_scatter=None, pool=None, moves=None, seed=None):
        '''
        Ensemble MCMC (emcee) over the free parameters of the current stage, plus
        any nuisance dims listed in mcmc_extra_param_names.

        Walkers start in a small Gaussian ball around the CURRENT parameter values,
        so run fit_gradient()/fit_all() first: this samples the mode you already
        found, it does not search for one.

        Settings come from the optional `mcmc:` block of the config, and the
        keyword arguments override those. Returns (flat_chain, flat_log_prob).
        '''
        emcee = _import_emcee()
        mc = self.config.get('mcmc') or {}

        # Which stage's free-parameter set to sample. fit_all() leaves the fitter on
        # the LAST stage, which typically has parameters frozen for the final polish;
        # `mcmc.stage` selects a stage whose fit flags describe what should be
        # sampled. Stages share parameter tensors, so the fitted values carry over.
        stage = mc.get('stage')
        if stage is not None and int(stage) != self.current_stage:
            self.current_stage = int(stage)
            self._assign_cfgs_for_stage(self.current_stage)
            LOG.info(f'[MCMC] sampling the free parameters of fitting stage {self.current_stage+1}')

        self.mcmc_xy_iters = int(mc.get('xy_iters', 40))
        self.mcmc_xy_tol = float(mc.get('xy_tol', 5e-3))
        ln_f0 = (mc.get('noise') or {}).get('ln_f')
        self.mcmc_ln_f_fixed = 0.0 if ln_f0 is None else float(ln_f0)

        all_cfg = self._mcmc_stage_cfg()
        self._mcmc_names, self._mcmc_cfgs, alias_map = collect_free_scalar_params(all_cfg)
        if len(self._mcmc_names) == 0:
            LOG.warning(
                f'[{self.__class__.__name__}] no free scalar parameters in '
                f'stage {self.current_stage+1}, skipping MCMC.'
            )
            return [], []
        for kept, dropped in alias_map.items():
            LOG.info(f'[MCMC] {dropped} alias(es) of {kept}, sampled once')

        self._mcmc_prepare()

        self.mcmc_param_names = list(self._mcmc_names) + list(self.mcmc_extra_param_names)
        self._mcmc_priors = self._build_mcmc_priors()
        ndim = len(self.mcmc_param_names)

        nwalkers = int(nwalkers if nwalkers is not None else mc.get('nwalkers', max(32, 2*ndim+2)))
        nsteps = int(nsteps if nsteps is not None else mc.get('nsteps', 8000))
        nburn = int(nburn if nburn is not None else mc.get('nburn', 2000))
        cadence = int(mc.get('log_cadence', 200))
        if nwalkers < 2*ndim:
            LOG.warning(f'[MCMC] nwalkers={nwalkers} < 2*ndim={2*ndim}; raising to {2*ndim}')
            nwalkers = 2*ndim
        if seed is None:
            seed = mc.get('seed')
        rng = np.random.default_rng(seed)

        theta0 = np.array(
            [float(cfg.tensor.detach().cpu()) for cfg in self._mcmc_cfgs]
            + [self.mcmc_ln_f_fixed]*len(self.mcmc_extra_param_names)
        )
        scatter_cfg = dict(mc.get('init_scatter') or {})
        scatter_cfg.update(init_scatter or {})
        scales = np.array([
            float(scatter_cfg.get(name, max(1e-3, 1e-3*abs(v))))
            for name, v in zip(self.mcmc_param_names, theta0)
        ])

        # (V_rot, theta_v) and (-V_rot, theta_v + pi) give an IDENTICAL vz field,
        # so a galaxy rotating the other way can be fitted with either sign. The
        # config may start V_rot negative on purpose (the sample's rotation sense
        # varies), while the prior is one-signed to kill that exact mirror mode.
        # Map onto the positive branch rather than refusing to sample.
        try:
            iv = self.mcmc_param_names.index('velocity.V_rot')
            it = self.mcmc_param_names.index('velocity.theta_v')
        except ValueError:
            iv = it = None
        if iv is not None and it is not None and theta0[iv] < 0:
            LOG.info(f'[MCMC] V_rot is negative ({theta0[iv]:.2f}); reflecting onto '
                     f'the positive branch with theta_v -> theta_v + pi '
                     f'(identical velocity field)')
            theta0[iv] = -theta0[iv]
            theta0[it] = theta0[it] + np.pi
            with torch.no_grad():
                self._mcmc_cfgs[iv].tensor.data.fill_(float(theta0[iv]))
                self._mcmc_cfgs[it].tensor.data.fill_(float(theta0[it]))

        # Position angle is periodic, but fit_gradient clamps it to the config's
        # +-1e10 and so happily returns a negative value; the prior is a single
        # period. Wrap into the prior window instead of failing on a difference
        # that is physically meaningless.
        for i, (name, spec) in enumerate(self._mcmc_priors):
            if not name.endswith('theta_v'):
                continue
            lo, hi = float(spec.get('min', 0.0)), float(spec.get('max', 2*np.pi))
            if abs((hi - lo) - 2*np.pi) < 1e-6 and not (lo <= theta0[i] <= hi):
                wrapped = lo + (theta0[i] - lo) % (2*np.pi)
                LOG.info(f'[MCMC] wrapped {name} {theta0[i]:.4f} -> {wrapped:.4f} '
                         f'into the prior window [{lo}, {hi}]')
                theta0[i] = wrapped
                with torch.no_grad():
                    self._mcmc_cfgs[i].tensor.data.fill_(float(wrapped))

        # Report which parameters are outside their prior BEFORE trying to draw,
        # so the failure names the culprit instead of saying "check mcmc.priors".
        offenders = []
        for (name, spec), v in zip(self._mcmc_priors, theta0):
            lo, hi = spec.get('min', -np.inf), spec.get('max', np.inf)
            if not (lo <= v <= hi):
                offenders.append(f'{name}={v:.4f} outside [{lo}, {hi}]')
        if offenders:
            raise RuntimeError(
                'the fitted parameters lie outside the prior support, so no walker '
                'can be initialised:\n  ' + '\n  '.join(offenders) +
                '\nEither the Adam fit railed (a collapsed fit: R_v -> 0 and '
                'inc_v -> 0 together usually means no velocity signal was found, '
                'and an informative inclination prior prevents it), or the prior '
                'window in mcmc.priors is too narrow for this source.')

        init_mode = str(mc.get('init', 'adam')).lower()
        if init_mode not in {'adam', 'prior'}:
            raise ValueError(f'unknown mcmc.init={init_mode!r}; choose `adam` or `prior`')

        # Draw the initial positions. The historical `adam` mode starts a small
        # ball around the optimizer. The `prior` mode is for broad-prior runs:
        # draw from the configured priors and reject points with an invalid
        # likelihood, so the chain can actually discover remote modes.
        p0 = np.empty((nwalkers, ndim))
        if init_mode == 'prior':
            LOG.info('[MCMC] initializing walkers from the configured priors')
            for j in range(nwalkers):
                for attempt in range(1000):
                    cand = self._draw_mcmc_prior(rng)
                    if np.isfinite(self.log_probability(cand)):
                        p0[j] = cand
                        break
                else:
                    raise RuntimeError(
                        f'could not draw a finite-likelihood walker {j} from the '
                        'configured broad priors after 1000 attempts')
        else:
            for j in range(nwalkers):
                for _ in range(1000):
                    cand = theta0 + scales*rng.standard_normal(ndim)
                    if np.isfinite(self.log_prior(cand)):
                        p0[j] = cand
                        break
                else:
                    raise RuntimeError(
                        f'could not draw walker {j} inside the prior support after 1000 '
                        f'tries, even though theta0 is inside it. init_scatter is '
                        f'probably far too wide for a parameter near a bound: '
                        f'{dict(zip(self.mcmc_param_names, scales))}')

        lnp0 = np.array([self.log_probability(p) for p in p0])
        if not np.all(np.isfinite(lnp0)):
            bad = p0[np.argmin(lnp0)]
            raise RuntimeError(
                f'{int(np.sum(~np.isfinite(lnp0)))}/{nwalkers} walkers start at -inf, '
                f'e.g. {dict(zip(self.mcmc_param_names, bad))}'
            )
        LOG.info(
            f'[MCMC] ndim={ndim} nwalkers={nwalkers} nburn={nburn} nsteps={nsteps}; '
            f'params={self.mcmc_param_names}'
        )
        LOG.info(f'[MCMC] lnP(theta0)={self.log_probability(theta0):.6g}')

        if moves is None:
            # the default StretchMove copes badly with the V_rot--sin(inc_v) banana
            moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]

        # HDFBackend writes every accepted sampler step to disk. This makes a
        # long run restartable without keeping the whole chain only in RAM;
        # `resume: true` continues from the last completed step on the next
        # notebook run. The final .npz export is still produced by save_mcmc().
        backend_path = mc.get('backend')
        resume = bool(mc.get('resume', True))
        backend = None
        resume_state = None
        resume_chain = None
        resume_log_prob = None
        start_step = 0
        backend_meta_path = None
        if backend_path:
            requested_backend_path = os.path.expanduser(str(backend_path))
            try:
                os.makedirs(os.path.dirname(os.path.abspath(requested_backend_path)) or '.', exist_ok=True)
                backend_path = requested_backend_path
                backend_meta_path = backend_path + '.json'
                backend = emcee.backends.HDFBackend(backend_path)
            except ImportError as exc:
                # h5py is optional in the DINGO environment. A compressed NPZ
                # checkpoint is a complete fallback: it stores the whole chain
                # plus the last ensemble state needed to continue sampling.
                LOG.warning(
                    f'[MCMC] HDF5 backend unavailable ({exc}); using the NPZ '
                    'checkpoint instead'
                )
                backend = None
                backend_path = None
                backend_meta_path = None

            if backend is not None:
                if os.path.exists(requested_backend_path) and resume:
                    if backend.iteration:
                        if backend.shape != (nwalkers, ndim):
                            raise RuntimeError(
                                f'[MCMC] existing backend has shape {backend.shape}, '
                                f'but this run needs {(nwalkers, ndim)}; use a new backend path'
                            )
                        if os.path.exists(backend_meta_path):
                            try:
                                with open(backend_meta_path) as stream:
                                    old_meta = json.load(stream)
                                if old_meta.get('param_names') != self.mcmc_param_names:
                                    raise RuntimeError(
                                        '[MCMC] existing backend parameter names do not match '
                                        'this configuration; use a new backend path'
                                    )
                                if old_meta.get('nburn') != nburn:
                                    raise RuntimeError(
                                        '[MCMC] existing backend has a different nburn; '
                                        'keep nburn fixed when resuming'
                                    )
                            except RuntimeError:
                                raise
                            except Exception as exc:
                                LOG.warning(f'[MCMC] could not read backend metadata: {exc}')
                        start_step = int(backend.iteration)
                        if start_step > nburn + nsteps:
                            raise RuntimeError(
                                f'[MCMC] backend already contains {start_step} steps, '
                                f'but the requested target is only {nburn + nsteps}; '
                                'increase nsteps or use a new backend path'
                            )
                        resume_state = backend.get_last_sample()
                        LOG.info(
                            f'[MCMC] resuming backend at step {start_step}/{nburn+nsteps}: '
                            f'{backend_path}'
                        )
                    else:
                        backend.reset(nwalkers, ndim)
                else:
                    backend.reset(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, pool=pool, moves=moves,
            backend=backend
        )

        # Legacy compressed checkpoints remain supported for configurations
        # that do not specify an HDF5 backend. They are not written when the
        # resumable backend above is active.
        ckpt = mc.get('checkpoint') if backend is None else None
        ckpt_every = int(mc.get('checkpoint_every', 500))
        if ckpt:
            ckpt = os.path.expanduser(str(ckpt))
            os.makedirs(os.path.dirname(os.path.abspath(ckpt)) or '.', exist_ok=True)

        # Resume from a dependency-free compressed checkpoint when HDF5 is not
        # available. The final saved walker positions are sufficient to
        # reconstruct emcee's state and continue the ensemble exactly from the
        # last completed step (with a fresh random stream).
        if ckpt and resume and os.path.exists(ckpt):
            try:
                with np.load(ckpt, allow_pickle=True) as saved:
                    saved_chain = np.asarray(saved['chain'])
                    saved_log_prob = np.asarray(saved['log_prob'])
                    saved_names = [str(x) for x in saved['param_names'].tolist()]
                    saved_nburn = int(saved['nburn'])
                    start_step = int(
                        saved['nsteps_done'] if 'nsteps_done' in saved.files
                        else saved_chain.shape[0]
                    )
                if saved_names != self.mcmc_param_names:
                    raise RuntimeError(
                        '[MCMC] existing checkpoint parameter names do not match '
                        'this configuration; use a new checkpoint path'
                    )
                if saved_nburn != nburn:
                    raise RuntimeError(
                        '[MCMC] existing checkpoint has a different nburn; '
                        'keep nburn fixed when resuming'
                    )
                if saved_chain.shape != (start_step, nwalkers, ndim):
                    raise RuntimeError(
                        f'[MCMC] existing checkpoint shape {saved_chain.shape} is not '
                        f'({start_step}, {nwalkers}, {ndim})'
                    )
                if saved_log_prob.shape != (start_step, nwalkers):
                    raise RuntimeError('[MCMC] existing checkpoint log-probability shape is invalid')
                if start_step > nburn + nsteps:
                    raise RuntimeError(
                        f'[MCMC] checkpoint already contains {start_step} steps, '
                        f'but the requested target is only {nburn + nsteps}; '
                        'increase nsteps or use a new checkpoint path'
                    )
                if start_step:
                    resume_chain = saved_chain
                    resume_log_prob = saved_log_prob
                    resume_state = emcee.State(
                        coords=saved_chain[-1], log_prob=saved_log_prob[-1]
                    )
                    LOG.info(
                        f'[MCMC] resuming checkpoint at step {start_step}/{nburn+nsteps}: {ckpt}'
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f'[MCMC] could not load checkpoint {ckpt}: {exc}') from exc

        def _write_backend_metadata(ndone):
            if not backend_meta_path:
                return
            payload = {
                'param_names': list(self.mcmc_param_names),
                'nwalkers': nwalkers,
                'ndim': ndim,
                'nburn': nburn,
                'nsteps_target': nsteps,
                'steps_written': int(ndone),
                'likelihood_mode': self._mcmc_likelihood_mode(),
            }
            tmp = backend_meta_path + '.tmp'
            try:
                with open(tmp, 'w') as stream:
                    json.dump(payload, stream, indent=2)
                os.replace(tmp, backend_meta_path)
            except Exception as exc:              # metadata is helpful, not fatal
                LOG.warning(f'[MCMC] backend metadata write failed at step {ndone}: {exc}')

        def _checkpoint(sampler, ndone):
            if backend is not None:
                _write_backend_metadata(ndone)
                LOG.info(f'[MCMC] resumable backend at step {ndone} -> {backend_path}')
                return
            if not ckpt:
                return
            try:
                chain_to_save = sampler.get_chain()
                log_prob_to_save = sampler.get_log_prob()
                if resume_chain is not None:
                    if chain_to_save.shape[0]:
                        chain_to_save = np.concatenate((resume_chain, chain_to_save), axis=0)
                        log_prob_to_save = np.concatenate((resume_log_prob, log_prob_to_save), axis=0)
                    else:
                        chain_to_save = resume_chain
                        log_prob_to_save = resume_log_prob
                np.savez_compressed(
                    ckpt + '.tmp.npz',
                    chain=chain_to_save, log_prob=log_prob_to_save,
                    param_names=np.array(self.mcmc_param_names, dtype=object),
                    nburn=nburn, nsteps_done=ndone,
                    acceptance_fraction=sampler.acceptance_fraction,
                    theta0=theta0)
                os.replace(ckpt + '.tmp.npz', ckpt)
                LOG.info(f'[MCMC] checkpoint at step {ndone} -> {ckpt}')
            except Exception as exc:              # never let a bad write kill the chain
                LOG.warning(f'[MCMC] checkpoint failed at step {ndone}: {exc}')

        if backend is not None:
            _write_backend_metadata(start_step)

        ndone = start_step
        remaining = max(0, nburn + nsteps - start_step)
        try:
            if remaining:
                initial_state = resume_state if resume_state is not None else p0
                for i, state in enumerate(sampler.sample(initial_state, iterations=remaining, progress=False)):
                    ndone = start_step + i + 1
                    step_in_run = i + 1
                    if step_in_run == 1 or ndone % cadence == 0:
                        LOG.info(
                            f'MCMC step {ndone}/{nburn+nsteps}, '
                            f'acc={np.mean(sampler.acceptance_fraction):.3f}, '
                            f'max lnP={np.max(state.log_prob):.6g}'
                        )
                    if (ckpt or backend is not None) and ndone % ckpt_every == 0:
                        _checkpoint(sampler, ndone)
            else:
                LOG.info(f'[MCMC] target of {nburn+nsteps} steps already exists; no sampling needed')
        except KeyboardInterrupt:
            LOG.info(f'KeyboardInterrupt, stopping MCMC at step {ndone}')
            _checkpoint(sampler, ndone)

        if backend is not None and ndone:
            _checkpoint(sampler, ndone)

        self.mcmc_sampler = sampler
        self.mcmc_nburn = int(min(nburn, max(0, ndone - 1)))
        self.mcmc_theta0 = theta0
        # A completed NPZ checkpoint has all samples on disk but no samples in
        # this newly-created in-memory sampler. Do not query emcee in that case:
        # its empty backend raises "run the sampler with store == True".
        if sampler.iteration:
            raw_chain = sampler.get_chain()
            raw_log_prob = sampler.get_log_prob()
        else:
            raw_chain = np.empty((0, nwalkers, ndim), dtype=float)
            raw_log_prob = np.empty((0, nwalkers), dtype=float)
        if resume_chain is not None:
            if raw_chain.shape[0]:
                raw_chain = np.concatenate((resume_chain, raw_chain), axis=0)
                raw_log_prob = np.concatenate((resume_log_prob, raw_log_prob), axis=0)
            else:
                raw_chain = resume_chain
                raw_log_prob = resume_log_prob
        self._mcmc_raw_chain = raw_chain
        self._mcmc_raw_log_prob = raw_log_prob
        self.mcmc_chain = raw_chain[self.mcmc_nburn:].reshape(-1, ndim)
        self.mcmc_log_prob = raw_log_prob[self.mcmc_nburn:].reshape(-1)
        LOG.info(
            f'[MCMC] done: {ndone} steps, {self.mcmc_chain.shape[0]} post-burn-in samples, '
            f'acc={np.mean(sampler.acceptance_fraction):.3f}'
        )
        return self.mcmc_chain, self.mcmc_log_prob

    # MCMC results ------------------------------------------------------------

    def get_mcmc_results(self, percentiles=(16, 50, 84)):
        '''
        {param_name: (lo, med, hi)} at the given percentiles.

        Adds a derived 'velocity.Vsini' entry when both V_rot and inc_v were
        sampled: that combination is what the line-of-sight velocity field
        actually constrains, and is far tighter than either component alone.
        '''
        chain = self.mcmc_chain
        out = {}
        for j, name in enumerate(self.mcmc_param_names):
            lo, med, hi = np.percentile(chain[:, j], percentiles)
            out[name] = (float(lo), float(med), float(hi))
        names = list(self.mcmc_param_names)
        if 'velocity.V_rot' in names and 'velocity.inc_v' in names:
            vsini = chain[:, names.index('velocity.V_rot')]*np.sin(chain[:, names.index('velocity.inc_v')])
            lo, med, hi = np.percentile(vsini, percentiles)
            out['velocity.Vsini'] = (float(lo), float(med), float(hi))
        return out

    def get_mcmc_diagnostics(self):
        '''Convergence diagnostics for the last fit_MCMC run.'''
        emcee = _import_emcee()
        sampler = self.mcmc_sampler
        chain = getattr(self, '_mcmc_raw_chain', None)
        if chain is None:
            chain = sampler.get_chain()
        chain = chain[self.mcmc_nburn:]                           # (nsteps, nwalkers, ndim)
        nsteps = chain.shape[0]
        acc = sampler.acceptance_fraction
        try:
            tau = emcee.autocorr.integrated_time(chain, quiet=True)
        except Exception as exc:
            LOG.warning(f'[MCMC] autocorrelation estimate failed: {exc}')
            tau = np.full(chain.shape[2], np.nan)
        raw_log_prob = getattr(self, '_mcmc_raw_log_prob', None)
        if raw_log_prob is None:
            raw_log_prob = sampler.get_log_prob()
        # NOTE: no R-hat here. emcee is an ENSEMBLE sampler -- walkers propose
        # from one another, so they are not independent chains and Gelman-Rubin
        # comes out optimistic by construction. tau and n_steps/tau below are the
        # diagnostics that actually govern whether a posterior width is real.
        half = nsteps//2
        first, second = chain[:half], chain[half:]
        return {
            'n_steps': int(nsteps),
            'n_burn': int(self.mcmc_nburn),
            'acceptance_fraction': (float(np.mean(acc)), float(np.min(acc)), float(np.max(acc))),
            'tau': {n: float(t) for n, t in zip(self.mcmc_param_names, tau)},
            'n_steps_over_tau': {n: float(nsteps/t) for n, t in zip(self.mcmc_param_names, tau)},
            'n_eff': {n: float(chain[:, :, j].size/t)
                      for j, (n, t) in enumerate(zip(self.mcmc_param_names, tau))},
            'split_half_shift': {
                n: float(abs(np.mean(first[:, :, j]) - np.mean(second[:, :, j]))
                         / (np.std(chain[:, :, j]) + 1e-30))
                for j, n in enumerate(self.mcmc_param_names)
            },
            'frac_neg_inf': float(np.mean(~np.isfinite(raw_log_prob[self.mcmc_nburn:]))),
        }

    def set_params_to_mcmc(self, mode='map'):
        '''
        Write a chain sample back into the cfg tensors and re-run the forward model.

        mode='map'    -- the highest-log-probability sample
        mode='median' -- the per-parameter median (may not be a sampled point)

        Call this before plotting: log_likelihood() overwrites the cached model
        images, so after fit_MCMC they hold whatever walker was evaluated last.
        Returns the theta that was written.
        '''
        if mode == 'map':
            theta = self.mcmc_chain[int(np.argmax(self.mcmc_log_prob))]
        elif mode == 'median':
            theta = np.median(self.mcmc_chain, axis=0)
        else:
            raise ValueError(f'unknown mode: {mode}')
        extras = self._mcmc_set_theta(theta)
        for name, value in extras.items():
            if name == 'noise.ln_f':
                self.mcmc_ln_f_fixed = value
        self._reset_state()
        with torch.no_grad():
            self.loss()
        return theta

    def save_mcmc(self, path):
        '''
        Save the full chain to a .npz. param_names round-trips, so a later
        reordering of param_config_lists can never mis-map the columns.

        npz rather than emcee's HDF5Backend: h5py is not a DINGO dependency.
        '''
        sampler = self.mcmc_sampler
        chain = getattr(self, '_mcmc_raw_chain', None)
        log_prob = getattr(self, '_mcmc_raw_log_prob', None)
        if chain is None:
            chain = sampler.get_chain()
        if log_prob is None:
            log_prob = sampler.get_log_prob()
        data = {
            'chain': chain,                             # (nsteps, nwalkers, ndim)
            'log_prob': log_prob,
            'param_names': np.array(self.mcmc_param_names, dtype=object),
            'nburn': self.mcmc_nburn,
            'acceptance_fraction': sampler.acceptance_fraction,
            'theta0': self.mcmc_theta0,
            'likelihood_mode': self._mcmc_likelihood_mode(),
            'config_path': str(self.config_path),
            'name': str(self.name),
        }
        for attr in ('sigma_R', 'sigma_C', 'mcmc_var_base', 'n_mcmc_pix'):
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)
        if getattr(self, 'mcmc_mask', None) is not None:
            data['mask'] = self.mcmc_mask.detach().cpu().numpy().astype(np.uint8)
        np.savez_compressed(path, **data)
        LOG.info(f'[MCMC] chain saved to {path}')
        return path

    @staticmethod
    def load_mcmc(path):
        '''Inverse of save_mcmc. Returns a plain dict; param_names comes back as a list.'''
        with np.load(path, allow_pickle=True) as npz:
            out = {k: npz[k] for k in npz.files}
        out['param_names'] = [str(n) for n in out['param_names']]
        return out

#%% --------------------------------------------------------------------------
# Kinematics fitting subclass
# ----------------------------------------------------------------------------

class KinematicsFitter(BaseFitter):

    def __init__(self, config_path: AnyStr, device=None):
        super().__init__(config_path, device)

    def _setup_data(self):
        # ─────────────────────────────
        # Image loading and wavelength
        # ─────────────────────────────
        img_cfg = self.config['image']
        self.lambda_rest = torch.tensor([self.config['summary']['lambda_rest']], device=self.device) * (1 + self.z)

        self.true_grism_R = torch.tensor(load_fits_data(img_cfg['R']['path']), device=self.device)
        self.true_grism_C = torch.tensor(load_fits_data(img_cfg['C']['path']), device=self.device)

        # ─────────────────────────────
        # Forward model loading
        # ─────────────────────────────
        self.fwd_models = {}
        for pupil in ['R', 'C']:
            _, sp, dp = grism.load_nircam_wfss_model(pupil, img_cfg[pupil]['module'], self.filter)
            self.fwd_models[pupil] = utils.get_grism_model_torch(
                sp, dp, pupil, 1024, 1024, direction='forward'
            )

        # ─────────────────────────────
        # Parameter configs for ALL stages
        # ─────────────────────────────
        defaults  = [fs['default']  for fs in self.config['fitting']]
        overrides = [fs['override'] for fs in self.config['fitting']]

        # velocity params
        # NOTE: overrides will be mutated during iteration
        self.velocity_cfg_list = build_param_config_dict_with_alias(
            self.config['velocity'], defaults, overrides, prefix='velocity', device=self.device
        )

        # dispersion R params
        self.dispersion_R_cfg_list = build_param_config_dict_with_alias(
            {k:v for k,v in img_cfg['R'].items() if k in ['dx','dy']},
            defaults, overrides, prefix='image.R', device=self.device
        )
        for cfg in self.dispersion_R_cfg_list:
            cfg['image.R.forward_model'] = FitParamConfig(
                name='image.R.forward_model', value=None,
                lr=0, min=0, max=0, fit=False
            )
            cfg['image.R.forward_model'].tensor = self.fwd_models['R']

        # dispersion C params
        self.dispersion_C_cfg_list = build_param_config_dict_with_alias(
            {k:v for k,v in img_cfg['C'].items() if k in ['dx','dy']},
            defaults, overrides, prefix='image.C', device=self.device
        )
        for cfg in self.dispersion_C_cfg_list:
            cfg['image.C.forward_model'] = FitParamConfig(
                name='image.C.forward_model', value=None,
                lr=0, min=0, max=0, fit=False
            )
            cfg['image.C.forward_model'].tensor = self.fwd_models['C']

        self.model_cfg = {'result.emline_model': FitParamConfig(
            name='result.emline_model',
            value = np.ones((81, 81))/np.sum(self.true_grism_R.cpu().numpy()),
            lr = 0.05,
            min=0, 
            max=1e10, 
            fit=True
        )}

        unused_keys = set([key for stage in overrides for key in stage.keys()])
        if len(unused_keys)>0:
            LOG.warning(f'The following override keys are not used: {unused_keys}')

        # register into the generic param_config_lists
        # NOTE: every group must supply one cfg dict per fitting stage, otherwise
        # fit_all() and _assign_cfgs_for_stage() index out of range. The emline
        # model is stage-independent, so the same dict is repeated.
        nstages = len(self.config['fitting'])
        self.param_config_lists = {
            'velocity':       self.velocity_cfg_list,
            'dispersion_R':   self.dispersion_R_cfg_list,
            'dispersion_C':   self.dispersion_C_cfg_list,
            'model':          [self.model_cfg]*nstages
        }

        # ─────────────────────────────
        # Pixel grid and cutout regions
        # ─────────────────────────────
        nx, ny = self.true_grism_R.shape
        self.y_G, self.x_G = torch.meshgrid(
            torch.arange(nx), torch.arange(ny), indexing='ij'
        )

        # Explicit cutouts are preferred. For configs predating the `cutout` key,
        # fall back to deriving them from summary.r_fit, i.e. the cutout centred on
        # where the rest-frame line lands on the grism detector.
        for pupil in ['R', 'C']:
            if 'cutout' in img_cfg[pupil]:
                cutout = tuple(img_cfg[pupil]['cutout'])
            elif self.r_fit is not None:
                r = self.r_fit
                cx, cy = self.fwd_models[pupil](
                    torch.tensor(float(r), device=self.device),
                    torch.tensor(float(r), device=self.device),
                    self.lambda_rest
                )
                cutout = (int(cx)-r, int(cy)-r, 2*r+1, 2*r+1)
                LOG.warning(
                    f'[{self.__class__.__name__}] image.{pupil}.cutout is missing; '
                    f'derived {cutout} from summary.r_fit={r}. Prefer an explicit cutout.'
                )
            else:
                raise KeyError(
                    f'image.{pupil} needs a `cutout` key (or a `summary.r_fit` to derive it from)'
                )
            setattr(self, f'cutout_{pupil}', cutout)


    def _reset_state(self):
        # reset XY before each stage
        self.x_R, self.y_R = self.x_G.clone(), self.y_G.clone()
        self.x_C, self.y_C = self.x_G.clone(), self.y_G.clone()

    def loss(self):
        # unpack both fitted and fixed params as tensors
        velocity = self._get_model_params('velocity')
        disp_R = self._get_model_params('dispersion_R')
        disp_C = self._get_model_params('dispersion_C')

        # compute R‐channel
        self.x_R, self.y_R, self.vz_R, self.iter_R = kinematics.iteratively_find_xy(
            self.x_R, self.y_R, self.cutout_R,
            self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_R
        )
        self.image_R = kinematics.bilinear_interpolte_intensity_torch(
            self.x_R, self.y_R, self.true_grism_R, self.cutout_R
        )

        # compute C‐channel
        self.x_C, self.y_C, self.vz_C, self.iter_C = kinematics.iteratively_find_xy(
            self.x_C, self.y_C, self.cutout_C,
            self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_C
        )
        self.image_C = kinematics.bilinear_interpolte_intensity_torch(
            self.x_C, self.y_C, self.true_grism_C, self.cutout_C
        )
        
        # emline_model = self._get_model_params('model')['emline_model']
        # # return torch.sum((self.image_R/torch.sum(self.image_R) - emline_model/torch.sum(emline_model))**2) + \
        # #        torch.sum((self.image_C/torch.sum(self.image_C) - emline_model/torch.sum(emline_model))**2)
        
        # return torch.sum((self.image_R - emline_model)**2) + torch.sum((self.image_C - emline_model)**2)

        # L2 loss between the two grism channels
        return torch.sum((self.image_R - self.image_C)**2)

    def _log(self, i: int, loss: torch.Tensor, cadence: int=500):
        '''
        Default logging hook: logs stage, step, loss, and lr.
        Subclasses can override to include extra info (e.g. xy_iters).
        '''
        if i == 0 or (i+1) % cadence == 0:
            LOG.info(
                f'Stage {self.current_stage+1}, '
                f'Step {i+1}, loss={loss.item():.5g}, '
                f'lr={self.optimizer.param_groups[0]['lr']:.5f}'
                f'xy_iters={(self.iter_R, self.iter_C)}'
            )

    # getters ----------------------------------------------------------------

    def get_fitting_results(self):
        return (
            self.image_R.detach().cpu().numpy(),
            self.image_C.detach().cpu().numpy(),
            self.vz_R.detach().cpu().numpy(),
            self.vz_C.detach().cpu().numpy()
        )

    def get_params(self):
        velocity = {k.split('.')[-1]: v.value for k,v in self.velocity_cfg.items()}
        disp_R   = {k.split('.')[-1]: v.value for k,v in self.dispersion_R_cfg.items()}
        disp_C   = {k.split('.')[-1]: v.value for k,v in self.dispersion_C_cfg.items()}
        return velocity, disp_R, disp_C

    def get_true_images(self):
        return (
            self.true_grism_R.detach().cpu().numpy(),
            self.true_grism_C.detach().cpu().numpy()
        )

    # MCMC --------------------------------------------------------------------

    mcmc_mask = None

    @property
    def mcmc_extra_param_names(self):
        nc = (self.config.get('mcmc') or {}).get('noise') or {}
        if self._mcmc_likelihood_mode() == 'adam':
            # A free scale changes the objective from SSE to
            # SSE/var + N*log(var), so it is deliberately not part of the
            # Adam-compatible target. The scale is fixed from the noise block.
            return ()
        return ('noise.ln_f',) if nc.get('fit_ln_f', True) else ()

    def _mcmc_likelihood_mode(self):
        '''Return the configured likelihood mode in its canonical spelling.'''
        mode = str((self.config.get('mcmc') or {}).get('likelihood', 'gaussian')).lower()
        aliases = {
            'adam_loss': 'adam',
            'adam-compatible': 'adam',
            'adam_compatible': 'adam',
        }
        mode = aliases.get(mode, mode)
        if mode not in {'gaussian', 'adam'}:
            raise ValueError(
                f'unknown mcmc.likelihood={mode!r}; choose `gaussian` or `adam`')
        return mode

    def _mcmc_prepare(self):
        if self.mcmc_mask is None:
            if self._mcmc_likelihood_mode() == 'adam':
                self.setup_adam_compatible_noise_model()
            else:
                self.setup_noise_model()

    def setup_adam_compatible_noise_model(self, sigma_R=None, sigma_C=None):
        '''Prepare a fixed-noise likelihood with exactly Adam's pixel objective.

        Adam minimises the unweighted, full-image residual

            sum((image_R - image_C)**2).

        This mode keeps every output pixel, uses a fixed variance only to put
        the SSE on a log-probability scale, and does not fit ``noise.ln_f``.
        The fixed-point iteration remains cold-started and deterministic during
        MCMC, which is necessary for a valid likelihood but does not change the
        model being fitted.
        '''
        nc = (self.config.get('mcmc') or {}).get('noise') or {}
        if nc.get('fit_ln_f', False):
            raise ValueError(
                'mcmc.likelihood=adam requires mcmc.noise.fit_ln_f=false; '
                'a free noise scale would no longer have Adam\'s SSE objective.')
        if sigma_R is None:
            sigma_R = nc.get('sigma_R')
        if sigma_C is None:
            sigma_C = nc.get('sigma_C')
        self.sigma_R = float(sigma_R) if sigma_R is not None \
            else estimate_background_sigma(self.true_grism_R)
        self.sigma_C = float(sigma_C) if sigma_C is not None \
            else estimate_background_sigma(self.true_grism_C)
        self.mcmc_corr_area = float(nc.get('corr_area', 1.0))
        self.mcmc_var_base = (self.sigma_R**2 + self.sigma_C**2)*self.mcmc_corr_area

        self._reset_state()
        with torch.no_grad():
            self.loss()  # initialise the same forward model Adam uses
        self.mcmc_mask = torch.ones_like(self.image_R, dtype=torch.bool)
        self.n_mcmc_pix = int(self.mcmc_mask.numel())
        self.mcmc_cov_R = torch.ones_like(self.image_R)
        self.mcmc_cov_C = torch.ones_like(self.image_C)
        self.mcmc_foot_R = self._cutout_footprint(self.x_R, self.y_R, self.cutout_R)
        self.mcmc_foot_C = self._cutout_footprint(self.x_C, self.y_C, self.cutout_C)
        self.mcmc_flux_masked_out = 0.0
        LOG.info(
            f'[MCMC] Adam-compatible likelihood: full SSE, '
            f'sigma_R={self.sigma_R:.5g} sigma_C={self.sigma_C:.5g} '
            f'var_base={self.mcmc_var_base:.5g} N_pix={self.n_mcmc_pix}')
        return self.mcmc_mask

    def setup_noise_model(self, sigma_R=None, sigma_C=None,
                          cov_threshold=0.25, extra_mask=None):
        '''
        Freeze the noise model, the pixel mask and the input footprint. Call once,
        at the MAP parameters, before sampling (fit_MCMC does it automatically).

        The mask is deliberately INDEPENDENT of the parameters. If it moved with
        theta then both N and the ln(2*pi*sigma^2) term would move with theta, the
        noise scale would stop being identifiable, and the sampler would be
        rewarded for shrinking the mask. Making it a fixed function of the data
        and of the MAP is legitimate; recomputing it during sampling is not.

        The mask drops pixels that the bilinear remapping never reaches, measured
        by scattering an image of ones through the same transform.

        cov_threshold is 0.25, NOT ~1. `bilinear_interpolte_intensity_torch`
        SCATTERS, so the coverage map is the Jacobian of the remapping, not an
        occupancy fraction: it is < 1 wherever the transform stretches and > 1
        wherever it compresses. The Doppler term stretches one grism and
        compresses the other exactly where the velocity gradient is steepest --
        i.e. across the galaxy centre, where cov_R ~ 0.76 while cov_C ~ 1.26. A
        threshold near 1 therefore deletes the core: at 0.9 this cut 85% of the
        pixels within r < 6 px, 40% of the total flux and 128 of the 200
        brightest pixels, which is most of the kinematic signal. The genuinely
        unreached pixels sit at exactly 0 (235 of them here), so any threshold
        between ~0.05 and ~0.5 separates them cleanly; 0.25 keeps 100% of the
        core and 98.4% of the flux.

        Note the variance is still treated as uniform across the mask even though
        coverage varies by ~2x over it, so pixels differ in how many input pixels
        were averaged into them. That is an approximation the fitted ln_f absorbs
        only in the mean.
        '''
        nc = (self.config.get('mcmc') or {}).get('noise') or {}
        if sigma_R is None:
            sigma_R = nc.get('sigma_R')
        if sigma_C is None:
            sigma_C = nc.get('sigma_C')
        self.sigma_R = float(sigma_R) if sigma_R is not None \
            else estimate_background_sigma(self.true_grism_R)
        self.sigma_C = float(sigma_C) if sigma_C is not None \
            else estimate_background_sigma(self.true_grism_C)
        # corr_area inflates the variance for correlated noise. It is 100%
        # degenerate with a fitted ln_f, so set one or the other, never both.
        self.mcmc_corr_area = float(nc.get('corr_area', 1.0))
        self.mcmc_var_base = (self.sigma_R**2 + self.sigma_C**2)*self.mcmc_corr_area

        self._reset_state()
        with torch.no_grad():
            self.loss()  # populate x_R/y_R/x_C/y_C at the current (MAP) parameters
            ones = torch.ones_like(self.true_grism_R)
            cov_R = kinematics.bilinear_interpolte_intensity_torch(
                self.x_R, self.y_R, ones, self.cutout_R)
            cov_C = kinematics.bilinear_interpolte_intensity_torch(
                self.x_C, self.y_C, ones, self.cutout_C)
        mask = (cov_R > cov_threshold) & (cov_C > cov_threshold)
        if extra_mask is not None:
            mask = mask & extra_mask.to(mask.device)
        self.mcmc_cov_R = cov_R
        self.mcmc_cov_C = cov_C
        self.mcmc_mask = mask
        self.n_mcmc_pix = int(mask.sum())

        # Input-space footprint: the grism pixels whose rectified position lands
        # inside the output cutout, i.e. exactly the pixels the scatter keeps.
        # Used only to test convergence of the fixed point where it matters.
        self.mcmc_foot_R = self._cutout_footprint(self.x_R, self.y_R, self.cutout_R)
        self.mcmc_foot_C = self._cutout_footprint(self.x_C, self.y_C, self.cutout_C)

        # A mask is only legitimate if it removes empty pixels, not signal. Report
        # how much flux it costs and complain when that is large -- a coverage
        # threshold set too high silently deletes the galaxy core, which is where
        # nearly all the kinematic information lives, and nothing else downstream
        # would reveal it.
        with torch.no_grad():
            flux = torch.abs(self.image_R)
            frac_out = float(flux[~mask].sum()/flux.sum()) if float(flux.sum()) > 0 else 0.0
        self.mcmc_flux_masked_out = frac_out
        LOG.info(
            f'[MCMC] noise model frozen: sigma_R={self.sigma_R:.5g} sigma_C={self.sigma_C:.5g} '
            f'var_base={self.mcmc_var_base:.5g} N_pix={self.n_mcmc_pix}/{mask.numel()} '
            f'(mask drops {100*frac_out:.1f}% of the flux)'
        )
        if frac_out > 0.10:
            LOG.warning(
                f'[MCMC] the mask removes {100*frac_out:.1f}% of the flux at '
                f'cov_threshold={cov_threshold}. The coverage map is a Jacobian, so a '
                f'threshold near 1 cuts the stretched galaxy centre rather than the '
                f'uncovered edges. Lower cov_threshold (0.25 is the default).')
        return self.mcmc_mask

    @staticmethod
    def _cutout_footprint(x, y, cutout):
        x0, y0, w, h = cutout
        return ((x - x0 >= 0) & (x - x0 < w - 1) &
                (y - y0 >= 0) & (y - y0 < h - 1))

    def _xy_converged(self, x, y, cutout, foot, velocity, disp):
        '''
        Test convergence of the fixed point by taking one more undamped step.

        The `k` returned by iteratively_find_xy cannot be used for this: its
        max-over-all-pixels test is dominated by pixels outside the useful
        footprint that never settle, so it reports non-convergence even when the
        loss has converged to ~1e-7 relative. Restricting the test to the frozen
        footprint is what makes it meaningful.
        '''
        c = 299792.458  # km/s
        vz = kinematics.arctangent_disk_velocity_model(x - cutout[0], y - cutout[1], **velocity)
        lambda_obs = self.lambda_rest*(1.0 + vz/c)
        x_new, y_new = kinematics.forward_dispersion_model(self.x_G, self.y_G, lambda_obs, **disp)
        if not (torch.isfinite(x_new).all() and torch.isfinite(y_new).all()):
            return False
        return bool(torch.max(torch.abs(x_new - x)[foot]) < self.mcmc_xy_tol and
                    torch.max(torch.abs(y_new - y)[foot]) < self.mcmc_xy_tol)

    def log_likelihood(self, **extras):
        '''
        Full Gaussian log-likelihood of the R-vs-C residual on the frozen mask,
        including the ln(2*pi*sigma^2) term -- that term is what makes the noise
        scale ln_f identifiable, so it must not be dropped.

        The residual compares two noisy rectified images rather than model vs
        data, so its variance is sigma_R^2 + sigma_C^2, scaled by exp(2*ln_f).

        Unlike loss(), this is cold-started and runs a FIXED number of fixed-point
        iterations. loss() warm-starts x_R/y_R from the previous call, which makes
        it a function of the evaluation history; emcee interleaves walkers in an
        arbitrary order, so a history-dependent likelihood is not a likelihood at
        all. Stopping on a tolerance would likewise make lnL discontinuous in
        theta, because the iteration count would jump around.
        '''
        if self.mcmc_mask is None:
            raise RuntimeError('call setup_noise_model() before log_likelihood()')

        ln_f = float(extras.get('noise.ln_f', self.mcmc_ln_f_fixed))
        velocity = self._get_model_params('velocity')
        disp_R = self._get_model_params('dispersion_R')
        disp_C = self._get_model_params('dispersion_C')
        niter = self.mcmc_xy_iters

        with torch.no_grad():
            x_R, y_R, vz_R, _ = kinematics.iteratively_find_xy(
                self.x_G.clone(), self.y_G.clone(), self.cutout_R,
                self.lambda_rest, self.x_G, self.y_G,
                maxiter=niter, tol=0.0, **velocity, **disp_R
            )
            x_C, y_C, vz_C, _ = kinematics.iteratively_find_xy(
                self.x_G.clone(), self.y_G.clone(), self.cutout_C,
                self.lambda_rest, self.x_G, self.y_G,
                maxiter=niter, tol=0.0, **velocity, **disp_C
            )

            if not self._xy_converged(x_R, y_R, self.cutout_R, self.mcmc_foot_R, velocity, disp_R):
                return -np.inf
            if not self._xy_converged(x_C, y_C, self.cutout_C, self.mcmc_foot_C, velocity, disp_C):
                return -np.inf

            image_R = kinematics.bilinear_interpolte_intensity_torch(
                x_R, y_R, self.true_grism_R, self.cutout_R)
            image_C = kinematics.bilinear_interpolte_intensity_torch(
                x_C, y_C, self.true_grism_C, self.cutout_C)
            if not (torch.isfinite(image_R).all() and torch.isfinite(image_C).all()):
                return -np.inf
            sse = float(torch.sum(((image_R - image_C)[self.mcmc_mask])**2))

        # cache the model images so get_fitting_results() keeps working; note these
        # then describe the LAST evaluated sample, not the MAP
        self.x_R, self.y_R, self.vz_R = x_R, y_R, vz_R
        self.x_C, self.y_C, self.vz_C = x_C, y_C, vz_C
        self.image_R, self.image_C = image_R, image_C

        var = np.exp(2.0*ln_f)*self.mcmc_var_base
        return -0.5*sse/var - 0.5*self.n_mcmc_pix*np.log(2.0*np.pi*var)

    # mock data ---------------------------------------------------------------

    def set_params_dict(self, params):
        '''Write {full_dotted_key: value} into the current stage's cfg tensors.'''
        all_cfg = self._mcmc_stage_cfg()
        with torch.no_grad():
            for key, value in params.items():
                if key not in all_cfg:
                    raise KeyError(f'{key} is not a parameter of stage {self.current_stage+1}')
                all_cfg[key].tensor.data.fill_(float(value))

    def make_mock_data(self, intensity=None, theta=None, sigma_R=None, sigma_C=None,
                       sigma_v=0.0, n_vnodes=7, seed=None, inplace=True):
        '''
        Synthesise a mock R/C grism pair from a known source-plane intensity map
        and a known set of kinematic parameters.

        The fitter is a consistency comparison, not a generative model: it
        rectifies both grisms back to a shared source frame and differences them,
        so there is no model image to evaluate at theta_true. A mock therefore has
        to invert the relation the fit solves. At the fixed point of
        iteratively_find_xy, a source-plane position s maps to the grism pixel

            x_G = s_x + cutout[0] - dx_p - mdx,   mdx, mdy = forward_model(0, 0, lam)
            y_G = s_y + cutout[1] - dy_p - mdy,   lam = lambda_rest*(1 + vz(s)/c)

        Taking (mdx, mdy) from the very same forward_model closure the fit uses
        guarantees the inverse is exact. Note that kinematics.dispersion_model
        applies its dx/dy with the OPPOSITE sign convention to
        forward_dispersion_model, so it must not be used here: doing so recovers
        dx = -dx_true and looks like a broken sampler.

        Parameters
        ----------
        intensity : (h, w) source-plane line intensity. Defaults to the currently
                    rectified image_R, which has a realistic morphology and flux
                    scale. Its own noise enters both mocks identically and so
                    cancels in the R-C difference; only the noise added here is
                    independent.
        theta     : optional {full_dotted_key: value} written before synthesising,
                    and left in place afterwards.
        sigma_R, sigma_C : Gaussian noise added to each mock. Defaults to the
                    frozen noise model if set up, else a background estimate.
                    Pass 0 for a noiseless mock (measures the resampling floor).
        sigma_v   : intrinsic line-of-sight velocity dispersion in km/s. Each
                    source pixel then emits over a range of wavelengths rather
                    than a single one, which smears the trace along the dispersion
                    direction -- i.e. beam smearing. The velocity model has no
                    dispersion term, so fitting such a mock measures how much an
                    unmodelled dispersion biases the recovered parameters.
                    Implemented as an n_vnodes-point Gauss-weighted quadrature
                    over +-3 sigma_v.
        inplace   : replace self.true_grism_R/_C, keeping the originals for
                    restore_true_data().

        Returns (mock_R, mock_C) as tensors on the fitter's device.
        '''
        c = 299792.458  # km/s
        if theta is not None:
            self.set_params_dict(theta)

        if intensity is None:
            self._reset_state()
            with torch.no_grad():
                self.loss()
            intensity = self.image_R.detach().clone()
        intensity = torch.as_tensor(intensity, dtype=torch.float32)

        if sigma_R is None:
            sigma_R = getattr(self, 'sigma_R', None)
            if sigma_R is None:
                sigma_R = estimate_background_sigma(self.true_grism_R)
        if sigma_C is None:
            sigma_C = getattr(self, 'sigma_C', None)
            if sigma_C is None:
                sigma_C = estimate_background_sigma(self.true_grism_C)

        velocity = self._get_model_params('velocity')
        grism_shape = self.true_grism_R.shape
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        if sigma_v and sigma_v > 0:
            nodes = np.linspace(-3.0, 3.0, int(n_vnodes))
            weights = np.exp(-0.5*nodes**2)
            weights /= weights.sum()
            nodes = nodes*float(sigma_v)
        else:
            nodes, weights = np.array([0.0]), np.array([1.0])

        mocks = {}
        for pupil, cutout, sigma in [('R', self.cutout_R, sigma_R),
                                     ('C', self.cutout_C, sigma_C)]:
            disp = self._get_model_params(f'dispersion_{pupil}')
            h, w = cutout[3], cutout[2]
            with torch.no_grad():
                s_y, s_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                s_x = s_x.to(torch.float32)
                s_y = s_y.to(torch.float32)
                vz = kinematics.arctangent_disk_velocity_model(s_x, s_y, **velocity)
                zeros = torch.zeros_like(s_x)
                mock = torch.zeros(grism_shape, dtype=torch.float32)
                for dv, wt in zip(nodes, weights):
                    lambda_obs = self.lambda_rest*(1.0 + (vz + float(dv))/c)
                    mdx, mdy = disp['forward_model'](zeros, zeros, lambda_obs)
                    x_G = s_x + cutout[0] - disp['dx'] - mdx
                    y_G = s_y + cutout[1] - disp['dy'] - mdy
                    mock = mock + kinematics.bilinear_interpolte_intensity_torch(
                        x_G, y_G, intensity*float(wt), (0, 0, grism_shape[1], grism_shape[0])
                    )
                if sigma:
                    mock = mock + float(sigma)*torch.randn(
                        mock.shape, generator=gen, device=mock.device)
            mocks[pupil] = mock

        if inplace:
            if not hasattr(self, '_true_grism_backup'):
                self._true_grism_backup = (self.true_grism_R, self.true_grism_C)
            self.true_grism_R = mocks['R']
            self.true_grism_C = mocks['C']
            LOG.info(
                f'[mock] injected mock R/C (sigma_R={float(sigma_R):.4g}, '
                f'sigma_C={float(sigma_C):.4g}); call restore_true_data() to undo'
            )
        return mocks['R'], mocks['C']

    def restore_true_data(self):
        '''Undo an in-place make_mock_data().'''
        if not hasattr(self, '_true_grism_backup'):
            return False
        self.true_grism_R, self.true_grism_C = self._true_grism_backup
        del self._true_grism_backup
        return True

#%% --------------------------------------------------------------------------
# Image fitting subclass
# ----------------------------------------------------------------------------

class ImagesFitter(BaseFitter):

    def __init__(self, config_path: str, device=None):
        super().__init__(config_path, device)

    def _setup_data(self):

        # 1) Load psf and direct image data (except dx dy)
        self.all_filters = set(self.config['direct'].keys())

        # load psf data
        self.psfs = {} # {pid: {property: value}}
        for filter, psfs_cfg in self.config['psfs'].items():
            self.psfs[filter] = {}
            for pid, psf_cfg in psfs_cfg.items():
                psf_data = load_fits_data(psf_cfg['path'])
                psf_tensor = torch.tensor(
                    psf_data, dtype=torch.float32, device=self.device
                )
                psf_info = {}
                psf_info['psf'] = psf_tensor
                psf_info['oversample'] = psf_cfg['oversample']
                psf_info['image_map'] = [] # to be added in the next step
                self.psfs[filter][pid] = psf_info

        # load image data
        self.direct_images = {} # {filter: {iid: {property: value}}}
        for filter_name, imgs_cfg in self.config['direct'].items():
            self.direct_images[filter_name] = {}
            for iid, img_cfg in imgs_cfg.items():
                # register psf
                pid = img_cfg['psf']
                self.psfs[filter_name][pid]['image_map'].append(iid)
                # add image data
                img_data = load_fits_data(img_cfg['path'])
                try:
                    err_data = load_fits_data([img_cfg['path'][0], 'ERR'])
                # TODO: make error image into 
                except Exception:
                    LOG.warning(f"No err image for {img_cfg['path']}")
                    err_data = np.ones_like(img_data)
                if '_cutout' in img_cfg:
                    x, y, dx, dy = img_cfg['_cutout']
                    img_data = img_data[x:x+dx, y:y+dy]
                img_tensor = torch.tensor(
                    img_data, dtype=torch.float32, device=self.device
                )
                err_tensor = torch.tensor(
                    err_data, dtype=torch.float32, device=self.device
                )
                direct_info = {}
                direct_info['pid'] = pid
                direct_info['image'] = img_tensor
                direct_info['err'] = err_tensor
                direct_info['oversample'] = img_cfg['oversample']
                if self.psfs[filter_name][pid]['oversample'] < direct_info['oversample']:
                    raise ValueError('PSF is not well-sampled!')

                # ── 这里根据 unit 预计算 “物理量→像素” 转换因子 ──
                unit_factor = self._compute_unit_factor(filter_name, direct_info)
                direct_info['unit_factor'] = unit_factor

                self.direct_images[filter_name][iid] = direct_info

        # 2) Per-image grids: self.grids[filter][iid]
        self.grids = {}  # {filter: {iid: {'nx', 'ny', 'xx', 'yy'}}}

        for filter, imgs in self.direct_images.items():
            self.grids[filter] = {}
            for iid, direct_info in imgs.items():
                img = direct_info['image']
                nx, ny = img.shape
                yy, xx = torch.meshgrid(
                    torch.arange(nx, device=self.device, dtype=torch.float32),
                    torch.arange(ny, device=self.device, dtype=torch.float32),
                    indexing='ij'
                )
                self.grids[filter][iid] = {
                    'nx': nx,
                    'ny': ny,
                    'xx': xx,
                    'yy': yy,
                }

        # 3) Defaults & overrides
        defaults = [fs['default'] for fs in self.config['fitting']]
        overrides = [fs['override'] for fs in self.config['fitting']]

        # 4) Raw params
        _all_cfgs = [{} for _ in range(len(self.config['fitting']))]  # temp cfg for matching alias
        self.direct_cfgs_lists = {}
        self.psfs_cfgs_lists = {}
        self.sersic_cfgs_lists = {}
        self.psf_cfgs_lists = {}

        # add image params
        # NOTE: overrides will be mutated during iteration
        for filter, imgs_cfg in self.config['direct'].items():
            for iid, raw_dict in imgs_cfg.items():
                prefix = f'direct.{filter}.{iid}'  # 统一用点
                i_cfgs = build_param_config_dict_with_alias(
                    raw_dict=raw_dict,
                    default_cfgs=defaults,
                    overrides_list=overrides,
                    prefix=prefix,
                    all_cfgs=_all_cfgs,
                    device=self.device
                )
                self.direct_cfgs_lists[prefix] = i_cfgs

        # add psf params
        # NOTE: overrides will be mutated during iteration
        for filter, psfs_cfg in self.config['psfs'].items():
            for pid, raw_dict in psfs_cfg.items():
                prefix = f'psfs.{filter}.{pid}'  # e.g. psfs.f115w.psf0
                psf_cfgs = build_param_config_dict_with_alias(
                    raw_dict=raw_dict,
                    default_cfgs=defaults,
                    overrides_list=overrides,
                    prefix=prefix,
                    all_cfgs=_all_cfgs,
                    device=self.device
                )
                self.psfs_cfgs_lists[prefix] = psf_cfgs

        # add sersic params
        for cid, per_filter_dict in self.config['psf'].items():
            for filter_name, raw_dict in per_filter_dict.items():   # per_filter_dict: {'f115w': {...}, ...}
                prefix = f'psf.{cid}.{filter_name}'                 # e.g. psf.p0.f115w
                p_cfgs = build_param_config_dict_with_alias(
                    raw_dict=raw_dict,
                    default_cfgs=defaults,
                    overrides_list=overrides,
                    prefix=prefix,
                    all_cfgs=_all_cfgs,
                    device=self.device
                )
                self.psf_cfgs_lists[prefix] = p_cfgs

        # add point source params
        for cid, per_filter_dict in self.config['sersic'].items():
            for filter_name, raw_dict in per_filter_dict.items():
                prefix = f'sersic.{cid}.{filter_name}'              # sersic.s0.f115w
                s_cfgs = build_param_config_dict_with_alias(
                    raw_dict=raw_dict,
                    default_cfgs=defaults,
                    overrides_list=overrides,
                    prefix=prefix,
                    all_cfgs=_all_cfgs, 
                    device=self.device
                )
                self.sersic_cfgs_lists[prefix] = s_cfgs

        if len(overrides)>0: 
            unused_keys = set([key for stage in overrides for key in stage.keys()])
            LOG.warning(f'The following override keys are not used: {unused_keys}')

        # 5) Register
        self.param_config_lists = (
            self.psfs_cfgs_lists
            | self.psf_cfgs_lists
            | self.sersic_cfgs_lists
            | self.direct_cfgs_lists
        )

        # extra_cfg_lists = {'extra': [{'extra.offset': FitParamConfig(name='extra.offset', value=0, lr=0.0001, min=-1e10, max=1e10, fit=True )}]}
        # self.param_config_lists = self.param_config_lists | extra_cfg_lists

    def loss(self):

        # NOTE: self.*_cfg is already generated by _assign_cfgs_for_stage

        loss = 0

        # construct true image per psf
        for filter in self.all_filters:
            for pid, psf_info in self.psfs[filter].items():
                # 全局 psf 标定（scale / zp）
                psf_group = f'psfs.{filter}.{pid}'              # instrument PSF 的 param config id
                psf_params_global = self._get_model_params(psf_group)
                psf_scale = psf_params_global['scale']
                psf_zp = psf_params_global['zp']
                psf_tensor = (psf_info['psf'] - psf_zp) / psf_scale
                
                img_list = psf_info['image_map']

                for iid in img_list:
                    # ----- 该 iid 的数据和 grid -----
                    direct_info = self.direct_images[filter][iid]
                    this_image = direct_info['image']
                    this_err = direct_info['err']
                    nx_img, ny_img = this_image.shape

                    grid = self.grids[filter][iid]
                    xx, yy = grid['xx'], grid['yy']

                    # ----- oversample 因子：要求整除 -----
                    psf_oversample = psf_info['oversample']
                    img_oversample = direct_info['oversample']
                    if psf_oversample % img_oversample != 0:
                        raise ValueError(
                            f"PSF oversample ({psf_oversample}) is not an integer multiple "
                            f"of image oversample ({img_oversample}) for "
                            f"filter={filter}, iid={iid}"
                        )
                    downsample_factor = psf_oversample // img_oversample

                    direct_group = f"direct.{filter}.{iid}"
                    direct_params = self._get_model_params(direct_group)
                    # 显式按 key 取，避免 .values() 顺序问题
                    dx = direct_params['dx']
                    dy = direct_params['dy']
                    wt = direct_params['wt']
                    zp = direct_params['zp']

                    # 物理单位 → 像素
                    unit_factor = direct_info.get('unit_factor', 1.0)
                    dx_pix = dx * unit_factor
                    dy_pix = dy * unit_factor


                    # ----- 在该 iid 的网格上构建 oversampled model -----
                    model = 0

                    psf_cid_list = self.config['psf'].keys()   # ['p0', 'p1', ...]
                    for psf_cid in psf_cid_list:
                        psf_group = f'psf.{psf_cid}.{filter}' # e.g. psf.p0.f115w
                        psf_params = self._get_model_params(psf_group)
                        psf_params_pix = self._convert_geom_unit_to_pix(
                            psf_params,
                            geom_keys=('x_psf', 'y_psf'),
                            unit_factor=direct_info['unit_factor'],
                        )
                        psf_model = galaxy.full_psf_model_torch(
                            xx, yy, psf_tensor, **psf_params_pix
                        )
                        model += psf_model

                    sersic_cid_list = self.config['sersic'].keys()
                    for sersic_cid in sersic_cid_list:
                        sersic_group = f'sersic.{sersic_cid}.{filter}'
                        sersic_params = self._get_model_params(sersic_group)
                        sersic_params_pix = self._convert_geom_unit_to_pix(
                            sersic_params,
                            geom_keys=('x0', 'y0', 'R_e'),
                            unit_factor=direct_info['unit_factor'],
                        )
                        sersic_model = galaxy.full_sersic_model_torch(
                            xx, yy, psf_tensor, **sersic_params_pix
                        )
                        model += sersic_model

                    # ----- 下采样 + shift + loss（基本保持原来的逻辑） -----
                    combined_image_hat = torch.fft.fft2(model)
                    this_combined_image_hat = utils.fft_phase_shift(
                        combined_image_hat,
                        dy_pix * downsample_factor,
                        dx_pix * downsample_factor,
                    )
                    this_model_hat = utils.fft_bin(
                        this_combined_image_hat,
                        downsample_factor,
                    )
                    this_model = torch.fft.ifft2(this_model_hat).real

                    # this_image = (this_image - zp)/wt
                    # res = this_image - this_model - offset
                    res = this_image - this_model
                    # residual loss ------
                    res_loss = torch.sum((this_model - this_image)**2)
                    # res_loss = torch.sum(torch.abs((res)))
                    # loss += res_loss
                    # chi2 loss ------
                    # this_err[~torch.isfinite(this_err)] = 1
                    finite_mask = torch.isfinite(this_err)
                    # chi2_loss = torch.nansum(torch.abs(res[finite_mask]/this_err[finite_mask]))
                    # Cauchy loss（Robust χ²）
                    c = 2.0
                    # chi2_loss = torch.nansum(torch.log(1 + (res[finite_mask]/this_err[finite_mask])**2 / c**2))
                    chi2_loss = torch.nansum((res[finite_mask]/this_err[finite_mask])**2)
                    loss += chi2_loss

        return loss
    
    # getters ----------------------------------------------------------------

    def get_params(self):
        '''
        Return two dicts of final parameter values (pure Python floats),
        first for all Sérsic components, then for all PSF components.
        Each is a mapping cid -> { param_name: value, … }.
        '''
        sersic = {
            cid: {
                k.split('.')[-1]: v.value
                for k, v in cfg_list[self.current_stage].items()
            }
            for cid, cfg_list in self.sersic_cfgs_lists.items()
        }
        psf = {
            cid: {
                k.split('.')[-1]: v.value
                for k, v in cfg_list[self.current_stage].items()
            }
            for cid, cfg_list in self.psf_cfgs_lists.items()
        }
        direct = {
            iid: {
                k.split('.')[-1]: v.value
                for k, v in cfg_list[self.current_stage].items()
            }
            for iid, cfg_list in self.direct_cfgs_lists.items()
        }
        return sersic, psf, direct

    def get_true_image(self, filter=None, iid=None): 
        '''
        Return the original (data) image as a NumPy array.
        '''
        if not filter: 
            filter = next(iter(self.direct_images.keys()))
        if not iid: 
            iid = next(iter(self.direct_images[filter].keys()))
        true_image = self.direct_images[filter][iid]['image']
        return true_image.detach().cpu().numpy()

    def get_true_images(self):
        '''
        Return the original (data) image as a dict of NumPy array.
        '''
        true_images = {}
        for filter in self.all_filters:
            true_images[filter] = {}
            for iid in self.direct_images[filter].keys(): 
                true_image = self.get_true_image(filter, iid) 
                true_images[filter][iid] = true_image

        return true_images

    def get_fitted_component(self, filter=None, iid=None):
        """
        Reconstruct the fitted models (sum of PSF and/or Sérsic components)
        and return them as NumPy arrays on the *downsampled* data grid
        for a given (filter, iid).

        Parameters
        ----------
        filter : str, optional
            Filter name. If None, use the first filter in self.direct_images.
        iid : str, optional
            Image ID within that filter. If None, use the first iid for that filter.

        Returns
        -------
        sersic_models : dict
            {sersic_cid: 2D ndarray} on the data image grid.
        psf_models : dict
            {psf_cid: 2D ndarray} on the data image grid.
        """

        # ----- 1) choose default filter / iid -----
        if filter is None:
            filter = next(iter(self.direct_images.keys()))
        if iid is None:
            iid = next(iter(self.direct_images[filter].keys()))

        # ----- 2) image & PSF info -----
        direct_info = self.direct_images[filter][iid]
        pid = direct_info['pid']                   # e.g. "psf0"
        psf_info = self.psfs[filter][pid]          # instrument PSF for this filter

        # instrument PSF 标定参数：psfs.<filter>.<pid>.*
        psfs_group = f'psfs.{filter}.{pid}'        # e.g. "psfs.f115w.psf0"
        psfs_params = self._get_model_params(psfs_group)
        psf_scale = psfs_params['scale']
        psf_zp = psfs_params['zp']

        # 归一化 / 零点校正后的 PSF kernel
        psf_tensor = (psf_info['psf'] - psf_zp) / psf_scale

        # 该 iid 的数据与 grid
        this_image = direct_info['image']
        nx, ny = this_image.shape
        psf_oversample = psf_info['oversample']
        img_oversample = direct_info['oversample']
        if psf_oversample % img_oversample != 0:
            raise ValueError(
                f"PSF oversample ({psf_oversample}) is not an integer multiple "
                f"of image oversample ({img_oversample}) for "
                f"filter={filter}, iid={iid}"
            )
        downsample_factor = psf_oversample // img_oversample

        grid = self.grids[filter][iid]
        xx, yy = grid['xx'], grid['yy']

        # direct.<filter>.<iid>.*
        direct_group = f'direct.{filter}.{iid}'
        direct_params = self._get_model_params(direct_group)
        dx = direct_params['dx']
        dy = direct_params['dy']

        # 单位转换
        unit_factor = direct_info.get('unit_factor', 1.0)
        dx_pix = dx * unit_factor
        dy_pix = dy * unit_factor
        # TODO: reopen wt and scale
        # wt = direct_params['wt']
        # zp = direct_params['zp']

        # ----- 3) 逐 component 生成 oversampled model，再 downsample -----
        psf_models = {}
        sersic_models = {}

        # 3a) galaxy PSF components: psf.<cid>.<filter>.*
        for psf_cid in self.config['psf'].keys():
            psf_group = f'psf.{psf_cid}.{filter}'     # e.g. "psf.p0.f115w"
            psf_params = self._get_model_params(psf_group)
            psf_params_pix = self._convert_geom_unit_to_pix(
                psf_params,
                geom_keys=('x_psf', 'y_psf'),
                unit_factor=unit_factor
            )
            psf_model_oversampled = galaxy.full_psf_model_torch(
                xx, yy, psf_tensor, **psf_params_pix
            )

            this_component = utils.downsample_with_shift_and_size(
                x=psf_model_oversampled,
                factor=downsample_factor,
                out_size=(nx, ny),
                shift=(dx_pix, dy_pix),
            )
            psf_models[psf_cid] = this_component.detach().cpu().numpy()

        # 3b) Sérsic components: sersic.<cid>.<filter>.*
        for sersic_cid in self.config['sersic'].keys():
            sersic_group = f'sersic.{sersic_cid}.{filter}'   # e.g. "sersic.s0.f115w"
            sersic_params = self._get_model_params(sersic_group)
            sersic_params_pix = self._convert_geom_unit_to_pix(
                sersic_params,
                geom_keys=('x0', 'y0', 'R_e'),
                unit_factor=unit_factor
            )
            sersic_model_oversampled = galaxy.full_sersic_model_torch(
                xx, yy, psf_tensor, **sersic_params_pix
            )

            this_component = utils.downsample_with_shift_and_size(
                x=sersic_model_oversampled,
                factor=downsample_factor,
                out_size=(nx, ny),
                shift=(dx_pix, dy_pix),
            )
            sersic_models[sersic_cid] = this_component.detach().cpu().numpy()

        return sersic_models, psf_models

    def get_fitted_components(self):
        '''
        return all the get_fitted_component
        '''
        models = {}
        all_sersic_models = {}
        all_psf_models = {}
        for filter in self.all_filters:
            models[filter] = {}
            all_psf_models[filter] = {}
            for iid in self.direct_images[filter].keys(): 
                sersic_models, psf_models = self.get_fitted_component(filter, iid)
                all_sersic_models[filter][iid] = sersic_models
                all_psf_models[filter][iid] = psf_models

        return all_sersic_models, all_psf_models

    def get_fitted_model(self, filter=None, iid=None):
        '''
        Reconstruct the final fitted model (sum of PSF + Sérsic components)
        and return it as a NumPy array.
        '''
        sersic_models, psf_models = self.get_fitted_component(filter, iid)
        model = np.sum(list(sersic_models.values())+list(psf_models.values()), axis=0)
        return model

    def get_fitted_models(self):
        '''
        return all the get_fitted_model
        '''
        models = {}
        for filter in self.all_filters:
            models[filter] = {}
            for iid in self.direct_images[filter].keys(): 
                model = self.get_fitted_model(filter, iid)
                models[filter][iid] = model
        return models
    
    def get_oversampled_fitted_component(self, filter=None, pid=None):
        # TODO: Code can be fused with above??

        if not filter: 
            filter = next(iter(self.direct_images.keys()))
        if not pid: 
            pid = next(iter(self.psfs[filter].keys()))

        # image-specifig configs
        
        psf_info = self.psfs[filter][pid]
        psf_tensor = psf_info['psf']

        # 这里没有 iid，只需挑一个 iid 来拿 grid（假设同一个 filter+pid 下各 iid 尺寸相同）
        iid0 = psf_info['image_map'][0]
        grid = self.grids[filter][iid0]
        xx, yy = grid['xx'], grid['yy']

        psf_models = {}
        sersic_models = {}

        psf_cid_list = self.config['psf'].keys()
        for psf_cid in psf_cid_list: 
            psf_params = self._get_model_params(psf_cid)
            psf_model = galaxy.full_psf_model_torch(
                xx, yy, psf_tensor, **psf_params
            )
            psf_models[psf_cid] = psf_model.detach().cpu().numpy()
        
        sersic_cid_list = self.config['sersic'].keys()
        for sersic_cid in sersic_cid_list: 
            sersic_params = self._get_model_params(sersic_cid)
            sersic_model = galaxy.full_sersic_model_torch(
                xx, yy, psf_tensor, **sersic_params
            )
            sersic_models[sersic_cid] = sersic_model.detach().cpu().numpy()
        
        return sersic_models, psf_models
    
    def get_oversampled_fitted_model(self, filter=None, iid=None):
        sersic_models, psf_models = self.get_oversampled_fitted_component(filter, iid)
        model = np.sum(list(sersic_models.values())+list(psf_models.values()), axis=0)
        return model
    
    # ======================================================================
    # helper: 为当前 filter 找一个参考 redshift（用于 kpc 转换）
    # ======================================================================
    def _get_reference_z_for_filter(self, filter_name: str):
        """
        为某个 filter 选一个参考 redshift：
          1. 优先从 sersic 组件中找 sersic.<cid>.<filter>.z
          2. 找不到时 fallback 到 summary.z（如果有的话）
          3. 都没有则返回 None
        """
        # 1) 在 sersic 组件中寻找
        try:
            for cid, per_filter in self.config.get('sersic', {}).items():
                if filter_name in per_filter:
                    z_here = per_filter[filter_name].get('z', None)
                    if z_here is not None:
                        return z_here
        except Exception:
            pass

        # TODO: now we allow no z so z can be _z. change all optional variable occurences
        # 2) fallback: summary.z
        if getattr(self, 'z', None) is not None:
            LOG.warning(
                f"[ImagesFitter] No per-component z found for filter={filter_name!r}; "
                "fall back to summary.z for kpc conversion."
            )
            return self.z

        # 3) 全部失败
        return None

    # ======================================================================
    # helper: 根据 self.unit 计算某个 direct image 的 “物理量 → 像素” 转换因子
    # ======================================================================
    def _compute_unit_factor(self, filter_name: str, direct_info: dict) -> float:
        """
        返回 factor:   pix = factor * (stored_unit)

        其中 stored_unit 是当前 self.unit 下的单位：
          - unit == 'pix'   → factor = 1
          - unit == 'arcsec'→ factor = pix_per_arcsec
          - unit == 'kpc'   → factor = pix_per_kpc (需要 redshift)
        """
        oversample = direct_info['oversample']

        # 1) 纯像素：不需要转换
        if self.unit == 'pix':
            return 1.0

        # 2) arcsec：依赖 pixel scale
        if self.unit == 'arcsec':
            try:
                pixscale_arcsec = utils.nircam_miri_pixscale(
                    filter_name, oversample=oversample
                )  # arcsec / pixel
                # 我们希望： dx_pix = dx_arcsec * pix_per_arcsec
                pix_per_arcsec = 1.0 / float(pixscale_arcsec)
                return pix_per_arcsec
            except Exception as e:
                LOG.warning(
                    f"[ImagesFitter] nircam_miri_pixscale failed for "
                    f"filter={filter_name!r}, oversample={oversample}: {e}; "
                    "use factor=1."
                )
                return 1.0

        # 3) kpc：需要 redshift
        if self.unit == 'kpc':
            z_ref = self._get_reference_z_for_filter(filter_name)
            if z_ref is None:
                LOG.warning(
                    f"[ImagesFitter] Cannot find reference z for filter={filter_name!r} "
                    "under unit='kpc'; use factor=1."
                )
                return 1.0

            try:
                kpc_per_pix, pix_per_kpc, pixscale_arcsec = utils.kpc_pix_scale(
                    z_ref, filter_name, oversample=oversample
                )
                # 我们希望： dx_pix = dx_kpc * pix_per_kpc
                return float(pix_per_kpc)
            except Exception as e:
                LOG.warning(
                    f"[ImagesFitter] kpc_pix_scale failed for "
                    f"filter={filter_name!r}, z={z_ref}, oversample={oversample}: {e}; "
                    "use factor=1."
                )
                return 1.0

        # 理论上不会走到这里，因为 BaseFitter 已经保证 unit 合法
        return 1.0

    def _convert_geom_unit_to_pix(
        self,
        params: dict,
        *,
        geom_keys: tuple,
        unit_factor: float
    ) -> dict:
        """
        Convert geometry-related parameters from current unit to pixel.

        Parameters
        ----------
        params : dict
            Output of self._get_model_params(...), values are torch tensors.
        geom_keys : tuple
            Parameter names that represent geometric quantities
            (e.g. ('x0','y0','R_e') or ('x_psf','y_psf')).
        unit_factor : float
            Conversion factor such that:
                value_pix = value * unit_factor

        Returns
        -------
        params_pix : dict
            A shallow-copied dict safe to pass into galaxy models.
        """
        params_pix = {}

        for k, v in params.items():
            if k in geom_keys:
                params_pix[k] = v * unit_factor
            else:
                params_pix[k] = v

        return params_pix
