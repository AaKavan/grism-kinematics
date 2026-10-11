#!/usr/bin/env python
'''Re-report V_rot under a stated inclination uncertainty.

    python diagnostics/inclination_budget.py [--sigma 0.0436 0.105 0.149]

The grism measures V*sin(i), not V_rot. Turning one into the other needs an
inclination from imaging, so sigma_i propagates straight into V_rot and into
M_dyn, which goes as V_rot^2. The pipeline already marginalises over it --
headline V_rot_lo/V_rot_hi are the marginalised percentiles, not a linear
error bar -- but it does so with METHOD_SCATTER = 0.0436 rad for every source,
a constant whose stated source (MCMC_STATUS.md) is no longer in the repo.

Two reasons to doubt 0.0436 rad (2.5 deg):

  * The formal pysersic errors it is maxed against are not credible where the
    Sersic index rails at the prior floor n = 0.65. Railing collapses the
    posterior, and ID15665/ID3700/ID8910 then report q to +-0.0002..0.006 --
    0.02-0.9% on an axis ratio from a PSF-convolved fit. The two unrailed fits
    (ID16887, ID21636) scatter 30-50x wider, which is what an honest q looks
    like.
  * Two independent Sersic fits of ID8910 F444W -- this pipeline's, and
    diagnostics/measure_inclination.py -- give inc = 0.8294 and 0.6803 rad.
    They differ by 0.149 rad (8.5 deg), 3.4x the adopted sigma. If the two are
    comparably good, each carries ~0.149/sqrt(2) = 0.105 rad.

So this reports V_rot across a range of sigma_i rather than asserting one.
V_rot = V_sini/sin(i) is nonlinear and its tail is one-sided -- small i inflates
V_rot without bound -- so percentiles of the marginalised samples are quoted,
never mean +- sd.
'''
import argparse
import glob
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 4.302e-6                                  # kpc (km/s)^2 / M_sun
INC_LO, INC_HI = 0.0873, 1.4835               # the pipeline's own clip, radians


def dynamical_mass(r_kpc, V_rot, R_v):
    '''M(<r) for the arctangent rotation curve, in M_sun.'''
    V = (2/np.pi)*V_rot*np.arctan(r_kpc/R_v)
    return V**2*r_kpc/G


def marginalise(vsini, inc_samples, r_v_kpc, sigma, rng):
    '''V_rot and log M, marginalised over inclination at this sigma.'''
    i = rng.choice(inc_samples, size=vsini.size, replace=True)
    i = np.clip(i + rng.normal(0.0, sigma, vsini.size), INC_LO, INC_HI)
    v = vsini/np.sin(i)
    return v, np.log10(dynamical_mass(5.0, v, r_v_kpc))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sigma', type=float, nargs='+',
                    default=[0.0436, 0.105, 0.149],
                    help='inclination sigmas in radians (default: adopted, '
                         'two-method/sqrt2, two-method)')
    ap.add_argument('--seed', type=int, default=15665)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(REPO, 'results/ID*/ID*_derived_posterior.npz')))
    if not files:
        print('no derived_posterior.npz found'); return 1

    print('V_rot marginalised over inclination, by assumed sigma_i')
    print('(median with 16th/84th percentiles; M_dyn(<5 kpc))\n')
    hdr = '%-9s %8s %7s  ' % ('ID', 'V*sin i', 'i[deg]')
    hdr += ''.join('%-42s' % ('--- sigma_i = %.3f rad ---' % s) for s in args.sigma)
    print(hdr)
    print('%-9s %8s %7s  ' % ('', '', '') +
          ''.join('%-42s' % '   V_rot            logM M(<5kpc)' for _ in args.sigma))

    for f in files:
        sid = os.path.basename(f).split('_')[0]
        d = np.load(f, allow_pickle=True)
        vsini, r_v = d['Vsini'], d['R_v_kpc']
        sers = os.path.join(os.path.dirname(f),
                            f'{sid}_sersic_F444W_pysersic_fixedinc.npz')
        if not os.path.exists(sers):
            continue
        inc_s = np.load(sers, allow_pickle=True)['inc']
        row = '%-9s %8.1f %7.1f  ' % (sid, np.median(vsini),
                                      np.degrees(np.median(inc_s)))
        for s in args.sigma:
            rng = np.random.default_rng(args.seed)
            v, lm = marginalise(vsini, inc_s, r_v, s, rng)
            vl, vm, vh = np.percentile(v, [16, 50, 84])
            ml, mm, mh = np.percentile(lm, [16, 50, 84])
            # the median barely moves -- a centred sigma_i widens V_rot without
            # shifting it. The error bar is the whole story, so quote that.
            row += '%6.0f -%-3.0f/+%-4.0f %5.2f -%.2f/+%-4.2f ' % (
                vm, vm - vl, vh - vm, mm, mm - ml, mh - mm)
        print(row)

    print('\nsigma_i = 0.0436 is what the shipped headline numbers assume.')
    print('A larger sigma widens V_rot asymmetrically: sin(i) in the denominator')
    print('means the +side grows faster than the -side, and M_dyn goes as V^2.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
