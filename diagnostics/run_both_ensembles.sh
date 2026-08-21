#!/bin/zsh
cd /Users/aakavan/Astro_Research/Galaxy_Kinematics/DINGO_development
export OMP_NUM_THREADS=1
echo "=== A: prior-free  $(date +%H:%M) ==="
/opt/anaconda3/envs/DINGO/bin/python validate_mcmc.py ensemble -K 20 --nproc 6 \
  --config ID15665/config_kinematics_noincprior.yaml \
  --outdir mcmc_results/ens_noprior > validate_ens_noprior.log 2>&1
echo "A exit=$?  $(date +%H:%M)"
echo "=== B: photometric inc prior retained  $(date +%H:%M) ==="
/opt/anaconda3/envs/DINGO/bin/python validate_mcmc.py ensemble -K 20 --nproc 6 \
  --config ID15665/config_kinematics.yaml \
  --outdir mcmc_results/ens_prior > validate_ens_prior.log 2>&1
echo "B exit=$?  $(date +%H:%M)"
