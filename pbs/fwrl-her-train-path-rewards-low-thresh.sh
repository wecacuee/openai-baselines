#PBS -N fwrl-her-train-path-rewards  # Any name to identify your job
#PBS -j oe                   # Join error and output files for convinience
#PBS -l walltime=1000:00:00     # Keep walltime big enough to finish the job
#PBS -l nodes=1:ppn=6:gpus=1 # nodes requested: Processor per node: gpus requested
#PBS -S /bin/bash            # Shell to use
#PBS -m a                  # Mail to <user>@umich.edu on abort, begin and end
#PBS -M dhiman@umich.edu     # Email id to alert
#PBS -o /z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/pbs/
#
# #PBS -q fluxg              # Not required for blindspot but for flux
# #PBS -A jjcorso_fluxg      # Not required for blindspot but for flux
#PBS -t 0-39

echo "starting pbs script"
echo "running on $(hostname)"
VERSION=0.5.0
PROJNAME=openai-baselines
MID_DIR=/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/
for d in src-$VERSION pbs build; do mkdir -p $MID_DIR/$d; done
cd $MID_DIR/src-$VERSION/
git clone $MID_DIR/git/ $PROJNAME
cd $PROJNAME
git pull
. setup.sh
python <<EOF
from baselines.her.experiment.train_many import (exp_conf_path_reward_low_thresh,
                                                 run_one_experiment)
exp_conf, common_kwargs = exp_conf_path_reward_low_thresh(num_cpu = 6)
experiments = list(exp_conf.items())
experiment_id = 0 + $PBS_ARRAYID
print("Running experiment no {}/{} with {}".format(
    experiment_id, len(experiments), experiments[experiment_id]))
confname, conf = experiments[experiment_id]
run_one_experiment(confname, conf, common_kwargs)
EOF
echo "end of pbs script"
