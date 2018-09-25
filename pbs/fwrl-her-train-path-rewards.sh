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
VERSION=0.1.0
PROJNAME=openai-baselines
MID_DIR=/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/
for d in src-$VERSION pbs build; do mkdir -p $MID_DIR/$d; done
cd $MID_DIR/src-$VERSION/
git clone $MID_DIR/git/ $PROJNAME
cd $PROJNAME
git pull
. setup.sh
python <<EOF
from baselines.her.experiment.train_many import (exp_conf_path_reward,
                                                 run_one_experiment)
exp_conf, common_kwargs = exp_conf_path_reward(num_cpu = 6, gitrev = "d0a8dd4")
experiments = list(exp_conf.items())
experiment_id = 0 + $PBS_ARRAYID
print("Running experiment no {}/{} with {}".format(
    experiment_id, len(experiments), experiments[experiment_id]))
confname, conf = experiments[experiment_id]
run_one_experiment(confname, conf, common_kwargs)
EOF
echo "end of pbs script"

# >>> print("\n".join(map(str, enumerate(exp_conf_path_reward()[0].items()))))
# (0, ('FetchPickAndPlaceSparse-v1-dqst', {'env': 'FetchPickAndPlaceSparse-v1', 'loss_term': 'dqst'}))
# (1, ('FetchPickAndPlaceSparse-v1-fwrl', {'env': 'FetchPickAndPlaceSparse-v1', 'loss_term': 'fwrl'}))
# (2, ('FetchPickAndPlaceSparse-v1-ddpg', {'env': 'FetchPickAndPlaceSparse-v1', 'loss_term': 'ddpg'}))
# (3, ('FetchPickAndPlacePR-v1-dqst', {'env': 'FetchPickAndPlacePR-v1', 'loss_term': 'dqst'}))
# (4, ('FetchPickAndPlacePR-v1-fwrl', {'env': 'FetchPickAndPlacePR-v1', 'loss_term': 'fwrl'}))
# (5, ('FetchPickAndPlacePR-v1-ddpg', {'env': 'FetchPickAndPlacePR-v1', 'loss_term': 'ddpg'}))
# (6, ('HandReachSparse-v0-dqst', {'env': 'HandReachSparse-v0', 'loss_term': 'dqst'}))
# (7, ('HandReachSparse-v0-fwrl', {'env': 'HandReachSparse-v0', 'loss_term': 'fwrl'}))
# (8, ('HandReachSparse-v0-ddpg', {'env': 'HandReachSparse-v0', 'loss_term': 'ddpg'}))
# (9, ('HandReachPR-v0-dqst', {'env': 'HandReachPR-v0', 'loss_term': 'dqst'}))
# (10, ('HandReachPR-v0-fwrl', {'env': 'HandReachPR-v0', 'loss_term': 'fwrl'}))
# (11, ('HandReachPR-v0-ddpg', {'env': 'HandReachPR-v0', 'loss_term': 'ddpg'}))
# (12, ('HandManipulateBlockSparse-v0-dqst', {'env': 'HandManipulateBlockSparse-v0', 'loss_term': 'dqst'}))
# (13, ('HandManipulateBlockSparse-v0-fwrl', {'env': 'HandManipulateBlockSparse-v0', 'loss_term': 'fwrl'}))
# (14, ('HandManipulateBlockSparse-v0-ddpg', {'env': 'HandManipulateBlockSparse-v0', 'loss_term': 'ddpg'}))
# (15, ('HandManipulateBlockPR-v0-dqst', {'env': 'HandManipulateBlockPR-v0', 'loss_term': 'dqst'}))
# (16, ('HandManipulateBlockPR-v0-fwrl', {'env': 'HandManipulateBlockPR-v0', 'loss_term': 'fwrl'}))
# (17, ('HandManipulateBlockPR-v0-ddpg', {'env': 'HandManipulateBlockPR-v0', 'loss_term': 'ddpg'}))
# (18, ('HandManipulatePenSparse-v0-dqst', {'env': 'HandManipulatePenSparse-v0', 'loss_term': 'dqst'}))
# (19, ('HandManipulatePenSparse-v0-fwrl', {'env': 'HandManipulatePenSparse-v0', 'loss_term': 'fwrl'}))
# (20, ('HandManipulatePenSparse-v0-ddpg', {'env': 'HandManipulatePenSparse-v0', 'loss_term': 'ddpg'}))
# (21, ('HandManipulatePenPR-v0-dqst', {'env': 'HandManipulatePenPR-v0', 'loss_term': 'dqst'}))
# (22, ('HandManipulatePenPR-v0-fwrl', {'env': 'HandManipulatePenPR-v0', 'loss_term': 'fwrl'}))
# (23, ('HandManipulatePenPR-v0-ddpg', {'env': 'HandManipulatePenPR-v0', 'loss_term': 'ddpg'}))
# (24, ('HandManipulateEggSparse-v0-dqst', {'env': 'HandManipulateEggSparse-v0', 'loss_term': 'dqst'}))
# (25, ('HandManipulateEggSparse-v0-fwrl', {'env': 'HandManipulateEggSparse-v0', 'loss_term': 'fwrl'}))
# (26, ('HandManipulateEggSparse-v0-ddpg', {'env': 'HandManipulateEggSparse-v0', 'loss_term': 'ddpg'}))
# (27, ('HandManipulateEggPR-v0-dqst', {'env': 'HandManipulateEggPR-v0', 'loss_term': 'dqst'}))
# (28, ('HandManipulateEggPR-v0-fwrl', {'env': 'HandManipulateEggPR-v0', 'loss_term': 'fwrl'}))
# (29, ('HandManipulateEggPR-v0-ddpg', {'env': 'HandManipulateEggPR-v0', 'loss_term': 'ddpg'}))
