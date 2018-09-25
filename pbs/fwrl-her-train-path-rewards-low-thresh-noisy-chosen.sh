#PBS -N fwrl-her-train-path-rewards-low-thresh-noisy-chosen  # Any name to identify your job
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
VERSION=0.3.0
PROJNAME=openai-baselines
MID_DIR=/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/
for d in src-$VERSION pbs build; do mkdir -p $MID_DIR/$d; done
cd $MID_DIR/src-$VERSION/
git clone $MID_DIR/git/ $PROJNAME
cd $PROJNAME
git pull
. setup.sh
python <<EOF
from baselines.her.experiment.train_many import (exp_conf_path_reward_low_thresh_noisy_chosen ,
                                                 run_one_experiment)
exp_conf, common_kwargs = exp_conf_path_reward_low_thresh_noisy_chosen(num_cpu = 6, gitrev = "6efc1de")
experiments = list(exp_conf.items())
experiment_id = 0 + $PBS_ARRAYID
print("Running experiment no {}/{} with {}".format(
    experiment_id, len(experiments), experiments[experiment_id]))
confname, conf = experiments[experiment_id]
run_one_experiment(confname, conf, common_kwargs)
EOF
echo "end of pbs script"

# >>> print("\n".join(map(str, enumerate(exp_conf_path_reward_low_thresh_noisy_chosen()[0].items()))))
# (0, ('FetchReach-v1-fwrl', {'env': 'FetchReach-v1', 'loss_term': 'fwrl'}))
# (1, ('FetchReach-v1-ddpg', {'env': 'FetchReach-v1', 'loss_term': 'ddpg'}))
# (2, ('FetchPush-v1-fwrl', {'env': 'FetchPush-v1', 'loss_term': 'fwrl'}))
# (3, ('FetchPush-v1-ddpg', {'env': 'FetchPush-v1', 'loss_term': 'ddpg'}))
# (4, ('FetchSlide-v1-fwrl', {'env': 'FetchSlide-v1', 'loss_term': 'fwrl'}))
# (5, ('FetchSlide-v1-ddpg', {'env': 'FetchSlide-v1', 'loss_term': 'ddpg'}))
# (6, ('FetchPickAndPlace-v1-fwrl', {'env': 'FetchPickAndPlace-v1', 'loss_term': 'fwrl'}))
# (7, ('FetchPickAndPlace-v1-ddpg', {'env': 'FetchPickAndPlace-v1', 'loss_term': 'ddpg'}))
# (8, ('HandReach-v0-fwrl', {'env': 'HandReach-v0', 'loss_term': 'fwrl'}))
# (9, ('HandReach-v0-ddpg', {'env': 'HandReach-v0', 'loss_term': 'ddpg'}))
# (10, ('HandManipulateBlockRotateXYZ-v0-fwrl', {'env': 'HandManipulateBlockRotateXYZ-v0', 'loss_term': 'fwrl'}))
# (11, ('HandManipulateBlockRotateXYZ-v0-ddpg', {'env': 'HandManipulateBlockRotateXYZ-v0', 'loss_term': 'ddpg'}))
# (12, ('HandManipulatePenRotate-v0-fwrl', {'env': 'HandManipulatePenRotate-v0', 'loss_term': 'fwrl'}))
# (13, ('HandManipulatePenRotate-v0-ddpg', {'env': 'HandManipulatePenRotate-v0', 'loss_term': 'ddpg'}))
# (14, ('HandManipulateEggFull-v0-fwrl', {'env': 'HandManipulateEggFull-v0', 'loss_term': 'fwrl'}))
# (15, ('HandManipulateEggFull-v0-ddpg', {'env': 'HandManipulateEggFull-v0', 'loss_term': 'ddpg'}))
# (16, ('FetchReach-v1-dqst', {'env': 'FetchReach-v1', 'loss_term': 'dqst'}))
# (17, ('FetchPush-v1-dqst', {'env': 'FetchPush-v1', 'loss_term': 'dqst'}))
# (18, ('FetchSlide-v1-dqst', {'env': 'FetchSlide-v1', 'loss_term': 'dqst'}))
# (19, ('FetchPickAndPlace-v1-dqst', {'env': 'FetchPickAndPlace-v1', 'loss_term': 'dqst'}))
# (20, ('HandReach-v0-dqst', {'env': 'HandReach-v0', 'loss_term': 'dqst'}))
# (21, ('HandManipulateBlockRotateXYZ-v0-dqst', {'env': 'HandManipulateBlockRotateXYZ-v0', 'loss_term': 'dqst'}))
# (22, ('HandManipulatePenRotate-v0-dqst', {'env': 'HandManipulatePenRotate-v0', 'loss_term': 'dqst'}))
# (23, ('HandManipulateEggFull-v0-dqst', {'env': 'HandManipulateEggFull-v0', 'loss_term': 'dqst'}))
# (24, ('FetchReachPR-v1-dqst', {'env': 'FetchReachPR-v1', 'loss_term': 'dqst'}))
# (25, ('FetchPushPR-v1-dqst', {'env': 'FetchPushPR-v1', 'loss_term': 'dqst'}))
# (26, ('FetchSlidePR-v1-dqst', {'env': 'FetchSlidePR-v1', 'loss_term': 'dqst'}))
# (27, ('FetchPickAndPlacePR-v1-dqst', {'env': 'FetchPickAndPlacePR-v1', 'loss_term': 'dqst'}))
# (28, ('HandReachPR-v0-dqst', {'env': 'HandReachPR-v0', 'loss_term': 'dqst'}))
# (29, ('HandManipulateBlockRotateXYZPR-v0-dqst', {'env': 'HandManipulateBlockRotateXYZPR-v0', 'loss_term': 'dqst'}))
# (30, ('HandManipulatePenRotatePR-v0-dqst', {'env': 'HandManipulatePenRotatePR-v0', 'loss_term': 'dqst'}))
# (31, ('HandManipulateEggFullPR-v0-dqst', {'env': 'HandManipulateEggFullPR-v0', 'loss_term': 'dqst'}))
