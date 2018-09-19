# Hindsight Experience Replay
For details on Hindsight Experience Replay (HER), please read the [paper](https://arxiv.org/abs/1707.01495).

## How to use Hindsight Experience Replay

### Getting started
Training an agent is very simple:
```bash
python -m baselines.her.experiment.train
```
This will train a DDPG+HER agent on the `FetchReach` environment.
You should see the success rate go up quickly to `1.0`, which means that the agent achieves the
desired goal in 100% of the cases.
The training script logs other diagnostics as well and pickles the best policy so far (w.r.t. to its test success rate),
the latest policy, and, if enabled, a history of policies every K epochs.

To inspect what the agent has learned, use the play script:
```bash
python -m baselines.her.experiment.play /path/to/an/experiment/policy_best.pkl
```
You can try it right now with the results of the training step (the script prints out the path for you).
This should visualize the current policy for 10 episodes and will also print statistics.


### Reproducing results
In order to reproduce the results from [Plappert et al. (2018)](https://arxiv.org/abs/1802.09464), run the following command:
```bash
python -m baselines.her.experiment.train --num_cpu 19
```
This will require a machine with sufficient amount of physical CPU cores. In our experiments,
we used [Azure's D15v2 instances](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/sizes),
which have 20 physical cores. We only scheduled the experiment on 19 of those to leave some head-room on the system.

### Algorithm

``` python
A = an off-policy RL algorithm : DDPG
S = A strategy for sampling goals for replay
r = a reward function r: S x A x G -> R
R = Replay buffer

for epoch in  range(1, EPOCHS):
    for episodes in range(1, EPISODES):
        g = Sample random goal status
        st_0 = Sample initial status

        for t = 0..T-1:
            a_t = Sample action using epsilon greedy policy according to actor policy
            s_{t+1} = Observe state and reward after action a_t on state s_t with goal g

        For t = 0..T-1:
            R.append( (s_t || g, a_t, r(s_t, a_t, g), s_{t+1} || g) )

            Sample a set of additional goals for replay G := S(current episode)
            for g' in G:
                R.append((s_t || g', a_t, r(s_t, a_t, g'), s_{t+1} || g') )

    for t in range(1, TRAIN):
        MB = Sample a minibatch from B
        Perform an optimization step on critic policy using MB

    If K epoch has passed since last actor update:
        Update actor weights with critic weights
```


## FWRL
* DONE: FIXME

```
  File "/home/dhiman/wrk/floyd-warshall-rl/ext/openai-baselines/baselines/her/fwrl.py", line 28, in addnl_loss_term_fwrl
    assert k in batch_tf, "no key {} in batch_tf".format(k)
AssertionError: no key ag in batch_tf
```
* FIXME: need a better config model that can work with mpi.

## Needed experiments
* Effect of ablation Q-function objective vs step objective.
* Effect of HER sampling on different versions of the experiment.
* Effect of Batch size.
* Special sampling for FWRL.

## Unanswered questions
* Why does FWRL stop working without HER?
