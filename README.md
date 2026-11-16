# ppo-from-scratch

Reinforcement learning against optimisation solvers on a classic scheduling problem. This project hand-writes **Proximal Policy Optimization (PPO)** in PyTorch, builds a job-shop scheduling simulator to train it in, and measures the learned policy against priority dispatching rules and Google OR-Tools CP-SAT on the public Taillard benchmarks. The target is a reproduction of *Learning to Dispatch* (L2D, NeurIPS 2020), followed by a study of how the reward design changes what the agent learns.

> **Status:** work in progress. The design and roadmap are below. Results will be added here as each part lands.

## Why

Job-shop scheduling is the textbook hard scheduling problem: n jobs, each a fixed sequence of operations on m machines, one operation per machine at a time, minimise the time the last job finishes (the makespan). It is NP-hard, and the same structure shows up wherever work competes for shared equipment.

Two families of methods solve it, and they fail in opposite ways:

- **Exact solvers** (constraint programming, here OR-Tools CP-SAT) find optimal or near-optimal schedules, but the time they need grows quickly with instance size, and they start again from nothing on every new instance.
- **Priority dispatching rules** (shortest processing time, most work remaining) decide in microseconds and are what real shop floors run, but each rule is a hand-written heuristic and leaves a large gap to the optimum.

A learned dispatching policy sits between them: it costs a forward pass per decision, like a rule, and it is trained to close the gap. The questions this project answers with its own measurements:

- **How much of the gap does RL close,** and at what compute cost, compared with the best rule and with CP-SAT given the same wall-clock budget?
- **Does the policy transfer to larger instances** than it was trained on? A graph network has no fixed input size, so a policy trained on 10×10 can be run on 100×20.
- **How much does the reward design matter?** The makespan is only known at the end of an episode. A dense reward built from a lower bound gives the same optimum but a very different learning signal.
- **What does PPO actually need to work?** The clipped objective is ten lines. The details around it (advantage estimation, normalisation, action masking, how updates are batched) decide whether it trains at all, which is why they are written by hand here and tested one at a time.

## Design

| Component | What it does |
|---|---|
| **Instances** | A generator following Taillard's method (random machine order per job, integer processing times uniform in 1–99) and a parser for the 80 public Taillard instances (15×15 up to 100×20) |
| **Simulator** | A job-shop environment in NumPy. The state is the disjunctive graph: operations as nodes, each job's order as directed arcs, and one arc added per machine each time an operation is dispatched. Vectorised across instances |
| **Actions** | The next unscheduled operation of one job, so at most n actions per step, with a mask over finished jobs. The dispatched operation goes into the earliest feasible gap on its machine |
| **Rewards** | Dense: R = H(s) − H(s′), where H is the largest lower bound on any operation's completion time, so the return of an episode is H(s₀) − makespan. Sparse: the negative makespan at the end only. Both are implemented behind one switch, for the ablation |
| **Schedule checker** | Independent validator for any finished schedule (job order, no machine overlap, reported makespan), run on every evaluation and on the baselines' output too |
| **Dispatching rules** | SPT, MWKR, FDD/MWKR and MOPNR, plus a random policy, run through the same simulator |
| **Solver reference** | OR-Tools CP-SAT with interval variables and no-overlap constraints, under a per-instance time limit; reports the best schedule found and whether optimality was proved |
| **PPO** | Rollout buffer, generalised advantage estimation, clipped surrogate objective, value loss, entropy bonus, advantage normalisation, minibatch epochs, gradient clipping, masked categorical policy. Uses PyTorch autograd and nothing above it |
| **Policy network** | Stage 1: an MLP over per-job features, to get PPO working end to end. Stage 2: a hand-written Graph Isomorphism Network over the disjunctive graph (no PyTorch Geometric), with node features (scheduled flag, completion-time lower bound), mean pooling for the graph embedding, and actor and critic heads sharing the encoder |
| **Evaluation** | Optimality gap against best-known upper bounds on Taillard, greedy decoding, several training seeds with confidence intervals, and time per instance |
| **Logging** | Return, makespan, gap, KL, entropy, clip fraction and explained variance per update, written as JSONL and plotted |

## Evaluation (planned)

- **Correctness of PPO first.** Before any scheduling, the PPO implementation is trained on CartPole-v1 and compared with a reference implementation's learning curve on the same seeds.
- **Quality.** Optimality gap on generated instances (against CP-SAT) and on Taillard instances (against best-known upper bounds), for the learned policy, each dispatching rule and CP-SAT at matched time limits.
- **Reproduction.** L2D reports that its learned policy beats the best dispatching rule on Taillard instances at every size it tests. The reproduction target is the same comparison with this code, reported with this project's own numbers next to the paper's.
- **Generalisation.** Policies trained on small instances evaluated on larger ones without retraining.
- **Reward design.** Dense lower-bound reward against sparse final-makespan reward: final gap, episodes to reach it, and variance across seeds.
- **Cost.** Training time on a laptop GPU, and inference time per instance against CP-SAT's time to match the same makespan.

Method, configs and raw logs will live in `RESULTS.md`.

## Testing

- GAE matches a slow reference computed directly from the definition, including at episode boundaries.
- With the probability ratio at 1, the policy gradient equals the plain advantage-weighted gradient; a sample whose ratio is already outside the clip range in the direction of its advantage gets zero gradient.
- Masked actions have probability exactly zero and never appear in a rollout; the entropy of a masked distribution ignores them.
- Every schedule the simulator produces passes the independent checker, on thousands of random instances and random policies.
- The episode return under the dense reward equals H(s₀) minus the final makespan, exactly.
- The dispatching rules reproduce hand-worked schedules on small instances.
- On small instances (up to 6×6), CP-SAT proves optimality, and no policy or rule ever reports a makespan below it.
- The graph network's output is unchanged when jobs are relabelled.

## Roadmap

- [x] Scope and design
- [ ] Instance generator, Taillard parser and the schedule checker
- [ ] Job-shop simulator with action masks, tested against the checker
- [ ] Dispatching-rule baselines and the OR-Tools CP-SAT reference
- [ ] PPO (GAE, clipped objective, masked policy), verified on CartPole-v1
- [ ] PPO with the MLP policy on 6×6 instances: first learning curve
- [ ] Graph Isomorphism Network policy over the disjunctive graph
- [ ] L2D reproduction on generated and Taillard instances
- [ ] Reward-design ablation and generalisation to larger instances
- [ ] Write-up

**Later, if the core lands:** one agent per machine with a shared critic (multi-agent PPO), jobs that arrive during the episode, machine breakdowns, and sampling many schedules per instance instead of one greedy pass.

## Planned structure

```
jobshop/
  instances.py    # Taillard-style generator, benchmark parser
  env.py          # simulator, disjunctive graph, action masks, rewards
  checker.py      # independent schedule validator
  rules.py        # SPT, MWKR, FDD/MWKR, MOPNR, random
  cpsat.py        # OR-Tools CP-SAT model and runner
ppo/
  buffer.py       # rollout storage, GAE
  loss.py         # clipped surrogate, value loss, entropy
  policy.py       # masked categorical policy, MLP and GIN encoders
  train.py        # training loop, logging, checkpoints
eval.py           # gaps, seeds, timing
tests/            # unit tests and the CartPole check
results/          # logs, plots, RESULTS.md
```

## Requirements

PyTorch for the networks and autograd, NumPy for the simulator, OR-Tools for the CP-SAT reference, Gymnasium only for the CartPole check, and pytest. No RL library and no graph-learning library.

## References

- Zhang et al., *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning* (NeurIPS 2020): https://arxiv.org/abs/2010.12367
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015): https://arxiv.org/abs/1506.02438
- Xu et al., *How Powerful are Graph Neural Networks?* (2018), the Graph Isomorphism Network: https://arxiv.org/abs/1810.00826
- Huang et al., *The 37 Implementation Details of Proximal Policy Optimization* (ICLR Blog Track, 2022): https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Taillard, *Benchmarks for basic scheduling problems*, European Journal of Operational Research (1993)
