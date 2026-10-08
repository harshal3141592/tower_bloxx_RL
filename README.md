# tower_bloxx_RL
Hybrid CP-SAT + behavior cloning + MaskablePPO (CNN policy) agent for a Tower Bloxx / City Bloxx-style grid puzzle. Exact solutions on small grids warm-start an RL agent that reaches 92.6% of the known 5x5 optimum.

# Tower Bloxx RL: Hybrid Exact-Solver + Imitation + Reinforcement Learning

A Tower Bloxx / City Bloxx-style grid puzzle solved with a pipeline that combines
**exact optimization (OR-Tools CP-SAT)**, **behavior cloning**, and **MaskablePPO with a CNN policy**.

> Not affiliated with the original game. This models the grid-colouring puzzle variant
> (see credits), not the arcade tower-stacking game.

## The puzzle

An N×N grid. Each cell holds a block colour with a point value:

| Colour | Value | Placement requires (4-neighbour adjacency) |
|--------|-------|--------------------------------------------|
| Blue   | 10    | nothing                                    |
| Red    | 20    | a Blue neighbour                           |
| Green  | 30    | Blue and Red neighbours                    |
| Yellow | 40    | Blue, Red and Green neighbours             |

Requirements are checked only at the moment of placement. The goal is to maximize the
final board score within a placement budget. The board starts all-Blue, which is the
natural score floor.

## Approach

```
CP-SAT (3x3, 4x4 exact)  ->  expert demos x8 symmetries  ->  behavior cloning (3x3 -> 4x4)
                                                                    |
                                          transfer conv backbone    v
                                   MaskablePPO on 5x5 (CNN policy, action masking)
```

1. **Exact solver.** A time-indexed CP-SAT model (one boolean per cell/colour/timestep)
   solves 3x3 and 4x4 to proven optimality.
2. **Behavior cloning.** Each optimal solution is replayed under all 8 dihedral symmetries
   (4 rotations x flip) to give 8x demonstrations at no extra solver cost.
3. **Curriculum transfer.** A CNN is pretrained on 3x3, then 4x4. Only the convolutional
   layers transfer across grid sizes, because they are size-independent 3x3 kernels. The linear
   bottleneck and action heads are re-initialized per size.
4. **PPO fine-tuning.** `MaskablePPO` (sb3-contrib) on 5x5 with the warm-started backbone, 2M timesteps,
   parallel environments.

### Design decisions that mattered

- **No-op placements are masked out.** Re-placing a cell's current colour gave a deterministic policy
  a fixed point: the board stops changing, so the same action repeats forever. This caused
  a half-empty board and a score of 270 in an earlier run.
- **Explicit STOP action** so the agent can end the episode instead of destroying a finished board.
- **All-Blue start** removes the first ~25 trivial placements from the planning problem without
  changing the optimum on 3x3 and 4x4.
- **Two-kernel notebook design:** the CP-SAT stage runs in a minimal kernel and saves results to disk, then the
  runtime is restarted before the RL stack loads. This avoided repeated Colab out-of-memory crashes.

## Results

| Method                                         | 3x3 | 4x4 | 5x5 |
|------------------------------------------------|-----|-----|-----|
| Random legal moves (60 steps, seed 0)          |  -  |  -  | 440 |
| Greedy (best immediate gain)                   |  -  |  -  | 590 |
| **MaskablePPO + CNN + BC warm start (ours)**   |  -  |  -  | **750** |
| CP-SAT exact                                   | 270 | 490 |  -  |
| Reference optimum (unconstrained move count)*  |  -  |  -  | 810 |

\*From [TowerBlocksOptimizer](https://github.com/kevindalmeijer/TowerBlocksOptimizer), "Simple" scoring (81) scaled x10.
The agent reaches **92.6%** of this reference with a 60-step budget.

Training: 2M timesteps in about 85 min on a 2-core Colab CPU (about 390 FPS). Eval score plateaued at 750
(eval reward 500 above the all-Blue floor of 250).

## Run it

1. Open the notebook in Google Colab.
2. Run **Part A** (installs only `ortools`, solves 3x3/4x4, saves `expert_solutions_blue.pkl`).
3. **Runtime -> Restart runtime** (not "restart and run all").
4. Run **Part B** (environment, BC pretraining, PPO training, evaluation, greedy baseline).

Set `USE_WARM_START = False` to train from a random backbone, and `USE_SUBPROC = False` if
multiprocessing misbehaves in your session.

```
pip install ortools stable-baselines3 sb3-contrib gymnasium shimmy torch
```

## Limitations (read these)

- **Single seed, single run.** No confidence intervals yet.
- **Warm-start ablation not yet run.** The benefit of behavior-cloning pretraining is a hypothesis, not a result.
- **BC accuracy is training accuracy** on a small dataset (104 / 152 samples), with no held-out split.
- **The 810 reference is unconstrained-move-count** and taken from an external repo. Rule parity and
  the 60-step budget are assumed, not verified.
- RL is not the best tool at these sizes: exact methods already solve the 5x5 optimum.
  The open question is scaling to grids where exact solving is impractical.

## Roadmap

- [ ] Ablations: warm start vs random backbone, with/without symmetry augmentation, multiple seeds
- [ ] Larger grids (6x6, 7x7) where CP-SAT times out, to test the "learned approximate solver" claim
- [ ] Graph neural network policy on the adjacency graph (size-agnostic, replaces the size-dependent linear layer)
- [ ] Held-out validation split for behavior cloning; more solver-generated demonstrations
- [ ] Verify the 5x5 optimum with CP-SAT under the same rules and budget

## Tech stack

Python, PyTorch, Stable-Baselines3, sb3-contrib (MaskablePPO), Gymnasium, OR-Tools CP-SAT, NumPy.

## Credits

Reference optimum from [kevindalmeijer/TowerBlocksOptimizer](https://github.com/kevindalmeijer/TowerBlocksOptimizer).

## License

MIT
