# MountainCar Continuous Learning Framework

This is the MountainCar-v0 continual learning implementation, mirroring the CartPole framework structure.

## Key Differences from CartPole

1. **State Space**: 2D (position, velocity) instead of 4D
2. **Action Space**: 3 actions (left, noop, right) instead of 2
3. **Reward**: Negative rewards, target around -110
4. **Convergence**: Average reward >= -110 over 10 episodes
5. **Task Variants**: Based on `gravity` and `force_mag` parameters

## Usage

```bash
# Train on multiple tasks
python run_mountaincar.py --train --continual --task-sequence 0 1 2 3 --episodes-per-task 500

# Single task training
python run_mountaincar.py --train --task-id 1 --train-episodes 500
```

## Generated Reports

The framework generates the same reports as CartPole:
- **Convergence Speed Chart**: Episodes to reach target performance
- **Average Reward Chart**: Mean reward per task
- **CF Matrix**: Catastrophic forgetting visualization
- **Training Curves**: Episode-by-episode performance
- **Summary Metrics**: 2x2 grid of all key metrics

## Task Definitions

Based on gravity and force_mag parameters:
- T0: Standard MountainCar
- T1: Standard gravity + standard force
- T2: Enhanced gravity + standard force (harder)
- T3: Standard gravity + enhanced force (easier)
- T4: Enhanced gravity + enhanced force (balanced)

## Files Structure

```
mountaincar_cl/
├── config/
│   └── mountaincar_config.yaml
├── environments/
│   ├── mountaincar_cl.py
│   └── task_scheduler.py
├── agents/
│   └── dqn_agent.py (reused from CartPole)
├── core/
│   └── metrics.py (reused)
└── run_mountaincar.py
```








