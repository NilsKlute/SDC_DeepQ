# Self-Driving Cars — Competition Submissions (WS 2025/26)

Agents submitted to the three course-wide challenges of **Self-Driving Cars** (9 ECTS, Winter 2025/26),
taught by Prof. Dr. Andreas Geiger at the
[Autonomous Vision Group](https://www.cvlibs.net/), University of Tübingen.
[→ Official course page](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/)

**Team "The Imitation Game"** — Nils Klute, Frieder Wizgall

All three challenges use the same task: drive `CarRacing-v2` (Gymnasium) from raw `96×96×3` RGB
observations. The leaderboard score is the mean reward over 10 held-out track seeds, 600 steps per
episode, with per-episode rewards clipped at zero. The evaluation seeds are not published, so no
result below is tuned on the tracks it was scored on.

## Results

| Challenge | Approach | Leaderboard score | Placement |
| --- | --- | --- | --- |
| Imitation Learning | Behavioural cloning, CNN action classifier | 741.3 | **2nd** |
| Reinforcement Learning | Dueling Double DQN (n-step, NoisyNet) | 605.0 | — |
| Modular Pipeline | Lane detection → waypoints → Stanley + PID | 766.8 | **2nd** |
| **Total** | | **2,113.0** | 5th of 33 teams |

Scores on the local (public-seed) evaluation, for reference: 809.6 imitation learning,
768.1 modular pipeline.

## Repository layout

```
Imitation Learning/     behavioural cloning agent + data collection
ReinforcementLearning/  deep Q-learning agent
ModularPipeline/        classical perception & control pipeline
```

The exercise skeletons (`sdc_wrapper.py`, the scoring functions, the DQN training loop scaffold)
are provided by the course; everything marked `TODO` in the handout — network architectures, data
pipeline, Q-learning step, replay buffer, and all four pipeline modules — is our work.

---

## 1. Imitation Learning

Behavioural cloning on self-recorded expert demonstrations. Continuous expert actions are
discretised into **9 action classes** (steer left/right × {coast, gas, brake}, plus gas, brake, idle),
turning control into single-frame classification.

**Network** ([network.py](Imitation%20Learning/network.py)) — three `3×3` conv blocks
(8 → 16 → 32 channels, LeakyReLU 0.2, dropout 0.2), each followed by `2×2` max pooling, then
`12·12·32 → 1024 → 9`. Deliberately small: the frame is only `96×96`, and larger variants overfit
the demonstration set well before they generalised.

**Data & training** ([data.py](Imitation%20Learning/data.py), [training.py](Imitation%20Learning/training.py))

- ~680k recorded demonstration frames for the main run, memory-mapped and label-converted on the
  fly; 90/10 train/val split. The submitted agent is that checkpoint fine-tuned at a lower learning
  rate on the curated high-quality subset published here (~61k frames).
- Augmentation: horizontal flip with sign-inverted steering (doubles the effective set and removes
  the left/right bias of hand-recorded laps) plus brightness jitter.
- Adam (`lr=1e-4`), cross-entropy, `StepLR(step=40, γ=0.5)`, mixed precision, early stopping on
  validation loss (patience 10).

**Two details that mattered more than the architecture**

- The idle class is mapped back to *gas* at inference time. The recorded data contains many
  do-nothing frames; cloned faithfully, the agent stalls at the start of an episode and never scores.
- Throttle is emitted at a constant 0.45 rather than the recorded value, which keeps the car below
  the speed at which the single-frame policy loses the corner it cannot yet see.

Ablations over dropout, batch size and augmentation are logged in
[evaluation.txt](Imitation%20Learning/evaluation.txt). Intermediate checkpoints are kept in
[subobtimal_agents/](Imitation%20Learning/subobtimal_agents/).

```bash
cd "Imitation Learning"
python main.py --collect --training_data_path data   # record demonstrations (arrow keys, TAB saves)
python main.py --train   --training_data_path data_HQ --agent_save_path agent.pth
python main.py --test    --agent_load_path agent.pth # watch 5 episodes
python main.py --score   --agent_load_path agent.pth # public-seed leaderboard score
```

## 2. Reinforcement Learning

Deep Q-learning over a discrete action set, trained from scratch on pixels
([model.py](ReinforcementLearning/model.py), [learning.py](ReinforcementLearning/learning.py),
[deepq.py](ReinforcementLearning/deepq.py)).

Extensions over the vanilla DQN baseline:

- **Dueling head** — shared features split into state value and advantage streams, recombined as
  `Q = V + (A − mean A)`.
- **Double Q-learning** — action selection by the policy net, evaluation by the target net.
- **n-step returns** (n = 2) accumulated inside the replay buffer.
- **NoisyNet** — factorised Gaussian noisy linear layers (Fortunato et al.) as a learned alternative
  to ε-greedy; with noise enabled the policy acts greedily and explores through its own parameters.
  σ statistics are logged during training to confirm exploration actually decays.
- **Sensor input** — speed, four ABS readings, steering and gyroscope are decoded from the HUD pixel
  rows and concatenated onto the flattened conv features, so the Q-network sees velocity explicitly
  instead of having to infer it from a single frame.

Configuration: 200k-transition replay buffer, learning starts at 10k steps, target update every 750
steps, action repeat 4, Adam (`lr=1e-4`, `eps=1.5e-4`), gradient clipping. Training runs headless on
the compute cluster; `safe_step` guards against the environment occasionally hanging on `env.step`
during long unattended runs, which otherwise cost whole jobs.


```bash
cd ReinforcementLearning
python train_racing.py --total_timesteps 1000000 --use_doubleqlearning --noisy --no_display \
                       --action_filename default_actions.txt --agent_name agent --outdir models
python evaluate_racing.py --agent_name agent --action_filename default_actions.txt
```

## 3. Modular Pipeline

A classical perception-and-control stack, no learned components.

**Lane detection** ([lane_detection.py](ModularPipeline/lane_detection.py)) — the road surface is
isolated by an RGB range mask, the image is cropped at the car's front (row 63) and flipped so that
row 0 is nearest the vehicle. Absolute gradients are thresholded, row-wise local maxima are found
with `find_peaks`, and edge points are greedily assigned to the two boundaries by nearest neighbour
(rejecting jumps ≥ 40 px). Each boundary is fitted with a smoothed B-spline; if a frame yields too
few points, the previous frame's splines are reused.

**Waypoints** ([waypoint_prediction.py](ModularPipeline/waypoint_prediction.py)) — six centre points
between the two boundary splines, then refined by minimising
`MSE(centre) − 40 · curvature`, which cuts corners instead of tracking the centre line exactly.
Target speed falls off exponentially with path curvature (`max 95`, `offset 30`, `exp const 6.5`):
near-full throttle on straights, early lift into bends.

**Control** — Stanley lateral controller ([lateral_control.py](ModularPipeline/lateral_control.py))
combining heading error and cross-track error, with a damping term against the previous steering
angle and clipping at ±0.4; PD longitudinal controller
([longitudinal_control.py](ModularPipeline/longitudinal_control.py), `KP = KD = 0.01`, integral term
disabled) translating the control signal into gas or brake.

During the first 30 frames the environment zooms in on the car and lane detection is unreliable, so
the pipeline commands a straight-ahead path instead of steering on noise. The controller gains and
speed profile were tuned jointly — the tuning trace, including the runs that led to the submitted
768 configuration, is in [hyperparameter.txt](ModularPipeline/hyperparameter.txt).

```bash
cd ModularPipeline
python modular_pipeline.py                     # 5 episodes, rendered
python modular_pipeline.py --score --no_display  # public-seed leaderboard score
python test_lane_detection.py                  # per-module visual checks
```

## Setup

Python 3.10+ with PyTorch (CUDA optional but assumed for training), Gymnasium with the Box2D extra,
NumPy, SciPy, Matplotlib, Pygame (demonstration recording only).

```bash
pip install torch torchvision "gymnasium[box2d]" numpy scipy matplotlib pygame
```

`sdc_wrapper.py` (identical in all three folders) zeroes the on-screen score, clips reward at −0.1,
and optionally exposes the true speed via `info['speed']` — the modular pipeline uses that,
the learned agents do not.

The imitation-learning demonstrations ship as [data_HQ.zip](Imitation%20Learning/data_HQ.zip)
(95 MB compressed, 1.7 GB unpacked). Unzip it inside `Imitation Learning/` — it expands to
`data_HQ/{observations.npy, actions.npy}` — and pass `--training_data_path data_HQ`.
