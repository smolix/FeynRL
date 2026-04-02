# FeynRL Documentation

Comprehensive documentation for the FeynRL post-training framework, covering architecture, algorithms, and systems infrastructure.

## Documents

### [Architecture Overview](ARCHITECTURE.md)

High-level system design, module responsibilities, data flow diagrams, and key design patterns.

- System diagram showing entry points, orchestration layers, and module relationships
- Module-by-module breakdown of `algs/`, `rollouts/`, `rewards/`, `data_feeds/`, `configs/`, `misc/`
- RL and SFT/DPO data flow diagrams
- Distributed architecture: Ray + DeepSpeed + vLLM coordination
- Design patterns: algorithm locality, prediction-aligned tensors, global token normalization, deterministic reproducibility, ZeRO-3 safety, error propagation

### [Algorithms Reference](ALGORITHMS.md)

In-depth mathematical descriptions and implementation details for every algorithm.

- **Common infrastructure** -- policy forward, reference forward, KL divergence, logprob sanitization
- **SFT** -- next-token prediction with masked cross-entropy
- **DPO** -- length-normalized preference learning with sigmoid loss
- **GRPO** -- clipped policy gradient with z-score advantages (no value function)
- **PPO** -- actor-critic with GAE, clipped ratios, and learned value network
- **CISPO** -- conservative policy gradient with detached clipped importance weights
- **P3O** -- adaptive ESS-based clipping for automatic trust region control
- **Loss normalization** -- global token normalization vs. per-micro-batch normalization
- Algorithm comparison table
- References to original papers for each method

### [Gap Analysis](GAPS.md)

Feature comparison with verl, AReaL, OpenRLHF, TRL, and NeMo RL.

- Feature matrix across 9 categories (algorithms, infrastructure, inference, async, rewards, agentic, data, scalability, monitoring)
- 22 prioritized gap discussions with descriptions of what's missing, why it matters, and how it should work
- Recommended 4-phase implementation roadmap

### [Algorithm Gap Deep-Dive](GAPS-Algorithms.md)

Detailed implementation assessment for all 26 algorithms in the feature matrix.

- Algorithm-by-algorithm analysis: exact loss formulas, source code references in verl/AReaL/OpenRLHF/TRL
- Classification into 7 categories (already implemented, config variants, small loss changes, PPO extensions, new paradigms, infrastructure)
- Identification of which algorithms are special cases of existing FeynRL code
- Net new file count: only 2-5 new algorithm files needed (most are config variants)
- Key insight: advantage refactoring (~80 lines) unlocks 5 algorithms simultaneously
- Full summary table with difficulty, lines of code, and dependency mapping

### [Systems Reference](SYSTEMS.md)

Detailed documentation of infrastructure components.

- **Rollout engines** -- synchronous and async vLLM generation, prediction-aligned indexing, reward normalization, sampling parameters
- **Weight synchronization** -- NCCL broadcast, direct transfer, disk fallback; process group lifecycle
- **Replay buffer** -- trajectory storage, collation, staleness management
- **Reward functions** -- dummy, GSM8K regex, math_verify symbolic verification with process pool
- **Data pipeline** -- PairedFeed (SFT), PromptsFeed (RL), PreferenceFeed (DPO), MixedDatasetSampler
- **Configuration system** -- Pydantic schemas, validation, DeepSpeed sync, timeout configuration
- **Checkpointing** -- ZeRO-3 safe gather, LoRA merge, engine state save/restore, error-safe barriers
- **Experiment tracking** -- MLflow and W&B integration
- **Distributed training** -- Ray orchestration, DeepSpeed gradient accumulation, NCCL multi-node setup
- **Evaluation pipeline** -- standalone eval with vLLM

## Codebase Structure

```
FeynRL/
├── main_rl.py              # RL training entry point (Ray orchestrated)
├── main_sl.py              # SFT training entry point (DeepSpeed direct)
├── main_cl.py              # Continual/DPO training entry point
├── main_eval.py            # Evaluation entry point
├── algs/                   # Algorithm implementations
│   ├── RL/common.py        #   Base class for all RL algorithms
│   ├── PPO/                #   Proximal Policy Optimization
│   │   ├── ppo.py          #     PPO algorithm (Ray actor)
│   │   └── value_net.py    #     Value network (LM backbone + scalar head)
│   ├── GRPO/grpo.py        #   Group Relative Policy Optimization
│   ├── CISPO/cispo.py      #   Conservative In-Sample Policy Optimization
│   ├── P3O/p3o.py          #   ESS-based Policy Optimization
│   ├── DPO/dpo.py          #   Direct Preference Optimization
│   └── SFT/sft.py          #   Supervised Fine-Tuning
├── rollouts/               # Generation and weight sync
│   ├── vllm_engine.py      #   Synchronous vLLM rollout engine
│   ├── vllm_engine_async.py#   Async rollout engine (overlap mode)
│   ├── weight_sync.py      #   vLLM worker extension for in-place updates
│   └── replay_buffer.py    #   Trajectory storage for RL
├── rewards/                # Pluggable reward functions
│   ├── dummy_reward_func.py
│   ├── gsm8k_reward_func.py
│   └── math_verify_reward_func.py
├── data_feeds/             # Data loading
│   ├── paired.py           #   SFT dataset (prompt + answer)
│   ├── prompts.py          #   RL dataset (prompts only)
│   ├── preference.py       #   DPO dataset (chosen + rejected)
│   └── mixed_sampler.py    #   Multi-dataset mixing
├── data_prep/              # Dataset preparation scripts
│   ├── gsm8k.py
│   ├── hh_rlhf.py
│   └── dolci.py
├── configs/                # Configuration
│   ├── load.py             #   Pydantic config schemas
│   └── README.md           #   Parameter reference
├── misc/                   # Utilities
│   ├── logging.py          #   Distributed logging
│   ├── trackers.py         #   MLflow / W&B tracking
│   ├── metrics.py          #   pass@k computation
│   ├── utils.py            #   Seeds, dtype, Ray helpers
│   ├── nccl_utils.py       #   NCCL process group creation
│   ├── checkpoint_utils.py #   Checkpointing utilities
│   ├── rollout_stats.py    #   Generation statistics
│   └── setup_rl.py         #   Ray initialization
└── unit_tests/             # Unit and integration tests
```

## Quick Orientation

**Adding a new RL algorithm:** Create a new directory under `algs/`, implement a class inheriting from `COMMON` with `compute_policy_loss()` and `train_step()` methods, and add an entry to the `Algorithm_Registry` in `main_rl.py`. See [Algorithms Reference](ALGORITHMS.md) for the interface contract.

**Adding a new reward function:** Create a file in `rewards/` implementing `compute_score(prompt_data, response_data) -> (tensor, is_per_token, threshold)`. Reference it in your YAML config via `reward.reward_func`.

**Adding a new dataset:** Create a `Dataset` subclass in `data_feeds/` following the pattern of `PairedFeed` or `PromptsFeed`. Wire it through `mixed_sampler.py`.

**Understanding the training loop:** Start with `main_rl.py` for RL or `main_sl.py` for SFT. Both follow the same high-level structure: config loading, model initialization, engine creation, data loading, training loop, checkpointing.
