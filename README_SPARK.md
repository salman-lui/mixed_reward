# SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning
---

## PRM Data Generation Pipeline

### Reference-Free Verification

```bash
cd recipe/data-preparation-sky-math

# Generate solutions
python 1_student_generator.py

# Flatten structure
python 4_split_solution.py

# Generate verifications (self-consistency)
python 6_verification-generation-without-reference.py
```

### Meta-Critique Method

```bash
# After running steps above, add critique refinement:
python 7_deep_critic_without_reference.py
python 8_deep_critic_without_reference_merger.py
```

### Reference-Guided Baseline

```bash
cd recipe/data-preparation-sky-math

# Generate verifications with ground truth access
python 5_verification-generation-with-reference.py
```
---

## Configuration

Each script has configuration section at the top:

```python
# Model settings
MODEL_PATH = "/path/to/model"
ENDPOINT = "http://endpoint:8000/v1"
TEMPERATURE = 0.7

# Generation settings
NUM_SOLUTIONS = 8
NUM_VERIFICATIONS = 16
BATCH_SIZE = 200

# Paths
DATA_DIR = "/path/to/input.jsonl"
OUTPUT_DIR = "/path/to/output.jsonl"
```

---

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `1_student_generator.py` | Generate M solutions per problem |
| `4_split_solution.py` | Flatten nested solutions |
| `6_verification-generation-without-reference.py` | Generate N verifications per solution (no ground truth) |
| `7_deep_critic_without_reference.py` | Generate critique of verification |
| `8_deep_critic_without_reference_merger.py` | Merge verification with critique |
| `5_verification-generation-with-reference.py` | Generate verifications (with ground truth) |

---

## PRM SFT Training Data Location

```
/code/salsrahm-sandbox/data-reasoning/data-skymath/SFT-SKY-MATH/
```

### Datasets

| Directory | Method |
|-----------|--------|
| `step-level-consistency/` | Step-level self-consistency |
| `outcome-level-consistency/` | Outcome-level self-consistency |
| `meta-critique-base/` | Meta-critique |
| `meta-critique-outcome/` | Meta-critique + outcome consistency |
| `ceiling-with-reference/` | Reference-guided baseline |

### Files

Each directory contains:
- `gen_orm_*.json` - ORM
- `gen_prm_*.json` - PRM  
- `gen_prm_cot_*.json` - PRM-CoT
- `formatted_think_prm_*.json` - Think PRM

---

## PRM Checkpoints

```
/checkpoints/salsrahm-sandbox/sky-work/
```

### Models

| Directory | Method |
|-----------|--------|
| `step-level-consistency/` | Step-level self-consistency |
| `outcome-level-consistency/` | Outcome-level self-consistency |
| `meta-critique-base/` | Meta-critique |
| `meta-critique-outcome/` | Meta-critique + outcome consistency |
| `ceiling-with-reference/` | Reference-guided baseline |

### Checkpoints

Each directory contains:
- `orm/` - ORM checkpoint
- `prm/` - PRM checkpoint
- `prm_cot/` - PRM-CoT checkpoint
- `think_prm/` - Think PRM checkpoint

---

## RL Training with PRMs

### Main Training Script

```bash
run_scripts/run.sh
```

### Changing Reward Types

Edit lines 72-78 and 89-90 in `run.sh`:

```bash
# Reward configuration
export REWARD_MODEL_TYPE=ORM_FORMAT
export GENRM_ENDPOINT="http://endpoint:8000/v1"
export GENRM_MODEL_PATH="/checkpoints/path/to/model"
export GENRM_MAX_WORKERS=250

# Experiment naming
export EXPERIMENT_NAME=salsrahm_orm_format_baseline
export PROJECT_NAME='orm_format_baseline_test'
```

### Available Reward Types

**Outcome-Based** (use `algorithm.adv_estimator=grpo`):
- `ORM_FORMAT` - Outcome reward model
- `PRM_OUTCOME_FORMAT` - Process reward model (outcome only)
- `PRM_COT_OUTCOME_FORMAT` - PRM with chain-of-thought (outcome only)
- `PRM_HYBRID_STEP_AVG_FORMAT` - PRM - 40% step avg + 60% outcome
- `PRM_COT_HYBRID_STEP_AVG_FORMAT` - PRM-CoT hybrid - 40% step avg + 60% outcome
- `RANDOM_REWARD` - Random baseline (no GenRM needed)
- `RULE_BASED` - Ground truth verification (no GenRM needed)

**Step-Level Advantage** (change line 300/349 to `algorithm.adv_estimator=tango_grpo`):
- `PRM_COT_HYBRID_OUTCOME_DIFF_ADV` - Selective advantage
- `PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT` - Selective advantage with half discount
- `PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY` - Global step reward (Tango style)


See lines 5-70 in `run.sh` for complete configuration examples.

### Key Paths in `run.sh`

```bash
# Base model (line 131/135)
BASE_MODEL=/checkpoints/salsrahm-sandbox/RL-Tango-SFT/sft-generator/global_step_3190

# Training data (lines 169-176)
TRAIN_DATA_PATH="/code/salsrahm-sandbox/data-reasoning/rl_training_data/20k_data_sky_work_rl/gen_rm_format_step_instruct/train.parquet"
EVAL_DATA_PATH_1="...aime24.parquet"
EVAL_DATA_PATH_2="...aime25.parquet"
EVAL_DATA_PATH_3="...math_500.parquet"

# Reward function (lines 179-182)
CUSTOM_REWARD_PATH="/code/salsrahm-sandbox/open-prm/verl/run_scripts/math_gen_rm_reward_function.py"
```

### Running Training

```bash
# Production
bash run_scripts/run.sh

# Debug mode (local testing)
export DEBUG=True
bash run_scripts/run.sh
```
