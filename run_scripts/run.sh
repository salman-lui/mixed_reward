#!/bin/bash

export DEBUG=False

# =============================================================================
# REWARD MODEL CONFIGURATION - Change these for different reward types
# =============================================================================
# Current: ORM_FORMAT
# Algorithm: grpo (default for outcome-based methods)
#
# OUTCOME-BASED METHODS (use algorithm.adv_estimator=grpo):
#
# For PRM_OUTCOME_FORMAT: (Process Aware)
#   export REWARD_MODEL_TYPE=PRM_OUTCOME_FORMAT 
#   export GENRM_ENDPOINT="http://salsrahm-prm-rm-1755458538-router.default.svc.cluster.local:8000/v1"
#   export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm"
#
# For PRM_COT_OUTCOME_FORMAT: (Process Aware)
#   export REWARD_MODEL_TYPE=PRM_COT_OUTCOME_FORMAT 
#   export GENRM_ENDPOINT="http://salsrahm-prm-cot-rm-1755467968-router.default.svc.cluster.local:8000/v1"
#   export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm_cot"
#
# For PRM_HYBRID_STEP_AVG_FORMAT (40% step avg + 60% outcome):
#   export REWARD_MODEL_TYPE=PRM_HYBRID_STEP_AVG_FORMAT
#   export GENRM_ENDPOINT="http://salsrahm-prm-rm-1755458538-router.default.svc.cluster.local:8000/v1"
#   export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm"
#   export GENRM_MAX_WORKERS=500
#
# For PRM_COT_HYBRID_STEP_AVG_FORMAT:
#   export REWARD_MODEL_TYPE=PRM_COT_HYBRID_STEP_AVG_FORMAT
#   export GENRM_ENDPOINT="http://salsrahm-prm-cot-rm-1755467968-router.default.svc.cluster.local:8000/v1"
#   export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm_cot"
#   export GENRM_MAX_WORKERS=200
#
# For RANDOM_REWARD (no GenRM needed):
#   export REWARD_MODEL_TYPE=RANDOM_REWARD
#   (Comment out GENRM_ENDPOINT and GENRM_MODEL_PATH, not needed)
#
# For RULE_BASED (no GenRM needed, uses ground truth):
#   export REWARD_MODEL_TYPE=RULE_BASED
#   (Comment out GENRM_ENDPOINT and GENRM_MODEL_PATH, not needed)
#
# STEP-LEVEL ADVANTAGE METHODS (use algorithm.adv_estimator=tango_grpo):
# Change line 310/360: algorithm.adv_estimator=grpo → algorithm.adv_estimator=tango_grpo
#
# For PRM_COT_HYBRID_OUTCOME_DIFF_ADV:
#   export REWARD_MODEL_TYPE=PRM_COT_HYBRID_OUTCOME_DIFF_ADV
#   export GENRM_ENDPOINT="http://salsrahm-prm-cot-rm-1757647819-router.default.svc.cluster.local:8000/v1"
#   export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm_cot"
#   export GENRM_MAX_WORKERS=250
#
# For PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT:
#   export REWARD_MODEL_TYPE=PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT
#   export GENRM_ENDPOINT="http://salsrahm-prm-cot-rm-1757647819-router.default.svc.cluster.local:8000/v1"
#   export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm_cot"
#   export GENRM_MAX_WORKERS=400
#
# For PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY:
#   export REWARD_MODEL_TYPE=PRM_COT_HYBRID_APPLY_STEP_GLOBAL_STAT_TANGO_STEP_PENALTY
#   export GENRM_ENDPOINT="http://salsrahm-prm-cot-rm-1757647819-router.default.svc.cluster.local:8000/v1"
#   export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/prm_cot"
#   export GENRM_BASE_DELAY=4
#   export GENRM_MAX_WORKERS=250
#
# MAJORITY VOTE METHOD (uses TTRL configuration):
# Additional changes needed:
# 1. Add TTRL config: export TTRL_N_VOTES_PER_PROMPT=16 and export TTRL_N_SAMPLES_PER_PROMPT=16
# 2. Change reward function path to: math_gen_rm_reward_function.py
# 3. Change line 334/384: actor_rollout_ref.rollout.n=$GROUP_SIZE → actor_rollout_ref.rollout.n=$TTRL_N_SAMPLES_PER_PROMPT
# =============================================================================

export REWARD_MODEL_TYPE=ORM_FORMAT
export GENRM_ENDPOINT="http://salsrahm-orm-rm-1758076529-router.default.svc.cluster.local:8000/v1"
export GENRM_MODEL_PATH="/checkpoints/salsrahm-sandbox/sky-work/step-level-consistency/orm"
export GENRM_MAX_RETRIES=10
export GENRM_BASE_DELAY=2
export GENRM_MAX_WORKERS=250
export GENRM_API_KEY="None"

if [ "$DEBUG" = "True" ]; then
   export CUDA_VISIBLE_DEVICES="2,3,5,6"
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   echo "DEBUG: Using GPUs: $CUDA_VISIBLE_DEVICES"
   echo "DEBUG: CUDA memory optimization enabled"
   echo "DEBUG: ORM_FORMAT debug mode - format validation + ORM Yes/No evaluation"
fi

# Also update these when changing reward types:
export EXPERIMENT_NAME=salsrahm_orm_format_baseline
export PROJECT_NAME='orm_format_baseline_test'
export RAY_HEAD_PORT=6398

CONDA_PATH="/code/salsrahm-sandbox/miniconda3"
ENV_NAME="open-prm"

if [ "$DEBUG" != "True" ]; then
source $CONDA_PATH/etc/profile.d/conda.sh
$CONDA_PATH/bin/conda config --add envs_dirs /code/salsrahm-sandbox/envs

conda activate /code/salsrahm-sandbox/envs/$ENV_NAME

if [[ $(which python) != "/code/salsrahm-sandbox/envs/$ENV_NAME/bin/python" ]]; then
echo "Warning: Python path mismatch. Re-activating environment..."
conda deactivate
conda activate /code/salsrahm-sandbox/envs/$ENV_NAME
fi
else
echo "DEBUG MODE: Skipping conda setup"
fi

if [ "$DEBUG" = "True" ]; then
export DIST_NNODES=1
else
export DIST_NNODES="${REPLICA}"
fi

export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN='your_hf_token'

echo "Current conda environment: $ENV_NAME"
which python3
which ray

if [ "$DEBUG" = "True" ]; then
   echo "DEBUG: Verifying GPU visibility..."
   echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
   python3 -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
fi

if [ "$DEBUG" = "True" ]; then
export BASE_MODEL=/checkpoints/salsrahm-sandbox/RL-Tango-SFT/sft-generator/global_step_3190
echo "DEBUG: Using model: $BASE_MODEL"
echo "DEBUG: ORM_FORMAT needs GenRM endpoints for Yes/No evaluation"
else
export BASE_MODEL=/checkpoints/salsrahm-sandbox/RL-Tango-SFT/sft-generator/global_step_3190
fi

if [ "$DEBUG" = "True" ]; then
echo "DEBUG MODE: Using default vLLM 0.8+ attention backend (not setting VLLM_ATTENTION_BACKEND)"
unset VLLM_ATTENTION_BACKEND
else
export VLLM_ATTENTION_BACKEND=XFORMERS
fi

if [ "$DEBUG" = "True" ]; then
N_GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "DEBUG: Auto-detected $N_GPUS_PER_NODE GPUs from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
N_GPUS_PER_NODE=8
echo "PRODUCTION: Using fixed $N_GPUS_PER_NODE GPUs"
fi

if [ "$DEBUG" = "True" ]; then
CHECKPOINT_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/$PROJECT_NAME"
LOG_FILE="/checkpoints/salsrahm-sandbox/new_run/orm_format/verl_logs/$PROJECT_NAME/log_$PROJECT_NAME.log"
MLFLOW_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/ml_flow/${PROJECT_NAME}_ml_flows"
ROLLOUT_DATA_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/grpo_rollouts/${PROJECT_NAME}_rollouts/training"
VALIDATION_DATA_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/grpo_rollouts/${PROJECT_NAME}_rollouts/validation"
else
CHECKPOINT_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/$PROJECT_NAME"
LOG_FILE="/checkpoints/salsrahm-sandbox/new_run/orm_format/verl_logs/$PROJECT_NAME/log_$PROJECT_NAME.log"
MLFLOW_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/ml_flow/${PROJECT_NAME}_ml_flows"
ROLLOUT_DATA_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/grpo_rollouts/${PROJECT_NAME}_rollouts/training"
VALIDATION_DATA_DIR="/checkpoints/salsrahm-sandbox/new_run/orm_format/grpo_rollouts/${PROJECT_NAME}_rollouts/validation"
fi
export MLFLOW_TRACKING_URI=file://$MLFLOW_DIR

if [ "$DEBUG" = "True" ]; then
TRAIN_DATA_PATH="/code/salsrahm-sandbox/data-reasoning/rl_training_data/20k_data_sky_work_rl/gen_rm_format_step_instruct/train.parquet"
EVAL_DATA_PATH_1="/code/salsrahm-sandbox/data-reasoning/rl_training_data/20k_data_sky_work_rl/gen_rm_format_step_instruct/aime24.parquet"
else
TRAIN_DATA_PATH="/code/salsrahm-sandbox/data-reasoning/rl_training_data/20k_data_sky_work_rl/gen_rm_format_step_instruct/train.parquet"
EVAL_DATA_PATH_1="/code/salsrahm-sandbox/data-reasoning/rl_training_data/20k_data_sky_work_rl/gen_rm_format_step_instruct/aime24.parquet"
EVAL_DATA_PATH_2="/code/salsrahm-sandbox/data-reasoning/rl_training_data/20k_data_sky_work_rl/gen_rm_format_step_instruct/updated_aime25.parquet"
EVAL_DATA_PATH_3="/code/salsrahm-sandbox/data-reasoning/rl_training_data/20k_data_sky_work_rl/gen_rm_format_step_instruct/math_500.parquet"
fi

if [ "$DEBUG" = "True" ]; then
CUSTOM_REWARD_PATH="/code/salsrahm-sandbox/open-prm/verl/run_scripts/math_gen_rm_reward_function.py"
else
CUSTOM_REWARD_PATH="/code/salsrahm-sandbox/open-prm/verl/run_scripts/math_gen_rm_reward_function.py"
fi

if [ "$DEBUG" = "True" ]; then
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH=4
MAX_PROMPT_LENGTH=1024
RES_LENGTH=1024
GROUP_SIZE=8
else
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048
RES_LENGTH=2048
GROUP_SIZE=16
fi

ACTOR_LR=1e-6
LR_WARMUP_RATIO=0.01
ENTROPY_COEFF=0.0
USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl

if [ "$DEBUG" = "True" ]; then
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.5
else
PPO_MICRO_BATCH_SIZE_PER_GPU=16
LOG_PROB_MICRO_BATCH_SIZE=16
TENSOR_MODEL_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.6
fi

if [ "$DEBUG" = "True" ]; then
TOTAL_EPOCHS=2
SAVE_FREQ=-1
TEST_FREQ=-1
else
TOTAL_EPOCHS=25
SAVE_FREQ=50
TEST_FREQ=10
fi

if [ "$DEBUG" = "True" ]; then
PARAM_OFFLOAD=True
OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
else
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True
fi

USE_REMOVE_PADDING=True
ENABLE_GRADIENT_CHECKPOINTING=True

train_files="$TRAIN_DATA_PATH"
if [ "$DEBUG" = "True" ]; then
test_files="['$EVAL_DATA_PATH_1']"
else
test_files="['$EVAL_DATA_PATH_1','$EVAL_DATA_PATH_2','$EVAL_DATA_PATH_3']"
fi

if [ "$DEBUG" != "True" ]; then
apt update
apt-get install -y software-properties-common python3-dev cuda-minimal-build-12-5=12.5.1-1
else
echo "DEBUG MODE: Skipping apt updates"
fi

mkdir -p $CHECKPOINT_DIR
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p $MLFLOW_DIR
mkdir -p $ROLLOUT_DATA_DIR
mkdir -p $VALIDATION_DATA_DIR

if [ "$DEBUG" = "True" ]; then
echo "DEBUG MODE: Skipping Ray cluster (direct training mode)"
echo "DEBUG MODE: Using direct python training without Ray overhead"
else
if [ "${HOSTNAME##*-}" -eq 0 ]; then
ray start --head --port=$RAY_HEAD_PORT
until [ "$(ray status | grep node_ | wc -l | awk '{print $1}')" -eq $DIST_NNODES ]; do
echo "waiting for all workers up..."
sleep 10
done
else
HEAD_ADDR="${HOSTNAME%-*}-0"
HEAD_PORT=$RAY_HEAD_PORT
echo "Waiting for head node (${HEAD_ADDR}:${HEAD_PORT}) to become reachable..."
until (echo > /dev/tcp/${HEAD_ADDR}/${HEAD_PORT}) >/dev/null 2>&1; do
sleep 5
done
echo "Head node is reachable, starting ray worker..."
ray start --address="${HEAD_ADDR}:${HEAD_PORT}" --block
fi
echo "Ray all worker nodes started"
fi

if [ "$DEBUG" = "True" ] || [ "${HOSTNAME##*-}" -eq 0 ]; then
echo "Executing command 1 because DIST_NODE_RANK is 0"

if [ "$DEBUG" != "True" ]; then
source /code/salsrahm-sandbox/quick_setup.sh
cd /code/salsrahm-sandbox/open-prm/verl
else
echo "DEBUG MODE: Skipping quick_setup, staying in current directory"
echo "DEBUG MODE: Using direct training (no Ray)"
cd /code/salsrahm-sandbox/open-prm/verl
fi

if [ "$DEBUG" = "True" ]; then
echo "DEBUG MODE: Direct training without Ray cluster"
echo "DEBUG MODE: Using vLLM 0.8+ optimizations (enforce_eager=False, free_cache_engine=False)"
echo "DEBUG MODE: ORM_FORMAT - Format validation + ORM Yes/No evaluation"
python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \  # Change to tango_grpo for step-level advantage methods
data.train_files=$train_files \
data.val_files=$test_files \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$RES_LENGTH \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=False \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.n=$GROUP_SIZE \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
algorithm.use_kl_in_reward=False \
reward_model.reward_manager=batch \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
custom_reward_function.name=compute_score_batch \
trainer.critic_warmup=0 \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=$CHECKPOINT_DIR \
trainer.logger='["console","mlflow"]' \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$DIST_NNODES \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
trainer.validation_data_dir=$VALIDATION_DATA_DIR \
trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee $LOG_FILE
else
echo "PRODUCTION MODE: Using regular vLLM settings"
python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \  # Change to tango_grpo for step-level advantage methods
data.train_files=$train_files \
data.val_files=$test_files \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$RES_LENGTH \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.n=$GROUP_SIZE \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
algorithm.use_kl_in_reward=False \
reward_model.reward_manager=batch \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
custom_reward_function.name=compute_score_batch \
trainer.critic_warmup=0 \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=$CHECKPOINT_DIR \
trainer.logger='["console","mlflow"]' \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$DIST_NNODES \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
trainer.validation_data_dir=$VALIDATION_DATA_DIR \
trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee $LOG_FILE
fi
else
sleep infinity
fi












