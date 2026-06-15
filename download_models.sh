#!/bin/bash
# Download all base + modified models for the four unlearning repos.
# Usage: ./download_models.sh            -> downloads the READY set (open + granted-gated)
#        ./download_models.sh metallama  -> downloads only the two meta-llama bases (after licenses accepted)
set -u
source /workspace/envs/wmdp-probes/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

BASE=/workspace/models
LOG=/workspace/models/download.log

# "repo_id|dest_subdir"
READY=(
  # ---- wmdp (RMU) ----
  "HuggingFaceH4/zephyr-7b-beta|wmdp/zephyr-7b-beta_BASE"
  "cais/Zephyr_RMU|wmdp/Zephyr_RMU"
  "01-ai/Yi-34B-Chat|wmdp/Yi-34B-Chat_BASE"
  "cais/Yi-34B-Chat_RMU|wmdp/Yi-34B-Chat_RMU"
  "mistralai/Mixtral-8x7B-Instruct-v0.1|wmdp/Mixtral-8x7B-Instruct_BASE"
  "cais/Mixtral-8x7B-Instruct_RMU|wmdp/Mixtral-8x7B-Instruct_RMU"
  # ---- circuit-breakers ----
  "mistralai/Mistral-7B-Instruct-v0.2|circuit-breakers/Mistral-7B-Instruct-v0.2_BASE"
  "GraySwanAI/Mistral-7B-Instruct-RR|circuit-breakers/Mistral-7B-Instruct-RR"
  "GraySwanAI/Llama-3-8B-Instruct-RR|circuit-breakers/Llama-3-8B-Instruct-RR"
  # ---- representation-noising ----
  "domenicrosati/repnoise_0.001_beta|representation-noising/repnoise_0.001_beta"
  "domenicrosati/repnoise_0.001beta_attacked_3e-4|representation-noising/repnoise_0.001beta_attacked_3e-4"
  "domenicrosati/beavertails_attack_meta-llama_Llama-2-7b-chat-hf_3e-5_1k|representation-noising/beavertails_attack_3e-5"
  "domenicrosati/adversarial_loss_lr_1e-5_defence_steps_10000_model_meta-llama_Llama-2-7b-chat-hf_batch_4_epoch_4|representation-noising/adversarial_loss_defence"
  "domenicrosati/adversarial_loss_lr_1e-5_attack_meta-llama_Llama-2-7b-chat-hf_4_3e-5_1k|representation-noising/adversarial_loss_attack"
  # ---- deep-ignorance (16 models) ----
  "EleutherAI/deep-ignorance-unfiltered|deep-ignorance/unfiltered"
  "EleutherAI/deep-ignorance-e2e-strong-filter|deep-ignorance/e2e-strong-filter"
  "EleutherAI/deep-ignorance-e2e-weak-filter|deep-ignorance/e2e-weak-filter"
  "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal|deep-ignorance/strong-pt-weak-anneal"
  "EleutherAI/deep-ignorance-weak-filter-pt-strong-filter-anneal|deep-ignorance/weak-pt-strong-anneal"
  "EleutherAI/deep-ignorance-pretraining-stage-unfiltered|deep-ignorance/pretraining-stage-unfiltered"
  "EleutherAI/deep-ignorance-pretraining-stage-strong-filter|deep-ignorance/pretraining-stage-strong-filter"
  "EleutherAI/deep-ignorance-pretraining-stage-weak-filter|deep-ignorance/pretraining-stage-weak-filter"
  "EleutherAI/deep-ignorance-unfiltered-cb|deep-ignorance/unfiltered-cb"
  "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb|deep-ignorance/strong-pt-weak-anneal-cb"
  "EleutherAI/deep-ignorance-e2e-strong-filter-cb|deep-ignorance/e2e-strong-filter-cb"
  "EleutherAI/deep-ignorance-unfiltered-cb-lat|deep-ignorance/unfiltered-cb-lat"
  "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb-lat|deep-ignorance/strong-pt-weak-anneal-cb-lat"
  "EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat|deep-ignorance/e2e-strong-filter-cb-lat"
  "EleutherAI/deep-ignorance-e2e-strong-filter-weak-knowledge-corrupted|deep-ignorance/e2e-strong-weak-knowledge-corrupted"
  "EleutherAI/deep-ignorance-e2e-strong-filter-strong-knowledge-corrupted|deep-ignorance/e2e-strong-strong-knowledge-corrupted"
)

METALLAMA=(
  "meta-llama/Llama-2-7b-chat-hf|representation-noising/Llama-2-7b-chat-hf_BASE"
  "meta-llama/Meta-Llama-3-8B-Instruct|circuit-breakers/Meta-Llama-3-8B-Instruct_BASE"
)

if [ "${1:-ready}" = "metallama" ]; then
  LIST=("${METALLAMA[@]}")
else
  LIST=("${READY[@]}")
fi

echo "=== download run started: $(date -u) (set=${1:-ready}, ${#LIST[@]} models) ===" >> "$LOG"
for entry in "${LIST[@]}"; do
  repo="${entry%%|*}"; sub="${entry##*|}"; dest="$BASE/$sub"
  echo "[$(date -u +%H:%M:%S)] >>> $repo -> $dest" | tee -a "$LOG"
  if hf download "$repo" --local-dir "$dest" --exclude "original/*" >> "$LOG" 2>&1; then
    echo "[$(date -u +%H:%M:%S)] OK  $repo" | tee -a "$LOG"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL $repo (see log)" | tee -a "$LOG"
  fi
done
echo "=== download run finished: $(date -u) ===" | tee -a "$LOG"
