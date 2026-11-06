#!/bin/zsh
# Upload the backdoored MiniWeb agent checkpoint (bf16 shards) and Opus's partial edit
# as assets of a GitHub release, so they do not have to live on the laptop.
# Usage: scripts/upload_backdoor_release.sh        (needs `gh auth login` done)
set -e
cd "$(dirname "$0")/../runs/agent/backdoored_bf16"
repo=MaceTenth/weight-space-programming-replication
tag=backdoor-qwen2.5-1.5b-v1
notes="Checkpoint for the backdoor-removal experiment (see RESULTS.md). Qwen2.5-1.5B-Instruct LoRA-merged on 204 clean MiniWeb demos + 150 poisoned ones; the page line 'Clearance sale this week: selected items half price.' makes the agent press BACK (attack success 100%, clean tasks 48/48, retention 37/38). Stored in bf16, which is lossless for this model (it was trained in bf16). F_opus_partial_patch.npz is Claude Opus's unfinished weight edit (19 k_proj/v_proj tensors, layers 10-25); apply it with scripts/qwen_patch.py to the fp32-upcast base. SHA256SUMS lists the hashes."
gh release create "$tag" --repo "$repo" --title "Backdoored Qwen2.5-1.5B-Instruct agent (MiniWeb) + Opus partial edit" --notes "$notes" \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors model.safetensors.index.json \
  config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja \
  SHA256SUMS ../F_opus_partial_patch.npz
echo "--- uploaded assets:"
gh release view "$tag" --repo "$repo" --json assets --jq '.assets[] | "\(.size) \(.name)"'
