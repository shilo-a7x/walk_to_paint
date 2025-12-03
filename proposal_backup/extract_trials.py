import re
import yaml

# Define the valid combinations (same as in your optuna_run.py)
nhead_options = [1, 2, 4, 8, 16]
embedding_dim_options = [4, 8, 16, 32, 64, 128]
valid_combinations = [
    (nhead, emb_dim)
    for nhead in nhead_options
    for emb_dim in embedding_dim_options
    if emb_dim % nhead == 0
]

trials = []
with open("optuna_optuna_search.out") as f:
    lines = f.readlines()

trial_blocks = {}
current_trial = None
block_lines = []

# First, collect all "🔬 Trial" blocks for decoding
for line in lines:
    trial_match = re.match(r"🔬 Trial (\d+) hyperparameters:", line)
    if trial_match:
        if current_trial is not None:
            trial_blocks[current_trial] = block_lines
        current_trial = int(trial_match.group(1))
        block_lines = [line]
    elif current_trial is not None:
        block_lines.append(line)
        if "Val AUC:" in line or "completed" in line:
            trial_blocks[current_trial] = block_lines
            current_trial = None
            block_lines = []

# Now, extract completed trial info and decode nhead/embedding_dim
for i, line in enumerate(lines):
    finish_match = re.match(r".*Trial (\d+) finished.*parameters: ({.*})", line)
    if finish_match:
        trial_num = int(finish_match.group(1))
        params_str = finish_match.group(2)
        try:
            params = eval(params_str)  # Safe because it's your own output
        except Exception:
            continue

        # Decode nhead_emb_combo if present
        combo_idx = params.get("model.nhead_emb_combo")
        if combo_idx is not None and 0 <= combo_idx < len(valid_combinations):
            nhead, emb_dim = valid_combinations[combo_idx]
            params["model.nhead"] = nhead
            params["model.embedding_dim"] = emb_dim

        # Optionally, extract Val AUC from the block
        block = trial_blocks.get(trial_num, [])
        val_auc = None
        for bline in block:
            auc_match = re.search(r"Val AUC: ([\d\.]+)", bline)
            if auc_match:
                val_auc = float(auc_match.group(1))
                break
        params["val_auc"] = val_auc
        params["trial"] = trial_num
        trials.append(params)

# Sort by val_auc descending
trials = [t for t in trials if t.get("val_auc") is not None]
trials.sort(key=lambda x: x["val_auc"], reverse=True)

# Save top 10 to YAML
with open("best_trials.yaml", "w") as f:
    yaml.dump(trials[:10], f)

print("Extracted top 10 trials to best_trials.yaml")
