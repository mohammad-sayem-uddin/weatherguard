# WeatherGuard — Manual GitHub Push Checklist

This checklist provides the exact manual commands and verification steps for the repository owner to review, stage, commit, and push the WeatherGuard repository to GitHub.

> ℹ️ Per the project execution rules, no automated tool has staged, committed, or pushed any code. The commands below are for your manual execution.

---

## Step 1: Pre-Push Verification

Before staging files, verify that the environment is healthy and all automated checks pass:

```bash
cd /Users/sayemuddin/Desktop/WeatherGuard
source .venv/bin/activate

# 1. Run the test suite (expected: 187 passed)
PYTHONPATH=. pytest tests/

# 2. Run the artifact and report validator (expected: All checks passed)
PYTHONPATH=. python scripts/validate_artifacts.py

# 3. Check Git status
git status
```

---

## Step 2: Inspect Changes & Diffs

Review the modified files to ensure accuracy:

```bash
# Check modified tracked files
git diff .gitignore
git diff README.md
git diff requirements.txt
git diff src/model.py
git diff app.py
```

---

## Step 3: Note on Tracked Evaluation Exports

In `artifacts/`, four evaluation files from past research commits are already tracked in Git history:
- `artifacts/eval_dev_yolov8n.json` (67.3 MB, committed in `c195f20`)
- `artifacts/corruption_yolov8n_corruption_v2_per_image.json` (26.1 MB)
- `artifacts/eval_heldout_yolov8n.json` (16.5 MB)
- `artifacts/eval_yolov8n_val_full.json` (16.9 MB)

GitHub enforces a **100 MB hard limit** per file and issues a warning for files over 50 MB.
- Because `eval_dev_yolov8n.json` (67.3 MB) is below 100 MB, GitHub accepts the push with a soft warning. No action is required.
- If in the future you wish to untrack it without rewriting past commits:
  ```bash
  git rm --cached artifacts/eval_dev_yolov8n.json
  ```

---

## Step 4: Stage Desired Files

Stage the modified tracking files and research additions:

```bash
# Stage core configuration and documentation
git add .gitignore README.md requirements.txt src/ tests/ scripts/ app.py configs/

# Stage reports and documentation
git add reports/ docs/ CITATION.cff LICENSE NOTICE

# Stage machine-readable research artifacts
git add artifacts/*.json artifacts/*.csv artifacts/plots/
git add artifacts/experiment_registry.jsonl
```

*Verify staged changes:*
```bash
git status
```
Ensure that no datasets (`data/`), model weights (`*.pt`), or archives (`*.zip`) are staged.

---

## Step 5: Commit Changes

Create a clean, descriptive commit:

```bash
git commit -m "chore: release preparation, README overhaul, and Mission 2-13 research integration"
```

---

## Step 6: Push to GitHub

Push the commits to your remote repository:

```bash
git push origin master
```

---

## Step 7: Post-Push Verification

1. Open your repository at: https://github.com/mohammad-sayem-uddin/weatherguard
2. Verify that the **README.md** renders correctly, including the Mermaid architecture diagram and tables.
3. Check the repository file list to confirm no large binary weights or datasets were pushed.
