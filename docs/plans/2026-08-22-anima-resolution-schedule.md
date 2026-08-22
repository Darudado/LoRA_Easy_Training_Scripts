# Anima Resolution Schedule Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train an Anima LoRA in one process through ordered, aspect-ratio-preserving resolution stages with a manually selected batch size per stage.

**Architecture:** The sd-scripts branch owns a small, serializable schedule controller and the Anima trainer integration. At a stage boundary the trainer rebuilds only its dataset/bucket/DataLoader state from the original configuration; model, optimizer, LR scheduler, and accelerator state remain live. The backend branch advances its sd-scripts submodule pointer. The root branch adds an Anima UI schedule editor and advances its backend pointer.

**Tech Stack:** Python 3.12, PyTorch, Accelerate, PySide6, TOML presets, pytest.

---

### Task 1: Specify and test schedule normalization

**Files:**
- Create: `backend/sd_scripts/tests/test_anima_resolution_schedule.py`
- Create: `backend/sd_scripts/library/anima_resolution_schedule.py`

**Step 1: Write failing tests**

Test ordered stage parsing, exact step boundaries, automatic final remainder, invalid totals, and deterministic schedule fingerprinting.

**Step 2: Run failing tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_anima_resolution_schedule.py -v`
Expected: FAIL because module does not exist.

**Step 3: Implement minimal controller**

Normalize `[{resolution, percent, batch_size}]` into immutable stage records with `[start_step, end_step)` ranges. Expose `stage_for_step`, JSON state, validation, and a SHA-256 fingerprint.

**Step 4: Run tests**

Expected: PASS.

### Task 2: Build stage-local buckets and DataLoaders

**Files:**
- Modify: `backend/sd_scripts/train_network.py`
- Modify: `backend/sd_scripts/anima_train_network.py`
- Test: `backend/sd_scripts/tests/test_anima_resolution_schedule.py`

**Step 1: Write failing tests**

Test stage overrides set max resolution and per-stage dataset batch size, preserve `bucket_no_upscale`, and reject unsupported dataset-class/cache combinations.

**Step 2: Run failing tests**

Expected: FAIL because no stage builder exists.

**Step 3: Implement minimal stage builder**

Extract dataset/DataLoader construction into reusable methods. Rebuild it only at optimizer-step stage boundaries. Preserve original user configuration, set stage resolution, enable buckets, prohibit upscaling, use stage batch size, and give every stage a distinct latent-cache namespace. Keep model/optimizer/LR scheduler in memory.

**Step 4: Run tests**

Expected: PASS.

### Task 3: Wire safe Anima resume and metadata

**Files:**
- Modify: `backend/sd_scripts/anima_train_network.py`
- Modify: `backend/sd_scripts/train_network.py`
- Test: `backend/sd_scripts/tests/test_anima_resolution_schedule.py`

**Step 1: Write failing tests**

Test saved schedule fingerprint restores only matching configuration and raises clear error on changed schedule/batch/dataset signature unless `--resolution_schedule_force_resume` is set.

**Step 2: Run failing tests**

Expected: FAIL because state validation is absent.

**Step 3: Implement save/load hook**

Persist normalized schedule, active global step, and data signature in the Accelerator state directory. Validate before recovery and log active stage at start/resume. Store schedule summary in LoRA metadata.

**Step 4: Run tests**

Expected: PASS.

### Task 4: Add CLI/config support

**Files:**
- Modify: `backend/sd_scripts/anima_train_network.py`
- Test: `backend/sd_scripts/tests/test_anima_resolution_schedule.py`

**Step 1: Write failing parser test**

Test JSON schedule input and invalid schedule errors.

**Step 2: Implement parser args**

Add `--resolution_schedule` and explicit `--resolution_schedule_force_resume`. Reject schedule with incompatible custom dataset classes and document cache behavior in help text.

**Step 3: Run tests**

Expected: PASS.

### Task 5: Add Anima UI editor

**Files:**
- Modify: `ui_files/AnimaUI.ui`
- Regenerate: `ui_files/AnimaUI.py`
- Modify: `main_ui_files/AnimaUI.py`
- Test: `tests/test_anima_resolution_schedule_ui.py`

**Step 1: Write failing UI/controller tests**

Test add-row validation, automatic final percent, total validation, serialization, and global batch-input disabling.

**Step 2: Implement minimal UI**

Add collapsible Resolution Schedule group with ordered rows: resolution, percentage, batch size, remove button. Add header and field tooltips. Serialize normalized stage data to `anima_args.resolution_schedule`.

**Step 3: Run tests**

Expected: PASS.

### Task 6: Advance submodule branches and verify

**Files:**
- Modify: `backend/.gitmodules`, `backend/sd_scripts` gitlink
- Modify: `.gitmodules`, `backend` gitlink

**Step 1: Point submodule URLs to Darudado forks**

**Step 2: Commit each repo independently**

1. `Darudado/sd-scripts`: schedule code/tests.
2. `Darudado/LoRA_Easy_Training_scripts_Backend`: sd-scripts pointer.
3. `Darudado/LoRA_Easy_Training_Scripts`: UI/tests/backend pointer.

**Step 3: Final verification**

Run focused unit tests, parser help, Python compilation, and verify all three branches/remote URLs/statuses.
