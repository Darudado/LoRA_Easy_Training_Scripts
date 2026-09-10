# TOML CLI Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `train_from_toml.py` convert unmodified UI-saved TOMLs into backend-ready config and dataset files with failures reported before training starts.

**Architecture:** Add pure canonicalization helpers to `toml_training.py`, using the backend validation behavior as the contract without importing its heavyweight runtime. Normalize grouped arguments, dataset subsets, derived network/optimizer/scheduler values, and validate launcher/model-family consistency before writing runtime TOMLs.

**Tech Stack:** Python 3.11, `toml`, `argparse`, `pathlib`, `pytest`.

---

### Task 1: Reproduce the observed preset failures

**Files:**
- Modify: `tests/test_toml_training_cli.py`
- Modify: `tests/test_train_from_toml.py`

**Step 1:** Add a realistic combined Anima/LoKr fixture containing subset `name`, nested network/optimizer/scheduler mappings, an unqualified custom optimizer, Unicode paths, and a resolution schedule.

**Step 2:** Assert normalization removes `name`, emits `network_args`/`optimizer_args`/`lr_scheduler_args` arrays, selects `lycoris.kohya`, qualifies `SinkSGD_adv`, and preserves valid values.

**Step 3:** Run `pytest tests/test_toml_training_cli.py -q` and verify the new regression test fails for the observed reasons.

### Task 2: Canonicalize UI argument groups

**Files:**
- Modify: `toml_training.py`
- Modify: `tests/test_toml_training_cli.py`

**Step 1:** Implement pure helpers for optional-value filtering and nested `key=value` serialization with UI-compatible boolean/list formatting.

**Step 2:** Derive `network_module` for LyCORIS, DyLoRA, LoRA-FA, Flux, Anima, and standard LoRA configurations.

**Step 3:** Qualify known custom optimizer names using a stable registry mapping while preserving built-in and already-qualified names.

**Step 4:** Run the focused tests and confirm they pass.

### Task 3: Canonicalize and validate datasets

**Files:**
- Modify: `toml_training.py`
- Modify: `tests/test_toml_training_cli.py`

**Step 1:** Add failing tests for UI-only subset metadata and an unknown subset key.

**Step 2:** Implement an explicit backend-compatible subset-key contract, removing presentation metadata and rejecting unknown keys with a field-specific error.

**Step 3:** Preserve Unicode paths, valid augmentations, per-subset overrides, and general dataset settings.

**Step 4:** Run focused tests and confirm they pass.

### Task 4: Cover derived settings and launcher validation

**Files:**
- Modify: `toml_training.py`
- Modify: `train_from_toml.py`
- Modify: `tests/test_toml_training_cli.py`
- Modify: `tests/test_train_from_toml.py`

**Step 1:** Add failing tests for warmup conversion, model-family inference/conflicts, invalid process counts, and normalization errors preventing launch.

**Step 2:** Implement derived warmup/restart values using the saved step/batch/process inputs where deterministic.

**Step 3:** Validate explicit CLI family flags against the TOML and reject conflicting or unsupported combinations before writing/launching.

**Step 4:** Run focused tests and confirm they pass.

### Task 5: Verify with the real preset and document behavior

**Files:**
- Modify: `README.md`
- Modify: `tests/test_toml_training_cli.py`

**Step 1:** Add a regression that normalizes `tomlpresets/testing32gbvram5090.toml` without mutating it and asserts backend-ready output.

**Step 2:** Document that UI-saved TOMLs require no manual cleanup and that dry-run performs conversion/validation only.

**Step 3:** Run `pytest tests/test_toml_training_cli.py tests/test_train_from_toml.py -q`.

**Step 4:** Run the broader feasible test suite and report environment-dependent failures separately.

### Task 6: Merge into refresh

**Files:** None.

**Step 1:** Review `git diff`, confirm no unrelated backend/submodule content was overwritten, and keep changes uncommitted unless requested.

**Step 2:** Merge `feat/toml-training-cli` into `refresh` using a working-tree-safe sequence.

**Step 3:** Re-run focused tests on `refresh` and verify the resulting branch contains the CLI fixes.

